//! winit event loop: window lifecycle, input, and driving the renderer.

use std::collections::HashMap;
use std::time::Instant;

use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use rtxmw_scene::{CellDetail, CellId, CellStreamer, LoadedCell, SceneError};

use crate::camera::{Camera, Movement};
use crate::cli::{Upscaling, WindowOptions};
use crate::renderer::Renderer;
use crate::scene_loader::{self, WantedCell};

/// Starting resolution — the internal render target from the design's performance budget.
const INITIAL_SIZE: (u32, u32) = (1920, 1080);

/// Cells taken from the streamer in one frame.
///
/// One at a time was right when the window was 49 cells; the distant ring is six hundred more, and
/// at one a frame the horizon would take ten seconds to arrive. What made one a time the rule was
/// the top-level rebuild, and that happens once per frame however many cells landed in it — so this
/// buys the whole ring for a handful of extra uploads on the frames that take them.
const CELLS_PER_FRAME: u32 = 16;

/// What came of taking one cell off the streamer.
///
/// Three outcomes rather than two because most of what the horizon asks for is open sea, and
/// "nothing arrived" and "what arrived was nothing" have to be told apart: stopping on the second
/// would drain one grid square a frame across six hundred of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Taken {
    /// Nothing had finished loading.
    Nothing,
    /// A cell arrived and was not placed — open sea, or a cell that failed to load or upload.
    Skipped,
    /// A cell arrived and is now on the device.
    Placed,
}

/// Held keys, tracked as flags so held-down movement is smooth rather than repeat-driven.
#[derive(Debug, Default)]
struct Keys {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    boost: bool,
}

impl Keys {
    fn movement(&self) -> Movement {
        Movement {
            forward: f32::from(self.forward) - f32::from(self.back),
            right: f32::from(self.right) - f32::from(self.left),
            up: f32::from(self.up) - f32::from(self.down),
            boost: if self.boost { 8.0 } else { 1.0 },
        }
    }
}

/// Application state for the winit event loop.
#[derive(Debug)]
pub(crate) struct App {
    /// What DLSS was asked to run at, kept because the renderer is built when the window appears
    /// rather than when the arguments are read.
    dlss: Upscaling,
    /// How much baked lighting to divide out, held for the renderer built when the window appears.
    delight: f32,
    /// How much of the cell's fog to apply, held for the renderer built when the window appears.
    fog: f32,
    /// Which cell to open in, from the command line.
    cell: CellId,
    /// **Before `window`, and that is load-bearing.** Fields drop in declaration order, and the
    /// renderer holds a `VkSurfaceKHR` and a swapchain built from this window — destroying the
    /// window first leaves the WSI layer dereferencing a freed surface as it tears them down, which
    /// segfaulted on exit. `Renderer` carries the same warning about its own fields; this is the
    /// same rule one level up.
    renderer: Option<Renderer>,
    window: Option<Window>,
    /// The exterior cell the camera was last known to be in, and the window's centre. Unset until
    /// the first frame outdoors, which is what makes that frame ask for the window.
    centre: Option<CellId>,
    /// Cells on the device and which tier each is at, so the window knows what it already has.
    ///
    /// A map rather than a list because the distant ring makes it six hundred entries, and the
    /// wanted set is scanned against it every time the camera crosses a boundary.
    resident: HashMap<CellId, CellDetail>,
    /// Cells asked for and not yet arrived, so the window does not ask twice while one is in
    /// flight. Cleared of a cell however it turns out, so a square that is open sea is asked for
    /// again the next time the window moves over it — which costs an index lookup and no file.
    requested: HashMap<CellId, CellDetail>,
    /// Scratch for the wanted set, refilled rather than reallocated. Only refilled when the camera
    /// crosses into another cell, which is what keeps sorting six hundred of them off the frame
    /// path.
    wanted: Vec<WantedCell>,
    streamer: CellStreamer,
    /// Frames to draw before exiting, for scripting the windowed path.
    exit_after: Option<u32>,
    frames_drawn: u32,
    camera: Camera,
    keys: Keys,
    /// Mouse look is only applied while the cursor is captured.
    mouse_captured: bool,
    last_frame: Instant,
    /// When the engine started, which is the clock the water's waves move against.
    started: Instant,
    last_title_update: Instant,
    frames_since_title: u32,
}

impl App {
    /// Keeps the cells around the camera resident, loading and evicting one at a time.
    ///
    /// Everything expensive is elsewhere: the reading and decoding happen on the streamer's thread,
    /// and what lands here is one cell's upload and one rebuild of the top level over what is
    /// placed. Sharing is what makes that small — a cell arriving next to forty-eight others names
    /// meshes and textures the device already holds, and uploads almost nothing.
    ///
    /// One cell per frame, deliberately. A camera crossing a boundary wants a whole column, and
    /// taking them all in one frame would rebuild the same structure seven times over while the
    /// frame it interrupted waited.
    fn stream_cells(&mut self) {
        // A door is the only way out of an interior, so nothing streams around one — and a
        // position inside a building does not meaningfully name a grid square anyway.
        if !matches!(self.cell, CellId::Exterior { .. }) {
            return;
        }
        let mut changed = false;
        for _ in 0..CELLS_PER_FRAME {
            match self.take_one_loaded() {
                Taken::Nothing => break,
                Taken::Skipped => {}
                Taken::Placed => changed = true,
            }
        }

        // The window only moves when the camera crosses into another cell, which is also what
        // keeps this frame-path free of the work — and the allocation — of sorting six hundred.
        let position = self.camera.position();
        let centre = CellId::containing(position.x, position.y);
        if self.centre.as_ref() != Some(&centre) {
            changed |= self.evict_beyond(&centre);
            self.request_missing(&centre);
            self.centre = Some(centre);
        }

        if changed
            && let Some(renderer) = self.renderer.as_mut()
            && let Err(e) = renderer.commit()
        {
            eprintln!("could not rebuild the scene: {e}");
        }
    }

    /// Uploads one cell that finished loading, if any has.
    fn take_one_loaded(&mut self) -> Taken {
        let Some(ready) = self.streamer.take_ready() else {
            return Taken::Nothing;
        };
        // Only when it answers what was last asked. A cell whose tier changed while it was in
        // flight has a newer request outstanding, and that one is what clears the entry.
        if self.requested.get(&ready.id) == Some(&ready.detail) {
            self.requested.remove(&ready.id);
        }
        let loaded = match ready.loaded {
            Ok(loaded) => loaded,
            // A grid square that is open sea has no record at all, and most of the six hundred the
            // horizon asks for are exactly that — so it is silent rather than a line each. Asked
            // for again the next time the window moves over it, and cheaply: the cell index
            // answers without touching the file.
            Err(SceneError::NoSuchCell(_)) => return Taken::Skipped,
            Err(e) => {
                eprintln!("could not load {}: {e}", ready.id);
                return Taken::Skipped;
            }
        };
        let Some(renderer) = self.renderer.as_mut() else {
            return Taken::Skipped;
        };
        // A cell changing tier arrives as a second copy of itself, so the one it replaces goes
        // first — otherwise the coarse terrain would sit inside the detailed terrain and every ray
        // near the ground would hit whichever the traversal reached. It goes first *here*, on
        // arrival, rather than when the camera crossed: the coarse copy is better than the hole
        // that dropping it early would leave.
        if self.resident.remove(&loaded.id).is_some() {
            renderer.remove_cell(&loaded.id);
        }
        match renderer.add_cell(loaded.id.clone(), &loaded.scene, &loaded.textures) {
            Ok(()) => {
                self.resident.insert(loaded.id, ready.detail);
                Taken::Placed
            }
            Err(e) => {
                eprintln!("could not upload {}: {e}", loaded.id);
                Taken::Skipped
            }
        }
    }

    /// Drops cells that have left the window altogether. Returns whether anything changed.
    fn evict_beyond(&mut self, centre: &CellId) -> bool {
        let Some(renderer) = self.renderer.as_mut() else {
            return false;
        };
        let before = self.resident.len();
        self.resident.retain(|id, _| {
            let keep =
                scene_loader::rings_from(centre, id).is_none_or(scene_loader::still_resident);
            if !keep {
                renderer.remove_cell(id);
            }
            keep
        });
        before != self.resident.len()
    }

    /// Asks for whichever cells around `centre` are missing, or resident at the wrong tier.
    fn request_missing(&mut self, centre: &CellId) {
        scene_loader::wanted_cells(centre, &mut self.wanted);
        for wanted in &self.wanted {
            let asked = match self.resident.get(&wanted.id) {
                Some(&resident) => match scene_loader::rebuild_as(resident, wanted.rings) {
                    Some(tier) => tier,
                    None => continue,
                },
                None => scene_loader::detail_at(wanted.rings),
            };
            if self.requested.get(&wanted.id) == Some(&asked) {
                continue;
            }
            self.streamer.request(wanted.id.clone(), asked);
            self.requested.insert(wanted.id.clone(), asked);
        }
    }

    /// An app configured from the command line.
    ///
    /// The frame limit exists because the shutdown path had no way to be exercised: the crash it
    /// was hiding — the window being destroyed before the surface built from it — only happens on a
    /// clean exit, which nothing but a person pressing a key could reach.
    pub(crate) fn opening_in(options: WindowOptions) -> Self {
        Self {
            cell: options.cell,
            dlss: options.dlss,
            delight: options.delight,
            fog: options.fog,
            exit_after: options.exit_after,
            ..Self::default()
        }
    }
}

impl Default for App {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            cell: scene_loader::cell_named(scene_loader::DEFAULT_CELL),
            dlss: Upscaling(None),
            delight: 1.0,
            fog: 1.0,
            renderer: None,
            window: None,
            centre: None,
            resident: HashMap::new(),
            requested: HashMap::new(),
            wanted: Vec::new(),
            streamer: CellStreamer::spawn(),
            exit_after: None,
            frames_drawn: 0,
            // Replaced by the loaded cell's own centre in `resumed`; this only matters if no game
            // data is configured and there is nothing to look at anyway.
            camera: Camera::new(Vec3::ZERO),
            keys: Keys::default(),
            mouse_captured: false,
            last_frame: now,
            started: now,
            last_title_update: now,
            frames_since_title: 0,
        }
    }
}

impl App {
    fn set_capture(&mut self, captured: bool) {
        let Some(window) = &self.window else {
            return;
        };
        // Wayland only offers locked; X11 only offers confined. Try both before giving up.
        let mode = if captured {
            window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
        } else {
            window.set_cursor_grab(CursorGrabMode::None)
        };
        if mode.is_err() && captured {
            return;
        }
        window.set_cursor_visible(!captured);
        self.mouse_captured = captured;
    }

    fn update_title(&mut self) {
        let Some(window) = &self.window else {
            return;
        };
        let elapsed = self.last_title_update.elapsed();
        if elapsed.as_secs_f32() < 0.5 {
            return;
        }

        let fps = self.frames_since_title as f32 / elapsed.as_secs_f32();
        let position = self.camera.position();
        let device = self
            .renderer
            .as_ref()
            .map(Renderer::device_name)
            .unwrap_or("no device");
        window.set_title(&format!(
            "rtxmw — {fps:.0} fps — {:.0}, {:.0}, {:.0} — {:.1} m/s — {} cells — {device}",
            position.x,
            position.y,
            position.z,
            self.camera.speed(),
            self.resident.len(),
        ));

        self.last_title_update = Instant::now();
        self.frames_since_title = 0;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("rtxmw")
            .with_inner_size(winit::dpi::PhysicalSize::new(
                INITIAL_SIZE.0,
                INITIAL_SIZE.1,
            ));
        let window = event_loop
            .create_window(attributes)
            .expect("failed to create window");

        let size = window.inner_size();
        match Renderer::new(
            &window,
            size.width,
            size.height,
            self.dlss,
            self.delight,
            self.fog,
        ) {
            Ok(renderer) => {
                println!("{}", renderer.capability_report());
                self.renderer = Some(renderer);
            }
            Err(e) => {
                eprintln!("{e}");
                event_loop.exit();
                return;
            }
        }

        // Content after the device, because uploading needs one. A missing install is not fatal:
        // the window still comes up and reports the device, which is what makes it obvious that the
        // path is what is wrong rather than the GPU.
        match LoadedCell::load_at(self.cell.clone()) {
            Ok(Some(cell)) => {
                let renderer = self.renderer.as_mut().expect("renderer was just created");
                if let Err(e) = renderer.load_scene(cell.id.clone(), &cell.scene, &cell.textures) {
                    eprintln!("could not upload {}: {e}", cell.id);
                    event_loop.exit();
                    return;
                }
                println!("{}", scene_loader::describe(&cell));
                self.camera = scene_loader::Viewpoint::entering(&cell).camera();
                // The cell the camera opens in is resident; outdoors, the rest of the window
                // streams in around it over the following frames. `centre` stays unset so the
                // first of those frames sees the camera's cell as a change and asks for the rest.
                self.resident.insert(cell.id, CellDetail::Full);
            }
            Ok(None) => eprintln!(
                "no game data configured — set MORROWIND_DATA_DIR, or put it in .env at the repo root"
            ),
            Err(e) => {
                eprintln!("could not load the cell: {e}");
                event_loop.exit();
                return;
            }
        }

        self.window = Some(window);

        // Device creation takes long enough that a delta measured from `App::default` would make
        // the first frame appear to span the whole startup.
        let now = Instant::now();
        self.last_frame = now;
        self.last_title_update = now;
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(_) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.invalidate_swapchain();
                }
            }

            WindowEvent::MouseInput { state, .. } => {
                if state == ElementState::Pressed && !self.mouse_captured {
                    self.set_capture(true);
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                if scroll != 0.0 {
                    self.camera.scale_speed(1.2f32.powf(scroll));
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                match code {
                    KeyCode::KeyW => self.keys.forward = pressed,
                    KeyCode::KeyS => self.keys.back = pressed,
                    KeyCode::KeyA => self.keys.left = pressed,
                    KeyCode::KeyD => self.keys.right = pressed,
                    KeyCode::Space => self.keys.up = pressed,
                    KeyCode::ControlLeft => self.keys.down = pressed,
                    KeyCode::ShiftLeft => self.keys.boost = pressed,
                    KeyCode::Escape if pressed => {
                        if self.mouse_captured {
                            self.set_capture(false);
                        } else {
                            event_loop.exit();
                        }
                    }
                    _ => {}
                }
            }

            WindowEvent::RedrawRequested => {
                let dt = self.last_frame.elapsed().as_secs_f32();
                self.last_frame = Instant::now();

                self.camera.fly(self.keys.movement(), dt);
                self.stream_cells();

                if let (Some(renderer), Some(window)) = (&mut self.renderer, &self.window) {
                    let size = window.inner_size();
                    renderer.set_time(self.started.elapsed().as_secs_f32());
                    let constants = renderer.frame_constants(
                        self.camera.view(),
                        self.camera.projection(renderer.aspect_ratio()),
                        self.camera.position(),
                    );
                    if let Err(e) = renderer.draw(size.width, size.height, &constants) {
                        eprintln!("draw failed: {e}");
                        event_loop.exit();
                        return;
                    }
                }

                self.frames_drawn += 1;
                if self
                    .exit_after
                    .is_some_and(|limit| self.frames_drawn >= limit)
                {
                    event_loop.exit();
                    return;
                }

                self.frames_since_title += 1;
                self.update_title();
            }

            _ => {}
        }
    }

    fn device_event(&mut self, _loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event
            && self.mouse_captured
        {
            self.camera.look(delta.0 as f32, delta.1 as f32);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}
