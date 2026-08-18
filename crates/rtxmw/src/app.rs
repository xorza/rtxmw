//! winit event loop: window lifecycle, input, and driving the renderer.

use std::time::Instant;

use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use rtxmw_scene::{CellId, CellStreamer, LoadedCell};

use crate::WindowOptions;
use crate::camera::{Camera, Movement};
use crate::renderer::Renderer;
use crate::scene_loader;

/// Starting resolution — the internal render target from the design's performance budget.
const INITIAL_SIZE: (u32, u32) = (1920, 1080);

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
    /// Cells on the device, so the window knows what it already has.
    resident: Vec<CellId>,
    /// Cells asked for and not yet arrived, so the window does not ask twice while one is in
    /// flight. Cleared of a cell however it turns out, so a square that is open sea is asked for
    /// again the next time the window moves over it — which costs an index lookup and no file.
    requested: Vec<CellId>,
    /// Scratch for the wanted set, refilled rather than reallocated. Only refilled when the camera
    /// crosses into another cell, which is what keeps sorting 49 of them off the frame path.
    wanted: Vec<CellId>,
    streamer: CellStreamer,
    /// Frames to draw before exiting, for scripting the windowed path.
    exit_after: Option<u32>,
    frames_drawn: u32,
    camera: Camera,
    keys: Keys,
    /// Mouse look is only applied while the cursor is captured.
    mouse_captured: bool,
    last_frame: Instant,
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
        let mut changed = self.take_one_loaded();

        // The window only moves when the camera crosses into another cell, which is also what
        // keeps this frame-path free of the work — and the allocation — of sorting 49 cells.
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

    /// Uploads one cell that finished loading, if any has. Returns whether anything changed.
    fn take_one_loaded(&mut self) -> bool {
        let Some(ready) = self.streamer.take_ready() else {
            return false;
        };
        self.requested.retain(|id| *id != ready.id);
        let loaded = match ready.loaded {
            Ok(loaded) => loaded,
            // Most often a grid square that is open sea, which has no record at all. Asked for
            // again the next time the window moves over it, and cheaply: the cell index answers
            // "no such cell" without touching the file.
            Err(e) => {
                eprintln!("could not load {}: {e}", ready.id);
                return false;
            }
        };
        let Some(renderer) = self.renderer.as_mut() else {
            return false;
        };
        match renderer.add_cell(loaded.id.clone(), &loaded.scene, &loaded.textures) {
            Ok(()) => {
                self.resident.push(loaded.id);
                true
            }
            Err(e) => {
                eprintln!("could not upload {}: {e}", loaded.id);
                false
            }
        }
    }

    /// Drops cells further from `centre` than the window keeps. Returns whether anything changed.
    fn evict_beyond(&mut self, centre: &CellId) -> bool {
        let Some(renderer) = self.renderer.as_mut() else {
            return false;
        };
        let before = self.resident.len();
        self.resident.retain(|id| {
            let keep = scene_loader::rings_from(centre, id)
                .is_none_or(|rings| rings <= scene_loader::KEEP_RADIUS);
            if !keep {
                renderer.remove_cell(id);
            }
            keep
        });
        before != self.resident.len()
    }

    /// Asks for whichever cells around `centre` are neither resident nor already on the way.
    fn request_missing(&mut self, centre: &CellId) {
        scene_loader::wanted_cells(centre, &mut self.wanted);
        for id in &self.wanted {
            if self.resident.contains(id) || self.requested.contains(id) {
                continue;
            }
            self.streamer.request(id.clone());
            self.requested.push(id.clone());
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
            exit_after: options.exit_after,
            ..Self::default()
        }
    }
}

impl Default for App {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            cell: scene_loader::cell_argument(None),
            renderer: None,
            window: None,
            centre: None,
            resident: Vec::new(),
            requested: Vec::new(),
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
        match Renderer::new(&window, size.width, size.height) {
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
                self.resident.push(cell.id);
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
