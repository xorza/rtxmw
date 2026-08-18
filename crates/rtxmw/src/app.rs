//! winit event loop: window lifecycle, input, and driving the renderer.

use std::time::Instant;

use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use rtxmw_scene::{CellId, LoadedCell};

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
    /// The exterior block currently resident, so the grid only reloads when the camera leaves it.
    /// `None` indoors, where there is nothing to stream.
    loaded_centre: Option<CellId>,
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
    /// Recentres the loaded block when the camera has settled in a different cell.
    ///
    /// Synchronous, so the reload is a visible hitch — about 30 ms of file work plus the
    /// acceleration structure rebuild. Doing it off the main thread is what M9's "no hitching"
    /// asks for and is the next piece; this is the part that makes the world traversable at all.
    fn stream_grid(&mut self) {
        let Some(centre) = &self.loaded_centre else {
            return;
        };
        let Some(CellId::Exterior { x, y }) =
            scene_loader::next_centre(self.camera.position(), centre)
        else {
            return;
        };

        let started = Instant::now();
        match LoadedCell::load_exterior_grid(x, y, scene_loader::GRID_RADIUS) {
            Ok(Some(cell)) => {
                let Some(renderer) = self.renderer.as_mut() else {
                    return;
                };
                if let Err(e) = renderer.load_scene(&cell.scene, &cell.textures) {
                    eprintln!("could not upload {}: {e}", cell.id);
                    return;
                }
                println!(
                    "streamed to {} in {:.0} ms: {} instances",
                    cell.id,
                    started.elapsed().as_secs_f32() * 1000.0,
                    cell.scene.instances.len()
                );
                self.loaded_centre = Some(cell.id);
            }
            // Off the edge of the world, where no cell record exists. The block stays where it is
            // rather than emptying, so flying out to sea leaves the coast behind rather than
            // nothing at all.
            Ok(None) => {}
            Err(e) => eprintln!("could not stream to ({x}, {y}): {e}"),
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
            loaded_centre: None,
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
            "rtxmw — {fps:.0} fps — {:.0}, {:.0}, {:.0} — {:.1} m/s — {device}",
            position.x,
            position.y,
            position.z,
            self.camera.speed(),
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
        match LoadedCell::load_at(self.cell.clone(), scene_loader::GRID_RADIUS) {
            Ok(Some(cell)) => {
                let renderer = self.renderer.as_mut().expect("renderer was just created");
                if let Err(e) = renderer.load_scene(&cell.scene, &cell.textures) {
                    eprintln!("could not upload {}: {e}", cell.id);
                    event_loop.exit();
                    return;
                }
                println!("{}", scene_loader::describe(&cell));
                self.camera = scene_loader::Viewpoint::entering(&cell).camera();
                self.loaded_centre = matches!(cell.id, CellId::Exterior { .. }).then_some(cell.id);
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
                self.stream_grid();

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
