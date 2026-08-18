//! winit event loop: window lifecycle, input, and driving the renderer.

use std::time::Instant;

use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::camera::{Camera, Movement};
use crate::renderer::Renderer;
use rtxmw_render::FrameConstants;

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
    window: Option<Window>,
    renderer: Option<Renderer>,
    camera: Camera,
    keys: Keys,
    /// Mouse look is only applied while the cursor is captured.
    mouse_captured: bool,
    last_frame: Instant,
    last_title_update: Instant,
    frames_since_title: u32,
}

impl Default for App {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            window: None,
            renderer: None,
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
        match scene_loader::load_default_cell() {
            Ok(Some(loaded)) => {
                let renderer = self.renderer.as_mut().expect("renderer was just created");
                if let Err(e) = renderer.load_scene(&loaded.scene, &loaded.textures) {
                    eprintln!("could not upload {}: {e}", loaded.name);
                    event_loop.exit();
                    return;
                }
                println!(
                    "{}: {} meshes, {} instances, {} lights",
                    loaded.name,
                    loaded.scene.meshes.len(),
                    loaded.scene.instances.len(),
                    loaded.scene.lights.len()
                );
                self.camera = Camera::new(loaded.viewpoint);
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

                if let (Some(renderer), Some(window)) = (&mut self.renderer, &self.window) {
                    let size = window.inner_size();
                    let constants = FrameConstants::new(
                        self.camera.view(),
                        self.camera.projection(renderer.aspect_ratio()),
                        self.camera.position(),
                        renderer.ambient(),
                        renderer.light_count(),
                    );
                    if let Err(e) = renderer.draw(size.width, size.height, &constants) {
                        eprintln!("draw failed: {e}");
                        event_loop.exit();
                        return;
                    }
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
