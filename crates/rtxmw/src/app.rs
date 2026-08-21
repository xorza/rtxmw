//! winit event loop: window lifecycle, input, and driving the renderer.

use std::collections::HashMap;
use std::time::Instant;

use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use rtxmw_scene::{
    CellDetail, CellId, CellStreamer, CloudSheet, LoadedCell, SceneError, Sky, Weather, WorldTime,
};

use crate::camera::{Camera, Movement};
use crate::cli::{Upscaling, WindowOptions};
use crate::renderer::{Conditions, Renderer};
use crate::scene_loader::{self, WantedCell};
use crate::world_clock::{ClockFace, WorldClock};

/// What the keys below do, for whoever is looking at the window rather than at this file.
///
/// Here rather than beside the clock they mostly drive, because the bindings are here: a key named
/// in one file and matched in another rots the moment either moves.
const KEYS: &str = concat!(
    "keys: WASD, space and ctrl to fly \u{b7} shift to hurry \u{b7} ",
    "[ ] time speed \u{b7} \\ pause time \u{b7} , . step the hour by half \u{b7} ",
    "; ' cycle the weather this region has \u{b7} k a new bolt \u{b7} l the same one again \u{b7} ",
    "p print this spot as arguments \u{b7} esc to release the mouse",
);

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
    /// How far painted relief tilts the normal, held for the renderer built when the window appears.
    relief: f32,
    /// How much of the cell's fog to apply, held for the renderer built when the window appears.
    fog: f32,
    /// How long the world has run and what hour it is there, which the keys below drive.
    clock: WorldClock,
    /// Which of the game's ten weathers the world is under, out of `--weather` and the keys.
    weather: Weather,
    /// The ones the region the camera is standing in allows, in the game's own order.
    ///
    /// **Refilled when the camera crosses into another cell**, because which weathers a place can
    /// have is a property of where you are: the Bitter Coast fogs and rains and never sees ash, and
    /// walking east into the Ashlands changes the answer. Empty until the first cell is resident,
    /// and the whole ten wherever nothing narrows them — see [`Weather::in_cell`].
    allowed: Vec<Weather>,
    /// What that weather's cloud sheet comes to on average, once the renderer has read it.
    ///
    /// Every sky is built with it — how much of the dome the deck hides is what the ground under it
    /// is dimmed by — so it is held here rather than fetched, and it is `NONE` until the textures
    /// have loaded, which is a sky with no layer in it.
    sheet: CloudSheet,
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
    last_title_update: Instant,
    frames_since_title: u32,
}

/// A place worth profiling, as the two lines `p` writes to standard output.
///
/// **A type rather than a pair of `println!`s in the handler**, so what it writes can be asserted
/// without a window, a device or a cell — the second line is arguments the binary has to be able to
/// read back, and a format that drifts from the parser is not something an eye catches in a log.
#[derive(Debug)]
struct Marker<'a> {
    /// Where this is, as the cell to open rather than the one the window was opened in.
    cell: CellId,
    position: Vec3,
    forward: Vec3,
    time: WorldTime,
    weather: &'a str,
}

impl<'a> Marker<'a> {
    /// Where `camera` is standing, given the cell the window was `opened` in.
    fn new(opened: &CellId, camera: &Camera, time: WorldTime, weather: &'a str) -> Self {
        let position = camera.position();
        Self {
            // **The camera's own square, not the one the window opened in** — the same cell only
            // until you fly out of it, and a marker naming the wrong one loads the wrong scene.
            // An interior has no grid position and stays named.
            cell: match opened {
                CellId::Interior(name) => CellId::Interior(name.clone()),
                CellId::Exterior { .. } => CellId::containing(position.x, position.y),
            },
            position,
            forward: camera.forward(),
            time,
            weather,
        }
    }

    /// How far round the compass the camera faces, in degrees clockwise from north.
    ///
    /// **North is +Y and east is +X**, so the arguments come the other way round from the usual
    /// `atan2`. It is the exact inverse of what a door's stored yaw goes through — `facing_from` in
    /// `rtxmw_scene`'s `door`, which is `(sin, cos, 0)` where a maths library would give
    /// `(cos, sin, 0)`.
    fn bearing(&self) -> f32 {
        self.forward
            .x
            .atan2(self.forward.y)
            .to_degrees()
            .rem_euclid(360.0)
    }

    /// How far above the horizon it looks, in degrees. `forward` is a unit vector, so its Z is the
    /// sine of the angle outright.
    fn climb(&self) -> f32 {
        self.forward.z.asin().to_degrees()
    }
}

impl std::fmt::Display for Marker<'_> {
    /// A line for a person and a line for the binary, in that order.
    ///
    /// The first is a comment, so a file of these can be fed to something that skips them. The
    /// second pastes after `--screenshot out.png` and renders this exact frame: which cell, where
    /// in it, facing where, at what hour and under what weather — everything that decides what the
    /// frame costs. What DLSS runs at is deliberately absent, being what a profiling run varies.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "# {} at {:.0}, {:.0}, {:.0} — bearing {:.0}°, climb {:.0}° — {}, {}",
            self.cell,
            self.position.x,
            self.position.y,
            self.position.z,
            self.bearing(),
            self.climb(),
            ClockFace(self.time),
            self.weather,
        )?;
        // How a cell is *written* rather than how it is shown: `Display` brackets an exterior for a
        // person to read, and the argument parser takes a bare pair. An interior's name is quoted
        // because they carry spaces and commas, and a comma unquoted reads as a grid position.
        match &self.cell {
            CellId::Interior(name) => write!(f, "{name:?}")?,
            CellId::Exterior { x, y } => write!(f, "{x},{y}")?,
        }
        write!(
            f,
            " --at {:.1},{:.1},{:.1} --look {:.4},{:.4},{:.4} --time {:.3} --weather {}",
            self.position.x,
            self.position.y,
            self.position.z,
            self.forward.x,
            self.forward.y,
            self.forward.z,
            // **The date as well as the clock face, which is what `--time` past 24 hours means.**
            // Two frames a day apart are lit alike and cost differently, because the moon that is
            // up has moved on a phase — and the running total is the only thing that says which
            // day this is.
            self.time.day() * 24.0,
            self.weather,
        )
    }
}

/// Which of `count` entries lands `by` places from `standing`, wrapping at both ends.
///
/// `standing` is `None` for a weather that is not in the list at all — which is what `--weather`
/// naming one the region never has leaves behind, and what walking out from under a storm does.
/// Stepping then enters the list at whichever end the step came from rather than refusing to move.
fn stepped(standing: Option<usize>, by: isize, count: usize) -> usize {
    debug_assert!(count > 0, "there is no entry to step to");
    match standing {
        Some(at) => (at as isize + by).rem_euclid(count as isize) as usize,
        None if by > 0 => 0,
        None => count - 1,
    }
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
            self.follow_region(&centre);
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

    /// Takes the weathers `cell`'s region allows, for the keys below to cycle through.
    ///
    /// **What the standing weather is does not change here**, however far the camera has walked. A
    /// blight storm that blew in over the Ashlands does not stop when you reach the coast — the
    /// region says what can *begin* here, and stepping the keys is what begins one.
    ///
    /// **Rebuilt per crossing rather than per region, which was measured rather than assumed.** The
    /// ten come out of the ini every time at **66 us**, against a crossing that already loads a
    /// cell, uploads it and rebuilds the top level — so remembering the last region to skip the
    /// work would be a cache for a thousandth of what the frame it lands in already costs.
    fn follow_region(&mut self, cell: &CellId) {
        match Weather::in_cell(cell) {
            Ok(allowed) => self.allowed = allowed,
            // The list is what the keys offer, so failing to read it costs the keys and nothing
            // else — the sky overhead is untouched and the session carries on.
            Err(failed) => eprintln!("could not read what weather this region has: {failed}"),
        }
    }

    /// Steps `by` places through the weathers this region allows, wrapping at either end.
    ///
    /// **From wherever the standing weather sits in that list, and from the end when it sits
    /// nowhere in it** — which is what `--weather blight` in a region that never blights leaves
    /// behind, and what walking out of the Ashlands under one does. Stepping then lands on the
    /// list's own first or last entry rather than refusing to move.
    fn cycle_weather(&mut self, by: isize) {
        let count = self.allowed.len();
        if count == 0 {
            return;
        }
        // **By name rather than by identity**, because the standing weather need not be one of
        // these at all — `--weather` can name one the region never has, and walking out from under
        // a storm leaves one behind.
        let standing = self
            .allowed
            .iter()
            .position(|weather| weather.name == self.weather.name);
        self.weather = self.allowed[stepped(standing, by, count)].clone();
        // The deck is a texture on the device, so the sheet has to be read before the next sky can
        // be built with it — and what it came to is asked for the same way `resumed` asks after the
        // renderer is built, rather than handed back by a second route.
        if let Some(renderer) = self.renderer.as_mut() {
            if let Err(failed) = renderer.set_weather(&self.weather) {
                eprintln!("could not put the weather's clouds on the device: {failed}");
            }
            self.sheet = renderer.cloud_sheet();
        }
        println!(
            "weather: {} (of {count} this region has)",
            self.weather.name
        );
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
            // An unrecognised name is clear rather than a refusal to start — `Weather::named` says
            // so, and a missing install cannot name anything at all.
            weather: Weather::named(&options.weather).unwrap_or_else(|_| Weather::clear()),
            cell: options.cell,
            dlss: options.dlss,
            delight: options.delight,
            relief: options.relief,
            fog: options.fog,
            clock: WorldClock::starting_at(options.time),
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
            relief: 1.0,
            fog: 1.0,
            clock: WorldClock::starting_at(WorldTime::default()),
            weather: Weather::clear(),
            allowed: Vec::new(),
            sheet: CloudSheet::NONE,
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

    /// Writes where the camera is standing to standard output — see [`Marker`].
    fn mark_viewpoint(&self) {
        let time = self.clock.time();
        println!(
            "{}",
            Marker::new(&self.cell, &self.camera, time, &self.weather.name)
        );
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
            "rtxmw — {fps:.0} fps — {} — {:.0}, {:.0}, {:.0} — {:.1} m/s — {} cells — {device}",
            self.clock,
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
            Conditions {
                delight: self.delight,
                relief: self.relief,
                fog: self.fog,
                sky: Sky::under(self.clock.time(), &self.weather, self.sheet),
                weather: &self.weather,
            },
        ) {
            Ok(renderer) => {
                // The sheet is only known once the archives have been read, which is inside the
                // renderer — so the first sky was built without a layer and every later one has one.
                self.sheet = renderer.cloud_sheet();
                println!("{}", renderer.capability_report());
                println!("{KEYS}");
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
                //
                // **Which is also what makes this call the interior's**: nothing streams around a
                // room, so the crossing that fills this outdoors never happens in one.
                self.follow_region(&cell.id);
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
                    // **Time.** The fog drifts against the same clock the sun climbs on, so one
                    // speed carries both — and the nudges move the hour alone, which is how two
                    // times of day are compared without the fog having blown somewhere else in
                    // between.
                    //
                    // These take the key's repeat as well as its press, so holding one scrubs
                    // rather than stepping — except the pause, which a repeat would flicker on and
                    // off thirty times a second.
                    KeyCode::BracketRight if pressed => self.clock.step_speed(1.0),
                    KeyCode::BracketLeft if pressed => self.clock.step_speed(-1.0),
                    KeyCode::Backslash if pressed && !event.repeat => self.clock.toggle_pause(),
                    KeyCode::Period if pressed => self.clock.nudge(1.0),
                    KeyCode::Comma if pressed => self.clock.nudge(-1.0),
                    // **Weather**, in the same punctuation row as the two time pairs because it is
                    // the same kind of thing: state of the world rather than of the camera.
                    //
                    // The repeat is dropped, which the time keys keep. Each step reads a cloud
                    // sheet onto the device and waits for the device to go idle to do it, so a held
                    // key would stall the frame thirty times a second to cycle past weathers
                    // nobody saw.
                    KeyCode::Quote if pressed && !event.repeat => self.cycle_weather(1),
                    KeyCode::Semicolon if pressed && !event.repeat => self.cycle_weather(-1),
                    // Nothing to hold: the strike moves the storm's own clock onto a flash, so the
                    // weather carries on from there rather than the key having to be remembered.
                    // **A new discharge, and the same one again.** Both move the storm's own clock
                    // rather than remembering a flash, so what comes back is drawn from the second it
                    // happened in — which is why the second key can exist at all.
                    KeyCode::KeyK if pressed && !event.repeat => {
                        let (eye, facing) = (self.camera.position(), self.camera.forward());
                        if let Some(renderer) = self.renderer.as_mut() {
                            renderer.strike(eye, facing);
                        }
                    }
                    KeyCode::KeyL if pressed && !event.repeat => {
                        if let Some(renderer) = self.renderer.as_mut() {
                            renderer.restrike();
                        }
                    }
                    // Nothing in the world changes, so the repeat is dropped only to keep a held
                    // key from filling the list with the same spot thirty times.
                    KeyCode::KeyP if pressed && !event.repeat => self.mark_viewpoint(),
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

                self.clock.advance(dt);

                if let (Some(renderer), Some(window)) = (&mut self.renderer, &self.window) {
                    let size = window.inner_size();
                    renderer.set_time(self.clock.seconds());
                    renderer.set_storm(self.clock.weather_seconds());
                    renderer.set_sky(Sky::under(self.clock.time(), &self.weather, self.sheet));
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

#[cfg(test)]
mod tests {
    use crate::cli::{Command, ScreenshotOptions};

    use super::*;

    #[test]
    fn stepping_through_a_regions_weathers_wraps_at_both_ends() {
        // The Bitter Coast's five, which is the list the keys actually cycle.
        const FIVE: usize = 5;
        assert_eq!(stepped(Some(0), 1, FIVE), 1);
        assert_eq!(stepped(Some(3), 1, FIVE), 4);
        // Off the top comes back to the bottom, and off the bottom to the top — `rem_euclid` rather
        // than `%`, which for a negative step would give a negative index.
        assert_eq!(stepped(Some(4), 1, FIVE), 0);
        assert_eq!(stepped(Some(0), -1, FIVE), 4);
        assert_eq!(stepped(Some(2), -1, FIVE), 1);

        // **A weather the region does not have is not in the list**, so a step enters at the end it
        // came from: forwards lands on the first, backwards on the last.
        assert_eq!(stepped(None, 1, FIVE), 0);
        assert_eq!(stepped(None, -1, FIVE), FIVE - 1);

        // A region with one weather goes nowhere, which beats going out of bounds.
        assert_eq!(stepped(Some(0), 1, 1), 0);
        assert_eq!(stepped(Some(0), -1, 1), 0);
    }

    /// The second line of a marker is arguments, and this is what says they are still readable.
    ///
    /// **Parsed back rather than only compared**, because a string that looks right and a string
    /// the binary accepts are different claims — the whole point of the line is that it can be
    /// pasted after `--screenshot`.
    fn arguments_of(marker: &Marker) -> ScreenshotOptions {
        let printed = marker.to_string();
        let (comment, arguments) = printed.split_once('\n').expect("a marker is two lines");
        assert!(comment.starts_with("# "), "the first line is a comment");
        // **A size before the cell, which is the shape the line is written to sit in**: the
        // screenshot command takes both positionally and in that order, so a marker pasted without
        // one would have its cell read as the size.
        let mut words = ["rtxmw", "--screenshot", "out.png", "1920x1080"]
            .map(str::to_owned)
            .to_vec();
        // The cell is one word however many spaces its name holds, which is what the quotes on it
        // are for — a shell would strip them, and this stands in for the shell.
        let (cell, flags) = match arguments.strip_prefix('"') {
            Some(quoted) => {
                let (name, rest) = quoted.split_once('"').expect("a quote is closed");
                (name.to_owned(), rest)
            }
            None => {
                let (bare, rest) = arguments
                    .split_once(' ')
                    .expect("a cell is followed by flags");
                (bare.to_owned(), rest)
            }
        };
        words.push(cell);
        words.extend(flags.split_whitespace().map(str::to_owned));
        match Command::parse_from(&words) {
            Command::Screenshot(options) => options,
            other => panic!("a marker should read back as a screenshot, not {other:?}"),
        }
    }

    #[test]
    fn a_marked_spot_prints_where_it_is_and_reads_back_as_arguments() {
        // A 3-4-5 triangle laid on its side: three east and four down over a hypotenuse of five, so
        // the direction is exactly (0.6, 0, -0.8) and the angles are hand-checkable — due east is a
        // bearing of 90, and a climb of `asin(-0.8)` is -53.13 degrees.
        let camera = Camera::looking(Vec3::new(12.3, -45.6, 78.9), Vec3::new(3.0, 0.0, -4.0));
        // Past a day, which is what says the marker carries the date: 38.25 hours is quarter past
        // two in the afternoon of the second day, and the moon over it is not the first day's.
        let indoors = CellId::Interior("Balmora, Guild of Mages".to_owned());
        let marker = Marker::new(&indoors, &camera, WorldTime::hours(38.25), "rain");
        assert_eq!(
            marker.to_string(),
            "# Balmora, Guild of Mages at 12, -46, 79 — bearing 90°, climb -53° — 14:15, rain\n\
             \"Balmora, Guild of Mages\" --at 12.3,-45.6,78.9 --look 0.6000,0.0000,-0.8000 \
             --time 38.250 --weather rain"
        );
        let read = arguments_of(&marker);
        assert_eq!(read.cell, indoors);
        assert_eq!(read.viewpoint.position, Some(Vec3::new(12.3, -45.6, 78.9)));
        assert_eq!(read.weather, "rain");
        assert_eq!(read.time, WorldTime::hours(38.25));

        // **Outdoors the marker names the square the camera is standing in**, which is not the cell
        // the window opened in once you have flown anywhere: -9000 and 20000 floor to -2 and 2 over
        // a grid of 8192, where truncation would have said -1 and 2.
        let flying = Camera::looking(Vec3::new(-9000.0, 20000.0, 512.0), Vec3::new(0.6, 0.8, 0.0));
        let opened = CellId::Exterior { x: 0, y: 0 };
        let marker = Marker::new(&opened, &flying, WorldTime::hours(0.0), "clear");
        assert_eq!(
            marker.to_string(),
            "# (-2, 2) at -9000, 20000, 512 — bearing 37°, climb 0° — 00:00, clear\n\
             -2,2 --at -9000.0,20000.0,512.0 --look 0.6000,0.8000,0.0000 --time 0.000 \
             --weather clear"
        );
        assert_eq!(arguments_of(&marker).cell, CellId::Exterior { x: -2, y: 2 });
    }
}
