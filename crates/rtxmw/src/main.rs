//! Entry point for the rtxmw engine.

mod app;
mod camera;
mod headless;
mod renderer;
mod scene_loader;

use std::path::PathBuf;

use winit::event_loop::{ControlFlow, EventLoop};

use crate::app::App;

/// Resolution of a `--screenshot` render when none is given, independent of any window.
///
/// Not the internal resolution the engine targets — see `docs/design.md` §5.3, which is 1920x1080 —
/// because a screenshot is usually being looked at rather than measured, and 720p renders faster
/// and reads the same. Pass `1920x1080` to measure against the budget.
const SCREENSHOT_SIZE: (u32, u32) = (1280, 720);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Both modes take the same optional cell: a pair of integers is an exterior and anything else
    // is an interior's name.
    //
    //     rtxmw [CELL]
    //     rtxmw --screenshot <path> [WIDTHxHEIGHT] [CELL]
    //
    // `--screenshot` renders one frame offscreen and exits, opening no window. The device it brings
    // up has no surface at all, so it works over ssh and in a script.
    let mut args = std::env::args().skip(1);
    let first = args.next();
    if first.as_deref() != Some("--screenshot") {
        let cell = scene_loader::cell_argument(first.as_deref());
        let event_loop = EventLoop::new()?;
        // Redraw continuously rather than only on OS-driven repaint.
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.run_app(&mut App::opening_in(cell))?;
        return Ok(());
    }

    let path = PathBuf::from(
        args.next()
            .ok_or("--screenshot needs a path to write the image to")?,
    );
    let (width, height) = match args.next() {
        None => SCREENSHOT_SIZE,
        Some(size) => {
            let malformed = || format!("expected a size like 1920x1080, got {size:?}");
            let (width, height) = size.split_once('x').ok_or_else(malformed)?;
            // The same message for a missing separator and an unparsable number: on its own
            // `parse` would report "invalid digit found in string" without saying of what.
            (
                width.parse().map_err(|_| malformed())?,
                height.parse().map_err(|_| malformed())?,
            )
        }
    };
    let cell = scene_loader::cell_argument(args.next().as_deref());
    let hit_fraction = headless::screenshot(&path, width, height, cell)?;
    println!(
        "wrote {} ({:.0}% of rays hit geometry)",
        path.display(),
        hit_fraction * 100.0
    );
    Ok(())
}
