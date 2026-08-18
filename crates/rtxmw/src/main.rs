//! Entry point for the rtxmw engine.

mod app;
mod camera;
mod headless;
mod renderer;
mod scene_loader;

use std::path::PathBuf;

use winit::event_loop::{ControlFlow, EventLoop};

use crate::app::App;

/// Resolution of a `--screenshot` render, independent of any window.
const SCREENSHOT_SIZE: (u32, u32) = (1280, 720);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `--screenshot <path>` renders one frame offscreen and exits, opening no window. The device it
    // brings up has no surface at all, so it works over ssh and in a script.
    let mut args = std::env::args().skip(1);
    if let Some(flag) = args.next() {
        if flag != "--screenshot" {
            return Err(format!("unknown argument {flag:?}; expected --screenshot <path>").into());
        }
        let path = PathBuf::from(
            args.next()
                .ok_or("--screenshot needs a path to write the image to")?,
        );
        let hit_fraction = headless::screenshot(&path, SCREENSHOT_SIZE.0, SCREENSHOT_SIZE.1)?;
        println!(
            "wrote {} ({:.0}% of rays hit geometry)",
            path.display(),
            hit_fraction * 100.0
        );
        return Ok(());
    }

    let event_loop = EventLoop::new()?;
    // Redraw continuously rather than only on OS-driven repaint.
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::default())?;
    Ok(())
}
