//! Entry point for the rtxmw engine.

mod app;
mod camera;
mod cli;
mod headless;
mod renderer;
mod scene_loader;

use clap::Parser;
use winit::event_loop::{ControlFlow, EventLoop};

use crate::app::App;
use crate::cli::{ScreenshotOptions, WindowOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // **Two commands rather than one with two shapes**, because the positionals differ: the
    // windowed mode takes a cell and the offscreen one a size before it. Which is which is settled
    // by the first argument, as it always has been. `--help` on either is the usage; there is no
    // copy of it here to fall out of date.
    //
    // `--screenshot` opens no window at all — the device it brings up has no surface extensions, so
    // it works over ssh and in a script.
    let arguments: Vec<String> = std::env::args().collect();
    if arguments
        .get(1)
        .is_some_and(|first| first == "--screenshot")
    {
        return screenshot(&ScreenshotOptions::parse_from(&arguments));
    }

    let event_loop = EventLoop::new()?;
    // Redraw continuously rather than only on OS-driven repaint.
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::opening_in(WindowOptions::parse_from(&arguments)))?;
    Ok(())
}

/// Renders offscreen from `options`.
fn screenshot(options: &ScreenshotOptions) -> Result<(), Box<dyn std::error::Error>> {
    let hit_fraction = headless::screenshot(options)?;
    println!(
        "wrote {} ({:.0}% of rays hit geometry)",
        options.path.display(),
        hit_fraction * 100.0
    );
    Ok(())
}
