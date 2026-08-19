//! Entry point for the rtxmw engine.

mod app;
mod camera;
mod cli;
mod headless;
mod renderer;
mod scene_loader;
mod texture_sheet;
mod upscaler;

use winit::event_loop::{ControlFlow, EventLoop};

use crate::app::App;
use crate::cli::{Command, ScreenshotOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().collect();
    match Command::parse_from(&arguments) {
        // The device this brings up has no surface extensions at all, so it works over ssh and in
        // a script.
        Command::Screenshot(options) => screenshot(&options),
        // Reads the cell and writes an image, with no device at all — the correction it shows is
        // arithmetic on the textures rather than anything a frame did.
        Command::TextureSheet(options) => texture_sheet::write(&options),
        Command::Window(options) => {
            let event_loop = EventLoop::new()?;
            // Redraw continuously rather than only on OS-driven repaint.
            event_loop.set_control_flow(ControlFlow::Poll);
            event_loop.run_app(&mut App::opening_in(options))?;
            Ok(())
        }
    }
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
