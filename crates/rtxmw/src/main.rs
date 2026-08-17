//! Entry point for the rtxmw engine.

mod app;
mod camera;
mod renderer;

use winit::event_loop::{ControlFlow, EventLoop};

use crate::app::App;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    // Redraw continuously rather than only on OS-driven repaint.
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::default())?;
    Ok(())
}
