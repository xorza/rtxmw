//! Rendering one frame without a window.
//!
//! The engine's own path — same `SceneRenderer`, same shader, same scene loading — brought up on a
//! device that was never asked to present. Useful for looking at a cell without a window appearing,
//! and for checking that the frame path is validation-clean in a script.

use std::path::Path;

use ash::vk;
use rtxmw_gpu::{
    Device, Instance, Memory, PhysicalDevice, Presentation, Uploader, Validation, readback,
};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{CellId, LoadedCell};

use crate::scene_loader;

/// Renders the default cell once and writes it to `path` as a PNG.
///
/// Returns the fraction of pixels that hit geometry, which is enough to tell "the cell rendered"
/// from "the camera was pointed at nothing" without opening the file.
pub(crate) fn screenshot(
    path: &Path,
    width: u32,
    height: u32,
    cell: CellId,
) -> Result<f32, Box<dyn std::error::Error>> {
    // No surface extensions and no swapchain: this device could not present if asked.
    let instance = Instance::new(c"rtxmw", &[], Validation::for_build())?;
    let physical = PhysicalDevice::select(&instance, Presentation::NotNeeded)?;
    let device = Device::new(&instance, &physical)?;
    let memory = Memory::new(&instance, &physical, &device)?;
    let mut uploader = Uploader::new(&device, &memory, physical.graphics_queue_family())?;

    let extent = vk::Extent2D { width, height };
    let mut renderer = SceneRenderer::new(&device, &physical, &memory, extent)?;

    let cell = LoadedCell::load_at(cell, scene_loader::GRID_RADIUS)?
        .ok_or("no game data configured — set MORROWIND_DATA_DIR, or put it in .env")?;
    renderer.load_scene(
        &device,
        &mut uploader,
        physical.limits(),
        &cell.scene,
        &cell.textures,
    )?;
    println!("{}", scene_loader::describe(&cell));

    let camera = scene_loader::Viewpoint::entering(&cell).camera();
    let constants = renderer.frame_constants(
        camera.view(),
        camera.projection(width as f32 / height as f32),
        camera.position(),
    );
    renderer.render_once(&mut uploader, &constants)?;

    // Zero everywhere means the queue cannot write timestamps, which is worth saying nothing about
    // rather than reporting as a frame that took no time.
    let timings = renderer.timings()?;
    if timings.total() > 0.0 {
        println!("  {timings}");
    }

    // The tonemapped output rather than the raw radiance, so the file holds exactly the bytes the
    // window would show — the whole verification loop rests on the screenshot being that.
    let pixels = readback::image_to_rgba8(
        &mut uploader,
        renderer.output(),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    )?;

    // Alpha carries the hit flag rather than coverage, so the file gets it flattened — otherwise a
    // viewer composites every missed ray as transparent and the image reads as blown out.
    let hits = pixels.chunks_exact(4).filter(|p| p[3] > 128).count();
    readback::write_png_opaque(path, &pixels, width, height);

    // Everything here is a local, so it drops in reverse declaration order — renderer, uploader,
    // memory, device — which is the order the device's allocations require. Worth knowing rather
    // than worth writing out: reordering the declarations would break it silently.
    Ok(hits as f32 / (width * height) as f32)
}
