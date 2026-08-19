//! Rendering one frame without a window.
//!
//! The engine's own path — same `SceneRenderer`, same shader, same scene loading — brought up on a
//! device that was never asked to present. Useful for looking at a cell without a window appearing,
//! and for checking that the frame path is validation-clean in a script.

use std::path::Path;
use std::time::Instant;

use ash::vk;
use rtxmw_gpu::{
    Device, Instance, Memory, PhysicalDevice, Presentation, RayTracingLimits, Uploader, Validation,
    readback,
};
use rtxmw_render::SceneRenderer;
#[cfg(feature = "dlss")]
use rtxmw_render::dlss::Upscaler;
use rtxmw_scene::{CellId, CellStreamer, LoadedCell};

use crate::scene_loader;

/// Stands in for the upscaler where DLSS is not compiled in, so one code path serves both builds.
#[cfg(not(feature = "dlss"))]
type Upscaler = std::convert::Infallible;
use crate::scene_loader::{Viewpoint, ViewpointOverride};

/// Renders the default cell once and writes it to `path` as a PNG.
///
/// Returns the fraction of pixels that hit geometry, which is enough to tell "the cell rendered"
/// from "the camera was pointed at nothing" without opening the file.
///
/// `viewpoint` says where the camera stands and which way it faces; whatever it leaves out is where
/// a traveller entering the cell would stand and what they would be looking at.
pub(crate) fn screenshot(
    path: &Path,
    width: u32,
    height: u32,
    cell: CellId,
    viewpoint: ViewpointOverride,
) -> Result<f32, Box<dyn std::error::Error>> {
    // No surface extensions and no swapchain: this device could not present if asked.
    let instance = Instance::new(c"rtxmw", &[], Validation::for_build())?;
    let physical = PhysicalDevice::select(&instance, Presentation::NotNeeded)?;
    // NGX names extensions of its own, and they have to be enabled when the device is created —
    // enabled where present, so a machine without them still builds a device that simply cannot
    // upscale. Empty when the feature is compiled out.
    let device = Device::new(&instance, &physical, &upscaler_extensions())?;
    let memory = Memory::new(&instance, &physical, &device)?;
    let mut uploader = Uploader::new(&device, &memory, physical.graphics_queue_family())?;

    // With an upscaler the size asked for is the *output* and DLSS says what to trace at, so it is
    // built before the renderer that has to be sized by its answer.
    let output = vk::Extent2D { width, height };
    let upscaler = build_upscaler(&instance, &physical, &device, &mut uploader, output)?;
    let extent = render_size(upscaler.as_ref(), output);
    let mut renderer = SceneRenderer::new(&device, &physical, &memory, extent)?;
    attach_upscaler(&memory, &mut renderer, upscaler)?;

    let cell = LoadedCell::load_at(cell)?
        .ok_or("no game data configured — set MORROWIND_DATA_DIR, or put it in .env")?;
    renderer.load_scene(
        &device,
        &mut uploader,
        physical.limits(),
        cell.id.clone(),
        &cell.scene,
        &cell.textures,
    )?;
    println!("{}", scene_loader::describe(&cell));
    fill_window(
        &device,
        &mut uploader,
        physical.limits(),
        &mut renderer,
        &cell.id,
    )?;

    let camera = viewpoint.over(Viewpoint::entering(&cell)).camera();
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

/// Streams in the cells around `centre` and waits for them, so a screenshot shows what the engine
/// shows rather than one cell surrounded by nothing.
///
/// Blocking, unlike the windowed path: there is no frame to keep running, and the image is not
/// worth taking until the world it is of has arrived. An interior has no window and returns at
/// once.
fn fill_window(
    device: &Device,
    uploader: &mut Uploader,
    limits: RayTracingLimits,
    renderer: &mut SceneRenderer,
    centre: &CellId,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wanted = Vec::new();
    scene_loader::wanted_cells(centre, &mut wanted);
    let streamer = CellStreamer::spawn();
    let mut outstanding = 0;
    for cell in wanted.iter().filter(|cell| cell.id != *centre) {
        streamer.request(cell.id.clone(), scene_loader::detail_at(cell.rings));
        outstanding += 1;
    }
    if outstanding == 0 {
        return Ok(());
    }

    let started = Instant::now();
    let mut loaded = 0;
    while outstanding > 0 {
        let Some(ready) = streamer.wait_ready() else {
            break;
        };
        outstanding -= 1;
        // A grid square with no cell record is open sea, and the commonest result out here.
        if let Ok(cell) = ready.loaded {
            renderer.add_cell(
                device,
                uploader,
                limits,
                cell.id,
                &cell.scene,
                &cell.textures,
            )?;
            loaded += 1;
        }
    }
    renderer.commit(device, uploader, limits)?;
    println!(
        "  window: {loaded} cells in {:.0} ms",
        started.elapsed().as_secs_f32() * 1000.0
    );
    Ok(())
}

/// Extensions NGX needs enabled on the device, or none when DLSS is not compiled in.
fn upscaler_extensions() -> Vec<&'static std::ffi::CStr> {
    #[cfg(feature = "dlss")]
    {
        rtxmw_render::dlss::Requirements::query()
            .map(|required| required.device)
            .unwrap_or_default()
    }
    #[cfg(not(feature = "dlss"))]
    Vec::new()
}

/// Builds an upscaler, where DLSS is compiled in and asked for.
///
/// **`RTXMW_DLSS` is read here and nowhere else.** Opt-in twice over — the feature to compile it and
/// the variable to attach it — so that a build carrying DLSS renders identically to one without it
/// until asked, which is what makes the two comparable.
fn build_upscaler(
    _instance: &Instance,
    _physical: &PhysicalDevice,
    _device: &Device,
    _uploader: &mut Uploader,
    _output: vk::Extent2D,
) -> Result<Option<Upscaler>, Box<dyn std::error::Error>> {
    #[cfg(feature = "dlss")]
    {
        if std::env::var("RTXMW_DLSS").is_err() {
            return Ok(None);
        }
        let upscaler = Upscaler::new(
            _instance,
            _physical,
            _device,
            _uploader,
            _output,
            upscaler_paths(),
        )
        .map_err(|e| format!("DLSS would not start: {e}"))?;
        Ok(Some(upscaler))
    }
    #[cfg(not(feature = "dlss"))]
    Ok(None)
}

/// What to trace at to produce `output`, which is `output` itself when nothing upscales it.
fn render_size(_upscaler: Option<&Upscaler>, output: vk::Extent2D) -> vk::Extent2D {
    #[cfg(feature = "dlss")]
    return _upscaler.map_or(output, Upscaler::render_size);
    #[cfg(not(feature = "dlss"))]
    output
}

/// Hands `upscaler` to the renderer, if there is one.
fn attach_upscaler(
    _memory: &Memory,
    _renderer: &mut SceneRenderer,
    _upscaler: Option<Upscaler>,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "dlss")]
    if let Some(upscaler) = _upscaler {
        let render = upscaler.render_size();
        let output = upscaler.output().extent();
        // Ray Reconstruction denoises as it upscales, so the à-trous passes would be filtering
        // something about to be filtered again.
        _renderer.set_denoise_passes(0);
        _renderer.set_upscaler(_memory, Some(upscaler))?;
        println!(
            "  DLSS Ray Reconstruction: {}x{} to {}x{}",
            render.width, render.height, output.width, output.height
        );
    }
    Ok(())
}

/// Where NGX may write and where its feature libraries are.
#[cfg(feature = "dlss")]
fn upscaler_paths() -> rtxmw_render::dlss::Paths<'static> {
    // Held for the life of the process, which is how long NGX keeps the pointer it is given.
    static DATA: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    let data = DATA.get_or_init(|| {
        let path = std::env::temp_dir().join("rtxmw-ngx");
        // NGX wants somewhere it may write; it says so as `FAIL_UnableToWriteToAppDataPath`.
        let _ = std::fs::create_dir_all(&path);
        path
    });
    rtxmw_render::dlss::Paths {
        data,
        feature_libraries: std::path::Path::new(rtxmw_render::dlss::FEATURE_LIBRARIES),
    }
}
