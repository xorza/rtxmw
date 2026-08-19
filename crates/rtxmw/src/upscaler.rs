//! Wiring DLSS Ray Reconstruction into a renderer, or standing in for it where there is none.
//!
//! **Shared by both front ends.** The windowed renderer and `--screenshot` bring up Vulkan
//! separately but need NGX brought up the same way, and a second copy of this would be a second
//! place for the two to disagree about what DLSS was told.

use ash::vk;
use rtxmw_gpu::{Device, Instance, Memory, PhysicalDevice, Uploader};
use rtxmw_render::SceneRenderer;

use crate::cli::Upscaling;
#[cfg(feature = "dlss")]
use rtxmw_render::dlss::Upscaler;

/// Stands in for the upscaler where DLSS is not compiled in, so one code path serves both builds.
#[cfg(not(feature = "dlss"))]
pub(crate) type Upscaler = std::convert::Infallible;

/// Extensions NGX needs enabled on the device, or none when DLSS is not compiled in.
pub(crate) fn device_extensions() -> Vec<&'static std::ffi::CStr> {
    #[cfg(feature = "dlss")]
    {
        rtxmw_render::dlss::Requirements::query()
            .map(|required| required.device)
            .unwrap_or_default()
    }
    #[cfg(not(feature = "dlss"))]
    Vec::new()
}

/// Builds an upscaler for `mode`, where DLSS is compiled in and the mode is not `off`.
///
/// **Takes the decision rather than making it.** What DLSS runs at is a setting, and settings are
/// read in `cli` — this module knowing how to consult the environment would be a second place for
/// the answer to come from.
pub(crate) fn build(
    _instance: &Instance,
    _physical: &PhysicalDevice,
    _device: &Device,
    _uploader: &mut Uploader,
    _output: vk::Extent2D,
    _mode: Upscaling,
) -> Result<Option<Upscaler>, Box<dyn std::error::Error>> {
    #[cfg(feature = "dlss")]
    {
        let Upscaling(Some(preset)) = _mode else {
            return Ok(None);
        };
        let upscaler = Upscaler::new(
            _instance,
            _physical,
            _device,
            _uploader,
            _output,
            preset,
            paths(),
        )
        .map_err(|e| format!("DLSS would not start: {e}"))?;
        Ok(Some(upscaler))
    }
    #[cfg(not(feature = "dlss"))]
    Ok(None)
}

/// What to trace at through `upscaler`, or `without` when there is none.
///
/// The fallback is not the output size: a window renders at a fraction of itself and the blit
/// bridges the two, so what an absent upscaler falls back to is the caller's own answer rather than
/// the size it is producing.
pub(crate) fn render_size(_upscaler: Option<&Upscaler>, without: vk::Extent2D) -> vk::Extent2D {
    #[cfg(feature = "dlss")]
    return _upscaler.map_or(without, Upscaler::render_size);
    #[cfg(not(feature = "dlss"))]
    without
}

/// Releases whatever upscaler `renderer` holds.
///
/// **Before a replacement is built, never after.** Each upscaler owns its own NGX, and dropping one
/// shuts NGX down for the whole device — so building the new one first leaves it orphaned the moment
/// the old one goes, which NGX reports as `FAIL_NotInitialized` at every evaluation and nowhere
/// else.
pub(crate) fn detach(
    memory: &Memory,
    renderer: &mut SceneRenderer,
) -> Result<(), Box<dyn std::error::Error>> {
    attach(memory, renderer, None)
}

/// Hands `upscaler` to the renderer, **or takes away the one it has**.
///
/// `None` releases rather than doing nothing, so that this and [`detach`] cannot disagree about what
/// an absent upscaler means.
pub(crate) fn attach(
    _memory: &Memory,
    _renderer: &mut SceneRenderer,
    _upscaler: Option<Upscaler>,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "dlss")]
    {
        let described = _upscaler
            .as_ref()
            .map(|upscaler| (upscaler.render_size(), upscaler.output().extent()));
        // `set_upscaler` owns what this does to the à-trous passes, in both directions.
        _renderer.set_upscaler(_memory, _upscaler)?;
        if let Some((render, output)) = described {
            println!(
                "  DLSS Ray Reconstruction: {}x{} to {}x{}",
                render.width, render.height, output.width, output.height
            );
        }
    }
    Ok(())
}

/// Where NGX may write and where its feature libraries are.
#[cfg(feature = "dlss")]
fn paths() -> rtxmw_render::dlss::Paths<'static> {
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
