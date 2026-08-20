//! Recombining the G-buffer into a finished frame.

use ash::vk;
use rtxmw_gpu::{Binding, ComputePipeline, Device, Image};

use crate::gbuffer::GBuffer;
use crate::shaders;

/// Matches `local_size_x`/`local_size_y` in `composite.comp`.
const WORKGROUP: u32 = 8;

/// Multiplies filtered lighting back by albedo, on top of what each surface emits.
#[derive(Debug)]
pub(crate) struct Composite {
    pass: ComputePipeline,
}

impl Composite {
    pub(crate) fn new(device: &Device) -> rtxmw_gpu::Result<Self> {
        Ok(Self {
            pass: ComputePipeline::new(
                device,
                &[
                    Binding::storage_image(0),
                    Binding::storage_image(1),
                    Binding::storage_image(2),
                    // What falls out of the sky, added here only when there is no upscaler to do
                    // it — see `Composite::record`.
                    Binding::storage_image(3),
                    Binding::storage_image(4),
                ],
                size_of::<f32>() as u32,
                shaders::composite(),
            )?,
        })
    }

    /// Points the pass at the target it adds into and the G-buffer it reads.
    ///
    /// Always the G-buffer's first illumination image, never the scratch: the filter runs an even
    /// number of passes and lands back on it.
    pub(crate) fn bind(&mut self, target: &Image, gbuffer: &GBuffer) {
        self.pass
            .bind_storage_images(0, &[target, gbuffer.albedo(), gbuffer.illumination()]);
        self.pass
            .bind_storage_images(3, &[gbuffer.transparency(), gbuffer.transparency_opacity()]);
    }

    /// # Safety
    /// `command_buffer` must be recording, [`Composite::bind`] must have run, and every image must
    /// be in `GENERAL`.
    /// **`overlay` is whether what falls still has to be put in**, which is exactly when there is
    /// no upscaler: Ray Reconstruction composites `DLSS.TransparencyLayer` itself, and doing it
    /// again would draw every streak twice. It belongs here rather than after the tone curve
    /// because the exposure pass meters what this leaves behind.
    pub(crate) unsafe fn record(
        &self,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        overlay: bool,
    ) {
        // SAFETY: the caller guarantees the command buffer is recording and the set is written.
        unsafe {
            self.pass.dispatch(
                command_buffer,
                [
                    extent.width.div_ceil(WORKGROUP),
                    extent.height.div_ceil(WORKGROUP),
                    1,
                ],
                &f32::from(u8::from(overlay)).to_ne_bytes(),
            );
        }
    }
}
