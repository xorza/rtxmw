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
                ],
                0,
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
    }

    /// # Safety
    /// `command_buffer` must be recording, [`Composite::bind`] must have run, and every image must
    /// be in `GENERAL`.
    pub(crate) unsafe fn record(&self, command_buffer: vk::CommandBuffer, extent: vk::Extent2D) {
        // SAFETY: the caller guarantees the command buffer is recording and the set is written.
        unsafe {
            self.pass.dispatch(
                command_buffer,
                [
                    extent.width.div_ceil(WORKGROUP),
                    extent.height.div_ceil(WORKGROUP),
                    1,
                ],
                &[],
            );
        }
    }
}
