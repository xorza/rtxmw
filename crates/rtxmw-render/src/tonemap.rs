//! Turning linear radiance into the bytes a display wants.

use ash::vk;
use rtxmw_gpu::{Binding, Buffer, ComputePipeline, Device, Image, Memory};

use crate::shaders;

/// Format of the tonemapped image.
///
/// Eight bits and **not** an `_SRGB` variant, because the shader does the encoding itself: sRGB
/// formats expose no storage capability, so a compute shader cannot write one at all. Storing the
/// encoded values means the swapchain must not encode them a second time — see
/// `Swapchain::choose_format`.
pub const OUTPUT_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// Matches `local_size_x`/`local_size_y` in `tonemap.comp`.
const WORKGROUP: u32 = 8;

/// Applies exposure and the tone curve, and writes display-ready bytes.
#[derive(Debug)]
pub(crate) struct Tonemap {
    pass: ComputePipeline,
    output: Image,
}

impl Tonemap {
    /// Creates the pass and the image it writes.
    pub(crate) fn new(
        device: &Device,
        memory: &Memory,
        extent: vk::Extent2D,
    ) -> rtxmw_gpu::Result<Self> {
        Ok(Self {
            pass: ComputePipeline::new(
                device,
                &[
                    Binding::storage_image(0),
                    Binding::storage_image(1),
                    Binding::storage_buffer(2),
                ],
                0,
                shaders::tonemap(),
            )?,
            output: Image::new(
                memory,
                "tonemapped output",
                extent,
                OUTPUT_FORMAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            )?,
        })
    }

    /// The display-ready image, in `TRANSFER_SRC_OPTIMAL` once [`Tonemap::record`] has run.
    pub(crate) fn output(&self) -> &Image {
        &self.output
    }

    /// Points the pass at the radiance it reads and the exposure to apply.
    pub(crate) fn bind(&mut self, source: &Image, exposure: &Buffer) {
        self.pass.bind_storage_images(0, &[source, &self.output]);

        let exposure_info = [vk::DescriptorBufferInfo::default()
            .buffer(exposure.raw())
            .range(vk::WHOLE_SIZE)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(self.pass.set())
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&exposure_info)];
        // SAFETY: every write names this pass's own set, and no dispatch using it is in flight.
        unsafe { self.pass.device().update_descriptor_sets(&writes, &[]) };
    }

    /// Records the dispatch. Both images must be in `GENERAL`.
    ///
    /// # Safety
    /// `command_buffer` must be recording and [`Tonemap::bind`] must have run.
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
