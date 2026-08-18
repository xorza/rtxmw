//! Measuring how bright the frame is, and what to scale it by.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use rtxmw_gpu::{
    Binding, Buffer, BufferMemory, ComputePipeline, Device, Image, Memory, memory_barrier,
};

use crate::shaders;

/// Bins in the luminance histogram. Matches `BINS` in both shaders, and the exposure pass's
/// workgroup is exactly this wide.
const BIN_COUNT: u32 = 256;

/// Darkest luminance the histogram resolves, as a power of two.
///
/// Two to the minus ten is about a thousandth of mid grey — below anything a lit surface reaches,
/// and well under the unlit interiors this has to expose.
const MIN_LOG_LUMINANCE: f32 = -10.0;

/// Brightest, as a power of two. Sixty-four times mid grey covers a torch flame seen directly.
const MAX_LOG_LUMINANCE: f32 = 6.0;

/// Matches `local_size_x`/`local_size_y` in `luminance_histogram.comp`.
const HISTOGRAM_WORKGROUP: u32 = 16;

/// What the binning pass needs to place a luminance.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct HistogramConstants {
    min_log_luminance: f32,
    inverse_log_range: f32,
}

/// What the reduction needs to undo the binning and weight the result.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ExposureConstants {
    min_log_luminance: f32,
    log_range: f32,
    pixels: u32,
}

/// Measures a frame's average luminance and turns it into an exposure multiplier.
///
/// Two dispatches rather than one: the first bins every pixel, the second reduces the 256 bins to a
/// single number. They cannot be merged, because the reduction has to see every pixel's
/// contribution before it can divide by the total, and a compute dispatch is the only barrier wide
/// enough to guarantee that.
#[derive(Debug)]
pub(crate) struct AutoExposure {
    histogram_pass: ComputePipeline,
    reduce_pass: ComputePipeline,
    /// One `u32` per bin. Zeroed on the host side of each frame, because a shader that cleared it
    /// would race with the workgroups already accumulating into it.
    histogram: Buffer,
    /// The single `f32` the tonemap pass reads.
    exposure: Buffer,
}

impl AutoExposure {
    /// Builds both passes and the buffers between them.
    pub(crate) fn new(device: &Device, memory: &Memory) -> rtxmw_gpu::Result<Self> {
        let histogram_pass = ComputePipeline::new(
            device,
            &[Binding::storage_image(0), Binding::storage_buffer(1)],
            size_of::<HistogramConstants>() as u32,
            shaders::luminance_histogram(),
        )?;
        let reduce_pass = ComputePipeline::new(
            device,
            &[Binding::storage_buffer(0), Binding::storage_buffer(1)],
            size_of::<ExposureConstants>() as u32,
            shaders::exposure(),
        )?;

        let histogram = Buffer::new(
            memory,
            "luminance histogram",
            (BIN_COUNT * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            BufferMemory::Device,
        )?;
        let exposure = Buffer::new(
            memory,
            "exposure",
            4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            BufferMemory::Device,
        )?;

        Ok(Self {
            histogram_pass,
            reduce_pass,
            histogram,
            exposure,
        })
    }

    /// The buffer holding the multiplier, for the pass that applies it.
    pub(crate) fn buffer(&self) -> &Buffer {
        &self.exposure
    }

    /// Points the passes at the image they measure.
    pub(crate) fn bind(&mut self, source: &Image) {
        let images = [vk::DescriptorImageInfo::default()
            .image_view(source.view())
            .image_layout(vk::ImageLayout::GENERAL)];
        let histogram_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.histogram.raw())
            .range(vk::WHOLE_SIZE)];
        let exposure_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.exposure.raw())
            .range(vk::WHOLE_SIZE)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.histogram_pass.set())
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&images),
            vk::WriteDescriptorSet::default()
                .dst_set(self.histogram_pass.set())
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&histogram_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.reduce_pass.set())
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&histogram_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.reduce_pass.set())
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&exposure_info),
        ];
        // SAFETY: every write names a set this struct owns, and no dispatch using them is in
        // flight — the caller rebinds only when the scene or the target changes.
        unsafe {
            self.histogram_pass
                .device()
                .update_descriptor_sets(&writes, &[])
        };
    }

    /// Records the clear, the binning and the reduction.
    ///
    /// # Safety
    /// `command_buffer` must be recording, [`AutoExposure::bind`] must have run, and the source
    /// image must be in `GENERAL`.
    pub(crate) unsafe fn record(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
    ) {
        let range = MAX_LOG_LUMINANCE - MIN_LOG_LUMINANCE;
        let histogram_constants = HistogramConstants {
            min_log_luminance: MIN_LOG_LUMINANCE,
            inverse_log_range: 1.0 / range,
        };
        let exposure_constants = ExposureConstants {
            min_log_luminance: MIN_LOG_LUMINANCE,
            log_range: range,
            pixels: extent.width * extent.height,
        };

        // SAFETY: the caller guarantees the command buffer is recording and the sets are written.
        unsafe {
            // Cleared with a transfer rather than a shader: the bins are accumulated into with
            // atomics from every workgroup at once, so anything zeroing them from inside the same
            // dispatch would be racing the accumulation.
            device.cmd_fill_buffer(command_buffer, self.histogram.raw(), 0, vk::WHOLE_SIZE, 0);
            memory_barrier::full(device, command_buffer);

            self.histogram_pass.dispatch(
                command_buffer,
                [
                    extent.width.div_ceil(HISTOGRAM_WORKGROUP),
                    extent.height.div_ceil(HISTOGRAM_WORKGROUP),
                    1,
                ],
                bytemuck::bytes_of(&histogram_constants),
            );
            memory_barrier::full(device, command_buffer);

            // One workgroup, because the reduction is over the bins and there are exactly as many
            // threads in it as there are bins.
            self.reduce_pass.dispatch(
                command_buffer,
                [1, 1, 1],
                bytemuck::bytes_of(&exposure_constants),
            );
            memory_barrier::full(device, command_buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_push_blocks_match_what_the_shaders_declare() {
        assert_eq!(size_of::<HistogramConstants>(), 8);
        assert_eq!(size_of::<ExposureConstants>(), 12);
    }

    #[test]
    fn the_histogram_spans_enough_of_the_range_to_expose_an_interior_and_a_flame() {
        // Seyda Neen's office traces at about 0.03 of mid grey and its flames reach past 1.0. Both
        // have to land inside the resolved range, or the exposure is measuring a clamp.
        let interior = 0.03f32.log2();
        let flame = 4.0f32.log2();
        assert!(interior > MIN_LOG_LUMINANCE && interior < MAX_LOG_LUMINANCE);
        assert!(flame > MIN_LOG_LUMINANCE && flame < MAX_LOG_LUMINANCE);
        // And with room to spare either side, since exteriors are far brighter still.
        assert!(MAX_LOG_LUMINANCE - flame > 3.0);
    }
}
