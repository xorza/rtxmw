//! Smoothing the lighting without touching the surface detail underneath it.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use rtxmw_gpu::{Binding, ComputePipeline, Device, memory_barrier};

use crate::gbuffer::GBuffer;
use crate::shaders;

/// Matches `local_size_x`/`local_size_y` in `denoise.comp`.
const WORKGROUP: u32 = 8;

/// How many à-trous passes to run unless a caller says otherwise.
///
/// The tap spacing doubles each pass, so four of them reach fifteen pixels either way for the cost
/// of a hundred taps — a direct blur of that radius would be nearly a thousand.
pub(crate) const DEFAULT_PASSES: u32 = 4;

/// The spacing of one pass, in pixels.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PassConstants {
    step: i32,
}

/// An edge-stopping à-trous filter over the G-buffer's lighting.
///
/// Two pipelines over one shader, because a pass reads a whole image and writes a whole image and
/// so cannot work in place. They differ only in which way round their descriptor sets point, and
/// alternating between them is the ping-pong.
#[derive(Debug)]
pub(crate) struct Denoiser {
    /// Reads the trace's illumination, writes the scratch.
    forward: ComputePipeline,
    /// And back again.
    backward: ComputePipeline,
}

impl Denoiser {
    pub(crate) fn new(device: &Device) -> rtxmw_gpu::Result<Self> {
        let pipeline = || {
            ComputePipeline::new(
                device,
                &[
                    Binding::storage_image(0),
                    Binding::storage_image(1),
                    Binding::storage_image(2),
                ],
                size_of::<PassConstants>() as u32,
                shaders::denoise(),
            )
        };
        Ok(Self {
            forward: pipeline()?,
            backward: pipeline()?,
        })
    }

    /// Points both directions at the G-buffer.
    pub(crate) fn bind(&mut self, gbuffer: &GBuffer) {
        for (pipeline, source, target) in [
            (&self.forward, gbuffer.illumination(), gbuffer.scratch()),
            (&self.backward, gbuffer.scratch(), gbuffer.illumination()),
        ] {
            pipeline.bind_storage_images(0, &[gbuffer.normal_depth(), source, target]);
        }
    }

    /// Records every pass, leaving the result back in the G-buffer's illumination image.
    ///
    /// # Safety
    /// `command_buffer` must be recording, [`Denoiser::bind`] must have run, and every G-buffer
    /// image must be in `GENERAL`.
    /// `passes` **must be even**: each pass reads one image and writes the other, so an odd count
    /// would leave the result in the scratch image, which nothing reads. Zero filters nothing,
    /// which is the honest A/B against filtering.
    pub(crate) unsafe fn record(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        passes: u32,
    ) {
        // The public entry point asserts this outright; this is a per-frame path, and a caller
        // cannot reach it without having gone through that one.
        debug_assert_eq!(passes % 2, 0);
        let groups = [
            extent.width.div_ceil(WORKGROUP),
            extent.height.div_ceil(WORKGROUP),
            1,
        ];
        for index in 0..passes {
            let pipeline = if index % 2 == 0 {
                &self.forward
            } else {
                &self.backward
            };
            let constants = PassConstants { step: 1 << index };
            // SAFETY: the caller guarantees the command buffer is recording and the sets are
            // written. Each pass reads what the one before it wrote, so they cannot overlap.
            unsafe {
                pipeline.dispatch(command_buffer, groups, bytemuck::bytes_of(&constants));
                memory_barrier::full(device, command_buffer);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_passes_end_where_they_started_and_reach_far_enough() {
        // An odd count would leave the filtered lighting in the scratch image, which nothing reads.
        assert_eq!(DEFAULT_PASSES % 2, 0);
        // The reach doubles each pass: the last spans two taps of `1 << (DEFAULT_PASSES - 1)`, so
        // four passes gather from sixteen pixels either way.
        assert_eq!(2 * (1 << (DEFAULT_PASSES - 1)), 16);
    }
}
