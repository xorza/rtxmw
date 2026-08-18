//! An offscreen colour target that can be rendered into and read back.

use ash::vk;

use crate::error::Result;
use crate::image::Image;
use crate::image_barrier::{self, COLOR_RANGE};
use crate::memory::Memory;
use crate::readback;
use crate::uploader::Uploader;

/// An offscreen image a test can fill and read back.
///
/// Cheap relative to device creation, so one per test is fine. The image is an ordinary [`Image`]
/// and the readback is the shared [`readback::image_to_rgba8`]; what this adds is the clear, and a
/// constructor with the usage flags a test target wants.
#[derive(Debug)]
pub struct RenderTarget {
    image: Image,
}

impl RenderTarget {
    pub(crate) fn new(
        memory: &Memory,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<Self> {
        // STORAGE so a compute or ray tracing shader can write it; TRANSFER_SRC for readback;
        // COLOR_ATTACHMENT so a raster pass can target it too.
        let image = Image::new(
            memory,
            "test render target",
            vk::Extent2D { width, height },
            format,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        Ok(Self { image })
    }

    /// Fills the whole image with `colour`, leaving it in `TRANSFER_DST_OPTIMAL`.
    ///
    /// Takes the uploader rather than reaching for [`crate::TestGpu::submit_and_wait`]: that locks
    /// the shared uploader, so a caller already holding the guard would deadlock. Threading it
    /// through makes that impossible to write instead of merely documented.
    pub fn clear(&self, uploader: &mut Uploader, colour: [f32; 4]) -> Result<()> {
        uploader.submit_and_wait(|device, cmd| {
            // SAFETY: the command buffer is recording and the image belongs to this device.
            unsafe {
                image_barrier::transition(
                    device,
                    cmd,
                    self.image.raw(),
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                device.cmd_clear_color_image(
                    cmd,
                    self.image.raw(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue { float32: colour },
                    &[COLOR_RANGE],
                );
            }
        })
    }

    /// Copies the image to host memory and returns it as 8-bit RGBA, row-major, top row first.
    ///
    /// `current_layout` is the layout the image is in when this is called; it is transitioned to
    /// `TRANSFER_SRC_OPTIMAL` and left there.
    pub fn read_rgba8(
        &self,
        uploader: &mut Uploader,
        current_layout: vk::ImageLayout,
    ) -> Result<Vec<u8>> {
        readback::image_to_rgba8(uploader, &self.image, current_layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::test_gpu::TestGpu;

    /// Chosen so the UNORM encode is exact: 0.2*255 = 51, 0.6*255 = 153, 0.8*255 = 204, 1.0 = 255.
    /// Values like 0.5 would land on 127.5, where the rounding direction is implementation-defined.
    const CLEAR: [f32; 4] = [0.2, 0.6, 0.8, 1.0];
    const CLEAR_RGBA8: [u8; 4] = [51, 153, 204, 255];

    #[test]
    fn clearing_a_target_reads_back_the_exact_colour() {
        let gpu = TestGpu::shared();
        let target = gpu
            .create_target(8, 4, vk::Format::R8G8B8A8_UNORM)
            .expect("could not create target");

        let mut uploader = gpu.uploader();
        target.clear(&mut uploader, CLEAR).expect("clear failed");
        let pixels = target
            .read_rgba8(&mut uploader, vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .expect("readback failed");
        drop(uploader);

        assert_eq!(pixels.len(), 8 * 4 * 4);
        for (index, pixel) in pixels.chunks_exact(4).enumerate() {
            assert_eq!(pixel, CLEAR_RGBA8, "pixel {index} differs");
        }
        gpu.assert_no_validation_errors();
    }

    #[test]
    fn a_cleared_target_matches_its_golden_image() {
        let gpu = TestGpu::shared();
        let target = gpu
            .create_target(16, 16, vk::Format::R8G8B8A8_UNORM)
            .expect("could not create target");

        let mut uploader = gpu.uploader();
        target.clear(&mut uploader, CLEAR).expect("clear failed");
        let pixels = target
            .read_rgba8(&mut uploader, vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .expect("readback failed");
        drop(uploader);

        crate::testing::golden::assert_matches("cleared_target", &pixels, 16, 16);
        gpu.assert_no_validation_errors();
    }

    #[test]
    fn bgra_targets_read_back_in_rgba_order() {
        let gpu = TestGpu::shared();
        let target = gpu
            .create_target(2, 2, vk::Format::B8G8R8A8_UNORM)
            .expect("could not create target");

        let mut uploader = gpu.uploader();
        target.clear(&mut uploader, CLEAR).expect("clear failed");
        let pixels = target
            .read_rgba8(&mut uploader, vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .expect("readback failed");
        drop(uploader);

        // The clear is specified in RGBA regardless of storage order, so readback must agree with
        // the R8G8B8A8 case rather than coming back swapped.
        for pixel in pixels.chunks_exact(4) {
            assert_eq!(pixel, CLEAR_RGBA8);
        }
        gpu.assert_no_validation_errors();
    }

    #[test]
    fn half_float_targets_read_back_quantised() {
        let gpu = TestGpu::shared();
        let target = gpu
            .create_target(2, 2, vk::Format::R16G16B16A16_SFLOAT)
            .expect("could not create target");

        let mut uploader = gpu.uploader();
        target.clear(&mut uploader, CLEAR).expect("clear failed");
        let pixels = target
            .read_rgba8(&mut uploader, vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .expect("readback failed");
        drop(uploader);

        // Half-float cannot hold 0.2 or 0.6 exactly, so allow one 8-bit step of slack.
        for pixel in pixels.chunks_exact(4) {
            for (actual, expected) in pixel.iter().zip(CLEAR_RGBA8) {
                assert!(
                    actual.abs_diff(expected) <= 1,
                    "expected ~{expected}, got {actual}"
                );
            }
        }
        gpu.assert_no_validation_errors();
    }
}
