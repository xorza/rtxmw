//! An offscreen colour target that can be rendered into and read back.

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use half::f16;

use crate::error::Result;
use crate::image_barrier::{self, COLOR_RANGE};
use crate::testing::test_gpu::TestGpu;

/// An offscreen image plus its view, sized and formatted for one test.
///
/// Cheap relative to device creation, so one per test is fine.
pub struct RenderTarget {
    gpu: &'static TestGpu,
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
    extent: vk::Extent2D,
    format: vk::Format,
}

impl std::fmt::Debug for RenderTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderTarget")
            .field("extent", &self.extent)
            .field("format", &self.format)
            .finish_non_exhaustive()
    }
}

impl RenderTarget {
    pub(crate) fn new(
        gpu: &'static TestGpu,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<Self> {
        let extent = vk::Extent2D { width, height };
        let device = gpu.device().raw();

        // STORAGE so a compute or ray tracing shader can write it; TRANSFER_SRC for readback;
        // COLOR_ATTACHMENT so a raster pass can target it too.
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        // SAFETY: `image_info` is fully initialised and the device is alive.
        let image = unsafe { device.create_image(&image_info, None)? };

        // SAFETY: `image` was just created on this device.
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = gpu
            .allocator()
            .allocate(&AllocationCreateDesc {
                name: "test render target",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::DedicatedImage(image),
            })
            .expect("failed to allocate render target memory");

        // SAFETY: the allocation matches the image's requirements.
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(COLOR_RANGE);
        // SAFETY: `image` is bound and alive.
        let view = unsafe { device.create_image_view(&view_info, None)? };

        Ok(Self {
            gpu,
            image,
            view,
            allocation: Some(allocation),
            extent,
            format,
        })
    }

    /// The underlying image.
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// A full-subresource colour view of the image.
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    /// Size in pixels.
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// The format the image was created with.
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Fills the whole image with `colour`, leaving it in `TRANSFER_DST_OPTIMAL`.
    ///
    /// Enough to exercise the harness before there is anything to render.
    pub fn clear(&self, colour: [f32; 4]) -> Result<()> {
        self.gpu.submit_and_wait(|device, cmd| {
            // SAFETY: the command buffer is recording and the image belongs to this device.
            unsafe {
                image_barrier::transition(
                    device,
                    cmd,
                    self.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                device.cmd_clear_color_image(
                    cmd,
                    self.image,
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
    pub fn read_rgba8(&self, current_layout: vk::ImageLayout) -> Result<Vec<u8>> {
        let pixel = PixelFormat::of(self.format);
        let pixels = (self.extent.width * self.extent.height) as usize;
        let size = (pixels * pixel.bytes_per_pixel()) as vk::DeviceSize;

        let device = self.gpu.device().raw();
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `buffer_info` is initialised and the device is alive.
        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        // SAFETY: `buffer` was just created on this device.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation = self
            .gpu
            .allocator()
            .allocate(&AllocationCreateDesc {
                name: "render target readback",
                requirements,
                location: MemoryLocation::GpuToCpu,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
            })
            .expect("failed to allocate readback memory");
        // SAFETY: the allocation matches the buffer's requirements.
        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())? };

        let copy_result = self.gpu.submit_and_wait(|device, cmd| {
            // SAFETY: the command buffer is recording and both resources are alive.
            unsafe {
                image_barrier::transition(
                    device,
                    cmd,
                    self.image,
                    current_layout,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
                let region = vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: self.extent.width,
                        height: self.extent.height,
                        depth: 1,
                    });
                device.cmd_copy_image_to_buffer(
                    cmd,
                    self.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    buffer,
                    &[region],
                );
            }
        });

        let rgba = copy_result.map(|()| {
            let bytes = allocation
                .mapped_slice()
                .expect("readback allocation is not host-visible");
            pixel.to_rgba8(&bytes[..size as usize], pixels)
        });

        self.gpu
            .allocator()
            .free(allocation)
            .expect("failed to free readback memory");
        // SAFETY: the submission completed, so nothing references the buffer.
        unsafe { device.destroy_buffer(buffer, None) };

        rgba
    }
}

impl Drop for RenderTarget {
    fn drop(&mut self) {
        let device = self.gpu.device().raw();
        // No `device_wait_idle` here: it requires external synchronisation of every queue, so
        // calling it while another test thread submits is itself a threading violation. Every
        // `submit_and_wait` blocks on a fence before returning, so this target's work is already
        // complete and nothing else can reference it.
        // SAFETY: as above — no in-flight work touches this image.
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            self.gpu
                .allocator()
                .free(allocation)
                .expect("failed to free render target memory");
        }
    }
}

/// How to turn raw target bytes into 8-bit RGBA.
#[derive(Debug, Clone, Copy)]
enum PixelFormat {
    /// Already 8-bit RGBA.
    Rgba8,
    /// 8-bit BGRA, needing a channel swap.
    Bgra8,
    /// Half-float RGBA, needing a linear-to-8-bit conversion.
    Rgba16Float,
}

impl PixelFormat {
    fn of(format: vk::Format) -> Self {
        match format {
            vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => Self::Rgba8,
            vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => Self::Bgra8,
            vk::Format::R16G16B16A16_SFLOAT => Self::Rgba16Float,
            other => panic!("readback is not implemented for {other:?}"),
        }
    }

    fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgba8 | Self::Bgra8 => 4,
            Self::Rgba16Float => 8,
        }
    }

    fn to_rgba8(self, bytes: &[u8], pixels: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(pixels * 4);
        match self {
            Self::Rgba8 => out.extend_from_slice(bytes),
            Self::Bgra8 => {
                for chunk in bytes.chunks_exact(4) {
                    out.extend_from_slice(&[chunk[2], chunk[1], chunk[0], chunk[3]]);
                }
            }
            Self::Rgba16Float => {
                for chunk in bytes.chunks_exact(2) {
                    let value = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                    out.push((value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bgra_readback_swaps_channels_and_keeps_alpha() {
        let bytes = [10u8, 20, 30, 40];
        let rgba = PixelFormat::Bgra8.to_rgba8(&bytes, 1);
        assert_eq!(rgba, vec![30, 20, 10, 40]);
    }

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

        target.clear(CLEAR).expect("clear failed");
        let pixels = target
            .read_rgba8(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .expect("readback failed");

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

        target.clear(CLEAR).expect("clear failed");
        let pixels = target
            .read_rgba8(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .expect("readback failed");

        crate::testing::golden::assert_matches("cleared_target", &pixels, 16, 16);
        gpu.assert_no_validation_errors();
    }

    #[test]
    fn bgra_targets_read_back_in_rgba_order() {
        let gpu = TestGpu::shared();
        let target = gpu
            .create_target(2, 2, vk::Format::B8G8R8A8_UNORM)
            .expect("could not create target");

        target.clear(CLEAR).expect("clear failed");
        let pixels = target
            .read_rgba8(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .expect("readback failed");

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

        target.clear(CLEAR).expect("clear failed");
        let pixels = target
            .read_rgba8(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .expect("readback failed");

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

    #[test]
    fn half_float_readback_clamps_and_quantises() {
        // 1.0, 0.0, 2.0 (clamps to 1.0), 0.5 -> 255, 0, 255, 128
        let bytes: Vec<u8> = [0x3c00u16, 0x0000, 0x4000, 0x3800]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let rgba = PixelFormat::Rgba16Float.to_rgba8(&bytes, 1);
        assert_eq!(rgba, vec![255, 0, 255, 128]);
    }
}
