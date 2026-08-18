//! Copying an image back to host memory, and writing it out as a PNG.
//!
//! Production rather than test-only: a screenshot is a feature, and the thing worth reading back is
//! the renderer's own output image, which no test owns. Golden-image comparison builds on this and
//! *is* test-only, so it stays behind the `internals` feature.

use std::path::Path;

use ash::vk;
use half::f16;

use crate::buffer::{Buffer, BufferMemory};
use crate::error::Result;
use crate::image::Image;
use crate::image_barrier;
use crate::uploader::Uploader;

/// Copies `image` to host memory and returns it as 8-bit RGBA, row-major, top row first.
///
/// `current_layout` is the layout the image is in on entry; it is transitioned to
/// `TRANSFER_SRC_OPTIMAL` and left there.
pub fn image_to_rgba8(
    uploader: &mut Uploader,
    image: &Image,
    current_layout: vk::ImageLayout,
) -> Result<Vec<u8>> {
    let pixel = PixelFormat::of(image.format());
    let extent = image.extent();
    let pixels = (extent.width * extent.height) as usize;
    let size = (pixels * pixel.bytes_per_pixel()) as vk::DeviceSize;

    let readback = Buffer::new(
        uploader.memory(),
        "image readback",
        size,
        vk::BufferUsageFlags::TRANSFER_DST,
        BufferMemory::Readback,
    )?;
    let destination = readback.raw();
    let source = image.raw();

    uploader.submit_and_wait(|device, cmd| {
        // SAFETY: the command buffer is recording and both resources are alive.
        unsafe {
            image_barrier::transition(
                device,
                cmd,
                source,
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
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                });
            device.cmd_copy_image_to_buffer(
                cmd,
                source,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                destination,
                &[region],
            );
        }
    })?;

    let bytes = readback
        .mapped()
        .expect("readback memory is host-visible by construction");
    Ok(pixel.to_rgba8(&bytes[..size as usize], pixels))
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

/// Writes 8-bit RGBA pixels to `path`.
///
/// Alpha is written as stored. A caller whose alpha channel carries something other than coverage —
/// the visibility shader puts a hit flag there — has to flatten it first, or a viewer composites
/// every miss as transparent and the image reads as blown out.
pub fn write_png(path: &Path, pixels: &[u8], width: u32, height: u32) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("could not create the image directory");
    }
    let file = std::fs::File::create(path).expect("could not create the image file");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .expect("could not write the png header")
        .write_image_data(pixels)
        .expect("could not write the png data");
}

/// Writes 8-bit RGBA pixels to `path` with every alpha forced opaque.
///
/// For images whose alpha is data rather than coverage.
pub fn write_png_opaque(path: &Path, pixels: &[u8], width: u32, height: u32) {
    let mut opaque = pixels.to_vec();
    for pixel in opaque.chunks_exact_mut(4) {
        pixel[3] = 0xFF;
    }
    write_png(path, &opaque, width, height);
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
