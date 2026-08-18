//! Every texture a cell needs, in one bindless array.

use ash::vk;
use rtxmw_gpu::{Device, Image, Uploader};
use rtxmw_texture::{Texture, TextureFormat};

/// Bindless sampled images plus the sampler they share.
///
/// One descriptor binding holding a runtime-sized array, indexed at a hit by a value read from the
/// material table. That indexing is divergent across a warp, which is why the shader has to mark it
/// `nonuniformEXT` and why the device enables non-uniform sampled-image indexing.
pub(crate) struct TextureArray {
    device: ash::Device,
    /// Slot zero, and the view every unresolvable slot points at. A material with no texture, or
    /// one naming a file that does not exist, samples this — 45 of the shipped library's 4,311
    /// references are dangling, so "missing" is a normal case rather than an error. Magenta because
    /// it has to be unmistakable: a missing texture rendering grey would read as a lighting problem
    /// and be chased in the wrong place.
    fallback: Image,
    /// One per texture the scene named, `None` where the file could not be decoded. Sharing the
    /// fallback's view rather than uploading a copy per miss keeps them a single image.
    slots: Vec<Option<Image>>,
    sampler: vk::Sampler,
}

// `ash::Device` is a table of function pointers and implements no `Debug`.
impl std::fmt::Debug for TextureArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextureArray")
            .field("slots", &self.slots.len())
            .finish_non_exhaustive()
    }
}

impl TextureArray {
    /// Uploads every texture, substituting the fallback wherever one is absent.
    ///
    /// Slot `n` is the texture the scene's path list names at `n`, offset by one — slot zero is
    /// always the fallback, so a material's texture id maps to `id + 1`.
    pub(crate) fn upload(
        device: &Device,
        uploader: &mut Uploader,
        textures: &[Option<Texture>],
    ) -> rtxmw_gpu::Result<Self> {
        let fallback = upload_one(uploader, &fallback_texture())?;
        let mut slots = Vec::with_capacity(textures.len());
        for texture in textures {
            slots.push(match texture {
                Some(texture) => Some(upload_one(uploader, texture)?),
                None => None,
            });
        }

        // Anisotropy is left off deliberately: it is a rasterizer's answer to a footprint problem
        // that a ray tracer solves with ray differentials instead, and turning it on here would
        // hide the absence of those rather than fix it.
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            // Morrowind tiles almost everything and authors UVs well outside [0, 1].
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .max_lod(vk::LOD_CLAMP_NONE);
        // SAFETY: `sampler_info` is fully initialised and the device is alive.
        let sampler = unsafe { device.raw().create_sampler(&sampler_info, None)? };

        Ok(Self {
            device: device.raw().clone(),
            fallback,
            slots,
            sampler,
        })
    }

    /// How many slots the array holds, fallback included.
    pub(crate) fn len(&self) -> u32 {
        self.slots.len() as u32 + 1
    }

    /// The descriptor image infos for the whole array, in slot order.
    ///
    /// Slot zero is the fallback, so a material's texture id addresses `id + 1`.
    pub(crate) fn descriptors(&self) -> Vec<vk::DescriptorImageInfo> {
        let info = |image: &Image| {
            vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(image.view())
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        };
        let mut out = Vec::with_capacity(self.slots.len() + 1);
        out.push(info(&self.fallback));
        out.extend(
            self.slots
                .iter()
                .map(|slot| info(slot.as_ref().unwrap_or(&self.fallback))),
        );
        out
    }
}

impl Drop for TextureArray {
    fn drop(&mut self) {
        // SAFETY: the caller waits for device idle before replacing a scene, and every upload
        // blocks on a fence.
        unsafe { self.device.destroy_sampler(self.sampler, None) };
    }
}

/// A 2x2 magenta texture, for the fallback slot.
fn fallback_texture() -> Texture {
    const MAGENTA: [u8; 4] = [255, 0, 255, 255];
    Texture::from_pixels(TextureFormat::Rgba8, 2, 2, MAGENTA.repeat(4))
}

/// Creates an image sized for `texture` and copies its whole chain in.
fn upload_one(uploader: &mut Uploader, texture: &Texture) -> rtxmw_gpu::Result<Image> {
    let image = Image::mipped(
        uploader.memory(),
        "scene texture",
        vk::Extent2D {
            width: texture.width(),
            height: texture.height(),
        },
        vulkan_format(texture.format()),
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        texture.levels().len() as u32,
    )?;

    let regions: Vec<vk::BufferImageCopy> = texture
        .levels()
        .iter()
        .enumerate()
        .map(|(level, mip)| {
            vk::BufferImageCopy::default()
                .buffer_offset(mip.offset as vk::DeviceSize)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(level as u32)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: mip.width,
                    height: mip.height,
                    depth: 1,
                })
        })
        .collect();

    uploader.upload_image(&image, texture.data(), &regions)?;
    Ok(image)
}

/// The Vulkan format to sample a decoded texture through.
///
/// sRGB throughout, because every one of these is albedo: vanilla Morrowind has no normal or
/// roughness maps at all. Sampling them as UNORM would feed gamma-encoded values to a renderer that
/// works in linear light, which darkens midtones by roughly a factor of two and cannot be tuned out.
fn vulkan_format(format: TextureFormat) -> vk::Format {
    match format {
        // BC1 with alpha rather than RGB-only: Morrowind uses its one-bit alpha for the foliage and
        // grates the alpha test needs, and the RGB view would discard it.
        TextureFormat::Bc1 => vk::Format::BC1_RGBA_SRGB_BLOCK,
        TextureFormat::Bc2 => vk::Format::BC2_SRGB_BLOCK,
        TextureFormat::Bgra8 => vk::Format::B8G8R8A8_SRGB,
        TextureFormat::Rgba8 => vk::Format::R8G8B8A8_SRGB,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shipped_format_maps_to_an_srgb_view() {
        // A UNORM view of albedo is the classic double-gamma bug, and it looks merely "a bit dark"
        // rather than obviously wrong, so the mapping is pinned rather than trusted.
        for format in [
            TextureFormat::Bc1,
            TextureFormat::Bc2,
            TextureFormat::Bgra8,
            TextureFormat::Rgba8,
        ] {
            let name = format!("{:?}", vulkan_format(format));
            assert!(name.contains("SRGB"), "{format:?} maps to {name}");
        }
        // BC1 must keep its alpha channel, which is where the cutout lives.
        assert_eq!(
            vulkan_format(TextureFormat::Bc1),
            vk::Format::BC1_RGBA_SRGB_BLOCK
        );
    }

    #[test]
    fn the_fallback_is_a_real_uploadable_texture() {
        let fallback = fallback_texture();
        assert_eq!((fallback.width(), fallback.height()), (2, 2));
        assert_eq!(fallback.levels().len(), 1);
        assert_eq!(fallback.data().len(), 16);
        assert_eq!(&fallback.data()[..4], &[255, 0, 255, 255]);
    }
}
