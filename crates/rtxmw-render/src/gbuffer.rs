//! What a surface is, kept apart from the light that reaches it.

use ash::vk;
use rtxmw_gpu::{Image, Memory};

/// Base colour.
///
/// Half float rather than the eight bits a reflectance in `0..1` would seem to need, because the
/// error does not stay small: the composite multiplies albedo by unbounded illumination, so a
/// quantisation step is scaled by however bright the light is. Measured against the same frame
/// rendered without the split, eight bits moved the mean pixel by 0.32 of 255 and the worst by 37;
/// half float moves the worst by 1. Exact recombination is the property this whole split rests on,
/// and eight megabytes at 1080p is not the constraint.
const ALBEDO_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// World normal in `xyz`, distance from the eye in `w`. Half floats because the depth is a world
/// distance in Morrowind units, which reach tens of thousands outdoors.
const NORMAL_DEPTH_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Light arriving at a surface, with its own albedo divided out. Unbounded, so half float.
const ILLUMINATION_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// The trace's output, split into what a surface is and what light reaches it.
///
/// **The split is what makes denoising possible.** All the noise is in the lighting — the albedo a
/// ray reads from a texture is exact — so filtering the lighting alone smooths the noise without
/// touching a texel of surface detail. A filter run on the combined image cannot tell one from the
/// other and blurs both.
///
/// The illumination is double-buffered because an à-trous filter reads a whole image and writes a
/// whole image, several times over; it cannot do that in place.
#[derive(Debug)]
pub(crate) struct GBuffer {
    albedo: Image,
    normal_depth: Image,
    /// The one the trace writes and the composite reads. The filter swaps which is which.
    illumination: [Image; 2],
}

impl GBuffer {
    /// Allocates every image at `extent`.
    pub(crate) fn new(memory: &Memory, extent: vk::Extent2D) -> rtxmw_gpu::Result<Self> {
        // Every one is written by a compute shader and read by another, so all of them are storage
        // images; none is ever copied, so none needs a transfer usage.
        let usage = vk::ImageUsageFlags::STORAGE;
        let image = |name: &str, format| Image::new(memory, name, extent, format, usage);
        Ok(Self {
            albedo: image("gbuffer albedo", ALBEDO_FORMAT)?,
            normal_depth: image("gbuffer normal and depth", NORMAL_DEPTH_FORMAT)?,
            illumination: [
                image("illumination", ILLUMINATION_FORMAT)?,
                image("illumination scratch", ILLUMINATION_FORMAT)?,
            ],
        })
    }

    pub(crate) fn albedo(&self) -> &Image {
        &self.albedo
    }

    pub(crate) fn normal_depth(&self) -> &Image {
        &self.normal_depth
    }

    /// The image the trace writes its lighting into, and the one a filter reads first.
    pub(crate) fn illumination(&self) -> &Image {
        &self.illumination[0]
    }

    /// The other one, which a filter writes into before swapping.
    pub(crate) fn scratch(&self) -> &Image {
        &self.illumination[1]
    }

    /// Every image, for a caller transitioning them together.
    pub(crate) fn images(&self) -> [&Image; 4] {
        [
            &self.albedo,
            &self.normal_depth,
            &self.illumination[0],
            &self.illumination[1],
        ]
    }
}
