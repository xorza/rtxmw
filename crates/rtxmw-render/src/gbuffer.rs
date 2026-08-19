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

/// The mirror-like part of a surface's response in `rgb`, how sharp it is in `a`.
///
/// Half float like the rest: both are fractions, and neither is a quantity a step of a thousandth
/// could be wrong about.
const MATERIAL_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Where each pixel's surface was on the previous frame's screen, in pixels.
///
/// **Full floats, unlike everything else here.** A motion vector spans the frame — up to a couple of
/// thousand pixels when the camera turns — and a half float's mantissa is eleven bits, so above 1024
/// it can only land on whole pixels and above 2048 on every other one. That is the range temporal
/// reuse cares most about getting right, and eight megabytes at 1080p is not the constraint.
const MOTION_FORMAT: vk::Format = vk::Format::R32G32_SFLOAT;

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
    motion: Image,
    material: Image,
}

impl GBuffer {
    /// Allocates every image at `extent`.
    pub(crate) fn new(memory: &Memory, extent: vk::Extent2D) -> rtxmw_gpu::Result<Self> {
        // Every one is written by a compute shader and read by another, so all of them are storage
        // images; the motion target is also the one thing here that leaves the device.
        let storage = vk::ImageUsageFlags::STORAGE;
        let image = |name: &str, format, usage| Image::new(memory, name, extent, format, usage);
        Ok(Self {
            albedo: image("gbuffer albedo", ALBEDO_FORMAT, storage)?,
            normal_depth: image("gbuffer normal and depth", NORMAL_DEPTH_FORMAT, storage)?,
            illumination: [
                image("illumination", ILLUMINATION_FORMAT, storage)?,
                image("illumination scratch", ILLUMINATION_FORMAT, storage)?,
            ],
            // Copied as well as stored: it goes to an upscaler that has not arrived yet, so reading
            // it back is the only way to assert on what the shader wrote.
            motion: image(
                "gbuffer motion",
                MOTION_FORMAT,
                storage | vk::ImageUsageFlags::TRANSFER_SRC,
            )?,
            material: image(
                "gbuffer material",
                MATERIAL_FORMAT,
                storage | vk::ImageUsageFlags::TRANSFER_SRC,
            )?,
        })
    }

    pub(crate) fn albedo(&self) -> &Image {
        &self.albedo
    }

    pub(crate) fn normal_depth(&self) -> &Image {
        &self.normal_depth
    }

    /// Where each pixel's surface was last frame, as a displacement in pixels.
    pub(crate) fn motion(&self) -> &Image {
        &self.motion
    }

    /// Specular albedo and roughness, which an upscaler reads alongside the colour.
    pub(crate) fn material(&self) -> &Image {
        &self.material
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
    pub(crate) fn images(&self) -> [&Image; 6] {
        [
            &self.albedo,
            &self.normal_depth,
            &self.illumination[0],
            &self.illumination[1],
            &self.motion,
            &self.material,
        ]
    }
}
