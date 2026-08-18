//! The primary visibility pass: one ray query per pixel, into an offscreen HDR image.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use rtxmw_gpu::{Binding, ComputePipeline, Device, Image};
use rtxmw_scene::Sun;

use crate::acceleration_structure::AccelerationStructure;
use crate::gbuffer::GBuffer;
use crate::geometry_buffers::GeometryBuffers;
use crate::light_buffer::LightBuffer;
use crate::material_buffers::MaterialBuffers;
use crate::shaders;
use crate::texture_array::TextureArray;

/// Everything lighting a cell, which arrives together and goes stale together.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Lighting {
    /// Sky light outdoors, the cell's own fixed term indoors.
    pub(crate) ambient: Vec3,
    /// How many entries of the light buffer to read.
    pub(crate) light_count: u32,
    /// The sun, for a cell with a sky.
    pub(crate) sun: Option<Sun>,
}

/// What the shader needs to turn a pixel into a ray, as its push constant block.
///
/// The combined inverse view-projection rather than the two matrices separately: Vulkan guarantees
/// only 128 bytes of push constants, and sending both plus the camera would need 144. Their product
/// carries everything unprojection requires.
///
/// **This block is now exactly 128 bytes — the guarantee, in full.** The field order is not
/// cosmetic either: std430 aligns a `vec3` to sixteen bytes, so each one has to land on a multiple
/// of sixteen or the shader reads the block shifted. That is why the sun's radius sits between its
/// direction and its colour rather than after both. Anything further has to move the block into a
/// buffer, which is also what motion vectors will need.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FrameConstants {
    inverse_view_projection: [f32; 16],
    camera_position: [f32; 3],
    light_count: u32,
    /// The cell's fixed lighting, standing in for every bounce the original engine could not
    /// compute. Interiors are mostly this.
    ambient: [f32; 3],
    /// How fast a pixel's ray cone widens with distance, in world units per unit travelled.
    ///
    /// A rasterizer gets its texture level from screen-space derivatives, which a compute shader
    /// does not have. This is what replaces them: the cone's width at the hit is its footprint on
    /// the surface, and the mip that matches that footprint is the one to sample.
    cone_spread: f32,
    /// Unit vector the sun's light travels along, from the sky toward the world.
    sun_direction: [f32; 3],
    /// Cosine of the sun's angular radius, which is what a shadow ray tests against.
    sun_cos_radius: f32,
    /// The sun's radiance. **Zero means no sun**, which is how an interior says it has no sky.
    sun_colour: [f32; 3],
    /// How many diffuse bounce rays each pixel casts to gather indirect light.
    ///
    /// Zero is not "no indirect light" but "every bounce ray escapes", which is the limit in which
    /// the estimator collapses back to a flat `albedo * ambient` fill — so it is exactly the
    /// lighting model this had before one bounce existed, and the A/B against it is honest.
    bounce_samples: u32,
}

impl FrameConstants {
    /// Builds the block from a camera's matrices.
    ///
    /// `projection` must be the reverse-Z Vulkan projection the shader assumes — it unprojects the
    /// near plane at depth 1, which is the only plane an infinite projection leaves invertible.
    pub(crate) fn new(
        view: Mat4,
        projection: Mat4,
        camera_position: Vec3,
        lighting: Lighting,
        cone_spread: f32,
        bounce_samples: u32,
    ) -> Self {
        // No sun is a black one: every term it feeds is a multiplication, so the shader needs no
        // flag to branch on and an interior costs nothing for having no sky.
        let sun = lighting.sun.unwrap_or(Sun {
            direction: Vec3::NEG_Z,
            colour: Vec3::ZERO,
            angular_radius: 0.0,
        });
        Self {
            inverse_view_projection: (projection * view).inverse().to_cols_array(),
            camera_position: camera_position.to_array(),
            light_count: lighting.light_count,
            ambient: lighting.ambient.to_array(),
            cone_spread,
            sun_direction: sun.direction.to_array(),
            sun_cos_radius: sun.angular_radius.cos(),
            sun_colour: sun.colour.to_array(),
            bounce_samples,
        }
    }

    /// How wide one pixel's ray cone grows per unit of distance.
    ///
    /// Two pixels' worth of vertical field of view divided by the height, which is the angle one
    /// pixel subtends. Derived from the projection rather than passed separately so it cannot
    /// disagree with the matrix the same frame is unprojected with.
    pub(crate) fn cone_spread_from(projection: Mat4, height: u32) -> f32 {
        // The projection's [1][1] is `1 / tan(fov_y / 2)`, negated by the Vulkan Y flip.
        let cot_half_fov = projection.y_axis.y.abs();
        2.0 / (cot_half_fov * height as f32)
    }
}

/// Everything a trace reads that changes when the cell does.
///
/// A bundle rather than seven more parameters: they arrive and go stale together — one cell's worth
/// of device data — and as a positional list of references, four of which are buffers, transposing
/// two would compile and then read the wrong table.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SceneBindings<'a> {
    pub(crate) structure: &'a AccelerationStructure,
    pub(crate) geometry: &'a GeometryBuffers,
    pub(crate) tables: &'a MaterialBuffers,
    pub(crate) lights: &'a LightBuffer,
    pub(crate) textures: &'a TextureArray,
}

/// Descriptors for tracing primary rays, over a compute pipeline.
///
/// A compute shader with inline ray queries rather than a ray tracing pipeline: there is one
/// material path and no recursion, so a shader binding table would add a build step and a dispatch
/// indirection for nothing. Revisit when a hit needs to launch further rays it cannot inline.
#[derive(Debug)]
pub(crate) struct VisibilityPass {
    pipeline: ComputePipeline,
}

/// Matches `local_size_x`/`local_size_y` in `primary_visibility.comp`.
///
/// Declared in the shader and needed here to size the dispatch, so the two have to agree; a
/// mismatch under-covers the image and leaves a band of it never written.
const WORKGROUP: u32 = 8;

impl VisibilityPass {
    /// Creates the descriptor set and pipeline.
    ///
    /// `max_textures` sizes the bindless array's binding. It is a ceiling rather than a count: the
    /// set is allocated with room for it and each scene writes as many slots as it actually has.
    pub(crate) fn new(device: &Device, max_textures: u32) -> rtxmw_gpu::Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(
                device,
                &[
                    Binding::acceleration_structure(0),
                    Binding::storage_image(1),
                    Binding::storage_buffer(2),
                    Binding::storage_buffer(3),
                    Binding::storage_buffer(4),
                    Binding::storage_buffer(5),
                    Binding::storage_buffer(6),
                    Binding::storage_buffer(7),
                    // The G-buffer the trace splits itself into.
                    Binding::storage_image(8),
                    Binding::storage_image(9),
                    Binding::storage_image(10),
                    // Last, because Vulkan allows a variable descriptor count only on a set's final
                    // binding. Adding anything after this one moves it — validation rejects the set
                    // outright, which is how this was caught.
                    Binding::variable_samplers(11, max_textures),
                ],
                size_of::<FrameConstants>() as u32,
                shaders::primary_visibility(),
            )?,
        })
    }

    /// Points the pass at the scene it traces and the image it writes.
    ///
    /// Takes `&mut self` because it rewrites the one descriptor set the pass owns, which must not
    /// happen while a dispatch using it is in flight. One set is deliberate: the scene changes per
    /// cell, not per frame.
    pub(crate) fn bind(&mut self, scene: SceneBindings<'_>, target: &Image, gbuffer: &GBuffer) {
        let structures = [scene.structure.raw()];
        let mut acceleration = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&structures);
        // Set by hand because the count lives on the write while the handles live on the chained
        // struct, and no builder method bridges the two. It must be non-zero: the write is rejected
        // outright otherwise, rather than binding nothing.
        let mut scene_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut acceleration);
        scene_write.descriptor_count = 1;

        let geometry_info = [vk::DescriptorBufferInfo::default()
            .buffer(scene.tables.geometries().raw())
            .range(vk::WHOLE_SIZE)];
        let geometry_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&geometry_info);

        let material_info = [vk::DescriptorBufferInfo::default()
            .buffer(scene.tables.materials().raw())
            .range(vk::WHOLE_SIZE)];
        let material_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&material_info);

        let texture_infos = scene.textures.descriptors();
        let texture_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(11)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&texture_infos);

        // The storage images go through the pipeline's own helper: the emissive-and-sky target at
        // binding one, and the G-buffer as a run from eight.
        self.pipeline.bind_storage_images(1, &[target]);
        self.pipeline.bind_storage_images(
            8,
            &[
                gbuffer.albedo(),
                gbuffer.normal_depth(),
                gbuffer.illumination(),
            ],
        );

        let position_info = [vk::DescriptorBufferInfo::default()
            .buffer(scene.geometry.positions().raw())
            .range(vk::WHOLE_SIZE)];
        let position_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(7)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&position_info);

        let light_info = [vk::DescriptorBufferInfo::default()
            .buffer(scene.lights.buffer().raw())
            .range(vk::WHOLE_SIZE)];
        let light_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&light_info);

        let index_info = [vk::DescriptorBufferInfo::default()
            .buffer(scene.geometry.indices().raw())
            .range(vk::WHOLE_SIZE)];
        let index_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&index_info);

        let attribute_info = [vk::DescriptorBufferInfo::default()
            .buffer(scene.geometry.attributes().raw())
            .range(vk::WHOLE_SIZE)];
        let attribute_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(5)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&attribute_info);

        // SAFETY: every write names this pass's own set, and no dispatch using it is in flight.
        unsafe {
            self.pipeline.device().update_descriptor_sets(
                &[
                    scene_write,
                    geometry_write,
                    material_write,
                    texture_write,
                    index_write,
                    attribute_write,
                    light_write,
                    position_write,
                ],
                &[],
            )
        };
    }

    /// Records one dispatch covering `extent`.
    ///
    /// The target must already be in `GENERAL`, and the caller is responsible for the barrier
    /// afterwards — it knows whether the image is next blitted, read back or sampled.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state, and [`VisibilityPass::bind`] must have run
    /// for the scene and target in use.
    pub(crate) unsafe fn record(
        &self,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        constants: &FrameConstants,
    ) {
        // SAFETY: the caller guarantees the command buffer is recording and the set is bound.
        unsafe {
            self.pipeline.dispatch(
                command_buffer,
                [
                    extent.width.div_ceil(WORKGROUP),
                    extent.height.div_ceil(WORKGROUP),
                    1,
                ],
                bytemuck::bytes_of(constants),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_push_block_fits_the_guaranteed_range() {
        // Vulkan promises 128 bytes and no more; exceeding it works on this GPU and fails elsewhere.
        assert!(size_of::<FrameConstants>() <= 128);
        // Exactly the guarantee, with nothing to spare. The field order is load-bearing too:
        // std430 aligns a `vec3` to sixteen bytes, so each has to land on a multiple of sixteen or
        // the shader reads the whole block shifted.
        assert_eq!(size_of::<FrameConstants>(), 128);
        let constants = FrameConstants::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::ZERO,
            Lighting {
                ambient: Vec3::ZERO,
                light_count: 0,
                sun: None,
            },
            0.0,
            0,
        );
        let base = std::ptr::from_ref(&constants).cast::<u8>();
        for (name, field) in [
            (
                "camera_position",
                std::ptr::from_ref(&constants.camera_position).cast::<u8>(),
            ),
            (
                "ambient",
                std::ptr::from_ref(&constants.ambient).cast::<u8>(),
            ),
            (
                "sun_direction",
                std::ptr::from_ref(&constants.sun_direction).cast::<u8>(),
            ),
            (
                "sun_colour",
                std::ptr::from_ref(&constants.sun_colour).cast::<u8>(),
            ),
        ] {
            // SAFETY: both pointers are into the same live value.
            let at = unsafe { field.offset_from(base) };
            assert_eq!(
                at % 16,
                0,
                "{name} sits at {at}, where std430 will not look for it"
            );
        }

        // One pixel of a 90-degree, 100-pixel-tall view subtends 2*tan(45)/100 = 0.02 units per
        // unit of distance.
        let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
            std::f32::consts::FRAC_PI_2,
            1.0,
            0.05,
        );
        let spread = FrameConstants::cone_spread_from(projection, 100);
        assert!((spread - 0.02).abs() < 1e-4, "got {spread}");
        // Twice the pixels, half the angle each.
        let finer = FrameConstants::cone_spread_from(projection, 200);
        assert!((finer - spread / 2.0).abs() < 1e-5);
    }

    #[test]
    fn the_inverse_maps_a_projected_point_back_to_where_it_started() {
        let eye = Vec3::new(1.0, 2.0, 3.0);
        let view = glam::camera::rh::view::look_to_mat4(eye, Vec3::X, Vec3::Z);
        // The Vulkan variant, matching the camera — it flips Y for Vulkan's Y-down NDC, and a test
        // built on the DirectX one would agree with itself while disagreeing with the renderer.
        let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
            75f32.to_radians(),
            16.0 / 9.0,
            0.05,
        );
        let constants = FrameConstants::new(
            view,
            projection,
            eye,
            Lighting {
                ambient: Vec3::ZERO,
                light_count: 0,
                sun: None,
            },
            0.0,
            0,
        );

        // Round-trip a world point through the forward transform and back through the stored
        // inverse. Anything that transposed or reordered the matrix survives multiplication but
        // not this.
        let world = glam::Vec4::new(40.0, -12.0, 7.0, 1.0);
        let clip = projection * view * world;
        let back = Mat4::from_cols_array(&constants.inverse_view_projection) * clip;
        let recovered = back.truncate() / back.w;
        assert!(
            (recovered - world.truncate()).length() < 1e-2,
            "recovered {recovered:?} from {world:?}"
        );
    }
}
