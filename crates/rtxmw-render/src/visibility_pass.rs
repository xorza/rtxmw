//! The primary visibility pass: one ray query per pixel, into an offscreen HDR image.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use rtxmw_gpu::{Device, Image};

use crate::acceleration_structure::AccelerationStructure;
use crate::material_buffers::MaterialBuffers;
use crate::shaders;

/// What the shader needs to turn a pixel into a ray, as its push constant block.
///
/// The combined inverse view-projection rather than the two matrices separately: Vulkan guarantees
/// only 128 bytes of push constants, and sending both plus the camera would need 144. Their product
/// carries everything unprojection requires.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FrameConstants {
    inverse_view_projection: [f32; 16],
    camera_position: [f32; 3],
    /// The block is a multiple of 16 bytes with this; the shader ignores it.
    padding: u32,
}

impl FrameConstants {
    /// Builds the block from a camera's matrices.
    ///
    /// `projection` must be the reverse-Z Vulkan projection the shader assumes — it unprojects the
    /// near plane at depth 1, which is the only plane an infinite projection leaves invertible.
    pub fn new(view: Mat4, projection: Mat4, camera_position: Vec3) -> Self {
        Self {
            inverse_view_projection: (projection * view).inverse().to_cols_array(),
            camera_position: camera_position.to_array(),
            padding: 0,
        }
    }
}

/// Compute pipeline and descriptors for tracing primary rays.
///
/// A compute shader with inline ray queries rather than a ray tracing pipeline: there is one
/// material path and no recursion, so a shader binding table would add a build step and a dispatch
/// indirection for nothing. Revisit when a hit needs to launch further rays it cannot inline.
pub struct VisibilityPass {
    /// A handle copy, not an owner: the real `Device` outlives this by construction.
    device: ash::Device,
    descriptor_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

// `ash::Device` is a table of function pointers and implements no `Debug`.
impl std::fmt::Debug for VisibilityPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisibilityPass")
            .field("pipeline", &self.pipeline)
            .field("set", &self.set)
            .finish_non_exhaustive()
    }
}

/// Matches `local_size_x`/`local_size_y` in `primary_visibility.comp`.
///
/// Declared in the shader and needed here to size the dispatch, so the two have to agree; a
/// mismatch under-covers the image and leaves a band of it never written.
const WORKGROUP: u32 = 8;

impl VisibilityPass {
    /// Creates the layout, pool, descriptor set and pipeline.
    pub fn new(device: &Device) -> rtxmw_gpu::Result<Self> {
        let raw = device.raw();

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: `layout_info` is fully initialised and the device is alive.
        let descriptor_layout = unsafe { raw.create_descriptor_set_layout(&layout_info, None)? };

        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&sizes)
            .max_sets(1);
        // SAFETY: same.
        let pool = unsafe { raw.create_descriptor_pool(&pool_info, None)? };

        let layouts = [descriptor_layout];
        let allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: the pool was just created with room for exactly this set.
        let set = unsafe { raw.allocate_descriptor_sets(&allocate)? }[0];

        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(size_of::<FrameConstants>() as u32)];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&ranges);
        // SAFETY: same.
        let pipeline_layout = unsafe { raw.create_pipeline_layout(&pipeline_layout_info, None)? };

        let module_info = vk::ShaderModuleCreateInfo::default().code(shaders::primary_visibility());
        // SAFETY: the build script only emits modules `spirv-val` accepted.
        let module = unsafe { raw.create_shader_module(&module_info, None)? };

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        // SAFETY: every referenced object is alive.
        let pipeline = unsafe {
            raw.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        };
        // The module is baked into the pipeline, so it can go regardless of the outcome.
        // SAFETY: pipeline creation has returned, and nothing else references the module.
        unsafe { raw.destroy_shader_module(module, None) };

        let pipeline = match pipeline {
            Ok(pipelines) => pipelines[0],
            Err((_, e)) => return Err(e.into()),
        };

        Ok(Self {
            device: raw.clone(),
            descriptor_layout,
            pool,
            set,
            pipeline_layout,
            pipeline,
        })
    }

    /// Points the pass at the scene it traces and the image it writes.
    ///
    /// Takes `&mut self` because it rewrites the one descriptor set the pass owns, which must not
    /// happen while a dispatch using it is in flight. One set is deliberate: the scene changes per
    /// cell, not per frame.
    pub fn bind(
        &mut self,
        scene: &AccelerationStructure,
        target: &Image,
        tables: &MaterialBuffers,
    ) {
        let structures = [scene.raw()];
        let mut acceleration = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&structures);
        // Set by hand because the count lives on the write while the handles live on the chained
        // struct, and no builder method bridges the two. It must be non-zero: the write is rejected
        // outright otherwise, rather than binding nothing.
        let mut scene_write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut acceleration);
        scene_write.descriptor_count = 1;

        let images = [vk::DescriptorImageInfo::default()
            .image_view(target.view())
            .image_layout(vk::ImageLayout::GENERAL)];
        let image_write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&images);

        let geometry_info = [vk::DescriptorBufferInfo::default()
            .buffer(tables.geometries().raw())
            .range(vk::WHOLE_SIZE)];
        let geometry_write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&geometry_info);

        let material_info = [vk::DescriptorBufferInfo::default()
            .buffer(tables.materials().raw())
            .range(vk::WHOLE_SIZE)];
        let material_write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&material_info);

        // SAFETY: every write names this pass's own set, and no dispatch using it is in flight.
        unsafe {
            self.device.update_descriptor_sets(
                &[scene_write, image_write, geometry_write, material_write],
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
    pub unsafe fn record(
        &self,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        constants: &FrameConstants,
    ) {
        // SAFETY: the caller guarantees the command buffer is recording and the set is bound.
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.set],
                &[],
            );
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(constants),
            );
            self.device.cmd_dispatch(
                command_buffer,
                extent.width.div_ceil(WORKGROUP),
                extent.height.div_ceil(WORKGROUP),
                1,
            );
        }
    }
}

impl Drop for VisibilityPass {
    fn drop(&mut self) {
        // SAFETY: the caller waits for device idle before dropping the renderer, and every test
        // submission blocks on a fence.
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            // Frees the set with it.
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
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
        assert_eq!(size_of::<FrameConstants>(), 80);
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
        let constants = FrameConstants::new(view, projection, eye);

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
