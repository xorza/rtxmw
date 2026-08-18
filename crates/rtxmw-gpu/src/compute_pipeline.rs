//! A compute pipeline with one descriptor set, for passes that need nothing exotic.

use ash::vk;

use crate::device::Device;
use crate::error::Result;

/// One descriptor a pass reads or writes, in binding order.
///
/// Everything here is visible to compute and nothing else, which is the one thing this shape will
/// not do; a graphics pass needs its own.
#[derive(Debug, Clone, Copy)]
pub struct Binding {
    binding: u32,
    kind: vk::DescriptorType,
    count: u32,
    /// Sized at allocation rather than at layout creation, and only partially filled.
    variable: bool,
}

impl Binding {
    fn single(binding: u32, kind: vk::DescriptorType) -> Self {
        Self {
            binding,
            kind,
            count: 1,
            variable: false,
        }
    }

    /// A storage image the shader writes, or reads and writes.
    pub fn storage_image(binding: u32) -> Self {
        Self::single(binding, vk::DescriptorType::STORAGE_IMAGE)
    }

    /// A storage buffer.
    pub fn storage_buffer(binding: u32) -> Self {
        Self::single(binding, vk::DescriptorType::STORAGE_BUFFER)
    }

    /// A top-level acceleration structure to trace against.
    pub fn acceleration_structure(binding: u32) -> Self {
        Self::single(binding, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
    }

    /// A bindless array of up to `max` combined image samplers.
    ///
    /// **Must be the last binding in the set.** Vulkan allows a variable descriptor count only on a
    /// set's final element, and moving it earlier is rejected outright rather than silently
    /// misbehaving — which is how that was found. Partially bound with it, so a scene writing a
    /// hundred descriptors into an array declared for thousands leaves the rest untouched instead
    /// of uploading placeholder views for slots no shader reaches.
    pub fn variable_samplers(binding: u32, max: u32) -> Self {
        assert!(max > 0, "a bindless array needs at least one slot");
        Self {
            binding,
            kind: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            count: max,
            variable: true,
        }
    }
}

/// Descriptor set layout, pool, set, pipeline layout and pipeline for one compute shader.
///
/// These five objects are created together, destroyed together, and are pure boilerplate in
/// between; every pass in the renderer was writing the same eighty lines to get them.
pub struct ComputePipeline {
    /// A handle copy, not an owner: the real `Device` outlives this by construction.
    device: ash::Device,
    descriptor_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

// `ash::Device` is a table of function pointers and implements no `Debug`.
impl std::fmt::Debug for ComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputePipeline")
            .field("pipeline", &self.pipeline)
            .finish_non_exhaustive()
    }
}

impl ComputePipeline {
    /// Builds the pipeline from `spirv`, with a descriptor set matching `bindings`.
    ///
    /// `push_constant_size` may be zero, which declares no push constant range at all rather than
    /// an empty one — a range of size zero is invalid.
    pub fn new(
        device: &Device,
        bindings: &[Binding],
        push_constant_size: u32,
        spirv: &[u32],
    ) -> Result<Self> {
        assert!(!bindings.is_empty(), "a pass with no descriptors");
        assert!(
            bindings[..bindings.len() - 1].iter().all(|b| !b.variable),
            "a variable descriptor count is only legal on a set's last binding"
        );
        let raw = device.raw();

        let layout_bindings: Vec<_> = bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(b.kind)
                    .descriptor_count(b.count)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let variable = bindings.last().filter(|b| b.variable).map(|b| b.count);
        let mut binding_flags: Vec<_> = bindings
            .iter()
            .map(|b| {
                if b.variable {
                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                } else {
                    vk::DescriptorBindingFlags::empty()
                }
            })
            .collect();
        let mut flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
        let mut layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        if variable.is_some() {
            layout_info = layout_info.push_next(&mut flags_info);
        }
        // SAFETY: `layout_info` is fully initialised and the device is alive.
        let descriptor_layout = unsafe { raw.create_descriptor_set_layout(&layout_info, None)? };
        binding_flags.clear();

        let sizes = pool_sizes(bindings);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&sizes)
            .max_sets(1);
        // SAFETY: same.
        let pool = unsafe { raw.create_descriptor_pool(&pool_info, None)? };

        let layouts = [descriptor_layout];
        let counts = [variable.unwrap_or(0)];
        let mut variable_count = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&counts);
        let mut allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        if variable.is_some() {
            allocate = allocate.push_next(&mut variable_count);
        }
        // SAFETY: the pool was just created with room for exactly this set.
        let set = unsafe { raw.allocate_descriptor_sets(&allocate)? }[0];

        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_constant_size)];
        let mut pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
        if push_constant_size > 0 {
            pipeline_layout_info = pipeline_layout_info.push_constant_ranges(&ranges);
        }
        // SAFETY: same.
        let pipeline_layout = unsafe { raw.create_pipeline_layout(&pipeline_layout_info, None)? };

        let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
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

    /// The set to write descriptors into, for a caller pointing the pass at its resources.
    pub fn set(&self) -> vk::DescriptorSet {
        self.set
    }

    /// The device that set belongs to, for the caller writing into it.
    ///
    /// Handed out rather than taken as a parameter because a caller holding this pipeline provably
    /// has the right device — it is the one the pipeline was built on — and asking for it again
    /// only creates the chance of passing a different one.
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Points a run of bindings starting at `first` at `images`, in order.
    ///
    /// The shape every pass here has somewhere in its set: consecutive storage images differing
    /// only in the binding number. Writing them out was thirty lines apiece of
    /// `DescriptorImageInfo` and `WriteDescriptorSet` saying the same thing.
    ///
    /// All in `GENERAL`, because a compute shader reading and writing images has no other layout
    /// available to it.
    pub fn bind_storage_images(&self, first: u32, images: &[&crate::image::Image]) {
        let infos: Vec<_> = images
            .iter()
            .map(|image| {
                [vk::DescriptorImageInfo::default()
                    .image_view(image.view())
                    .image_layout(vk::ImageLayout::GENERAL)]
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(first + binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(info)
            })
            .collect();
        // SAFETY: every write names this pipeline's own set, and the caller rebinds only when the
        // resources change, never with a dispatch in flight.
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Binds the pipeline and its set, pushes `constants`, and dispatches.
    ///
    /// # Safety
    /// `command_buffer` must be recording, and the descriptor set must have been written with live
    /// resources in the layouts the shader expects.
    pub unsafe fn dispatch(
        &self,
        command_buffer: vk::CommandBuffer,
        groups: [u32; 3],
        constants: &[u8],
    ) {
        // SAFETY: the caller guarantees the command buffer is recording and the set is written.
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
            if !constants.is_empty() {
                self.device.cmd_push_constants(
                    command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    constants,
                );
            }
            self.device
                .cmd_dispatch(command_buffer, groups[0], groups[1], groups[2]);
        }
    }
}

/// How many descriptors of each kind the pool must hold.
///
/// One entry per *kind*, not per binding: a pool promising one descriptor to a set that declares
/// two of the same kind fails allocation, and a pool listing the same kind twice is invalid in its
/// own right. The tonemap pass reads one storage image and writes another, so this is not a
/// hypothetical.
fn pool_sizes(bindings: &[Binding]) -> Vec<vk::DescriptorPoolSize> {
    let mut sizes: Vec<vk::DescriptorPoolSize> = Vec::new();
    for binding in bindings {
        match sizes.iter_mut().find(|size| size.ty == binding.kind) {
            Some(size) => size.descriptor_count += binding.count,
            None => sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(binding.kind)
                    .descriptor_count(binding.count),
            ),
        }
    }
    sizes
}

impl Drop for ComputePipeline {
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
    fn descriptors_of_one_kind_are_pooled_together() {
        // The tonemap pass's own shape: two storage images and a storage buffer. Counted per
        // binding this would ask for three pools of one, and allocating the set would fail because
        // the second image had no descriptor left to take.
        let sizes = pool_sizes(&[
            Binding::storage_image(0),
            Binding::storage_image(1),
            Binding::storage_buffer(2),
        ]);
        assert_eq!(sizes.len(), 2, "one entry per kind, not per binding");

        let of = |kind| {
            sizes
                .iter()
                .find(|size| size.ty == kind)
                .unwrap_or_else(|| panic!("no pool entry for {kind:?}"))
                .descriptor_count
        };
        assert_eq!(of(vk::DescriptorType::STORAGE_IMAGE), 2);
        assert_eq!(of(vk::DescriptorType::STORAGE_BUFFER), 1);
    }

    #[test]
    fn every_binding_is_accounted_for() {
        // Whatever the mix, the counts have to sum to the number of bindings — a set is allocated
        // against exactly the descriptors its layout declares.
        for bindings in [
            vec![Binding::storage_buffer(0)],
            vec![Binding::storage_buffer(0), Binding::storage_buffer(1)],
            vec![
                Binding::storage_image(0),
                Binding::storage_buffer(1),
                Binding::storage_buffer(2),
                Binding::storage_image(3),
            ],
        ] {
            let total: u32 = pool_sizes(&bindings)
                .iter()
                .map(|size| size.descriptor_count)
                .sum();
            assert_eq!(total as usize, bindings.len(), "{bindings:?}");
        }
    }

    #[test]
    fn a_bindless_array_reserves_every_slot_it_declares() {
        // The pool has to promise the whole array even though a scene writes a fraction of it: the
        // allocation is checked against the layout's count, not against what is later bound.
        let sizes = pool_sizes(&[
            Binding::acceleration_structure(0),
            Binding::variable_samplers(1, 8192),
        ]);
        let samplers = sizes
            .iter()
            .find(|size| size.ty == vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .expect("no pool entry for the sampler array");
        assert_eq!(samplers.descriptor_count, 8192);
    }
}
