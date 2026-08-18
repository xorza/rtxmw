//! Image layout transitions.

use ash::vk;

/// The whole colour image: one mip level, one array layer.
pub const COLOR_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

/// Inserts a layout transition that blocks on everything and makes everything visible.
///
/// Deliberately coarse. Precise stage and access masks belong with the passes that need them; at
/// one or two transitions per frame the difference is unmeasurable, and getting them wrong is a
/// class of bug that is miserable to find.
///
/// # Safety
/// `command_buffer` must be in the recording state, and `image` must belong to `device`.
pub unsafe fn transition(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    // SAFETY: forwarded unchanged.
    unsafe { transition_range(device, command_buffer, image, from, to, COLOR_RANGE) }
}

/// As [`transition`], for an image whose subresource range is not a single mip level.
///
/// # Safety
/// `command_buffer` must be in the recording state, and `image` must belong to `device`.
pub unsafe fn transition_range(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
    range: vk::ImageSubresourceRange,
) {
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(from)
        .new_layout(to)
        .image(image)
        .subresource_range(range);
    let barriers = [barrier];
    let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
    // SAFETY: the caller guarantees the command buffer is recording.
    unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency) };
}
