//! Barriers that order memory access without naming a specific resource.

use ash::vk;

/// Inserts a barrier making every prior write visible to everything recorded after it.
///
/// Deliberately coarse, for the same reason [`crate::image_barrier::transition`] is: precise stage
/// and access masks belong with the passes that need them, and every caller here is on a load path
/// where one barrier costs nothing against the transfer or acceleration structure build either side
/// of it.
///
/// What it is *for* is the dependency a fence does not create. Waiting on a fence orders the host
/// against the work, but a later submission reading the same memory on the device needs the
/// dependency recorded — an uploaded buffer read by a build, or a compacted structure read by a
/// top-level build.
///
/// # Safety
/// `command_buffer` must be in the recording state.
pub unsafe fn full(device: &ash::Device, command_buffer: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
    let barriers = [barrier];
    let dependency = vk::DependencyInfo::default().memory_barriers(&barriers);
    // SAFETY: the caller guarantees the command buffer is recording.
    unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency) };
}
