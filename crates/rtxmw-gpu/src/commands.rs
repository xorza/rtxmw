//! One-shot command submission for work outside the frame loop.

use ash::vk;

use crate::device::Device;
use crate::error::Result;

/// A command pool, buffer and fence used to record and complete one submission at a time.
///
/// Deliberately not the frame ring: uploads and acceleration structure builds happen at load time,
/// where blocking until the GPU is done is the simplest correct thing.
pub(crate) struct Commands {
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

// `ash::Device` is a table of function pointers and implements no `Debug`.
impl std::fmt::Debug for Commands {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Commands")
            .field("pool", &self.pool)
            .field("queue", &self.queue)
            .finish_non_exhaustive()
    }
}

impl Commands {
    /// Allocates the pool, buffer and fence on `device`'s graphics queue.
    pub(crate) fn new(device: &Device, queue_family: u32) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: `pool_info` is fully initialised and the device is alive.
        let pool = unsafe { device.raw().create_command_pool(&pool_info, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool was just created on this device.
        let command_buffer = unsafe { device.raw().allocate_command_buffers(&alloc_info)? }[0];

        // SAFETY: same.
        let fence = unsafe {
            device
                .raw()
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };

        Ok(Self {
            device: device.raw().clone(),
            queue: device.graphics_queue(),
            pool,
            command_buffer,
            fence,
        })
    }

    /// Records `record` into the command buffer, submits it, and blocks until it completes.
    ///
    /// `&mut self` because the pool and its one buffer are reused: two overlapping calls would
    /// record into the same buffer and reset each other's fence.
    pub(crate) fn submit_and_wait(
        &mut self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<()> {
        // SAFETY: the fence and command buffer are idle — any previous submission was waited on
        // before this call could be made.
        unsafe {
            self.device.reset_fences(&[self.fence])?;
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(self.command_buffer, &begin)?;
        }

        record(&self.device, self.command_buffer);

        // SAFETY: the command buffer is recording and every handle is alive.
        unsafe {
            self.device.end_command_buffer(self.command_buffer)?;

            let buffers =
                [vk::CommandBufferSubmitInfo::default().command_buffer(self.command_buffer)];
            let submit = vk::SubmitInfo2::default().command_buffer_infos(&buffers);
            self.device
                .queue_submit2(self.queue, &[submit], self.fence)?;

            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        Ok(())
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        // SAFETY: `submit_and_wait` blocks on the fence before returning, so nothing this owns is
        // in flight once the last call has returned.
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}
