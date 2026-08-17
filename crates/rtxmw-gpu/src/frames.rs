//! Per-frame command buffers and synchronisation for a frames-in-flight loop.

use ash::vk;

use crate::device::Device;
use crate::error::Result;

/// How many frames the CPU may run ahead of the GPU.
pub(crate) const FRAMES_IN_FLIGHT: usize = 2;

/// Command buffers and sync objects for the frame ring.
///
/// `image_available` and the fence are per frame-in-flight, but `render_finished` is per swapchain
/// image: a present operation waits on a semaphore that must not be reused until that specific
/// image comes back, and there can be more images than frames in flight.
pub struct Frames {
    device: ash::Device,
    pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    render_finished: Vec<vk::Semaphore>,
    current: usize,
}

impl std::fmt::Debug for Frames {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Frames")
            .field("pool", &self.pool)
            .field("in_flight", &self.in_flight.len())
            .field("render_finished", &self.render_finished.len())
            .field("current", &self.current)
            .finish_non_exhaustive()
    }
}

/// Handles for one frame in flight.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub command_buffer: vk::CommandBuffer,
    pub image_available: vk::Semaphore,
    pub in_flight: vk::Fence,
}

impl Frames {
    /// Allocates the command pool, buffers and sync objects.
    pub fn new(device: &Device, queue_family: u32, swapchain_images: usize) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: `pool_info` is fully initialised and the device is alive.
        let pool = unsafe { device.raw().create_command_pool(&pool_info, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32);
        // SAFETY: the pool was just created on this device.
        let command_buffers = unsafe { device.raw().allocate_command_buffers(&alloc_info)? };

        let mut image_available = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut in_flight = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            // Created signalled so the first wait on each frame returns immediately.
            let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            // SAFETY: the device is alive and both infos are initialised.
            unsafe {
                image_available.push(device.raw().create_semaphore(&semaphore_info, None)?);
                in_flight.push(device.raw().create_fence(&fence_info, None)?);
            }
        }

        let mut frames = Self {
            device: device.raw().clone(),
            pool,
            command_buffers,
            image_available,
            in_flight,
            render_finished: Vec::new(),
            current: 0,
        };
        frames.resize_present_semaphores(swapchain_images)?;
        Ok(frames)
    }

    /// Reallocates the per-image present semaphores after the swapchain image count changes.
    ///
    /// A plain resize usually keeps the image count, and destroying a semaphore the presentation
    /// engine may still hold is a hazard `vkDeviceWaitIdle` does not fully cover — so leave them
    /// alone unless the count actually moved.
    pub fn resize_present_semaphores(&mut self, swapchain_images: usize) -> Result<()> {
        if self.render_finished.len() == swapchain_images {
            return Ok(());
        }
        self.destroy_present_semaphores();
        self.render_finished.reserve_exact(swapchain_images);
        for _ in 0..swapchain_images {
            let info = vk::SemaphoreCreateInfo::default();
            // SAFETY: the device is alive.
            self.render_finished
                .push(unsafe { self.device.create_semaphore(&info, None)? });
        }
        Ok(())
    }

    /// Blocks until the current frame's previous submission has completed, then resets its fence.
    pub fn wait_for_current(&self) -> Result<Frame> {
        let frame = self.current();
        // SAFETY: the fence belongs to this device and is either signalled or pending.
        unsafe {
            self.device
                .wait_for_fences(&[frame.in_flight], true, u64::MAX)?;
        }
        Ok(frame)
    }

    /// Resets the current frame's fence. Call only once the frame is certain to be submitted.
    pub fn reset_current_fence(&self) -> Result<()> {
        // SAFETY: the fence belongs to this device and is not in use by a pending submit.
        unsafe { self.device.reset_fences(&[self.current().in_flight])? };
        Ok(())
    }

    /// Handles for the frame currently being recorded.
    pub fn current(&self) -> Frame {
        Frame {
            command_buffer: self.command_buffers[self.current],
            image_available: self.image_available[self.current],
            in_flight: self.in_flight[self.current],
        }
    }

    /// The semaphore signalled when rendering into `image_index` is finished.
    pub fn render_finished(&self, image_index: u32) -> vk::Semaphore {
        self.render_finished[image_index as usize]
    }

    /// Advances to the next frame in the ring.
    pub fn advance(&mut self) {
        self.current = (self.current + 1) % FRAMES_IN_FLIGHT;
    }

    fn destroy_present_semaphores(&mut self) {
        for semaphore in self.render_finished.drain(..) {
            // SAFETY: the caller waited for device idle.
            unsafe { self.device.destroy_semaphore(semaphore, None) };
        }
    }
}

impl Drop for Frames {
    fn drop(&mut self) {
        self.destroy_present_semaphores();
        // SAFETY: the caller waited for device idle before dropping the renderer.
        unsafe {
            for &semaphore in &self.image_available {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &fence in &self.in_flight {
                self.device.destroy_fence(fence, None);
            }
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}
