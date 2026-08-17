//! Owns every GPU object and draws one frame.

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use rtxmw_gpu::image_barrier::{self, COLOR_RANGE};
use rtxmw_gpu::{
    Device, Frames, Instance, PhysicalDevice, Presentation, Surface, Swapchain, Validation,
};

/// The whole GPU side of the engine.
///
/// Field order is load-bearing: fields drop in declaration order, and every device-owned object
/// must be destroyed before the device itself.
#[derive(Debug)]
pub(crate) struct Renderer {
    frames: Frames,
    swapchain: Swapchain,
    surface: Surface,
    device: Device,
    physical: PhysicalDevice,
    /// Never read, but must outlive every object created from it — dropping it early would destroy
    /// the `VkInstance` while the device and surface still reference it.
    #[allow(dead_code)]
    instance: Instance,
    needs_recreate: bool,
}

impl Renderer {
    /// Brings up Vulkan for `window` and allocates the frame ring.
    pub(crate) fn new<W>(window: &W, width: u32, height: u32) -> rtxmw_gpu::Result<Self>
    where
        W: HasDisplayHandle + HasWindowHandle,
    {
        let extensions = Surface::required_extensions(window)?;
        let instance = Instance::new(c"rtxmw", extensions, Validation::for_build())?;
        let physical = PhysicalDevice::select(&instance, Presentation::Required)?;
        let surface = Surface::new(&instance, window)?;

        assert!(
            surface.supports_present(&physical)?,
            "selected device cannot present to this surface"
        );

        let device = Device::new(&instance, &physical)?;
        let extent = vk::Extent2D { width, height };
        let swapchain = Swapchain::new(&instance, &physical, &device, &surface, extent)?;
        let frames = Frames::new(
            &device,
            physical.graphics_queue_family(),
            swapchain.images().len(),
        )?;

        Ok(Self {
            frames,
            swapchain,
            surface,
            device,
            physical,
            instance,
            needs_recreate: false,
        })
    }

    /// Name of the physical device in use.
    pub(crate) fn device_name(&self) -> &str {
        self.physical.name()
    }

    /// A one-shot summary of what the chosen device and swapchain can do.
    ///
    /// Worth printing at startup rather than only in a debugger: these limits decide shader binding
    /// table layout and acceleration structure budgets, and they differ across drivers.
    pub(crate) fn capability_report(&self) -> String {
        let support = self.physical.support();
        let limits = self.physical.limits();
        let extent = self.swapchain.extent();
        format!(
            "{}\n  \
             swapchain              {:?} {}x{}, {} images\n  \
             position fetch         {}\n  \
             rt maintenance1        {}\n  \
             opacity micromap       {}\n  \
             max ray recursion      {}\n  \
             shader group handle    {} bytes, {} byte alignment\n  \
             max BLAS geometries    {}\n  \
             max TLAS instances     {}",
            self.physical.name(),
            self.swapchain.format(),
            extent.width,
            extent.height,
            self.swapchain.images().len(),
            support.position_fetch,
            support.maintenance1,
            support.opacity_micromap,
            limits.max_ray_recursion_depth,
            limits.shader_group_handle_size,
            limits.shader_group_base_alignment,
            limits.max_geometry_count,
            limits.max_instance_count,
        )
    }

    /// Current swapchain aspect ratio, for the projection matrix.
    ///
    /// Unused until the camera's projection feeds ray generation at M3.
    #[allow(dead_code)]
    pub(crate) fn aspect_ratio(&self) -> f32 {
        let extent = self.swapchain.extent();
        extent.width as f32 / extent.height.max(1) as f32
    }

    /// Flags the swapchain as stale, e.g. after the window is resized.
    pub(crate) fn invalidate_swapchain(&mut self) {
        self.needs_recreate = true;
    }

    /// Records and submits one frame, clearing the swapchain image.
    ///
    /// Returns without drawing when the surface has zero area, which happens while minimised.
    pub(crate) fn draw(
        &mut self,
        width: u32,
        height: u32,
        clear: [f32; 4],
    ) -> rtxmw_gpu::Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        if self.needs_recreate {
            self.recreate(width, height)?;
        }

        let frame = self.frames.wait_for_current()?;

        let Some(image_index) = self.swapchain.acquire_next_image(frame.image_available)? else {
            self.needs_recreate = true;
            return Ok(());
        };

        // Reset only once acquisition succeeded, or an early return would leave the fence
        // unsignalled and the next wait would deadlock.
        self.frames.reset_current_fence()?;

        let image = self.swapchain.images()[image_index as usize];
        let raw = self.device.raw();

        // SAFETY: the command buffer is not in use — its fence was just waited on.
        unsafe {
            raw.reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            raw.begin_command_buffer(frame.command_buffer, &begin)?;

            image_barrier::transition(
                raw,
                frame.command_buffer,
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            let color = vk::ClearColorValue { float32: clear };
            raw.cmd_clear_color_image(
                frame.command_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &color,
                &[COLOR_RANGE],
            );

            image_barrier::transition(
                raw,
                frame.command_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );

            raw.end_command_buffer(frame.command_buffer)?;
        }

        let render_finished = self.frames.render_finished(image_index);
        let wait = [vk::SemaphoreSubmitInfo::default()
            .semaphore(frame.image_available)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(render_finished)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(frame.command_buffer)];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait)
            .command_buffer_infos(&buffers)
            .signal_semaphore_infos(&signal);

        // SAFETY: every referenced object is alive and owned by this device.
        unsafe {
            self.device.raw().queue_submit2(
                self.device.graphics_queue(),
                &[submit],
                frame.in_flight,
            )?;
        }

        let healthy =
            self.swapchain
                .present(self.device.graphics_queue(), image_index, render_finished)?;
        if !healthy {
            self.needs_recreate = true;
        }

        self.frames.advance();
        Ok(())
    }

    fn recreate(&mut self, width: u32, height: u32) -> rtxmw_gpu::Result<()> {
        // SAFETY: no further work is submitted until this returns.
        unsafe { self.device.raw().device_wait_idle()? };

        let extent = vk::Extent2D { width, height };
        self.swapchain
            .recreate(&self.physical, &self.surface, extent)?;
        self.frames
            .resize_present_semaphores(self.swapchain.images().len())?;
        self.needs_recreate = false;
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // Fields are destroyed after this body runs, so everything must be idle first.
        // SAFETY: no further submissions happen once the renderer is being dropped.
        unsafe {
            let _ = self.device.raw().device_wait_idle();
        }
    }
}
