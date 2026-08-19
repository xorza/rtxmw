//! Owns every GPU object and draws one frame.

use ash::vk;
use glam::Vec3;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use rtxmw_gpu::image_barrier::{self, COLOR_RANGE};
use rtxmw_gpu::{
    Device, Frames, Instance, Memory, PhysicalDevice, Presentation, Surface, Swapchain, Uploader,
    Validation, image_blit,
};
use rtxmw_render::{FrameConstants, OUTPUT_FORMAT, SceneRenderer, TARGET_FORMAT};
use rtxmw_scene::{CellId, StaticScene};
use rtxmw_texture::Texture;

use crate::cli::Upscaling;
use crate::upscaler;

/// Rows the trace renders, independent of the window's own height.
///
/// The design budgets for 1080 internal rows upscaled to a 2160-row display (`docs/design.md`
/// §5.3), and the blit to the swapchain is what bridges the two.
const RENDER_HEIGHT: u32 = 1080;

/// The internal size for a given window.
///
/// **The width follows the window's aspect ratio rather than being fixed.** A fixed 16:9 target
/// blitted into a window of any other shape is stretched — which is what an ultrawide or a
/// half-screen window got — and no projection can undo that, because the distortion happens after
/// the image is drawn. Matching the aspect makes the blit a pure scale.
///
/// Never taller than the window either: rendering 1080 rows into a 400-row window is work whose
/// result is thrown away by the downscale.
fn internal_extent(window: vk::Extent2D) -> vk::Extent2D {
    let height = window.height.clamp(1, RENDER_HEIGHT);
    let width = (height as f64 * window.width as f64 / window.height.max(1) as f64).round();
    vk::Extent2D {
        width: (width as u32).max(1),
        height,
    }
}

/// The whole GPU side of the engine.
///
/// Field order is load-bearing: fields drop in declaration order, and every device-owned object
/// must be destroyed before the device itself. `Memory` hands out clones to every buffer and image,
/// so everything holding one — the uploader included — must precede `device`.
#[derive(Debug)]
pub(crate) struct Renderer {
    /// Everything that does not care about a window: the pass, the target, the loaded cell.
    scene: SceneRenderer,
    /// What DLSS was asked to run at, so a resize can build the next one.
    ///
    /// The feature is built for one pair of resolutions and cannot be told a new pair, so a window
    /// that changes size needs a new one — and by then the old one has been handed to the scene.
    dlss: Upscaling,
    /// The size everything downstream of the swapchain was built for.
    ///
    /// A compositor sends a resize on first map that changes nothing, and a drag sends one a frame.
    /// Rebuilding Ray Reconstruction means uploading its weights again, so the size is remembered
    /// rather than assumed to have changed.
    display: vk::Extent2D,
    /// Holds the only `Memory` clone the renderer needs: every buffer and image keeps its own, and
    /// everything that creates one goes through the uploader to reach it.
    uploader: Uploader,
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
    pub(crate) fn new<W>(
        window: &W,
        width: u32,
        height: u32,
        dlss: Upscaling,
        delight: f32,
        fog: f32,
    ) -> rtxmw_gpu::Result<Self>
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

        // NGX names device extensions of its own and they have to be enabled at creation, so the
        // decision to upscale is made before there is anything to upscale. Empty without the
        // feature, and empty on a machine whose driver does not offer them.
        let device = Device::new(&instance, &physical, &upscaler::device_extensions())?;
        let extent = vk::Extent2D { width, height };
        let swapchain = Swapchain::new(&instance, &physical, &device, &surface, extent)?;
        let frames = Frames::new(
            &device,
            physical.graphics_queue_family(),
            swapchain.images().len(),
        )?;

        let memory = Memory::new(&instance, &physical, &device)?;
        let mut uploader = Uploader::new(&device, &memory, physical.graphics_queue_family())?;

        // **The window's own size is the output**, so the blit that follows is a pure copy rather
        // than the upscale it is without one. What to trace at is then DLSS's answer, not this
        // crate's — `internal_extent` is the fallback for a frame nothing else will resize.
        let display = swapchain.extent();
        let upscaler = upscaler::build(&instance, &physical, &device, &mut uploader, display, dlss)
            .map_err(|failed| {
                eprintln!("DLSS did not start, rendering without it: {failed}");
            })
            .unwrap_or_default();

        let mut scene = SceneRenderer::new(
            &device,
            &physical,
            &memory,
            upscaler::render_size(upscaler.as_ref(), internal_extent(display)),
        )?;
        scene.set_delight(delight);
        scene.set_fog(fog);
        if let Err(failed) = upscaler::attach(&memory, &mut scene, upscaler) {
            eprintln!("DLSS did not attach: {failed}");
        }

        Ok(Self {
            scene,
            dlss,
            display,
            uploader,
            frames,
            swapchain,
            surface,
            device,
            physical,
            instance,
            needs_recreate: false,
        })
    }

    /// Uploads `scene` and makes it the only resident cell.
    pub(crate) fn load_scene(
        &mut self,
        id: CellId,
        scene: &StaticScene,
        textures: &[Option<Texture>],
    ) -> rtxmw_gpu::Result<()> {
        // SAFETY: replacing the scene frees structures a queued frame could still be reading.
        unsafe { self.device.raw().device_wait_idle()? };
        self.scene.load_scene(
            &self.device,
            &mut self.uploader,
            self.physical.limits(),
            id,
            scene,
            textures,
        )
    }

    /// Makes one more cell resident, drawn from the next [`Renderer::commit`].
    pub(crate) fn add_cell(
        &mut self,
        id: CellId,
        scene: &StaticScene,
        textures: &[Option<Texture>],
    ) -> rtxmw_gpu::Result<()> {
        // SAFETY: an upload can move a buffer a queued frame is reading through its descriptor.
        unsafe { self.device.raw().device_wait_idle()? };
        self.scene.add_cell(
            &self.device,
            &mut self.uploader,
            self.physical.limits(),
            id,
            scene,
            textures,
        )
    }

    /// Drops a resident cell, taking effect at the next [`Renderer::commit`].
    pub(crate) fn remove_cell(&mut self, id: &CellId) {
        self.scene.remove_cell(id);
    }

    /// Rebuilds the top level over whatever is resident now.
    pub(crate) fn commit(&mut self) -> rtxmw_gpu::Result<()> {
        // SAFETY: the rebuild frees the structure a queued frame traces against.
        unsafe { self.device.raw().device_wait_idle()? };
        self.scene
            .commit(&self.device, &mut self.uploader, self.physical.limits())
    }

    /// Sets the clock the water's waves move against, in seconds since the engine started.
    pub(crate) fn set_time(&mut self, seconds: f32) {
        self.scene.set_time(seconds);
    }

    /// The frame constants for a camera, filled in with the loaded cell's lighting.
    ///
    /// `&mut` because the scene renderer remembers this camera to measure the next frame's motion
    /// vectors against — one call per frame, which is what the frame loop does.
    pub(crate) fn frame_constants(
        &mut self,
        view: glam::Mat4,
        projection: glam::Mat4,
        camera_position: Vec3,
    ) -> FrameConstants {
        self.scene
            .frame_constants(view, projection, camera_position)
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
             internal target        {:?} {}x{}\n  \
             tonemapped output      {:?}\n  \
             position fetch         {}\n  \
             rt maintenance1        {}\n  \
             opacity micromap       {}\n  \
             max ray recursion      {}\n  \
             shader group handle    {} bytes, {} byte alignment\n  \
             max BLAS geometries    {}\n  \
             max TLAS instances     {}\n  \
             scratch alignment      {} bytes",
            self.physical.name(),
            self.swapchain.format(),
            extent.width,
            extent.height,
            self.swapchain.images().len(),
            TARGET_FORMAT,
            self.scene.target().extent().width,
            self.scene.target().extent().height,
            OUTPUT_FORMAT,
            support.position_fetch,
            support.maintenance1,
            support.opacity_micromap,
            limits.max_ray_recursion_depth,
            limits.shader_group_handle_size,
            limits.shader_group_base_alignment,
            limits.max_geometry_count,
            limits.max_instance_count,
            limits.min_scratch_offset_alignment,
        )
    }

    /// Aspect ratio of the offscreen target, which is what the projection must match.
    ///
    /// Read from the target rather than from the window: they now agree by construction, and taking
    /// it from the image that is actually traced means they cannot drift if that ever stops being
    /// true.
    pub(crate) fn aspect_ratio(&self) -> f32 {
        let extent = self.scene.target().extent();
        extent.width as f32 / extent.height as f32
    }

    /// Flags the swapchain as stale, e.g. after the window is resized.
    pub(crate) fn invalidate_swapchain(&mut self) {
        self.needs_recreate = true;
    }

    /// Records and submits one frame: trace into the offscreen target, then blit it to the screen.
    ///
    /// Returns without drawing when the surface has zero area, which happens while minimised.
    pub(crate) fn draw(
        &mut self,
        width: u32,
        height: u32,
        constants: &FrameConstants,
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
        let traced = self.scene.has_scene();

        // SAFETY: the command buffer is not in use — its fence was just waited on — and every
        // image and pipeline referenced below is alive.
        unsafe {
            raw.reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            raw.begin_command_buffer(frame.command_buffer, &begin)?;

            self.scene.record(raw, frame.command_buffer, constants);

            image_barrier::transition(
                raw,
                frame.command_buffer,
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            if traced {
                image_blit::stretch(
                    raw,
                    frame.command_buffer,
                    self.scene.output().raw(),
                    self.scene.output().extent(),
                    image,
                    self.swapchain.extent(),
                );
            } else {
                // No cell loaded yet: the clear keeps the window from showing undefined memory.
                let color = vk::ClearColorValue {
                    float32: [0.05, 0.07, 0.10, 1.0],
                };
                raw.cmd_clear_color_image(
                    frame.command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &color,
                    &[COLOR_RANGE],
                );
            }

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
        // Against the *swapchain's* extent rather than the one asked for: a compositor may hand
        // back something else, and the internal image has to match what is actually presented or
        // the aspect correction is computed for a window that does not exist.
        let display = self.swapchain.extent();
        if display == self.display {
            // The swapchain still had to be rebuilt — it may have been out of date for its own
            // reasons — but nothing sized by the window has changed.
            self.needs_recreate = false;
            return Ok(());
        }
        self.display = display;

        // **The old feature goes first.** NGX builds Ray Reconstruction for a fixed pair of
        // resolutions and cannot be told another, so a resize needs a new one — and releasing the
        // old one shuts NGX down for the device, which would orphan a replacement built before it.
        upscaler::detach(self.uploader.memory(), &mut self.scene)
            .map_err(|failed| eprintln!("DLSS did not release: {failed}"))
            .ok();
        let upscaler = upscaler::build(
            &self.instance,
            &self.physical,
            &self.device,
            &mut self.uploader,
            display,
            self.dlss,
        )
        .map_err(|failed| eprintln!("DLSS did not survive the resize: {failed}"))
        .unwrap_or_default();
        self.scene.resize(
            self.uploader.memory(),
            upscaler::render_size(upscaler.as_ref(), internal_extent(display)),
        )?;
        if let Err(failed) = upscaler::attach(self.uploader.memory(), &mut self.scene, upscaler) {
            eprintln!("DLSS did not reattach: {failed}");
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn extent(width: u32, height: u32) -> vk::Extent2D {
        vk::Extent2D { width, height }
    }

    #[test]
    fn the_internal_size_keeps_the_windows_shape() {
        // A 16:9 window at or above the design's height renders exactly the budgeted 1920x1080.
        assert_eq!(internal_extent(extent(1920, 1080)), extent(1920, 1080));
        assert_eq!(internal_extent(extent(3840, 2160)), extent(1920, 1080));

        // An ultrawide gets a wider target, not a stretched one: 1080 rows at 21:9.
        assert_eq!(internal_extent(extent(3440, 1440)), extent(2580, 1080));
        // And a tall window a narrower one.
        assert_eq!(internal_extent(extent(1080, 1920)), extent(608, 1080));

        // Every case keeps the window's aspect ratio to within a pixel of rounding, which is the
        // whole point — the blit is a scale, and a scale cannot undo a shape change.
        for (w, h) in [
            (1920, 1080),
            (3440, 1440),
            (1080, 1920),
            (2560, 1600),
            (800, 600),
        ] {
            let internal = internal_extent(extent(w, h));
            let window = f64::from(w) / f64::from(h);
            let target = f64::from(internal.width) / f64::from(internal.height);
            assert!(
                (window - target).abs() < 0.01,
                "{w}x{h} window against a {}x{} target",
                internal.width,
                internal.height
            );
        }
    }

    #[test]
    fn a_window_shorter_than_the_budget_is_not_oversampled() {
        // Rendering more rows than the window shows is work the downscale throws away.
        assert_eq!(internal_extent(extent(800, 600)), extent(800, 600));
        assert_eq!(internal_extent(extent(640, 360)), extent(640, 360));

        // And nothing degenerate at the limits, where a zero would divide by it or allocate an
        // image with no pixels.
        assert_eq!(internal_extent(extent(1, 1)), extent(1, 1));
        assert_eq!(internal_extent(extent(0, 0)), extent(1, 1));
    }
}
