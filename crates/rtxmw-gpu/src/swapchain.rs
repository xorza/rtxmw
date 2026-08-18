//! Swapchain creation, image views, and recreation on resize.

use ash::vk;

use crate::device::Device;
use crate::error::Result;
use crate::instance::Instance;
use crate::physical_device::PhysicalDevice;
use crate::surface::Surface;

/// The presentation swapchain and its images.
///
/// No image views are created: the ray tracing output reaches the screen through
/// `cmd_blit_image` from an offscreen target, which takes images directly. Add views here if a
/// pass ever renders into the swapchain as a colour attachment.
pub struct Swapchain {
    loader: ash::khr::swapchain::Device,
    raw: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
}

// The loaders are function tables and implement no `Debug`.
impl std::fmt::Debug for Swapchain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Swapchain")
            .field("raw", &self.raw)
            .field("format", &self.format)
            .field("extent", &self.extent)
            .field("image_count", &self.images.len())
            .finish_non_exhaustive()
    }
}

impl Swapchain {
    /// Creates a swapchain sized to `preferred_extent`, clamped to what the surface allows.
    pub fn new(
        instance: &Instance,
        physical: &PhysicalDevice,
        device: &Device,
        surface: &Surface,
        preferred_extent: vk::Extent2D,
    ) -> Result<Self> {
        let loader = ash::khr::swapchain::Device::new(instance.raw(), device.raw());
        let mut swapchain = Self {
            loader,
            raw: vk::SwapchainKHR::null(),
            images: Vec::new(),
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D::default(),
        };
        swapchain.build(
            physical,
            surface,
            preferred_extent,
            vk::SwapchainKHR::null(),
        )?;
        Ok(swapchain)
    }

    /// Rebuilds at a new size, reusing the old swapchain so presentation is not interrupted.
    pub fn recreate(
        &mut self,
        physical: &PhysicalDevice,
        surface: &Surface,
        preferred_extent: vk::Extent2D,
    ) -> Result<()> {
        let old = self.raw;
        self.build(physical, surface, preferred_extent, old)?;
        if old != vk::SwapchainKHR::null() {
            // SAFETY: the new swapchain has been created from it, so the old one is retired.
            unsafe { self.loader.destroy_swapchain(old, None) };
        }
        Ok(())
    }

    fn build(
        &mut self,
        physical: &PhysicalDevice,
        surface: &Surface,
        preferred_extent: vk::Extent2D,
        old: vk::SwapchainKHR,
    ) -> Result<()> {
        let caps = surface.capabilities(physical)?;

        // A current extent of u32::MAX means the surface defers the choice to us.
        let extent = if caps.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: preferred_extent
                    .width
                    .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: preferred_extent
                    .height
                    .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        } else {
            caps.current_extent
        };

        let surface_format = Self::choose_format(&surface.formats(physical)?);
        let present_mode = Self::choose_present_mode(&surface.present_modes(physical)?);

        // One more than the minimum lets the driver hand us an image while another is displayed.
        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            image_count = image_count.min(caps.max_image_count);
        }

        // No STORAGE: sRGB formats do not expose VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT, so a compute
        // shader can never write these images directly. Ray tracing output goes to an offscreen HDR
        // image and reaches the swapchain through the tonemap blit, which is the wanted shape
        // regardless.
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST;

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.raw())
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old);

        // SAFETY: every referenced object outlives the call.
        let raw = unsafe { self.loader.create_swapchain(&create_info, None)? };
        // SAFETY: `raw` was just created by this loader.
        let images = unsafe { self.loader.get_swapchain_images(raw)? };

        self.raw = raw;
        self.images = images;
        self.format = surface_format.format;
        self.extent = extent;
        Ok(())
    }

    /// Prefers a `UNORM` target in the sRGB colour space, so the presentation engine passes the
    /// bytes through untouched.
    ///
    /// **Not `_SRGB`, and that is the point.** The tonemap pass encodes sRGB itself, because it
    /// writes through a storage image and sRGB formats expose no storage capability. Presenting
    /// those already-encoded bytes through an `_SRGB` swapchain would encode them a second time and
    /// wash the whole frame out. The colour space is still `SRGB_NONLINEAR`, which is what tells
    /// the display how to read them.
    fn choose_format(available: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        available
            .iter()
            .copied()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or_else(|| {
                *available
                    .first()
                    .expect("surface reported no formats, which the spec forbids")
            })
    }

    /// Mailbox where available, otherwise FIFO, which is always supported.
    fn choose_present_mode(available: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        if available.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        }
    }

    /// Acquires the next image, returning `None` when the swapchain needs recreating.
    ///
    /// `signal` is signalled once the image is actually available for rendering.
    pub fn acquire_next_image(&self, signal: vk::Semaphore) -> Result<Option<u32>> {
        // SAFETY: the swapchain and semaphore are alive and belong to this device.
        let result = unsafe {
            self.loader
                .acquire_next_image(self.raw, u64::MAX, signal, vk::Fence::null())
        };
        match result {
            Ok((index, _suboptimal)) => Ok(Some(index)),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Presents `image_index` once `wait` is signalled.
    ///
    /// Returns `false` when the swapchain needs recreating.
    pub fn present(&self, queue: vk::Queue, image_index: u32, wait: vk::Semaphore) -> Result<bool> {
        let wait_semaphores = [wait];
        let swapchains = [self.raw];
        let indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&indices);

        // SAFETY: every referenced object is alive and owned by this device.
        let result = unsafe { self.loader.queue_present(queue, &present_info) };
        match result {
            Ok(suboptimal) => Ok(!suboptimal),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// Swapchain images, one per presentable frame.
    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }

    /// The format the images were created with.
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Current swapchain size in pixels.
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        // SAFETY: the caller waited for device idle before dropping the renderer.
        unsafe { self.loader.destroy_swapchain(self.raw, None) };
    }
}
