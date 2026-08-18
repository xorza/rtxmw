//! A 2D image with its device memory and a full-subresource view.

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::error::Result;
use crate::image_barrier::COLOR_RANGE;
use crate::memory::Memory;

/// A device-local colour image, its allocation and its view, freed together.
///
/// One array layer. Render targets take one mip level; textures take the chain their file carries,
/// which matters more in a ray tracer than a rasterizer — there is no screen-space derivative to
/// pick a level from, so an unmipped texture aliases into noise at distance.
#[derive(Debug)]
pub struct Image {
    memory: Memory,
    raw: vk::Image,
    view: vk::ImageView,
    /// `None` only after `Drop` has taken it.
    allocation: Option<Allocation>,
    extent: vk::Extent2D,
    format: vk::Format,
    mip_levels: u32,
}

impl Image {
    /// Creates a single-level image of `extent` in `format` and binds memory to it.
    pub fn new(
        memory: &Memory,
        name: &str,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self> {
        Self::mipped(memory, name, extent, format, usage, 1)
    }

    /// Creates an image with `mip_levels` levels, largest first.
    pub fn mipped(
        memory: &Memory,
        name: &str,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        mip_levels: u32,
    ) -> Result<Self> {
        assert!(
            mip_levels > 0,
            "`{name}`: an image needs at least one level"
        );
        assert!(
            extent.width > 0 && extent.height > 0,
            "`{name}`: Vulkan rejects an image with a zero dimension, got {}x{}",
            extent.width,
            extent.height
        );

        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: `info` is fully initialised and the device is alive.
        let raw = unsafe { memory.device().create_image(&info, None)? };

        // SAFETY: `raw` was just created on this device.
        let requirements = unsafe { memory.device().get_image_memory_requirements(raw) };
        let allocation = memory.lock().allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            // Optimal tiling is opaque to the host, which is what `linear: false` tells the
            // allocator so it keeps these away from buffers in the same memory type.
            linear: false,
            allocation_scheme: AllocationScheme::DedicatedImage(raw),
        });
        let allocation = match allocation {
            Ok(allocation) => allocation,
            Err(e) => {
                // SAFETY: nothing has referenced the image, and it is not returned to the caller.
                unsafe { memory.device().destroy_image(raw, None) };
                return Err(e.into());
            }
        };

        // SAFETY: the handle is only handed back to `bind_image_memory` for this image.
        let device_memory = unsafe { allocation.memory() };
        let offset = allocation.offset();

        // Assembled before the calls that can fail, so the error paths drop it rather than leaking
        // the image and its allocation.
        let mut image = Self {
            memory: memory.clone(),
            raw,
            view: vk::ImageView::null(),
            allocation: Some(allocation),
            extent,
            format,
            mip_levels,
        };

        // SAFETY: the allocation was made against this image's own requirements.
        unsafe {
            memory
                .device()
                .bind_image_memory(raw, device_memory, offset)?
        };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(raw)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                level_count: mip_levels,
                ..COLOR_RANGE
            });
        // SAFETY: `raw` is bound and alive.
        image.view = unsafe { memory.device().create_image_view(&view_info, None)? };

        Ok(image)
    }

    /// The underlying `VkImage`.
    pub fn raw(&self) -> vk::Image {
        self.raw
    }

    /// A full-subresource colour view, for binding as a storage or sampled image.
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    /// Size in pixels.
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// The format the image was created with.
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// How many mip levels the image holds.
    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    /// The whole image, for a barrier that has to cover every level.
    pub fn full_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            level_count: self.mip_levels,
            ..COLOR_RANGE
        }
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let device = self.memory.device();
        // SAFETY: callers keep an image alive until the work reading or writing it has completed.
        // A null view is the window between binding failing and this running, which `destroy_*`
        // accepts as a no-op.
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.raw, None);
        }
        if let Some(allocation) = self.allocation.take() {
            self.memory
                .lock()
                .free(allocation)
                .expect("failed to free image memory");
        }
    }
}
