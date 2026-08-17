//! Window surface creation and capability queries.

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::error::Result;
use crate::instance::Instance;
use crate::physical_device::PhysicalDevice;

/// A `VkSurfaceKHR` for a window, plus the loader used to query and destroy it.
pub struct Surface {
    loader: ash::khr::surface::Instance,
    raw: vk::SurfaceKHR,
}

// `ash::khr::surface::Instance` is a function table and implements no `Debug`.
impl std::fmt::Debug for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surface")
            .field("raw", &self.raw)
            .finish_non_exhaustive()
    }
}

impl Surface {
    /// Creates a surface for `window`.
    ///
    /// The instance must already have the extensions returned by [`required_extensions`] enabled.
    pub fn new<W>(instance: &Instance, window: &W) -> Result<Self>
    where
        W: HasDisplayHandle + HasWindowHandle,
    {
        let display = window
            .display_handle()
            .expect("window has no display handle")
            .as_raw();
        let handle = window
            .window_handle()
            .expect("window has no window handle")
            .as_raw();

        // SAFETY: `display` and `handle` come from a live window that outlives this surface.
        let raw = unsafe {
            ash_window::create_surface(instance.entry(), instance.raw(), display, handle, None)?
        };

        Ok(Self {
            loader: ash::khr::surface::Instance::new(instance.entry(), instance.raw()),
            raw,
        })
    }

    /// Instance extensions needed to create a surface for `display`.
    ///
    /// This is the list that must be handed to [`Instance::new`]; only the windowing system knows
    /// which platform surface extension applies.
    pub fn required_extensions<D>(display: &D) -> Result<&'static [*const std::ffi::c_char]>
    where
        D: HasDisplayHandle,
    {
        let handle = display
            .display_handle()
            .expect("no display handle")
            .as_raw();
        Ok(ash_window::enumerate_required_extensions(handle)?)
    }

    /// Whether `physical`'s graphics queue family can present to this surface.
    pub fn supports_present(&self, physical: &PhysicalDevice) -> Result<bool> {
        // SAFETY: both handles are alive and came from the same instance.
        let supported = unsafe {
            self.loader.get_physical_device_surface_support(
                physical.raw(),
                physical.graphics_queue_family(),
                self.raw,
            )?
        };
        Ok(supported)
    }

    /// Surface capabilities, which carry the permitted swapchain extents and image counts.
    pub fn capabilities(&self, physical: &PhysicalDevice) -> Result<vk::SurfaceCapabilitiesKHR> {
        // SAFETY: same.
        let caps = unsafe {
            self.loader
                .get_physical_device_surface_capabilities(physical.raw(), self.raw)?
        };
        Ok(caps)
    }

    /// Formats this surface can be presented in.
    pub fn formats(&self, physical: &PhysicalDevice) -> Result<Vec<vk::SurfaceFormatKHR>> {
        // SAFETY: same.
        let formats = unsafe {
            self.loader
                .get_physical_device_surface_formats(physical.raw(), self.raw)?
        };
        Ok(formats)
    }

    /// Present modes this surface supports.
    pub fn present_modes(&self, physical: &PhysicalDevice) -> Result<Vec<vk::PresentModeKHR>> {
        // SAFETY: same.
        let modes = unsafe {
            self.loader
                .get_physical_device_surface_present_modes(physical.raw(), self.raw)?
        };
        Ok(modes)
    }

    /// The underlying handle.
    pub fn raw(&self) -> vk::SurfaceKHR {
        self.raw
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        // SAFETY: the swapchain built on this surface is destroyed first — `Renderer` declares it
        // before the surface, and fields drop in declaration order.
        unsafe { self.loader.destroy_surface(self.raw, None) };
    }
}
