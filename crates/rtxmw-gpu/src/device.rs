//! Logical device creation with the ray tracing feature set enabled.

use ash::vk;

use crate::error::Result;
use crate::instance::Instance;
use crate::physical_device::PhysicalDevice;

/// A logical device plus the queue and extension function tables the renderer needs.
pub struct Device {
    raw: ash::Device,
    graphics_queue: vk::Queue,
    acceleration_structure: ash::khr::acceleration_structure::Device,
    ray_tracing_pipeline: ash::khr::ray_tracing_pipeline::Device,
}

// The `ash` device and extension tables are function pointers and implement no `Debug`.
impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("handle", &self.raw.handle())
            .field("graphics_queue", &self.graphics_queue)
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Creates the logical device, enabling every feature the renderer's design depends on.
    ///
    /// Buffer device address and descriptor indexing are not optional here: acceleration structure
    /// builds need the former, and bindless material lookup at a hit needs the latter.
    /// `extra` names extensions beyond this crate's own, **enabled only where the device has
    /// them**, which is the rule the optional set already follows. Something outside this crate
    /// knows it needs them and decides what to do when they are missing, so an absent one is not an
    /// error here.
    pub fn new(
        instance: &Instance,
        physical: &PhysicalDevice,
        extra: &[&std::ffi::CStr],
    ) -> Result<Self> {
        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(physical.graphics_queue_family())
            .queue_priorities(&priorities)];

        let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_indexing(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .runtime_descriptor_array(true)
            .scalar_block_layout(true);
        let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let mut acceleration_structure =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);
        let mut ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
        let mut ray_tracing_pipeline =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
        let mut position_fetch = vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR::default()
            .ray_tracing_position_fetch(true);

        let mut extensions = physical.extensions_to_enable();
        extensions.extend(
            extra
                .iter()
                .filter(|name| physical.supports(name))
                .map(|name| name.as_ptr()),
        );
        let mut create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extensions)
            .push_next(&mut vulkan12)
            .push_next(&mut vulkan13)
            .push_next(&mut acceleration_structure)
            .push_next(&mut ray_query)
            .push_next(&mut ray_tracing_pipeline);
        if physical.support().position_fetch {
            create_info = create_info.push_next(&mut position_fetch);
        }

        // SAFETY: every pushed feature struct and the extension name array outlive this call.
        let raw = unsafe {
            instance
                .raw()
                .create_device(physical.raw(), &create_info, None)?
        };

        // SAFETY: the family index came from this device's own queue family enumeration.
        let graphics_queue = unsafe { raw.get_device_queue(physical.graphics_queue_family(), 0) };

        let acceleration_structure =
            ash::khr::acceleration_structure::Device::new(instance.raw(), &raw);
        let ray_tracing_pipeline =
            ash::khr::ray_tracing_pipeline::Device::new(instance.raw(), &raw);

        Ok(Self {
            raw,
            graphics_queue,
            acceleration_structure,
            ray_tracing_pipeline,
        })
    }

    /// The underlying `VkDevice` wrapper.
    pub fn raw(&self) -> &ash::Device {
        &self.raw
    }

    /// The graphics + compute queue.
    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    /// `VK_KHR_acceleration_structure` entry points.
    pub fn acceleration_structure(&self) -> &ash::khr::acceleration_structure::Device {
        &self.acceleration_structure
    }

    /// `VK_KHR_ray_tracing_pipeline` entry points.
    pub fn ray_tracing_pipeline(&self) -> &ash::khr::ray_tracing_pipeline::Device {
        &self.ray_tracing_pipeline
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: callers must have destroyed device-owned objects first; waiting idle ensures no
        // work is in flight when the device goes away.
        unsafe {
            let _ = self.raw.device_wait_idle();
            self.raw.destroy_device(None);
        }
    }
}
