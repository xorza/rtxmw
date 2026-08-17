//! Physical device selection against the ray tracing requirements.

use ash::vk;

use crate::error::{GpuError, Rejection, RejectionReason, Result};
use crate::instance::Instance;

/// Device extensions without which the renderer cannot work at all.
const REQUIRED: &[&std::ffi::CStr] = &[
    ash::khr::acceleration_structure::NAME,
    ash::khr::ray_tracing_pipeline::NAME,
    ash::khr::ray_query::NAME,
    ash::khr::deferred_host_operations::NAME,
];

/// Additionally required to present to a window.
///
/// Kept separate because `VK_KHR_swapchain` depends on the `VK_KHR_surface` *instance* extension:
/// enabling it on a headless instance is a specification violation, not merely wasteful.
const PRESENTATION: &[&std::ffi::CStr] = &[ash::khr::swapchain::NAME];

/// Whether the chosen device has to be able to present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presentation {
    /// The device will drive a window; require and enable the swapchain extension.
    Required,
    /// Offscreen only, as in tests and tools.
    NotNeeded,
}

/// Extensions the renderer uses when present and works without otherwise.
const OPTIONAL: &[&std::ffi::CStr] = &[
    ash::khr::ray_tracing_position_fetch::NAME,
    ash::khr::ray_tracing_maintenance1::NAME,
    ash::ext::opacity_micromap::NAME,
];

/// A chosen `VkPhysicalDevice` together with what it can do.
#[derive(Debug)]
pub struct PhysicalDevice {
    raw: vk::PhysicalDevice,
    name: String,
    device_type: vk::PhysicalDeviceType,
    graphics_queue_family: u32,
    support: RayTracingSupport,
    /// Everything that must be enabled, including presentation when it was asked for.
    required_extensions: Vec<&'static std::ffi::CStr>,
    /// Exactly the subset of [`OPTIONAL`] this device offers, resolved once during inspection so
    /// the enable list cannot drift out of step with the capability flags.
    optional_extensions: Vec<&'static std::ffi::CStr>,
    limits: RayTracingLimits,
}

/// Device limits that acceleration structure and shader binding table layout depend on.
///
/// Copied out of the Vulkan property structs rather than storing those: their `p_next` fields
/// point at the locals used to build the query chain, which are gone by the time this is returned.
/// Nothing dereferences them today, but keeping dangling pointers in a long-lived struct is a trap,
/// and it is what forced `unsafe impl Send`/`Sync` on anything holding a `PhysicalDevice`.
#[derive(Debug, Clone, Copy)]
pub struct RayTracingLimits {
    pub max_ray_recursion_depth: u32,
    pub shader_group_handle_size: u32,
    pub shader_group_base_alignment: u32,
    pub max_geometry_count: u64,
    pub max_instance_count: u64,
}

/// Which optional ray tracing capabilities this device offers.
#[derive(Debug, Default, Clone, Copy)]
pub struct RayTracingSupport {
    /// `VK_KHR_ray_tracing_position_fetch` — read hit triangle vertices without a vertex buffer.
    pub position_fetch: bool,
    /// `VK_KHR_ray_tracing_maintenance1`.
    pub maintenance1: bool,
    /// `VK_EXT_opacity_micromap` — the hardware path for alpha-tested geometry.
    pub opacity_micromap: bool,
}

impl PhysicalDevice {
    /// Picks the first discrete GPU meeting the required extension set, falling back to any
    /// device that does.
    ///
    /// Every rejected candidate is reported, because "no suitable device" without a reason is the
    /// least useful error a renderer can produce.
    pub fn select(instance: &Instance, presentation: Presentation) -> Result<Self> {
        // SAFETY: the instance is alive for the duration of the call.
        let candidates = unsafe { instance.raw().enumerate_physical_devices()? };

        let mut rejections = Vec::new();
        let mut accepted: Vec<Self> = Vec::new();

        for raw in candidates {
            match Self::inspect(instance, raw, presentation) {
                Ok(device) => accepted.push(device),
                Err(rejection) => rejections.push(rejection),
            }
        }

        // A discrete GPU beats an integrated one even when both qualify.
        accepted
            .into_iter()
            .min_by_key(|d| !d.is_discrete())
            .ok_or(GpuError::NoSuitableDevice(rejections))
    }

    fn inspect(
        instance: &Instance,
        raw: vk::PhysicalDevice,
        presentation: Presentation,
    ) -> std::result::Result<Self, Rejection> {
        // SAFETY: `raw` came from this instance's enumeration.
        let properties = unsafe { instance.raw().get_physical_device_properties(raw) };
        let name = properties
            .device_name_as_c_str()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|_| String::from("<unnamed device>"));

        let reject = |reason| Rejection {
            device_name: name.clone(),
            reason,
        };

        // SAFETY: same.
        let available = unsafe {
            instance
                .raw()
                .enumerate_device_extension_properties(raw)
                .map_err(|e| reject(RejectionReason::QueryFailed(e)))?
        };
        let has = |wanted: &std::ffi::CStr| {
            available
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(wanted))
        };

        let required_extensions: Vec<&'static std::ffi::CStr> = match presentation {
            Presentation::Required => REQUIRED.iter().chain(PRESENTATION).copied().collect(),
            Presentation::NotNeeded => REQUIRED.to_vec(),
        };

        let missing: Vec<&'static str> = required_extensions
            .iter()
            .filter(|w| !has(w))
            .map(|w| w.to_str().unwrap_or("<non-utf8 extension>"))
            .collect();
        if !missing.is_empty() {
            return Err(reject(RejectionReason::MissingExtensions(missing)));
        }

        let mut acceleration_structure =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
        let mut ray_tracing_pipeline = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut acceleration_structure)
            .push_next(&mut ray_query)
            .push_next(&mut ray_tracing_pipeline);
        // SAFETY: every pushed struct outlives the call.
        unsafe {
            instance
                .raw()
                .get_physical_device_features2(raw, &mut features)
        };

        if acceleration_structure.acceleration_structure == vk::FALSE {
            return Err(reject(RejectionReason::MissingFeature(
                "accelerationStructure",
            )));
        }
        if ray_query.ray_query == vk::FALSE {
            return Err(reject(RejectionReason::MissingFeature("rayQuery")));
        }
        if ray_tracing_pipeline.ray_tracing_pipeline == vk::FALSE {
            return Err(reject(RejectionReason::MissingFeature(
                "rayTracingPipeline",
            )));
        }

        // SAFETY: same.
        let families = unsafe {
            instance
                .raw()
                .get_physical_device_queue_family_properties(raw)
        };
        let graphics_queue_family = families
            .iter()
            .position(|f| {
                f.queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
            .ok_or_else(|| reject(RejectionReason::NoGraphicsQueue))?
            as u32;

        let mut acceleration_structure_properties =
            vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut ray_tracing_pipeline_properties =
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut props = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut acceleration_structure_properties)
            .push_next(&mut ray_tracing_pipeline_properties);
        // SAFETY: same.
        unsafe {
            instance
                .raw()
                .get_physical_device_properties2(raw, &mut props)
        };

        let optional_extensions: Vec<&'static std::ffi::CStr> =
            OPTIONAL.iter().copied().filter(|n| has(n)).collect();

        Ok(Self {
            raw,
            name,
            device_type: properties.device_type,
            graphics_queue_family,
            support: RayTracingSupport {
                position_fetch: has(ash::khr::ray_tracing_position_fetch::NAME),
                maintenance1: has(ash::khr::ray_tracing_maintenance1::NAME),
                opacity_micromap: has(ash::ext::opacity_micromap::NAME),
            },
            required_extensions,
            optional_extensions,
            limits: RayTracingLimits {
                max_ray_recursion_depth: ray_tracing_pipeline_properties.max_ray_recursion_depth,
                shader_group_handle_size: ray_tracing_pipeline_properties.shader_group_handle_size,
                shader_group_base_alignment: ray_tracing_pipeline_properties
                    .shader_group_base_alignment,
                max_geometry_count: acceleration_structure_properties.max_geometry_count,
                max_instance_count: acceleration_structure_properties.max_instance_count,
            },
        })
    }

    fn is_discrete(&self) -> bool {
        self.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
    }

    /// The underlying handle.
    pub fn raw(&self) -> vk::PhysicalDevice {
        self.raw
    }

    /// Human-readable device name, for logs and the debug overlay.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Index of a queue family supporting both graphics and compute.
    pub fn graphics_queue_family(&self) -> u32 {
        self.graphics_queue_family
    }

    /// Which optional ray tracing extensions this device exposes.
    pub fn support(&self) -> RayTracingSupport {
        self.support
    }

    /// Extension names to enable on the logical device: everything required, plus what is offered.
    pub fn extensions_to_enable(&self) -> Vec<*const std::ffi::c_char> {
        let mut names =
            Vec::with_capacity(self.required_extensions.len() + self.optional_extensions.len());
        names.extend(self.required_extensions.iter().map(|n| n.as_ptr()));
        names.extend(self.optional_extensions.iter().map(|n| n.as_ptr()));
        names
    }

    /// Limits that acceleration structure and shader binding table layout depend on.
    pub fn limits(&self) -> RayTracingLimits {
        self.limits
    }
}
