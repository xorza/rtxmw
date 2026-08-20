//! Logical device creation with the ray tracing feature set enabled.

use ash::vk;

use crate::error::Result;
use crate::instance::Instance;
use crate::physical_device::PhysicalDevice;

/// A logical device plus the queue and extension function tables the renderer needs.
pub struct Device {
    raw: ash::Device,
    pipeline_cache: vk::PipelineCache,
    /// Whether this run's compiled pipelines have already gone back to disk.
    cache_stored: std::sync::atomic::AtomicBool,
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

        // **Seeded from disk, because compiling one of these is most of what starting up costs.**
        // `primary_visibility.comp` is 172,043 words of SPIR-V and the driver takes three seconds
        // turning it into machine code; the other five modules together take one millisecond. With
        // a cache in hand a second renderer in the same process pays about 20 ms instead of 3,100,
        // and with the file behind it a fresh process pays 72 ms rather than 3,169 — which the test
        // suite was paying fourteen times over.
        //
        // Best effort throughout. A cache that cannot be read or written is a slow start, never a
        // wrong picture, so nothing here reports failure — and a blob from another driver or another
        // card is *safe* rather than merely unlikely: its header carries the vendor, device and
        // driver it was built by, and an implementation that does not recognise its own ignores the
        // contents and compiles from source.
        let stored = std::fs::read(Self::cache_path()).unwrap_or_default();
        let cache_info = vk::PipelineCacheCreateInfo::default().initial_data(&stored);
        // SAFETY: the device is alive, and `stored` outlives the call that borrows it.
        let pipeline_cache = unsafe { raw.create_pipeline_cache(&cache_info, None)? };
        Ok(Self {
            raw,
            pipeline_cache,
            cache_stored: std::sync::atomic::AtomicBool::new(false),
            graphics_queue,
            acceleration_structure,
            ray_tracing_pipeline,
        })
    }

    /// Where the driver keeps what it has already compiled.
    pub(crate) fn pipeline_cache(&self) -> vk::PipelineCache {
        self.pipeline_cache
    }

    /// The file the compiled pipelines are kept in between runs.
    ///
    /// **Keyed by nothing here, because Vulkan keys it already.** What goes in the blob is indexed
    /// by the whole pipeline — its layout and the SPIR-V it was built from — so a shader that
    /// changed simply misses and is compiled once more, and the entry it replaces is dropped when
    /// the driver next prunes. Hashing the source to name the file would key it a second time,
    /// coarser, and throw away every *other* pipeline whenever one of them changed.
    fn cache_path() -> std::path::PathBuf {
        if let Some(named) = std::env::var_os("RTXMW_PIPELINE_CACHE") {
            return named.into();
        }
        // The system's own temporary directory, which every platform has and names for itself. A
        // cache is exactly what belongs there: losing it costs one slow start and nothing else.
        std::env::temp_dir().join("rtxmw-pipelines.bin")
    }

    /// Writes what the driver compiled this run back out, for the next one to start from.
    ///
    /// **Called once the pipelines exist rather than only at teardown**, because a device held in a
    /// `static` — which is how the tests share one — is never dropped and so never got here at all.
    /// Once per run *once it succeeds*, and the latch is set on the write rather than on entry: a
    /// call that arrives before anything has been compiled has nothing to save, and burning the
    /// latch on it would mean the run never saved at all. Half a megabyte on this driver, so once is
    /// worth arranging.
    ///
    /// **Through a temporary and a rename**, because the test suite runs a dozen processes at once
    /// and they all finish into the same file: a rename is atomic, so a reader sees one whole blob
    /// or the other and never half of each. Whichever writes last wins, and what it wins with is a
    /// superset of what it started from.
    pub fn store_pipeline_cache(&self) {
        use std::sync::atomic::Ordering;
        if self.cache_stored.load(Ordering::Relaxed) {
            return;
        }
        let at = Self::cache_path();
        // SAFETY: the device is alive and the cache belongs to it.
        let Ok(data) = (unsafe { self.raw.get_pipeline_cache_data(self.pipeline_cache) }) else {
            return;
        };
        if data.is_empty() {
            return;
        }
        let Some(directory) = at.parent() else {
            return;
        };
        if std::fs::create_dir_all(directory).is_err() {
            return;
        }
        let staged = at.with_extension(format!("{}.tmp", std::process::id()));
        if std::fs::write(&staged, &data).is_err() {
            let _ = std::fs::remove_file(&staged);
            return;
        }
        match std::fs::rename(&staged, &at) {
            Ok(()) => self.cache_stored.store(true, Ordering::Relaxed),
            Err(_) => {
                let _ = std::fs::remove_file(&staged);
            }
        }
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
        self.store_pipeline_cache();
        unsafe {
            let _ = self.raw.device_wait_idle();
            self.raw.destroy_pipeline_cache(self.pipeline_cache, None);
            self.raw.destroy_device(None);
        }
    }
}
