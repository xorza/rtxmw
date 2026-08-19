//! The process-wide Vulkan device shared by every test.

use std::sync::{LazyLock, Mutex, MutexGuard};

use ash::vk;

use crate::device::Device;
use crate::error::Result;
use crate::instance::{Instance, Validation};
use crate::memory::Memory;
use crate::physical_device::{PhysicalDevice, Presentation};
use crate::testing::render_target::RenderTarget;
use crate::uploader::Uploader;
use crate::validation_log::ValidationLog;

/// A Vulkan device shared by every test in the process.
///
/// Device creation costs on the order of 100 ms and the test harness runs tests in parallel
/// threads, so creating one per test would dominate the suite and risk exhausting driver
/// resources. This is built once on first use and never torn down — the process exiting is the
/// teardown.
pub struct TestGpu {
    /// Behind a `Mutex` because a command pool is externally synchronised and a `VkQueue` may not
    /// be submitted to from two threads at once.
    uploader: Mutex<Uploader>,
    memory: Memory,
    device: Device,
    physical: PhysicalDevice,
    instance: Instance,
}

impl std::fmt::Debug for TestGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestGpu")
            .field("device", &self.physical.name())
            .finish_non_exhaustive()
    }
}

static SHARED: LazyLock<TestGpu> =
    LazyLock::new(|| TestGpu::new().expect("failed to bring up the test GPU"));

impl TestGpu {
    /// The shared device, created on first call.
    ///
    /// Panics if Vulkan cannot be brought up, because every test depending on it would fail anyway
    /// and one clear panic beats a cascade of confusing errors.
    pub fn shared() -> &'static TestGpu {
        &SHARED
    }

    fn new() -> Result<Self> {
        // Headless: no surface extensions. `Record` rather than `AbortOnError` — an abort would
        // take down the whole suite, and each test asserts on its own thread's errors instead.
        let instance = Instance::new(c"rtxmw-test", &[], Validation::Record)?;
        let physical = PhysicalDevice::select(&instance, Presentation::NotNeeded)?;
        let device = Device::new(&instance, &physical, &[])?;
        let memory = Memory::new(&instance, &physical, &device)?;
        let uploader = Uploader::new(&device, &memory, physical.graphics_queue_family())?;

        Ok(Self {
            uploader: Mutex::new(uploader),
            memory,
            device,
            physical,
            instance,
        })
    }

    /// The logical device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The physical device the tests run against.
    pub fn physical(&self) -> &PhysicalDevice {
        &self.physical
    }

    /// The captured validation output, when validation is available.
    pub fn validation_log(&self) -> Option<&ValidationLog> {
        self.instance.validation_log()
    }

    /// Fails the calling test if validation flagged any Vulkan call made on this thread.
    ///
    /// Call at the end of any test that submits work: a render that emits validation errors and
    /// still produces plausible pixels would otherwise pass silently. Scoped to the current thread
    /// because the log is shared by every test in the process.
    #[track_caller]
    pub fn assert_no_validation_errors(&self) {
        let Some(log) = self.validation_log() else {
            return;
        };
        let errors = log.errors_on_this_thread();
        assert!(
            errors.is_empty(),
            "{} validation error(s):\n{}",
            errors.len(),
            errors
                .iter()
                .map(|e| format!("  {}", e.text))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// The device's memory allocator, for creating buffers and images.
    pub fn memory(&self) -> &Memory {
        &self.memory
    }

    /// Locks the shared uploader for the caller.
    ///
    /// Serialised across threads: the queue and command pool are shared, and a test suite gains
    /// nothing from overlapping submissions. [`TestGpu::submit_and_wait`] takes the same lock, so
    /// calling it while holding this guard deadlocks — go through the guard instead.
    pub fn uploader(&self) -> MutexGuard<'_, Uploader> {
        match self.uploader.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Records `record` into a one-shot command buffer, submits it, and blocks until it completes.
    pub fn submit_and_wait(
        &self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<()> {
        self.uploader().submit_and_wait(record)
    }

    /// Creates an offscreen colour target to render into.
    pub fn create_target(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<RenderTarget> {
        RenderTarget::new(&self.memory, width, height, format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_returns_one_device_for_the_whole_process() {
        let first = TestGpu::shared();
        let second = TestGpu::shared();
        assert!(
            std::ptr::eq(first, second),
            "each call built a separate device; the whole point is to share one"
        );
        assert!(!first.physical().name().is_empty());
    }

    #[test]
    fn validation_is_active_so_errors_can_fail_a_test() {
        let gpu = TestGpu::shared();
        assert!(
            gpu.validation_log().is_some(),
            "validation layers are unavailable, so render tests cannot catch API misuse"
        );
    }

    #[test]
    fn submitting_an_empty_command_buffer_succeeds() {
        let gpu = TestGpu::shared();
        gpu.submit_and_wait(|_device, _cmd| {})
            .expect("empty submission failed");
        gpu.assert_no_validation_errors();
    }
}
