//! The process-wide Vulkan device shared by every test.

use std::sync::{LazyLock, Mutex, MutexGuard};

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use crate::device::Device;
use crate::error::Result;
use crate::instance::{Instance, Validation};
use crate::physical_device::{PhysicalDevice, Presentation};
use crate::testing::render_target::RenderTarget;
use crate::validation_log::ValidationLog;

/// Command pool, buffer and fence for one-shot submissions.
///
/// Vulkan command pools are externally synchronised and a `VkQueue` may not be submitted to from
/// two threads at once, so this is only ever reached through a `Mutex`.
#[derive(Debug)]
struct Submitter {
    /// Owned so the buffer stays valid. Never read: `TestGpu` lives in a `static` and is never
    /// dropped, so process exit is the teardown.
    #[allow(dead_code)]
    pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

/// A Vulkan device shared by every test in the process.
///
/// Device creation costs on the order of 100 ms and the test harness runs tests in parallel
/// threads, so creating one per test would dominate the suite and risk exhausting driver
/// resources. This is built once on first use and never torn down — the process exiting is the
/// teardown.
pub struct TestGpu {
    submitter: Mutex<Submitter>,
    allocator: Mutex<Allocator>,
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
        let device = Device::new(&instance, &physical)?;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.raw().clone(),
            device: device.raw().clone(),
            physical_device: physical.raw(),
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })
        .expect("failed to create the device memory allocator");

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(physical.graphics_queue_family())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: the device is alive and `pool_info` is initialised.
        let pool = unsafe { device.raw().create_command_pool(&pool_info, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool was just created on this device.
        let command_buffer = unsafe { device.raw().allocate_command_buffers(&alloc_info)? }[0];

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: same.
        let fence = unsafe { device.raw().create_fence(&fence_info, None)? };

        Ok(Self {
            submitter: Mutex::new(Submitter {
                pool,
                command_buffer,
                fence,
            }),
            allocator: Mutex::new(allocator),
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

    /// Locks the allocator for the caller.
    pub(crate) fn allocator(&self) -> MutexGuard<'_, Allocator> {
        match self.allocator.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Records `record` into a one-shot command buffer, submits it, and blocks until it completes.
    ///
    /// Serialised across threads: the queue and command pool are shared, and a test suite gains
    /// nothing from overlapping submissions.
    pub fn submit_and_wait(
        &self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<()> {
        let submitter = match self.submitter.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let raw = self.device.raw();

        // SAFETY: the fence and command buffer are idle — any previous submission was waited on
        // before the lock was released.
        unsafe {
            raw.reset_fences(&[submitter.fence])?;
            raw.reset_command_buffer(
                submitter.command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )?;

            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            raw.begin_command_buffer(submitter.command_buffer, &begin)?;
        }

        record(raw, submitter.command_buffer);

        // SAFETY: the command buffer is recording and every handle is alive.
        unsafe {
            raw.end_command_buffer(submitter.command_buffer)?;

            let buffers =
                [vk::CommandBufferSubmitInfo::default().command_buffer(submitter.command_buffer)];
            let submit = vk::SubmitInfo2::default().command_buffer_infos(&buffers);
            raw.queue_submit2(self.device.graphics_queue(), &[submit], submitter.fence)?;

            raw.wait_for_fences(&[submitter.fence], true, u64::MAX)?;
        }

        Ok(())
    }

    /// Creates an offscreen colour target to render into.
    pub fn create_target(
        &'static self,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<RenderTarget> {
        RenderTarget::new(self, width, height, format)
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
