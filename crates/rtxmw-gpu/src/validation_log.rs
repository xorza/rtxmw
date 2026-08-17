//! Capture of Vulkan validation errors via `VK_EXT_debug_utils`.

use std::sync::Mutex;
use std::thread::ThreadId;

/// One error reported by the validation layers.
#[derive(Debug, Clone)]
pub struct ValidationMessage {
    pub text: String,
    /// The thread the offending Vulkan call was made on.
    ///
    /// Validation callbacks fire synchronously on the calling thread, and the test harness runs
    /// tests in parallel against one shared log — without this, the first test to provoke an error
    /// would fail every test that checked afterwards.
    pub thread: ThreadId,
}

/// Thread-safe sink for validation errors.
///
/// Printing to stderr is not enough for tests: a render that emits validation errors and still
/// produces plausible pixels would otherwise pass. Recording them lets a test assert on them.
///
/// **Errors only.** Warnings are printed but not stored — nothing reads them, and a long debug
/// session would otherwise accumulate them without bound. Add a policy here if warnings ever need
/// to fail a test.
#[derive(Debug, Default)]
pub struct ValidationLog {
    errors: Mutex<Vec<ValidationMessage>>,
}

impl ValidationLog {
    /// Records an error. Called from the Vulkan debug callback, on whichever thread it fires.
    pub(crate) fn record_error(&self, message: ValidationMessage) {
        self.lock().push(message);
    }

    /// Errors raised by Vulkan calls made on the calling thread.
    pub fn errors_on_this_thread(&self) -> Vec<ValidationMessage> {
        let current = std::thread::current().id();
        self.lock()
            .iter()
            .filter(|m| m.thread == current)
            .cloned()
            .collect()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<ValidationMessage>> {
        // A poisoned lock means another thread panicked mid-record; the errors are worth keeping
        // regardless, and panicking inside a Vulkan callback would unwind across the FFI boundary.
        match self.errors.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}
