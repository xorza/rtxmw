//! Vulkan instance creation and the validation debug messenger.

use std::io::Write;
use std::sync::Arc;

use ash::vk;

use crate::error::Result;
use crate::validation_log::{ValidationLog, ValidationMessage};

const VALIDATION_LAYER: &std::ffi::CStr = c"VK_LAYER_KHRONOS_validation";

/// How much validation to run, and what to do when it complains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Validation {
    /// No layers, no messenger.
    Disabled,
    /// Record messages and print them; let the process continue.
    ///
    /// For the test harness, which asserts per test so one failure does not take the suite down.
    Record,
    /// Record, print, and **abort the process** on the first error.
    ///
    /// Aborting rather than unwinding is not a stylistic choice: the callback is `extern "system"`
    /// and a panic crossing that boundary is undefined behaviour. `abort` stops at the offending
    /// Vulkan call with the stack intact, which is what a debugger needs.
    AbortOnError,
}

impl Validation {
    /// [`Self::AbortOnError`] in debug builds, [`Self::Disabled`] otherwise.
    ///
    /// Validation costs far too much to ship, and a release build has no way to act on a report.
    pub fn for_build() -> Self {
        if cfg!(debug_assertions) {
            Self::AbortOnError
        } else {
            Self::Disabled
        }
    }

    fn wants_layer(self) -> bool {
        self != Self::Disabled
    }

    fn aborts(self) -> bool {
        self == Self::AbortOnError
    }
}

/// Shared with the Vulkan debug callback through its user-data pointer.
#[derive(Debug)]
struct MessengerContext {
    log: ValidationLog,
    abort_on_error: bool,
}

/// Owns the Vulkan loader, the `VkInstance`, and the debug messenger when validation is on.
pub struct Instance {
    // Declaration order is drop order: the messenger must go before the instance that owns it.
    debug: Option<DebugMessenger>,
    entry: ash::Entry,
    raw: ash::Instance,
}

// `ash::Entry` and `ash::Instance` are function tables and implement no `Debug`.
impl std::fmt::Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("handle", &self.raw.handle())
            .field("validation", &self.debug.is_some())
            .finish_non_exhaustive()
    }
}

struct DebugMessenger {
    loader: ash::ext::debug_utils::Instance,
    raw: vk::DebugUtilsMessengerEXT,
    /// Kept alive because the messenger's user-data pointer aliases it.
    context: Arc<MessengerContext>,
}

impl Instance {
    /// Loads the Vulkan loader and creates an instance.
    ///
    /// `surface_extensions` are the instance extensions the windowing system requires; pass an
    /// empty slice for headless use. Anything validation needs that the machine does not provide is
    /// skipped with a warning, so a host without the Vulkan SDK still runs.
    pub fn new(
        app_name: &std::ffi::CStr,
        surface_extensions: &[*const std::ffi::c_char],
        validation: Validation,
    ) -> Result<Self> {
        // SAFETY: loads the system Vulkan loader; unsound only if the library is malicious.
        let entry = unsafe { ash::Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(app_name)
            // ash 0.38 ships Vulkan 1.3.281 headers. 1.3 plus the KHR ray tracing extensions
            // covers everything the renderer needs; 1.4 waits for ash 0.39.
            .api_version(vk::API_VERSION_1_3);

        let mut layers: Vec<*const std::ffi::c_char> = Vec::new();
        let mut extensions: Vec<*const std::ffi::c_char> = surface_extensions.to_vec();

        let mut layer_available = false;
        if validation.wants_layer() {
            layer_available = Self::validation_layer_present(&entry)?;
            if !layer_available {
                eprintln!("validation requested but VK_LAYER_KHRONOS_validation is not installed");
            }
        }

        let mut sync_validation = false;
        if layer_available {
            layers.push(VALIDATION_LAYER.as_ptr());
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());

            // SAFETY: the entry is loaded; this queries the layer's own extensions.
            let layer_extensions =
                unsafe { entry.enumerate_instance_extension_properties(Some(VALIDATION_LAYER))? };
            sync_validation = layer_extensions
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::validation_features::NAME));
            if sync_validation {
                extensions.push(ash::ext::validation_features::NAME.as_ptr());
            }
        }

        let context = layer_available.then(|| {
            Arc::new(MessengerContext {
                log: ValidationLog::default(),
                abort_on_error: validation.aborts(),
            })
        });

        // Synchronisation validation catches missing and incorrect barriers, which is the failure
        // mode a raytracer with hand-written barriers actually hits. It is off by default.
        let enabled_features = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
        let mut validation_features =
            vk::ValidationFeaturesEXT::default().enabled_validation_features(&enabled_features);

        // Chaining a messenger into the create-info covers `vkCreateInstance` and
        // `vkDestroyInstance` themselves, which the standalone messenger below cannot see.
        let mut create_messenger = context
            .as_ref()
            .map(|context| messenger_create_info(Arc::as_ptr(context)));

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        if sync_validation {
            create_info = create_info.push_next(&mut validation_features);
        }
        if let Some(create_messenger) = create_messenger.as_mut() {
            create_info = create_info.push_next(create_messenger);
        }

        // SAFETY: `create_info` and everything it points at outlive this call.
        let raw = unsafe { entry.create_instance(&create_info, None)? };

        let debug = match context {
            Some(context) => Some(DebugMessenger::new(&entry, &raw, context)?),
            None => None,
        };

        Ok(Self { debug, entry, raw })
    }

    fn validation_layer_present(entry: &ash::Entry) -> Result<bool> {
        // SAFETY: the entry is loaded.
        let available = unsafe { entry.enumerate_instance_layer_properties()? };
        Ok(available
            .iter()
            .any(|l| l.layer_name_as_c_str() == Ok(VALIDATION_LAYER)))
    }

    /// The loaded Vulkan entry points.
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    /// The underlying `VkInstance` wrapper.
    pub fn raw(&self) -> &ash::Instance {
        &self.raw
    }

    /// Captured validation errors, when validation is enabled. Warnings are printed, not stored.
    pub fn validation_log(&self) -> Option<&ValidationLog> {
        self.debug.as_ref().map(|d| &d.context.log)
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        if let Some(debug) = self.debug.take() {
            // SAFETY: the messenger belongs to this instance and nothing else references it.
            unsafe { debug.loader.destroy_debug_utils_messenger(debug.raw, None) };
        }
        // SAFETY: every object created from this instance is dropped before it, by construction —
        // `Device` borrows `Instance`, so the borrow checker enforces the ordering.
        unsafe { self.raw.destroy_instance(None) };
    }
}

impl DebugMessenger {
    fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        context: Arc<MessengerContext>,
    ) -> Result<Self> {
        let loader = ash::ext::debug_utils::Instance::new(entry, instance);
        let create_info = messenger_create_info(Arc::as_ptr(&context));

        // SAFETY: `create_info` outlives the call, and the user-data pointer stays valid because
        // `context` is stored alongside the messenger and outlives it.
        let raw = unsafe { loader.create_debug_utils_messenger(&create_info, None)? };

        Ok(Self {
            loader,
            raw,
            context,
        })
    }
}

fn messenger_create_info(
    context: *const MessengerContext,
) -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback))
        .user_data(context as *mut std::ffi::c_void)
}

/// Records the message, mirrors it to stderr, and aborts on an error when asked to.
///
/// Returning `FALSE` is required: `TRUE` aborts the offending Vulkan call.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    // SAFETY: Vulkan guarantees `data` is valid for the duration of the callback.
    let text = unsafe {
        data.as_ref()
            .and_then(|d| {
                (!d.p_message.is_null()).then(|| {
                    std::ffi::CStr::from_ptr(d.p_message)
                        .to_string_lossy()
                        .into_owned()
                })
            })
            .unwrap_or_default()
    };

    let is_error = severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR);
    eprintln!(
        "[vulkan {}] {text}",
        if is_error { "error" } else { "warning" }
    );

    if !is_error || user_data.is_null() {
        return vk::FALSE;
    }
    // SAFETY: the pointer was set from an `Arc<MessengerContext>` that outlives the messenger.
    let context = unsafe { &*(user_data as *const MessengerContext) };

    context.log.record_error(ValidationMessage {
        text,
        thread: std::thread::current().id(),
    });

    if context.abort_on_error {
        eprintln!(
            "\naborting on a Vulkan validation error.\n\
             run under a debugger for the call that produced it, or use `Validation::Record` to \
             continue past it."
        );
        let _ = std::io::stderr().flush();
        std::process::abort();
    }

    vk::FALSE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_policy_matches_its_documented_behaviour() {
        // Tests always run with `debug_assertions`, so only this branch is reachable here; the
        // release branch is pinned by the `cfg!` in `for_build` itself.
        assert_eq!(Validation::for_build(), Validation::AbortOnError);

        assert!(Validation::AbortOnError.wants_layer());
        assert!(Validation::AbortOnError.aborts());

        // `Record` still loads the layer but must never take the process down — the test harness
        // depends on that.
        assert!(Validation::Record.wants_layer());
        assert!(!Validation::Record.aborts());

        assert!(!Validation::Disabled.wants_layer());
        assert!(!Validation::Disabled.aborts());
    }
}
