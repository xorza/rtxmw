//! DLSS Ray Reconstruction, through NVIDIA's NGX SDK.
//!
//! **Hand-written FFI against the C header**, per `docs/design.md` §6. NGX's parameter map is a C++
//! class with a vtable, but every call this needs is exported with C linkage — the SDK's own helper
//! headers reach the map through `NVSDK_NGX_Parameter_SetI` and friends rather than through the
//! vtable, and so does this. Verified against `nm` on `libnvsdk_ngx.a`: no mangled symbol is named
//! here.
//!
//! The whole module is behind the `dlss` feature, because the SDK is NVIDIA's and is not in this
//! repository. Without it the renderer keeps its own à-trous denoiser and does not upscale.

// **Reached only by its own tests so far.** This is the first slice of the NGX integration: the
// extension query has to exist before the instance and the device can be created with what it asks
// for, and it is that creation — in the binary, through `rtxmw-gpu` — that will be its first real
// caller. Kept rather than deferred because the answer is what the next slice is built from.
#![allow(dead_code)]

use std::ffi::{CStr, c_char, c_uint};

mod ffi;

/// What NGX wants enabled on the instance and the device before it is initialised.
///
/// Queried rather than hardcoded: these are the extensions *this* SDK build needs, and the list has
/// changed between versions. It takes no Vulkan objects, so it can be asked before either exists —
/// which is the only order that works, since they have to be created with the answer.
#[derive(Debug)]
struct Requirements {
    instance: Vec<&'static CStr>,
    device: Vec<&'static CStr>,
}

impl Requirements {
    /// Asks the SDK what it needs.
    ///
    /// The lists belong to the SDK and live as long as it is loaded, which is the whole process —
    /// hence `'static`, and hence nothing is copied out of them.
    fn query() -> Result<Self, Status> {
        let mut instance_count: c_uint = 0;
        let mut instance: *const *const c_char = std::ptr::null();
        let mut device_count: c_uint = 0;
        let mut device: *const *const c_char = std::ptr::null();

        // SAFETY: every pointer is to a live local, and the SDK writes only through them. It reads
        // no Vulkan state, so there is nothing to have initialised first.
        let status = Status(unsafe {
            ffi::NVSDK_NGX_VULKAN_RequiredExtensions(
                &mut instance_count,
                &mut instance,
                &mut device_count,
                &mut device,
            )
        });
        status.ok()?;

        // SAFETY: the SDK reported these counts for these arrays, and both are static storage
        // inside it — the header documents the names as owned by the SDK.
        Ok(unsafe {
            Self {
                instance: names(instance, instance_count),
                device: names(device, device_count),
            }
        })
    }
}

/// Borrows `count` C strings out of an array the SDK owns.
///
/// # Safety
/// `array` must point to `count` valid, nul-terminated pointers that outlive the process.
unsafe fn names(array: *const *const c_char, count: c_uint) -> Vec<&'static CStr> {
    if array.is_null() {
        return Vec::new();
    }
    // SAFETY: the caller guarantees the array and its contents.
    unsafe {
        std::slice::from_raw_parts(array, count as usize)
            .iter()
            .map(|&name| CStr::from_ptr(name))
            .collect()
    }
}

/// An `NVSDK_NGX_Result`, kept as the raw value.
///
/// Not mapped to a Rust enum: the SDK defines several dozen codes as a bitfield — success is the
/// single value `1` and every failure carries `0xBAD00000` with the reason in its low bits — and
/// this needs to *report* them faithfully far more than it needs to match on them. The SDK names
/// them all, including ones it has not been told about, so `Display` asks rather than guessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Status(u32);

impl Status {
    const SUCCESS: u32 = 1;

    /// Whether the call succeeded.
    fn is_ok(self) -> bool {
        self.0 == Self::SUCCESS
    }

    fn ok(self) -> Result<(), Self> {
        if self.is_ok() { Ok(()) } else { Err(self) }
    }
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_ok() {
            return write!(f, "ok");
        }
        // The SDK's own name for it, which is more use in a log than the number.
        // SAFETY: the SDK returns a static wide string for any value, including unknown ones.
        let text = unsafe { ffi::GetNGXResultAsString(self.0) };
        if text.is_null() {
            return write!(f, "NGX error {:#x}", self.0);
        }
        let mut wide = text;
        let mut out = String::new();
        // SAFETY: nul-terminated static storage owned by the SDK, walked to its own terminator.
        unsafe {
            while *wide != 0 {
                out.push(char::from_u32(*wide).unwrap_or('?'));
                wide = wide.add(1);
            }
        }
        write!(f, "{out} ({:#x})", self.0)
    }
}

impl std::error::Error for Status {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sdk_says_which_extensions_it_needs() {
        // **The first thing that has to work**, and the only NGX call that takes no Vulkan objects:
        // the instance and device have to be *created* with these enabled, so nothing can be built
        // before the answer is in hand. It doubles as the proof that the static library linked and
        // that the symbols are the C ones rather than the C++ vtable beside them.
        let required = Requirements::query().expect("the SDK should answer");
        println!("NGX wants instance {:?}", required.instance);
        println!("NGX wants device   {:?}", required.device);

        // Vulkan extension names are `VK_`-prefixed by the spec, so this catches a list read at the
        // wrong stride or off the end of the array — which is what a hand-written binding gets
        // wrong, and which would otherwise show up as an unexplained device-creation failure.
        for name in required.instance.iter().chain(&required.device) {
            let text = name.to_str().expect("extension names are ASCII");
            assert!(text.starts_with("VK_"), "{text:?} is not an extension name");
        }
        // Both lists are non-empty on Vulkan: NGX needs at least the two NVX extensions on the
        // device, and an empty answer would mean the query silently did nothing.
        assert!(
            !required.device.is_empty(),
            "NGX asked for no device extensions"
        );
    }

    #[test]
    fn a_failure_carries_the_name_the_sdk_gives_it() {
        // Errors from here reach a log and nothing else, so the one thing that matters is that they
        // say what happened rather than a bare number.
        assert!(Status(1).is_ok());
        let denied = Status(0xBAD0_0000 | 0x0C);
        assert!(!denied.is_ok());
        let text = denied.to_string();
        println!("a failure reads {text:?}");
        // **The SDK's own name, in full.** A `wchar_t` is 32 bits here and reading it at 16 gives
        // back its first character and stops — which looks like a name until it is compared against
        // one, so the check is for the real string rather than for "some letters".
        assert!(
            text.starts_with("NVSDK_NGX_Result_"),
            "a failure rendered as {text:?}, which is not what the SDK calls it"
        );
        assert!(
            text.contains("0xbad0000c"),
            "{text:?} does not carry the code"
        );
    }
}
