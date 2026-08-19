//! The NGX entry points this crate calls, declared by hand.
//!
//! One declaration per symbol actually used, checked against `include/nvsdk_ngx_vk.h` and against
//! `nm` on `libnvsdk_ngx.a`. Nothing here is generated: the header is C++ with default arguments and
//! a vtable-bearing parameter type, and a binding generator produces far more surface than this
//! needs while still requiring the same review of the parts it does use.

use std::ffi::{c_char, c_uint};

unsafe extern "C" {
    /// Instance and device extensions NGX needs enabled. Takes no Vulkan objects.
    pub(super) fn NVSDK_NGX_VULKAN_RequiredExtensions(
        out_instance_count: *mut c_uint,
        out_instance: *mut *const *const c_char,
        out_device_count: *mut c_uint,
        out_device: *mut *const *const c_char,
    ) -> u32;

    /// The SDK's own name for a result code, as a wide string it owns.
    ///
    /// `wchar_t`, which on Linux is **32 bits** — declaring it as `u16` reads UTF-32 at half the
    /// stride and every name comes back one character long, its second byte being the terminator.
    pub(super) fn GetNGXResultAsString(result: u32) -> *const u32;
}
