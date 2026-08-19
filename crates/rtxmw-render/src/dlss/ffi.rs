//! The NGX entry points this crate calls, declared by hand.
//!
//! One declaration per symbol actually used, checked against `include/nvsdk_ngx_vk.h` and against
//! `nm` on `libnvsdk_ngx.a`. Nothing here is generated: the header is C++ with default arguments and
//! a vtable-bearing parameter type, and a binding generator produces far more surface than this
//! needs while still requiring the same review of the parts it does use.

use std::ffi::{c_char, c_int, c_uint, c_void};

/// The optimal-settings query, as the driver's feature library provides it.
///
/// Reads its inputs from the parameter map and writes its answers back into the same map — nothing
/// is passed or returned but the map itself.
pub(super) type OptimalSettingsCallback = unsafe extern "C" fn(*mut c_void) -> u32;

/// `NVSDK_NGX_PathListInfo`, from `nvsdk_ngx_defs.h:352`.
#[repr(C)]
pub(super) struct PathList {
    /// `wchar_t const* const*`, so an array of UTF-32 strings on Linux.
    pub(super) paths: *const *const u32,
    pub(super) length: c_uint,
}

/// `NVSDK_NGX_FeatureCommonInfo`, from `nvsdk_ngx_defs.h:400`.
///
/// The logging block is by value rather than behind a pointer, so the whole struct has to be
/// declared even though only the path list is set — a short one would leave NGX reading past it.
#[repr(C)]
pub(super) struct FeatureCommonInfo {
    pub(super) paths: PathList,
    /// `NVSDK_NGX_FeatureCommonInfo_Internal*`, which NGX fills in. Null on the way in.
    pub(super) internal: *mut c_void,
    /// `NVSDK_NGX_LoggingInfo`: a callback, a level, and a flag.
    pub(super) logging_callback: *const c_void,
    pub(super) minimum_logging_level: c_int,
    pub(super) disable_other_logging_sinks: bool,
}

/// `NVSDK_NGX_ImageViewInfo_VK`, from `nvsdk_ngx_defs_vk.h:52`.
#[repr(C)]
pub(super) struct ImageViewInfo {
    pub(super) view: ash::vk::ImageView,
    pub(super) image: ash::vk::Image,
    pub(super) subresource: ash::vk::ImageSubresourceRange,
    pub(super) format: ash::vk::Format,
    pub(super) width: c_uint,
    pub(super) height: c_uint,
}

/// `NVSDK_NGX_Resource_VK`, from `nvsdk_ngx_defs_vk.h:93`.
///
/// Its first member is a union of an image view and a buffer. Only the image arm is used here and it
/// is the larger of the two, so declaring it directly gives the union its correct size without a
/// second arm nothing reads.
#[repr(C)]
pub(super) struct Resource {
    pub(super) view: ImageViewInfo,
    /// `NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW`, which is zero.
    pub(super) kind: c_int,
    /// True where the image was created with `STORAGE` usage — every one of ours is.
    pub(super) read_write: bool,
}

unsafe extern "C" {
    /// Instance and device extensions NGX needs enabled. Takes no Vulkan objects.
    pub(super) fn NVSDK_NGX_VULKAN_RequiredExtensions(
        out_instance_count: *mut c_uint,
        out_instance: *mut *const *const c_char,
        out_device_count: *mut c_uint,
        out_device: *mut *const *const c_char,
    ) -> u32;

    /// Brings NGX up on an existing Vulkan device.
    ///
    /// `nvsdk_ngx_vk.h:260`, in its C form — the C++ declaration beside it gives the last four
    /// parameters defaults, and passing null for all of them is what those defaults are. The data
    /// path is `wchar_t`, 32 bits here.
    pub(super) fn NVSDK_NGX_VULKAN_Init_with_ProjectID(
        project_id: *const c_char,
        engine: c_int,
        engine_version: *const c_char,
        application_data_path: *const u32,
        instance: ash::vk::Instance,
        physical_device: ash::vk::PhysicalDevice,
        device: ash::vk::Device,
        get_instance_proc_addr: *const c_void,
        get_device_proc_addr: *const c_void,
        feature_info: *const c_void,
        sdk_version: c_uint,
    ) -> u32;

    /// Releases NGX and everything it holds for `device`.
    pub(super) fn NVSDK_NGX_VULKAN_Shutdown1(device: ash::vk::Device) -> u32;

    /// The parameter map describing what this driver and device can do. Owned by NGX.
    pub(super) fn NVSDK_NGX_VULKAN_GetCapabilityParameters(out: *mut *mut c_void) -> u32;

    /// Writes an unsigned integer into a parameter map.
    pub(super) fn NVSDK_NGX_Parameter_SetUI(
        parameters: *mut c_void,
        name: *const c_char,
        value: c_uint,
    );

    /// Reads an unsigned integer out of a parameter map.
    pub(super) fn NVSDK_NGX_Parameter_GetUI(
        parameters: *mut c_void,
        name: *const c_char,
        out: *mut c_uint,
    ) -> u32;

    /// Writes an integer into a parameter map.
    pub(super) fn NVSDK_NGX_Parameter_SetI(
        parameters: *mut c_void,
        name: *const c_char,
        value: c_int,
    );

    /// Reads a pointer out of a parameter map. **How the optimal-settings query is reached**: it is
    /// not an exported symbol but a function pointer the driver's feature library puts here.
    pub(super) fn NVSDK_NGX_Parameter_GetVoidPointer(
        parameters: *mut c_void,
        name: *const c_char,
        out: *mut *mut c_void,
    ) -> u32;

    /// Reads an integer out of a parameter map.
    pub(super) fn NVSDK_NGX_Parameter_GetI(
        parameters: *mut c_void,
        name: *const c_char,
        out: *mut c_int,
    ) -> u32;

    /// A parameter map of the caller's own, for a feature rather than for queries.
    pub(super) fn NVSDK_NGX_VULKAN_AllocateParameters(out: *mut *mut c_void) -> u32;

    /// Releases one of those.
    pub(super) fn NVSDK_NGX_VULKAN_DestroyParameters(parameters: *mut c_void) -> u32;

    /// Builds a feature. Records into `cmd`, which must be recording.
    ///
    /// The `1` variant, which the SDK's own helper prefers whenever a device is to hand — the
    /// older one leaves NGX to find the device itself.
    pub(super) fn NVSDK_NGX_VULKAN_CreateFeature1(
        device: ash::vk::Device,
        cmd: ash::vk::CommandBuffer,
        feature: c_int,
        parameters: *mut c_void,
        out: *mut *mut c_void,
    ) -> u32;

    /// Writes a pointer into a parameter map. How a resource is handed to a feature.
    pub(super) fn NVSDK_NGX_Parameter_SetVoidPointer(
        parameters: *mut c_void,
        name: *const c_char,
        value: *mut c_void,
    );

    /// Writes a float into a parameter map.
    pub(super) fn NVSDK_NGX_Parameter_SetF(
        parameters: *mut c_void,
        name: *const c_char,
        value: f32,
    );

    /// Runs a feature. **The `_C` variant**, which is the one the SDK's own helper calls: the
    /// unsuffixed symbol beside it takes a C++ callback type.
    pub(super) fn NVSDK_NGX_VULKAN_EvaluateFeature_C(
        cmd: ash::vk::CommandBuffer,
        handle: *mut c_void,
        parameters: *mut c_void,
        progress: *const c_void,
    ) -> u32;

    /// Releases a feature and everything it holds.
    pub(super) fn NVSDK_NGX_VULKAN_ReleaseFeature(handle: *mut c_void) -> u32;

    /// The SDK's own name for a result code, as a wide string it owns.
    ///
    /// `wchar_t`, which on Linux is **32 bits** — declaring it as `u16` reads UTF-32 at half the
    /// stride and every name comes back one character long, its second byte being the terminator.
    pub(super) fn GetNGXResultAsString(result: u32) -> *const u32;
}
