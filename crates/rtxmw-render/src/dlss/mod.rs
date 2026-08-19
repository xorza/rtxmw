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

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};

use rtxmw_gpu::{Device, Image, Instance, PhysicalDevice};

mod ffi;

/// How much NGX says about what it is doing, from `RTXMW_NGX_LOG`.
///
/// **Off unless asked, and worth asking.** Every failure this API has is a one-word status —
/// `FAIL_InvalidParameter` names no parameter — while NGX's own log says exactly what it disliked:
/// "Low resolution Motion Vectors required" is reachable no other way, and cost two wrong guesses
/// before it was found.
///
/// It is not on by default because it cannot be had quietly: the feature libraries write 422 lines
/// to the console on a single successful run, which is enough to bury the assertion message of
/// whatever failure sent someone looking. Set `RTXMW_NGX_LOG=1` for that, or `2` for everything;
/// the files land beside the logs in the data path.
fn logging_level() -> c_int {
    std::env::var("RTXMW_NGX_LOG")
        .ok()
        .and_then(|level| level.parse().ok())
        .unwrap_or(0)
}

/// `NVSDK_NGX_Feature_RayReconstruction`, from `nvsdk_ngx_defs.h:214`.
const FEATURE_RAY_RECONSTRUCTION: c_int = 13;

/// How the frame is described to DLSS, from `nvsdk_ngx_defs.h:291`.
///
/// `IsHDR` because the trace's output is scene-referred radiance rather than a tone-mapped image —
/// the whole point of the G-buffer split — and `DepthInverted` because the projection is reverse-Z,
/// so the near plane is 1 and the far plane 0.
///
/// **`MVLowRes` reads as a description, not a request**: it says the motion vectors *are* at the
/// low — render — resolution, which §8.13 writes them at. Reasoning it the other way round and
/// leaving it out is what Ray Reconstruction rejects with "Low resolution Motion Vectors required",
/// a message reachable only from NGX's own log.
const CREATE_FLAGS: c_int = (1 << 0) | (1 << 1) | (1 << 3);

/// Extensions NGX names that Vulkan 1.2 provides itself, and which must therefore not be enabled.
///
/// Enabling one beside the core feature it was promoted into is invalid rather than redundant.
const SUPERSEDED_BY_CORE: &[&CStr] = &[c"VK_EXT_buffer_device_address"];

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
        let mut asked = unsafe {
            Self {
                instance: names(instance, instance_count),
                device: names(device, device_count),
            }
        };
        // **NGX still asks for extensions Vulkan has since absorbed**, because it supports drivers
        // older than this one does. `VK_EXT_buffer_device_address` is the case that bites: the same
        // capability is core in Vulkan 1.2 and this device enables it there, and the spec forbids
        // having both — `vkCreateDevice` rejects the pair outright rather than ignoring one.
        // Dropping the superseded name leaves NGX the capability under its core name.
        asked
            .device
            .retain(|name| !SUPERSEDED_BY_CORE.contains(name));
        Ok(asked)
    }
}

/// Writes an unsigned into a parameter map.
///
/// A free function taking the map, because there are two of them and they are not interchangeable:
/// NGX owns the capability map and answers questions from it, while a feature is built from one the
/// caller allocates. Both are written the same way and neither belongs to the other.
///
/// # Safety
/// `parameters` must be a live NGX parameter map.
fn set_u32(parameters: *mut c_void, name: &CStr, value: u32) {
    // SAFETY: the caller guarantees the map; the name is a nul-terminated static.
    unsafe { ffi::NVSDK_NGX_Parameter_SetUI(parameters, name.as_ptr(), value) };
}

/// Writes an integer into a parameter map. See [`set_u32`].
///
/// # Safety
/// `parameters` must be a live NGX parameter map.
fn set_i32(parameters: *mut c_void, name: &CStr, value: c_int) {
    // SAFETY: as above.
    unsafe { ffi::NVSDK_NGX_Parameter_SetI(parameters, name.as_ptr(), value) };
}

/// Writes a float into a parameter map. See [`set_u32`].
///
/// # Safety
/// `parameters` must be a live NGX parameter map.
fn set_f32(parameters: *mut c_void, name: &CStr, value: f32) {
    // SAFETY: as above.
    unsafe { ffi::NVSDK_NGX_Parameter_SetF(parameters, name.as_ptr(), value) };
}

/// Hands a resource to a parameter map. See [`set_u32`].
///
/// # Safety
/// `parameters` must be a live NGX parameter map, and `value` must outlive every call that reads the
/// map — the map keeps the pointer rather than what it points at.
fn set_resource(parameters: *mut c_void, name: &CStr, value: &mut ffi::Resource) {
    // SAFETY: as above, and the caller guarantees the resource's lifetime.
    unsafe {
        ffi::NVSDK_NGX_Parameter_SetVoidPointer(
            parameters,
            name.as_ptr(),
            (value as *mut ffi::Resource).cast(),
        )
    };
}

/// A path as NGX takes one: nul-terminated UTF-32.
///
/// **`wchar_t` is 32 bits here**, not the 16 it is on Windows — the width that truncated every error
/// name to one character when `GetNGXResultAsString` was first declared. Written once because it is
/// the conversion this binding is most likely to get wrong twice.
fn wide(path: &std::path::Path) -> Vec<u32> {
    path.to_string_lossy()
        .chars()
        .map(u32::from)
        .chain(std::iter::once(0))
        .collect()
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
    /// `NVSDK_NGX_Result_FAIL_OutOfDate`. The one failure this crate raises itself, and the one the
    /// SDK's own helper raises for the same reason — a feature library too old, or never found.
    const OUT_OF_DATE: Self = Self(0xBAD0_0000 | 12);

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

/// NGX, brought up on a device and shut down with it.
///
/// **One per device and one per process.** The SDK keeps its state globally keyed by the `VkDevice`
/// it was initialised with, so this owns that lifetime: constructing it initialises, dropping it
/// shuts down.
#[derive(Debug)]
struct Ngx {
    device: ash::vk::Device,
    /// The capability map. NGX owns the allocation; this is a borrowed handle into it.
    capabilities: *mut c_void,
}

impl Ngx {
    /// Brings NGX up on `device`, which must have been created with [`Requirements::query`]'s
    /// extensions enabled.
    /// `data` is a directory NGX may write to. **Not optional**, despite the header giving it a
    /// default of null: passing none comes back as `FAIL_UnableToWriteToAppDataPath`. NGX keeps its
    /// logs and its downloaded feature libraries there.
    fn new(
        instance: &Instance,
        physical: &PhysicalDevice,
        device: &Device,
        data: &std::path::Path,
        feature_libraries: &std::path::Path,
    ) -> Result<Self, Status> {
        // NVIDIA's handle on an application, for their own telemetry and driver overrides. **A
        // UUID, and parsed as one** — a memorable string in the last group instead of hex comes back
        // as `FAIL_InvalidParameter` with nothing to say which parameter. Ours, not borrowed from
        // another project, and `CUSTOM` below because this is not one of the engines they know.
        const PROJECT: &CStr = c"7b2c9f1e-4a63-4d58-9c07-5e3f18ab24d6";
        const VERSION: &CStr = c"0.1.0";
        const ENGINE_CUSTOM: c_int = 0;
        // `NVSDK_NGX_VERSION_API_MACRO`, from `nvsdk_ngx_defs.h:56`. The SDK rejects a mismatch, so
        // this is the one number that has to track the headers in `.refs/`.
        const API_VERSION: c_uint = 0x0000015;

        let data_path = wide(data);

        // **Where NGX looks for the feature library**, and the reason Ray Reconstruction reports
        // itself unavailable without it: `libnvidia-ngx-dlssd.so` is not on the loader path and not
        // beside this binary, so the only way NGX finds it is being told. The default search is the
        // application folder alone.
        let library_path = wide(feature_libraries);
        let search = [library_path.as_ptr()];
        let common = ffi::FeatureCommonInfo {
            paths: ffi::PathList {
                paths: search.as_ptr(),
                length: 1,
            },
            internal: std::ptr::null_mut(),
            logging_callback: std::ptr::null(),
            minimum_logging_level: logging_level(),
            disable_other_logging_sinks: false,
        };

        // SAFETY: the handles are live for the call, the strings are nul-terminated, the paths,
        // `search` and `common` all outlive it, and null for the two remaining optional
        // parameters is what the C++ overload defaults them to.
        let status = Status(unsafe {
            ffi::NVSDK_NGX_VULKAN_Init_with_ProjectID(
                PROJECT.as_ptr(),
                ENGINE_CUSTOM,
                VERSION.as_ptr(),
                data_path.as_ptr(),
                instance.raw().handle(),
                physical.raw(),
                device.raw().handle(),
                std::ptr::null(),
                std::ptr::null(),
                &raw const common as *const c_void,
                API_VERSION,
            )
        });
        status.ok()?;

        let mut capabilities: *mut c_void = std::ptr::null_mut();
        // SAFETY: NGX is initialised, and it writes a pointer it owns through this one.
        let status =
            Status(unsafe { ffi::NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut capabilities) });
        if let Err(failed) = status.ok() {
            // SAFETY: initialisation succeeded, so this is the matching shutdown.
            unsafe { ffi::NVSDK_NGX_VULKAN_Shutdown1(device.raw().handle()) };
            return Err(failed);
        }

        Ok(Self {
            device: device.raw().handle(),
            capabilities,
        })
    }

    /// An integer the capability map holds, or `None` where it holds none by that name.
    fn capability(&self, name: &CStr) -> Option<c_int> {
        let mut value: c_int = 0;
        // SAFETY: the map is NGX's and outlives this, and the name is a nul-terminated static.
        let status = Status(unsafe {
            ffi::NVSDK_NGX_Parameter_GetI(self.capabilities, name.as_ptr(), &mut value)
        });
        status.is_ok().then_some(value)
    }

    /// Whether this driver and device offer Ray Reconstruction.
    ///
    /// **Asked of NGX rather than inferred from the hardware.** It depends on the driver, the SDK
    /// and the GPU together, and NGX is the only one that knows all three.
    fn ray_reconstruction(&self) -> bool {
        self.capability(c"SuperSamplingDenoising.Available") == Some(1)
    }
}

/// How aggressively DLSS upscales, and so what it costs.
///
/// §5.3 settles on **Performance** — 1920×1080 internal to 3840×2160 — which is the mode the whole
/// frame budget is written against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Preset {
    Performance,
    Balanced,
    Quality,
}

impl Preset {
    /// The `NVSDK_NGX_PerfQuality_Value` this is, from `nvsdk_ngx_defs.h:251`.
    fn value(self) -> c_int {
        match self {
            Self::Performance => 0,
            Self::Balanced => 1,
            Self::Quality => 2,
        }
    }
}

/// What DLSS wants to be handed, for an output size and a quality level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OptimalSettings {
    /// The resolution to render at.
    render: (u32, u32),
    /// The range a dynamic-resolution renderer may move between. Equal to `render` where the
    /// feature library does not support varying it.
    lowest: (u32, u32),
    highest: (u32, u32),
}

impl Ngx {
    /// Asks DLSS what to render at to produce `output` under `preset`.
    ///
    /// **Not an exported symbol.** The query lives as a function pointer inside the capability map,
    /// put there by the driver's feature library — which is why it is absent when that library is
    /// not found, and why the SDK's own helper answers `FAIL_OutOfDate` in that case rather than
    /// something about paths.
    fn optimal_settings(
        &self,
        output: (u32, u32),
        preset: Preset,
    ) -> Result<OptimalSettings, Status> {
        let mut callback: *mut c_void = std::ptr::null_mut();
        // SAFETY: the map is NGX's and outlives this; the name is a nul-terminated static.
        Status(unsafe {
            ffi::NVSDK_NGX_Parameter_GetVoidPointer(
                self.capabilities,
                c"DLSSDOptimalSettingsCallback".as_ptr(),
                &mut callback,
            )
        })
        .ok()?;
        if callback.is_null() {
            // What the SDK's own helper returns here, and it means the feature library is missing
            // or too old rather than anything about this call.
            return Err(Status::OUT_OF_DATE);
        }

        set_u32(self.capabilities, c"Width", output.0);
        set_u32(self.capabilities, c"Height", output.1);
        set_i32(self.capabilities, c"PerfQualityValue", preset.value());
        // Older feature libraries still read this, and the SDK's helper always clears it.
        set_i32(self.capabilities, c"RTXValue", 0);

        // SAFETY: the pointer came from the map under the name the SDK documents for exactly this
        // signature, and it is called with that map.
        let query =
            unsafe { std::mem::transmute::<*mut c_void, ffi::OptimalSettingsCallback>(callback) };
        // SAFETY: as above; the callback reads and writes only the map it is given.
        Status(unsafe { query(self.capabilities) }).ok()?;

        // The query writes its answers back into the map, so an absent one here means it did not
        // run — which the callback would already have reported, hence the expect rather than a
        // second error path.
        let render = (
            self.get_u32(c"OutWidth").expect("the query wrote a width"),
            self.get_u32(c"OutHeight")
                .expect("the query wrote a height"),
        );
        // A feature library that does not vary resolution leaves these unset, and the SDK's own
        // helper falls back to the optimal size — so absent means "no range", not "zero".
        Ok(OptimalSettings {
            render,
            lowest: (
                self.get_u32(c"DLSS.Get.Dynamic.Min.Render.Width")
                    .unwrap_or(render.0),
                self.get_u32(c"DLSS.Get.Dynamic.Min.Render.Height")
                    .unwrap_or(render.1),
            ),
            highest: (
                self.get_u32(c"DLSS.Get.Dynamic.Max.Render.Width")
                    .unwrap_or(render.0),
                self.get_u32(c"DLSS.Get.Dynamic.Max.Render.Height")
                    .unwrap_or(render.1),
            ),
        })
    }

    /// An unsigned the map holds, or `None` where it holds none by that name.
    ///
    /// `None` rather than zero, matching [`Self::capability`]: a render resolution of zero and a
    /// resolution the map never carried are different answers, and only one of them is a failure.
    fn get_u32(&self, name: &CStr) -> Option<u32> {
        let mut value: c_uint = 0;
        // SAFETY: as above.
        let status = Status(unsafe {
            ffi::NVSDK_NGX_Parameter_GetUI(self.capabilities, name.as_ptr(), &mut value)
        });
        status.is_ok().then_some(value)
    }
}

/// A built DLSS Ray Reconstruction feature, and the parameter map it was built from.
///
/// **Both belong to it.** The map a feature is created with is not the capability map — that one is
/// NGX's and is for asking questions — and it has to outlive the feature, so the two are released
/// together and in that order.
#[derive(Debug)]
struct Feature {
    handle: *mut c_void,
    parameters: *mut c_void,
}

impl Ngx {
    /// Builds Ray Reconstruction to take `render` and produce `output`.
    ///
    /// Records into `cmd`, which must be recording and must be submitted before the feature is
    /// used: NGX uploads its weights here.
    fn build(
        &self,
        cmd: ash::vk::CommandBuffer,
        render: (u32, u32),
        output: (u32, u32),
        preset: Preset,
    ) -> Result<Feature, Status> {
        let mut parameters: *mut c_void = std::ptr::null_mut();
        // SAFETY: NGX is initialised and writes a map it allocates through this pointer.
        Status(unsafe { ffi::NVSDK_NGX_VULKAN_AllocateParameters(&mut parameters) }).ok()?;
        // **Owned from here on**, so a failure below releases the map rather than leaking it — and
        // owned by exactly one value, so success does not release it out from under the caller.
        let mut feature = Feature {
            handle: std::ptr::null_mut(),
            parameters,
        };

        // What `NGX_VULKAN_CREATE_DLSSD_EXT1` sets, which is a header-only helper rather than a
        // symbol — `nvsdk_ngx_helpers_dlssd_vk.h:100`.
        // One GPU, so both masks are the first node.
        set_u32(parameters, c"CreationNodeMask", 1);
        set_u32(parameters, c"VisibilityNodeMask", 1);
        set_u32(parameters, c"Width", render.0);
        set_u32(parameters, c"Height", render.1);
        set_u32(parameters, c"OutWidth", output.0);
        set_u32(parameters, c"OutHeight", output.1);
        set_i32(parameters, c"PerfQualityValue", preset.value());
        set_i32(parameters, c"DLSS.Feature.Create.Flags", CREATE_FLAGS);
        set_i32(parameters, c"DLSS.Enable.Output.Subrects", 0);
        // The only denoise mode Ray Reconstruction has, and the helper hardcodes it too.
        set_i32(parameters, c"DLSS.Denoise.Mode", 1);
        // Roughness comes from the normal target's fourth channel — see the G-buffer's layout.
        set_u32(parameters, c"DLSS.Roughness.Mode", 1);
        // **The enum is about the depth's shape, not where it came from**: `Linear` is 0 and `HW`
        // is 1, and what §8.20 writes is clip depth — projected and reverse-Z — whoever computed it.
        // Saying `Linear` for it is what `FAIL_InvalidParameter` meant.
        set_u32(parameters, c"DLSS.Use.HW.Depth", 1);

        // SAFETY: the command buffer is recording, the map is ours and fully populated, and 13 is
        // `NVSDK_NGX_Feature_RayReconstruction`.
        Status(unsafe {
            ffi::NVSDK_NGX_VULKAN_CreateFeature1(
                self.device,
                cmd,
                FEATURE_RAY_RECONSTRUCTION,
                parameters,
                &mut feature.handle,
            )
        })
        .ok()?;
        Ok(feature)
    }
}

/// Everything one evaluation reads and the one image it writes.
///
/// Borrowed rather than owned: the images belong to the G-buffer and the frame decides what to hand
/// over, which is the shape the pass will take when it is wired into the renderer.
#[derive(Debug, Clone, Copy)]
struct Inputs<'a> {
    /// The trace's radiance, **undenoised** — Ray Reconstruction is the denoiser.
    colour: &'a Image,
    diffuse_albedo: &'a Image,
    specular_albedo: &'a Image,
    /// Normal in `xyz`, roughness in `w`.
    normal_roughness: &'a Image,
    /// Clip depth in `r`.
    depth: &'a Image,
    motion: &'a Image,
    output: &'a Image,
    /// Where inside its pixel this frame sampled, in render pixels.
    jitter: glam::Vec2,
    /// Whether the previous frame is usable. True after a jump no motion vector can describe.
    reset: bool,
}

impl Feature {
    /// Records one evaluation into `cmd`.
    ///
    /// Every image must be in `GENERAL`, which is where the renderer's own barrier leaves the
    /// G-buffer, and must have been written by the frame this is upscaling.
    fn evaluate(&self, cmd: ash::vk::CommandBuffer, inputs: Inputs<'_>) -> Result<(), Status> {
        // Held by value for the whole call: the map keeps the pointers rather than the contents, so
        // these have to outlive `EvaluateFeature`.
        let mut colour = resource(inputs.colour);
        let mut diffuse = resource(inputs.diffuse_albedo);
        let mut specular = resource(inputs.specular_albedo);
        let mut normals = resource(inputs.normal_roughness);
        let mut depth = resource(inputs.depth);
        let mut motion = resource(inputs.motion);
        let mut output = resource(inputs.output);

        let map = self.parameters;
        set_resource(map, c"Color", &mut colour);
        set_resource(map, c"Output", &mut output);
        set_resource(map, c"Depth", &mut depth);
        set_resource(map, c"MotionVectors", &mut motion);
        // Ray Reconstruction's own names, which are not the generic `GBuffer.Albedo` beside them.
        set_resource(map, c"DLSS.Input.DiffuseAlbedo", &mut diffuse);
        set_resource(map, c"DLSS.Input.SpecularAlbedo", &mut specular);
        set_resource(map, c"GBuffer.Normals", &mut normals);

        set_f32(map, c"Jitter.Offset.X", inputs.jitter.x);
        set_f32(map, c"Jitter.Offset.Y", inputs.jitter.y);
        // Ours are already in pixels — see §8.13 — so nothing needs scaling. The SDK's helper reads
        // zero here as "one", which would work by accident; saying it is clearer.
        set_f32(map, c"MV.Scale.X", 1.0);
        set_f32(map, c"MV.Scale.Y", 1.0);
        set_i32(map, c"Reset", i32::from(inputs.reset));

        let render = inputs.colour.extent();
        set_u32(map, c"DLSS.Render.Subrect.Dimensions.Width", render.width);
        set_u32(map, c"DLSS.Render.Subrect.Dimensions.Height", render.height);

        // SAFETY: the command buffer is recording, the handle and map are this feature's, and every
        // resource named in the map is alive until this returns. No progress callback.
        Status(unsafe {
            ffi::NVSDK_NGX_VULKAN_EvaluateFeature_C(
                cmd,
                self.handle,
                self.parameters,
                std::ptr::null(),
            )
        })
        .ok()
    }
}

/// An image as NGX takes one.
///
/// Read-write because every image here was created with `STORAGE` usage, which is what that flag
/// means — not that this evaluation writes to it.
fn resource(image: &Image) -> ffi::Resource {
    let extent = image.extent();
    ffi::Resource {
        view: ffi::ImageViewInfo {
            view: image.view(),
            image: image.raw(),
            subresource: ash::vk::ImageSubresourceRange {
                aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            format: image.format(),
            width: extent.width,
            height: extent.height,
        },
        kind: 0,
        read_write: true,
    }
}

impl Drop for Feature {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: built by `Ngx::build` and released once.
            unsafe { ffi::NVSDK_NGX_VULKAN_ReleaseFeature(self.handle) };
        }
        // SAFETY: allocated by `Ngx::build`, and after the feature that was built from it.
        unsafe { ffi::NVSDK_NGX_VULKAN_DestroyParameters(self.parameters) };
    }
}

impl Drop for Ngx {
    fn drop(&mut self) {
        // SAFETY: this type exists only after a successful init on this device, and nothing else
        // shuts NGX down.
        unsafe { ffi::NVSDK_NGX_VULKAN_Shutdown1(self.device) };
    }
}

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
        // And what Vulkan 1.2 absorbed is gone, or `vkCreateDevice` rejects the whole list: the core
        // feature and the extension it replaced cannot both be enabled.
        for gone in SUPERSEDED_BY_CORE {
            assert!(
                !required.device.contains(gone),
                "{gone:?} survived, and enabling it beside the core feature is invalid"
            );
        }
    }

    #[test]
    fn ngx_comes_up_and_agrees_with_the_frame_budget() {
        use rtxmw_gpu::{Presentation, Validation};

        // **The whole point of the extension query, closed.** NGX names what it needs, the device is
        // created with it, and NGX is asked whether Ray Reconstruction is actually available — which
        // depends on the driver, the SDK and the GPU together, so nothing but NGX can answer it —
        // and then what it wants to be handed.
        //
        // One test rather than three because NGX is global to the process and keyed by device:
        // initialising it twice concurrently is not something the SDK promises to survive, and the
        // three questions share the one expensive setup.
        //
        // Its own instance and device rather than the shared test one: those extensions have to be
        // present at *creation*, and the shared device is built without them for every other test in
        // the workspace.
        // 1. What NGX needs before any Vulkan object exists.
        let required = Requirements::query().expect("the SDK should answer");
        let instance =
            Instance::new(c"rtxmw-ngx", &[], Validation::Record).expect("an instance should build");
        let Ok(physical) = PhysicalDevice::select(&instance, Presentation::NotNeeded) else {
            eprintln!("skipping: no device this renderer can use");
            return;
        };

        // Absent extensions are not an error at device creation, so they are checked here — a device
        // silently missing them would fail later inside NGX with something far less specific.
        for name in &required.device {
            assert!(
                physical.supports(name),
                "{} does not offer {name:?}, which NGX says it needs",
                physical.name()
            );
        }
        let device = Device::new(&instance, &physical, &required.device)
            .expect("a device should build with what NGX asked for");

        // Somewhere NGX may write its logs and any feature library it downloads.
        let data = std::env::temp_dir().join("rtxmw-ngx");
        std::fs::create_dir_all(&data).expect("a scratch directory should be creatable");
        // Where `build.rs` found them, so `DLSS_SDK_DIR` is honoured here too — deriving the path
        // again would work on this machine and quietly look in the wrong place on one that overrides
        // it. Shipping a copy beside the binary is what a release would do instead.
        let libraries = std::path::Path::new(env!("NGX_FEATURE_DIR"));
        // 2. NGX itself, on a device built from that answer.
        let ngx = Ngx::new(&instance, &physical, &device, &data, libraries)
            .unwrap_or_else(|e| panic!("NGX should initialise: {e}"));
        let available = ngx.ray_reconstruction();
        println!(
            "{}: DLSS Ray Reconstruction {}",
            physical.name(),
            if available {
                "available"
            } else {
                "unavailable"
            }
        );
        // The design targets Ada-class NVIDIA hardware (§ hardware requirements), and this driver
        // ships the NGX core. If this is ever false on such a machine, DLSS-RR is off the table and
        // §5.2's whole denoising decision has to be reopened — so it is asserted, not reported.
        assert!(
            available,
            "NGX says this device cannot do Ray Reconstruction, which M7 is built on"
        );

        // **§5.3's number, asked of DLSS rather than assumed.** The design settles on 1920x1080
        // internal to 3840x2160 output, and Performance is the mode that ratio comes from — so if
        // DLSS asks for something else, the frame budget the whole project is written against is
        // measured at the wrong resolution.
        // 3. What to render at, which the frame budget is measured against.
        const OUTPUT: (u32, u32) = (3840, 2160);
        let settings = ngx
            .optimal_settings(OUTPUT, Preset::Performance)
            .unwrap_or_else(|e| panic!("DLSS should say what it wants: {e}"));
        println!("for {OUTPUT:?} at Performance, DLSS wants {settings:?}");
        assert_eq!(
            settings.render,
            (1920, 1080),
            "DLSS asks to render at {:?} for a {OUTPUT:?} output, not the 1920x1080 §5.3 budgets \
             for",
            settings.render
        );

        // The other modes have to differ from it, and in the direction their names claim — a query
        // that ignored the quality value would return the same size for all three and still look
        // plausible on its own.
        let balanced = ngx
            .optimal_settings(OUTPUT, Preset::Balanced)
            .expect("balanced");
        let quality = ngx
            .optimal_settings(OUTPUT, Preset::Quality)
            .expect("quality");
        println!(
            "balanced {:?}, quality {:?}",
            balanced.render, quality.render
        );
        assert!(
            settings.render.0 < balanced.render.0 && balanced.render.0 < quality.render.0,
            "the three modes render at {:?}, {:?} and {:?}, which is not an ordering",
            settings.render,
            balanced.render,
            quality.render
        );

        // Every mode has to fit inside the range it reports, or a dynamic-resolution renderer would
        // be handed a size DLSS then refuses.
        for mode in [settings, balanced, quality] {
            assert!(
                mode.lowest.0 <= mode.render.0 && mode.render.0 <= mode.highest.0,
                "{:?} is outside its own reported range",
                mode
            );
        }

        // **And the feature builds.** NGX uploads its weights into the command buffer, so this is
        // the first call that does real work on the device rather than answering from a table — and
        // the first place a wrong parameter map shows up as something other than a query result.
        let memory = rtxmw_gpu::Memory::new(&instance, &physical, &device)
            .expect("device memory should open");
        let mut uploader =
            rtxmw_gpu::Uploader::new(&device, &memory, physical.graphics_queue_family())
                .expect("an uploader should build");
        // 4. The feature, whose creation uploads weights into a command buffer.
        let mut built = None;
        uploader
            .submit_and_wait(|_, cmd| {
                built = Some(ngx.build(cmd, settings.render, OUTPUT, Preset::Performance));
            })
            .expect("the build should submit");
        let feature = built
            .expect("the recording ran")
            .unwrap_or_else(|e| panic!("Ray Reconstruction should build: {e}"));

        assert!(
            !feature.handle.is_null(),
            "the build reported success and produced no feature"
        );

        // **And it runs.** Every input is a real image of the right size and format, in `GENERAL`,
        // and the output is the 4K one DLSS was built to fill. What this proves is the resource
        // wrapper and the parameter names — the picture is the *next* thing, once the renderer's own
        // frame feeds this instead of a set of empty targets.
        let image = |name: &str, size: (u32, u32), format: ash::vk::Format| {
            Image::new(
                &memory,
                name,
                ash::vk::Extent2D {
                    width: size.0,
                    height: size.1,
                },
                format,
                ash::vk::ImageUsageFlags::STORAGE,
            )
            .expect("an image should allocate")
        };
        // 5. One evaluation over real images.
        let half = ash::vk::Format::R16G16B16A16_SFLOAT;
        let colour = image("colour", settings.render, half);
        let diffuse = image("diffuse", settings.render, half);
        let specular = image("specular", settings.render, half);
        let normals = image("normals", settings.render, half);
        let depth = image("depth", settings.render, ash::vk::Format::R32G32_SFLOAT);
        let motion = image("motion", settings.render, ash::vk::Format::R32G32_SFLOAT);
        let output = image("output", OUTPUT, half);

        let mut ran = None;
        uploader
            .submit_and_wait(|device, cmd| {
                // DLSS reads and writes them as storage images, which is the layout the renderer's
                // own frame leaves the G-buffer in.
                for target in [
                    &colour, &diffuse, &specular, &normals, &depth, &motion, &output,
                ] {
                    // SAFETY: the command buffer is recording and every image is alive.
                    unsafe {
                        rtxmw_gpu::image_barrier::transition(
                            device,
                            cmd,
                            target.raw(),
                            ash::vk::ImageLayout::UNDEFINED,
                            ash::vk::ImageLayout::GENERAL,
                        );
                    }
                }
                ran = Some(feature.evaluate(
                    cmd,
                    Inputs {
                        colour: &colour,
                        diffuse_albedo: &diffuse,
                        specular_albedo: &specular,
                        normal_roughness: &normals,
                        depth: &depth,
                        motion: &motion,
                        output: &output,
                        jitter: glam::Vec2::ZERO,
                        // The first frame has no history, which is exactly what a reset means.
                        reset: true,
                    },
                ));
            })
            .expect("the evaluation should submit");
        ran.expect("the recording ran")
            .unwrap_or_else(|e| panic!("Ray Reconstruction should evaluate: {e}"));
        println!(
            "evaluated {:?} to {:?}",
            settings.render,
            (output.extent().width, output.extent().height)
        );

        // **DLSS records its own commands into that buffer**, and a status of success says only
        // that NGX was happy with the map — not that what it recorded was valid. The layer is what
        // has an opinion about the resources it then touched.
        if let Some(log) = instance.validation_log() {
            let errors = log.errors_on_this_thread();
            assert!(
                errors.is_empty(),
                "{} validation error(s) from the evaluation:\n{}",
                errors.len(),
                errors
                    .iter()
                    .map(|e| format!("  {}", e.text))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
        // Released here rather than at the end of the scope, so a failure to release is this test's
        // and not the next one's — NGX keeps its state per device and a leaked feature outlives the
        // handle that owned it.
        drop(feature);
    }

    #[test]
    fn a_failure_carries_the_name_the_sdk_gives_it() {
        // Errors from here reach a log and nothing else, so the one thing that matters is that they
        // say what happened rather than a bare number.
        assert!(Status(1).is_ok());
        let denied = Status::OUT_OF_DATE;
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
