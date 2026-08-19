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

use rtxmw_gpu::{Device, Instance, PhysicalDevice};

mod ffi;

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
            minimum_logging_level: 0,
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

        self.set_u32(c"Width", output.0);
        self.set_u32(c"Height", output.1);
        self.set_i32(c"PerfQualityValue", preset.value());
        // Older feature libraries still read this, and the SDK's helper always clears it.
        self.set_i32(c"RTXValue", 0);

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

    fn set_u32(&self, name: &CStr, value: u32) {
        // SAFETY: the map outlives this and the name is a nul-terminated static.
        unsafe { ffi::NVSDK_NGX_Parameter_SetUI(self.capabilities, name.as_ptr(), value) };
    }

    fn set_i32(&self, name: &CStr, value: c_int) {
        // SAFETY: as above.
        unsafe { ffi::NVSDK_NGX_Parameter_SetI(self.capabilities, name.as_ptr(), value) };
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
