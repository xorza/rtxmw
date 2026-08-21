//! What both Ray Reconstruction tests need before they can measure anything.
//!
//! They are separate binaries on purpose — NGX is global per device and the SDK does not promise to
//! survive being initialised twice at once — which is exactly why the bring-up they share has to
//! live somewhere neither of them owns.
//!
//! `dead_code` because `common` is compiled into all fifteen test binaries and this is used by two
//! — the same reason `under_the_sky` next door carries it.
#![allow(dead_code)]

use ash::vk;
use glam::{Mat4, Vec3};
use rtxmw_gpu::{Device, Instance, Memory, PhysicalDevice, Presentation, Uploader, Validation};
use rtxmw_render::dlss::Requirements;
use rtxmw_scene::StaticScene;

/// The cell both tests look at, and the one every other renderer test looks at too.
pub(crate) const CELL: &str = "Seyda Neen, Census and Excise Office";

/// A device NGX will run on, and the pieces a renderer is built from on top of it.
///
/// **Brought up together because NGX asks for its Vulkan extensions at device-creation time**, so
/// there is no adding them to a device that already exists — which is why these tests cannot use
/// the shared `TestGpu` the rest of the suite runs on.
///
/// Fields rather than accessors: a caller needs `&instance` and `&mut uploader` in the same
/// expression, which methods on one owner cannot give it.
pub(crate) struct NgxGpu {
    pub(crate) instance: Instance,
    pub(crate) physical: PhysicalDevice,
    pub(crate) device: Device,
    pub(crate) memory: Memory,
    pub(crate) uploader: Uploader,
}

impl std::fmt::Debug for NgxGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NgxGpu")
            .field("device", &self.physical.name())
            .finish_non_exhaustive()
    }
}

impl NgxGpu {
    /// Brings one up, or `None` where this machine has no device the renderer can use.
    ///
    /// `named` reaches the validation layer's output, so a message says which test it came from.
    pub(crate) fn new(named: &'static std::ffi::CStr) -> Option<Self> {
        let required = Requirements::query().expect("the SDK should answer");
        let instance =
            Instance::new(named, &[], Validation::Record).expect("an instance should build");
        let Ok(physical) = PhysicalDevice::select(&instance, Presentation::NotNeeded) else {
            eprintln!("skipping: no device this renderer can use");
            return None;
        };
        for name in &required.device {
            assert!(
                physical.supports(name),
                "{} does not offer {name:?}, which NGX says it needs",
                physical.name()
            );
        }
        let device = Device::new(&instance, &physical, &required.device)
            .expect("a device should build with what NGX asked for");
        let memory = Memory::new(&instance, &physical, &device).expect("memory should come up");
        let uploader = Uploader::new(&device, &memory, physical.graphics_queue_family())
            .expect("an uploader should build");
        Some(Self {
            instance,
            physical,
            device,
            memory,
            uploader,
        })
    }

    /// Where NGX keeps its own scratch files, which it is handed rather than choosing.
    pub(crate) fn scratch() -> std::path::PathBuf {
        let data = std::env::temp_dir().join("rtxmw-ngx");
        std::fs::create_dir_all(&data).expect("a scratch directory should be creatable");
        data
    }
}

/// One camera as the three things a frame is built from.
///
/// **Placed by the scene rather than by hand**, so it faces geometry whichever cell loaded, and
/// never moved: both tests are about what a resolve does with a still frame.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Viewing {
    pub(crate) eye: Vec3,
    pub(crate) view: Mat4,
    pub(crate) projection: Mat4,
}

impl Viewing {
    /// Standing in the middle of `scene` and looking north, for a frame `size` across.
    pub(crate) fn of(scene: &StaticScene, size: vk::Extent2D) -> Self {
        let eye = scene.bounds().map_or(Vec3::ZERO, |bounds| bounds.centre());
        Self {
            eye,
            view: glam::camera::rh::view::look_to_mat4(eye, Vec3::X, Vec3::Z),
            projection: glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
                75f32.to_radians(),
                size.width as f32 / size.height as f32,
                0.05,
            ),
        }
    }
}
