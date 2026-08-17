//! Errors raised while bringing up Vulkan.

use ash::vk;

/// Anything that can go wrong between loading the loader and owning a logical device.
#[derive(Debug)]
pub enum GpuError {
    /// The Vulkan loader itself could not be found or loaded.
    LoaderMissing(ash::LoadingError),
    /// A Vulkan call failed.
    Vulkan(vk::Result),
    /// No physical device met the renderer's requirements.
    NoSuitableDevice(Vec<Rejection>),
}

/// Why one physical device was passed over, kept so the failure message can name every candidate.
#[derive(Debug)]
pub struct Rejection {
    pub device_name: String,
    pub reason: RejectionReason,
}

/// The specific requirement a physical device failed to meet.
#[derive(Debug)]
pub enum RejectionReason {
    /// Device extensions the design depends on that this device does not expose.
    MissingExtensions(Vec<&'static str>),
    /// A required feature bit is not set even though the extension is present.
    MissingFeature(&'static str),
    /// No queue family supports graphics and compute.
    NoGraphicsQueue,
    /// Querying the device failed, so it could not be assessed either way.
    QueryFailed(vk::Result),
}

impl From<vk::Result> for GpuError {
    fn from(value: vk::Result) -> Self {
        Self::Vulkan(value)
    }
}

impl From<ash::LoadingError> for GpuError {
    fn from(value: ash::LoadingError) -> Self {
        Self::LoaderMissing(value)
    }
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoaderMissing(e) => write!(f, "could not load the Vulkan loader: {e}"),
            Self::Vulkan(e) => write!(f, "vulkan call failed: {e}"),
            Self::NoSuitableDevice(rejections) => {
                writeln!(f, "no physical device meets the ray tracing requirements")?;
                for r in rejections {
                    write!(f, "\n  {}: ", r.device_name)?;
                    match &r.reason {
                        RejectionReason::MissingExtensions(names) => {
                            write!(f, "missing extensions: {}", names.join(", "))?;
                        }
                        RejectionReason::MissingFeature(name) => {
                            write!(f, "feature not supported: {name}")?;
                        }
                        RejectionReason::NoGraphicsQueue => {
                            write!(f, "no graphics + compute queue family")?;
                        }
                        RejectionReason::QueryFailed(e) => {
                            write!(f, "could not be queried: {e}")?;
                        }
                    }
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for GpuError {}

/// Result alias for GPU bring-up.
pub type Result<T> = std::result::Result<T, GpuError>;
