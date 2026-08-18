//! Vulkan bring-up: instance, device, surface, swapchain and the frame ring.
//!
//! Nothing above this crate should touch `ash` directly.

/// Image layout transitions, qualified as `image_barrier::transition(..)` at call sites.
pub mod image_barrier;

mod device;
mod error;
mod frames;
mod instance;
mod physical_device;
mod surface;
mod swapchain;
mod validation_log;

pub use crate::device::Device;
pub use crate::error::{GpuError, Result};
pub use crate::frames::{Frame, Frames};
pub use crate::instance::{Instance, Validation};
pub use crate::physical_device::{
    PhysicalDevice, Presentation, RayTracingLimits, RayTracingSupport,
};
pub use crate::surface::Surface;
pub use crate::swapchain::Swapchain;
pub use crate::validation_log::{ValidationLog, ValidationMessage};

#[cfg(any(test, feature = "internals"))]
mod testing;

// The gate must match the module's: a module compiled under cfg(test) but not re-exported
// there would be `pub` yet unreachable, which `unreachable_pub` rightly rejects.
#[cfg(any(test, feature = "internals"))]
pub use crate::testing::{golden, render_target::RenderTarget, test_gpu::TestGpu};
