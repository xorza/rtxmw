//! What building an upscaler can fail with.

use crate::dlss::Status;

/// A failure on the way to a working upscaler.
///
/// **Two sources, kept apart.** NGX reports its own codes and this crate reports Vulkan's, and
/// flattening the second onto the first is how an out-of-memory comes out reading
/// `NVSDK_NGX_Result_FAIL_OutOfDate` — a diagnostic that sends the reader to the driver version when
/// the machine was simply out of memory.
#[derive(Debug)]
pub enum UpscalerError {
    /// NGX refused. Carries the SDK's own code, which names itself.
    Ngx(Status),
    /// A Vulkan allocation or submission failed before NGX was reached.
    Gpu(rtxmw_gpu::GpuError),
}

impl std::fmt::Display for UpscalerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ngx(status) => write!(f, "{status}"),
            Self::Gpu(failed) => write!(f, "{failed}"),
        }
    }
}

impl std::error::Error for UpscalerError {}

impl From<Status> for UpscalerError {
    fn from(status: Status) -> Self {
        Self::Ngx(status)
    }
}

impl From<rtxmw_gpu::GpuError> for UpscalerError {
    fn from(failed: rtxmw_gpu::GpuError) -> Self {
        Self::Gpu(failed)
    }
}
