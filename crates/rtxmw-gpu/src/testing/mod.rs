//! Offscreen test harness: a shared device, render targets, and golden-image comparison.
//!
//! Gated behind the `internals` feature so nothing here reaches a release build.

pub mod golden;
pub(crate) mod render_target;
pub(crate) mod test_gpu;
