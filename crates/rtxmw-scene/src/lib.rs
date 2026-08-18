//! Turning content files and models into placed geometry.
//!
//! This is the seam between the format crates and the renderer: nothing below it knows about
//! Vulkan, and nothing above it knows about ESM records or NIF blocks.

mod error;
mod mesh;
mod static_scene;

pub use crate::error::{Result, SceneError};
pub use crate::mesh::{Bounds, Mesh};
pub use crate::static_scene::{Instance, MeshId, ModelIndex, StaticScene};
