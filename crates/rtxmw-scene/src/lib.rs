//! Turning content files and models into placed geometry.
//!
//! This is the seam between the format crates and the renderer: nothing below it knows about
//! Vulkan, and nothing above it knows about ESM records or NIF blocks.

mod error;
mod material;
mod material_table;
mod mesh;
mod static_scene;

pub use crate::error::{Result, SceneError};
pub use crate::material::{AlphaMode, Material, TextureId};
pub use crate::material_table::MaterialTable;
pub use crate::mesh::{Bounds, Mesh, Submesh};
pub use crate::static_scene::{Instance, MeshId, ModelIndex, StaticScene};
