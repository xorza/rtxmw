//! Turning content files and models into placed geometry.
//!
//! This is the seam between the format crates and the renderer: nothing below it knows about
//! Vulkan, and nothing above it knows about ESM records or NIF blocks.

mod cell_streamer;
mod clouds;
mod door;
mod error;
mod game_data;
mod ini;
mod light;
mod lightning;
mod loaded_cell;
mod material;
mod material_table;
mod mesh;
mod moon;
mod precipitation;
mod sky;
mod sky_textures;
mod srgb;
mod static_scene;
mod sun;
mod weather;
mod world_time;

// Re-exported because `Door` carries one and `Door::leading_to` takes one, so a crate above this
// seam cannot use either without naming the type — and the layering says nothing above here knows
// about `rtxmw-esm`. This is the published surface saying so.
pub use rtxmw_esm::CellId;

pub use crate::cell_streamer::{CellStreamer, StreamedCell};
pub use crate::clouds::{CloudSheet, Clouds};
pub use crate::door::Door;
pub use crate::error::{Result, SceneError};
pub use crate::light::{Ambient, Light};
pub use crate::lightning::{Discharge, Flash, Lightning};
pub use crate::loaded_cell::LoadedCell;
pub use crate::material::{AlphaMode, Material, MaterialKind, TerrainLayers, TextureId};
pub use crate::material_table::MaterialTable;
pub use crate::mesh::{Bounds, Mesh, Submesh};
pub use crate::moon::Moon;
pub use crate::precipitation::Precipitation;
pub use crate::sky::{Sky, Veil};
pub use crate::sky_textures::SkyTextures;
pub use crate::srgb::{LUMA, to_linear};
pub use crate::static_scene::{CellDetail, Instance, MeshId, ModelIndex, StaticScene};
pub use crate::sun::Sun;
pub use crate::weather::{Schedule, Weather};
pub use crate::world_time::WorldTime;
