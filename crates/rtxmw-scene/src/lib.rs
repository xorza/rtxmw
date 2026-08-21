//! Turning content files and models into placed geometry.
//!
//! This is the seam between the format crates and the renderer: nothing below it knows about
//! Vulkan, and nothing above it knows about ESM records or NIF blocks.

mod assembled_actor;
pub mod blackbody;
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
mod particle_emitter;
mod precipitation;
mod rig;
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
// The key types a rig's channels are made of, re-exported so a caller describing one by hand does
// not have to name the NIF crate to do it.
pub use rtxmw_nif::{FloatKey, QuaternionKey, VectorKey};

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
pub use crate::particle_emitter::ParticleEmitter;
pub use crate::precipitation::{Falling, Precipitation};
pub use crate::rig::{
    Bone, Channel, INFLUENCES, Influence, Keyframe, NO_PARENT, Pose, Rig, affine_of,
};
pub use crate::sky::{Sky, Veil};
pub use crate::sky_textures::SkyTextures;
pub use crate::srgb::{LUMA, to_linear};
pub use crate::static_scene::{
    CellDetail, DeformingInstance, Instance, MeshId, ModelIndex, RigId, StaticScene,
};
pub use crate::sun::Sun;
pub use crate::weather::{Schedule, Weather};
pub use crate::world_time::WorldTime;
