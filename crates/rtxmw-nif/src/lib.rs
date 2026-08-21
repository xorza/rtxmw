//! Reader for Morrowind-era NIF models.

mod block;
mod cursor;
mod error;
mod keyframe_data;
mod nif_file;
mod skin_data;
mod skin_instance;
mod text_keys;
mod time_controller;

pub use crate::block::{
    AlphaProperty, AvObject, Block, Geometry, GeometryData, MaterialProperty, Node, ObjectNet,
    SourceTexture, TextureSlot, TexturingProperty, Transform,
};
pub use crate::cursor::{Cursor, Link};
pub use crate::error::{NifError, Result};
pub use crate::keyframe_data::{
    FloatKey, Interpolation, Keyed, KeyframeData, QuaternionKey, Track, VectorKey,
};
pub use crate::nif_file::{NifFile, VER_MORROWIND, version};
pub use crate::skin_data::{BoneSkin, SkinData, VertexWeight};
pub use crate::skin_instance::SkinInstance;
pub use crate::text_keys::{TextKey, TextKeys};
pub use crate::time_controller::{ControllerKind, TimeController};
