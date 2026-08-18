//! Reader for Morrowind-era NIF models.

mod block;
mod cursor;
mod error;
mod nif_file;

pub use crate::block::{
    AlphaProperty, AvObject, Block, Geometry, GeometryData, MaterialProperty, Node, ObjectNet,
    SourceTexture, TextureSlot, TexturingProperty, Transform,
};
pub use crate::cursor::{Cursor, Link};
pub use crate::error::{NifError, Result};
pub use crate::nif_file::{NifFile, VER_MORROWIND, version};
