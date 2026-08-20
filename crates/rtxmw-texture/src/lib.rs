//! Decoding the texture formats Morrowind ships.
//!
//! The compressed formats are not decompressed on the way to the GPU: DXT1 and DXT3 are BC1 and BC2,
//! which every target samples natively, so decoding means parsing a header and handing the blocks on
//! untouched. [`Texture::to_rgba8`] exists beside that for something that has to *look* at a
//! texture rather than sample it, and is not on any path a frame takes.

mod colour;
mod dds;
mod error;
mod rgba8;
mod shading_map;
mod texture;
mod tga;

pub use crate::colour::{LUMA, channel_to_linear};
pub use crate::error::{Result, TextureError};
pub use crate::texture::{MipLevel, Texture, TextureFormat};

// The gate must match the module's, or the module is `pub` yet unreachable under cfg(test).
#[cfg(any(test, feature = "internals"))]
pub use crate::texture::internals as texture_internals;
