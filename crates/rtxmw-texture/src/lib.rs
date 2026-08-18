//! Decoding the texture formats Morrowind ships.
//!
//! The compressed formats are not decompressed: DXT1 and DXT3 are BC1 and BC2, which every target
//! GPU samples natively, so decoding means parsing a header and handing the blocks on untouched.

mod dds;
mod error;
mod texture;
mod tga;

pub use crate::error::{Result, TextureError};
pub use crate::texture::{MipLevel, Texture, TextureFormat};
