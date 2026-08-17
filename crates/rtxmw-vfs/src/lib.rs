//! Morrowind's virtual file system: BSA archives and loose directories behind one
//! case-insensitive index.

mod bsa_archive;
mod directory_archive;
mod error;
mod normalized_path;
mod vfs;

pub use crate::bsa_archive::BsaArchive;
pub use crate::directory_archive::DirectoryArchive;
pub use crate::error::{Result, VfsError};
pub use crate::normalized_path::NormalizedPath;
pub use crate::vfs::Vfs;

#[cfg(feature = "internals")]
pub use crate::bsa_archive::internals as bsa_internals;
