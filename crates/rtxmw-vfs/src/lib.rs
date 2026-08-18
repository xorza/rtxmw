//! Morrowind's virtual file system: BSA archives and loose directories behind one
//! case-insensitive index.

mod bsa_archive;
mod directory_archive;
mod error;
#[cfg(any(test, feature = "internals"))]
mod game_data;
mod normalized_path;
mod vfs;

pub use crate::bsa_archive::BsaArchive;
pub use crate::directory_archive::DirectoryArchive;
pub use crate::error::{Result, VfsError};
pub use crate::normalized_path::NormalizedPath;
pub use crate::vfs::Vfs;

// The gate must match the module's, or the module is `pub` yet unreachable under cfg(test).
#[cfg(any(test, feature = "internals"))]
pub use crate::bsa_archive::internals as bsa_internals;
#[cfg(any(test, feature = "internals"))]
pub use crate::game_data::{DATA_DIR_VAR, morrowind_data_dir};
