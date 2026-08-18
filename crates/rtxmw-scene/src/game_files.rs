//! The installed game's content files, opened.

use rtxmw_vfs::{Vfs, morrowind_data_dir};

use crate::error::{Result, SceneError};

/// `Morrowind.esm` in memory, and the archives its meshes and textures come out of.
///
/// The bytes rather than a reader over them: `EsmReader` borrows what it reads, and the cell index
/// and model table borrow the reader in turn, so whatever owns the bytes has to outlive all three.
/// Handing back the buffer lets each caller build that stack as locals — which is what makes the
/// streamer a loop in a thread rather than a struct that would have to borrow from itself.
#[derive(Debug)]
pub(crate) struct GameFiles {
    pub(crate) esm: Vec<u8>,
    pub(crate) vfs: Vfs,
}

impl GameFiles {
    /// Opens the installed game, or `None` when none is configured.
    ///
    /// No game data is a normal state — the engine still comes up and says so — while a configured
    /// directory whose archives will not open is an error, because it means the path is wrong and
    /// silently drawing nothing would hide that.
    pub(crate) fn open() -> Result<Option<Self>> {
        let Some(data) = morrowind_data_dir() else {
            return Ok(None);
        };
        let vfs = rtxmw_vfs::morrowind_archives().ok_or_else(|| {
            SceneError::Io(std::io::Error::other(
                "the game directory is configured but its archives could not be opened",
            ))
        })?;
        let esm = std::fs::read(data.join("Morrowind.esm")).map_err(SceneError::Io)?;
        Ok(Some(Self { esm, vfs }))
    }
}
