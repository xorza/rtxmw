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
        let Some(vfs) = Self::archives()? else {
            return Ok(None);
        };
        let esm = std::fs::read(data.join("Morrowind.esm")).map_err(SceneError::Io)?;
        Ok(Some(Self { esm, vfs }))
    }

    /// Opens the archives alone, without the master file.
    ///
    /// **For a caller that wants a texture and nothing else.** `Morrowind.esm` is eighty megabytes
    /// and [`Self::open`] reads all of it; the moons' two portraits come out of a BSA, so going
    /// through the full open to reach them cost 23 ms and an eighty-megabyte allocation that was
    /// dropped unread — twice over, because the cell load then reads the master file for itself.
    ///
    /// Same two states as [`Self::open`]: no game data is `None` and a configured directory whose
    /// archives will not open is an error.
    pub(crate) fn archives() -> Result<Option<Vfs>> {
        if morrowind_data_dir().is_none() {
            return Ok(None);
        }
        rtxmw_vfs::morrowind_archives()
            .ok_or_else(|| {
                SceneError::Io(std::io::Error::other(
                    "the game directory is configured but its archives could not be opened",
                ))
            })
            .map(Some)
    }
}
