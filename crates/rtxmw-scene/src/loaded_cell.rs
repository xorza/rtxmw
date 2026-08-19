//! Everything a cell needs, assembled from the installed game.

use rtxmw_esm::CellId;
use rtxmw_texture::Texture;
use rtxmw_vfs::Vfs;

use crate::door::Door;
use crate::error::Result;
use crate::game_data::GameData;
use crate::static_scene::{CellDetail, StaticScene};

/// A cell's geometry and the textures its materials name.
///
/// The two arrive together because they come from the same content file and the same archives, and
/// separating them only moves the decode loop to whoever asked. Every caller wanting real data
/// wants both.
#[derive(Debug)]
pub struct LoadedCell {
    /// Which cell this is. Carried rather than remembered by the caller, so a report about it
    /// cannot name a different one than was loaded.
    pub id: CellId,
    pub scene: StaticScene,
    /// One entry per path in the scene's texture list, `None` where the file could not be decoded.
    ///
    /// A miss is normal rather than an error: the shipped meshes carry a tail of references to art
    /// that was removed, and a renderer substitutes a fallback for them.
    pub textures: Vec<Option<Texture>>,
    /// Every door elsewhere in the file that leads into this cell.
    ///
    /// Each one names a place the game itself puts the player, which is what makes any of them a
    /// defensible spawn point — see [`Door`]. Empty for a cell nothing leads to, which a cell
    /// reached only by script or by coordinates legitimately is — **and empty for a streamed cell,
    /// where the search is skipped rather than coming back with nothing**. Only the cell opened
    /// through [`LoadedCell::load_at`] is asked the question.
    pub entrances: Vec<Door>,
}

impl LoadedCell {
    /// Loads whichever cell `id` names, or `None` when no game data is configured.
    ///
    /// **The expensive way in, and what makes it expensive is the doors rather than the file.**
    /// `GameData::shared` has the master file open and both indices built by the time this runs,
    /// so what is left here is one cell and a pass over the whole file for everything that opens
    /// into it — which is why this is the right call exactly once, for the cell the camera starts
    /// in. Anything loading a second cell wants [`CellStreamer`], which skips the door search.
    ///
    /// [`CellStreamer`]: crate::cell_streamer::CellStreamer
    pub fn load_at(id: CellId) -> Result<Option<Self>> {
        let Some(game) = GameData::shared()? else {
            return Ok(None);
        };
        let esm = game.reader();
        let scene = StaticScene::load_cell(
            &esm,
            game.cells(),
            game.models(),
            game.vfs(),
            &id,
            CellDetail::Full,
        )?;
        let entrances = Door::leading_to(&esm, game.models(), &id)?;
        Ok(Some(Self::assemble(id, scene, game.vfs(), entrances)))
    }

    /// Loads the named interior from the installed game, or `None` when none is configured.
    pub fn load_interior(cell_name: &str) -> Result<Option<Self>> {
        Self::load_at(CellId::Interior(cell_name.to_owned()))
    }

    /// One cell without searching for its doors.
    ///
    /// What streaming loads. The door search is a pass over the whole file costing as much again
    /// as building the cell, and it answers "where would someone arriving here appear" — which a
    /// cell the camera is walking into has already answered for itself.
    ///
    /// Took the reader, both indices and the archives as four arguments until they all began coming
    /// from one place. Building the reader per cell rather than once for the streamer is what that
    /// costs, and it is a header parse that measures zero.
    pub(crate) fn streamed(game: &GameData, id: CellId, detail: CellDetail) -> Result<Self> {
        let esm = game.reader();
        let scene =
            StaticScene::load_cell(&esm, game.cells(), game.models(), game.vfs(), &id, detail)?;
        Ok(Self::assemble(id, scene, game.vfs(), Vec::new()))
    }

    /// Decodes the textures a scene's materials name and packages the result.
    ///
    /// A texture that fails to read or decode becomes `None` rather than aborting the load: one
    /// dangling reference in a cell of hundreds is not a reason to have no cell.
    fn assemble(id: CellId, scene: StaticScene, vfs: &Vfs, entrances: Vec<Door>) -> Self {
        let textures = scene
            .materials
            .textures()
            .iter()
            .map(|path| {
                vfs.read(path)
                    .ok()
                    .and_then(|bytes| Texture::decode(&bytes).ok())
            })
            .collect();
        Self {
            id,
            scene,
            textures,
            entrances,
        }
    }

    /// How many of the cell's texture references resolved to nothing.
    pub fn missing_textures(&self) -> usize {
        self.textures.iter().filter(|t| t.is_none()).count()
    }
}
