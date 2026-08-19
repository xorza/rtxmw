//! Everything a cell needs, assembled from the installed game.

use rtxmw_esm::{CellId, CellIndex, EsmReader};
use rtxmw_texture::Texture;
use rtxmw_vfs::Vfs;

use crate::door::Door;
use crate::error::Result;
use crate::game_files::GameFiles;
use crate::static_scene::{CellDetail, ModelIndex, StaticScene};

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
    /// Opens the content file, indexes it, and reads one cell out — so this is the *expensive* way
    /// in, and the right one exactly once: at startup, for the cell the camera opens in. Anything
    /// loading a second cell wants [`CellStreamer`], which keeps the file open and the index built
    /// and pays neither again.
    ///
    /// [`CellStreamer`]: crate::cell_streamer::CellStreamer
    pub fn load_at(id: CellId) -> Result<Option<Self>> {
        let Some(files) = GameFiles::open()? else {
            return Ok(None);
        };
        let vfs = &files.vfs;
        let esm = EsmReader::new(&files.esm)?;
        let index = CellIndex::build(&esm)?;
        let models = ModelIndex::build(&esm)?;

        let scene = StaticScene::load_cell(&esm, &index, &models, vfs, &id, CellDetail::Full)?;
        let entrances = Door::leading_to(&esm, &models, &id)?;
        Ok(Some(Self::assemble(id, scene, vfs, entrances)))
    }

    /// Loads the named interior from the installed game, or `None` when none is configured.
    pub fn load_interior(cell_name: &str) -> Result<Option<Self>> {
        Self::load_at(CellId::Interior(cell_name.to_owned()))
    }

    /// Assembles one cell from files a caller already has open, without searching for its doors.
    ///
    /// What streaming loads. The door search is a pass over the whole file costing as much again
    /// as building the cell, and it answers "where would someone arriving here appear" — which a
    /// cell the camera is walking into has already answered for itself.
    pub fn streamed(
        esm: &EsmReader<'_>,
        index: &CellIndex,
        models: &ModelIndex,
        vfs: &Vfs,
        id: CellId,
        detail: CellDetail,
    ) -> Result<Self> {
        let scene = StaticScene::load_cell(esm, index, models, vfs, &id, detail)?;
        Ok(Self::assemble(id, scene, vfs, Vec::new()))
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
