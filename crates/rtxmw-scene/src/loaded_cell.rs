//! Everything a cell needs, assembled from the installed game.

use rtxmw_esm::EsmReader;
use rtxmw_texture::Texture;
use rtxmw_vfs::morrowind_data_dir;

use crate::error::{Result, SceneError};
use crate::static_scene::{ModelIndex, StaticScene};

/// A cell's geometry and the textures its materials name.
///
/// The two arrive together because they come from the same content file and the same archives, and
/// separating them only moves the decode loop to whoever asked. Every caller wanting real data
/// wants both.
#[derive(Debug)]
pub struct LoadedCell {
    pub scene: StaticScene,
    /// One entry per path in the scene's texture list, `None` where the file could not be decoded.
    ///
    /// A miss is normal rather than an error: the shipped meshes carry a tail of references to art
    /// that was removed, and a renderer substitutes a fallback for them.
    pub textures: Vec<Option<Texture>>,
}

impl LoadedCell {
    /// Loads the named interior from the installed game, or `None` when none is configured.
    ///
    /// Reads the whole of `Morrowind.esm` each call, so loading several cells this way rereads a
    /// 79 MB file each time. That is fine for opening one cell at startup and wrong for streaming;
    /// when cells start streaming, the reader wants hoisting above this.
    pub fn load_interior(cell_name: &str) -> Result<Option<Self>> {
        let Some(data) = morrowind_data_dir() else {
            return Ok(None);
        };
        let vfs = rtxmw_vfs::morrowind_archives().ok_or_else(|| {
            SceneError::Io(std::io::Error::other(
                "the game directory is configured but its archives could not be opened",
            ))
        })?;
        let bytes = std::fs::read(data.join("Morrowind.esm")).map_err(SceneError::Io)?;
        let esm = EsmReader::new(&bytes)?;
        let models = ModelIndex::build(&esm)?;
        let scene = StaticScene::load_interior(&esm, &models, &vfs, cell_name)?;

        // A texture that fails to read or decode becomes `None` rather than aborting the load: one
        // dangling reference in a cell of hundreds is not a reason to have no cell.
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

        Ok(Some(Self { scene, textures }))
    }

    /// How many of the cell's texture references resolved to nothing.
    pub fn missing_textures(&self) -> usize {
        self.textures.iter().filter(|t| t.is_none()).count()
    }
}
