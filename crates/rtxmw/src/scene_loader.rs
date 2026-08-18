//! Getting a cell out of the game's content files and into memory.

use glam::Vec3;
use rtxmw_esm::EsmReader;
use rtxmw_scene::{ModelIndex, StaticScene};
use rtxmw_texture::Texture;
use rtxmw_vfs::morrowind_data_dir;

/// The cell the engine opens in until there is a way to choose one.
const DEFAULT_CELL: &str = "Seyda Neen, Census and Excise Office";

/// A loaded cell and where to stand in it.
#[derive(Debug)]
pub(crate) struct LoadedCell {
    pub(crate) name: &'static str,
    pub(crate) scene: StaticScene,
    /// One entry per path in the scene's texture list. `None` where the file does not exist, which
    /// is normal: the shipped meshes carry a tail of references to art that was removed.
    pub(crate) textures: Vec<Option<Texture>>,
    /// Centre of the cell's geometry — inside the room for this cell, though there is no general
    /// rule; a spawn point needs the cell's own markers, which is a later milestone's problem.
    pub(crate) viewpoint: Vec3,
}

/// Loads [`DEFAULT_CELL`], or `None` when no install is configured.
pub(crate) fn load_default_cell() -> Result<Option<LoadedCell>, Box<dyn std::error::Error>> {
    let Some(data) = morrowind_data_dir() else {
        return Ok(None);
    };
    let vfs = rtxmw_vfs::morrowind_archives().ok_or("archives could not be opened")?;
    let bytes = std::fs::read(data.join("Morrowind.esm"))?;
    let esm = EsmReader::new(&bytes)?;
    let models = ModelIndex::build(&esm)?;
    let scene = StaticScene::load_interior(&esm, &models, &vfs, DEFAULT_CELL)?;

    let viewpoint = scene.bounds().map_or(Vec3::ZERO, |b| b.centre());

    let mut textures = Vec::with_capacity(scene.materials.textures().len());
    for path in scene.materials.textures() {
        textures.push(
            vfs.read(path)
                .ok()
                .and_then(|bytes| Texture::decode(&bytes).ok()),
        );
    }

    Ok(Some(LoadedCell {
        name: DEFAULT_CELL,
        scene,
        textures,
        viewpoint,
    }))
}
