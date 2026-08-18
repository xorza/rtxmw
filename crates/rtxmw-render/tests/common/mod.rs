//! Fixtures shared by the renderer's integration tests.
//!
//! Loading a cell and decoding its textures is a sequence every test wanting real data needs, and
//! writing it out per file is how the copies drift apart.

use rtxmw_esm::EsmReader;
use rtxmw_scene::{Instance, Material, MaterialTable, Mesh, ModelIndex, StaticScene};
use rtxmw_texture::Texture;
use rtxmw_vfs::{morrowind_archives, morrowind_data_dir};

/// A cell and the textures its materials name.
pub(crate) struct LoadedCell {
    pub(crate) scene: StaticScene,
    /// One entry per path in the scene's texture list, `None` where the file could not be decoded.
    ///
    /// 45 of the shipped library's 4,311 references name files that were removed, so a miss is
    /// normal: it becomes the texture array's fallback slot rather than a failure.
    pub(crate) textures: Vec<Option<Texture>>,
}

/// Loads the named interior, or `None` when the game is not installed.
///
/// Everything past the install check panics rather than returning: a configured path that does not
/// parse is a broken setup, and a test that skipped on it would be indistinguishable from a pass.
pub(crate) fn load_cell(name: &str) -> Option<LoadedCell> {
    let data = morrowind_data_dir()?;
    let vfs = morrowind_archives().expect("the game directory is configured but unreadable");
    let bytes = std::fs::read(data.join("Morrowind.esm")).expect("Morrowind.esm should read");
    let esm = EsmReader::new(&bytes).expect("Morrowind.esm should parse");
    let models = ModelIndex::build(&esm).expect("model index should build");
    let scene = StaticScene::load_interior(&esm, &models, &vfs, name).expect("cell should load");

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
    Some(LoadedCell { scene, textures })
}

/// Assembles a scene from loose parts, so a test can describe one without a content file.
pub(crate) fn scene_of(
    meshes: &[Mesh],
    materials: &[Material],
    instances: &[Instance],
) -> StaticScene {
    let mut table = MaterialTable::default();
    for material in materials {
        table.intern(*material);
    }
    StaticScene {
        meshes: meshes.to_vec(),
        instances: instances.to_vec(),
        materials: table,
        lights: Vec::new(),
        ambient: None,
        without_model: Vec::new(),
    }
}
