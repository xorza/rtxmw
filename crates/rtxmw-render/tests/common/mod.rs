//! Fixtures shared by the renderer's integration tests.
//!
//! Loading a real cell lives in `rtxmw_scene::LoadedCell`, because the engine needs it too. What is
//! left here is describing a scene without a content file, which only a test wants.

use rtxmw_scene::{Instance, Material, MaterialTable, Mesh, StaticScene};

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
