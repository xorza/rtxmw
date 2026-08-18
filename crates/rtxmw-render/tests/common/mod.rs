//! Fixtures shared by the renderer's integration tests.
//!
//! Loading a real cell lives in `rtxmw_scene::LoadedCell`, because the engine needs it too. What is
//! left here is describing a scene without a content file, which only a test wants.

use glam::Vec3;
use rtxmw_scene::{Ambient, Instance, Light, Material, MaterialTable, Mesh, StaticScene};

/// Assembles a scene from loose parts, so a test can describe one without a content file.
///
/// Lighting is part of the description rather than something a caller sets afterwards: every trace
/// needs an ambient term and most need lights, and three places had grown the same
/// build-then-poke-two-fields sequence. Only the ambient *colour* is taken because only the colour
/// is ever read — the sunlight and fog an `Ambient` also carries reach no renderer yet.
pub(crate) fn scene_of(
    meshes: &[Mesh],
    materials: &[Material],
    instances: &[Instance],
    lights: &[Light],
    ambient: Vec3,
) -> StaticScene {
    let mut table = MaterialTable::default();
    for material in materials {
        table.intern(*material);
    }
    StaticScene {
        meshes: meshes.to_vec(),
        instances: instances.to_vec(),
        materials: table,
        lights: lights.to_vec(),
        ambient: Some(Ambient {
            colour: ambient,
            ..Ambient::default()
        }),
        without_model: Vec::new(),
    }
}
