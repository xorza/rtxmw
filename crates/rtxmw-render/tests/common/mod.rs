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
    // A fixture names a texture by id and hands the decoded image over separately, with no file
    // anywhere. The renderer keys textures by where they came from, so each id needs a path to be
    // known by — and the ids have to line up with the images the caller passes alongside.
    let named = materials
        .iter()
        .filter_map(|material| material.base_colour)
        .map(|id| id.0 + 1)
        .max()
        .unwrap_or(0);
    for id in 0..named {
        table.intern_texture(&format!("fixture-texture-{id}"));
    }
    for material in materials {
        table.intern(*material);
    }
    StaticScene {
        // Fixtures build meshes by hand, so their identity is only ever their position here.
        mesh_sources: (0..meshes.len()).map(|i| format!("fixture:{i}")).collect(),
        meshes: meshes.to_vec(),
        instances: instances.to_vec(),
        materials: table,
        lights: lights.to_vec(),
        ambient: Some(Ambient {
            colour: ambient,
            ..Ambient::default()
        }),
        // Set by the caller when a fixture wants one; most are indoors and have none.
        sun: None,
        // Likewise: a fixture that places water sets the level its quad sits at.
        water_level: None,
        without_model: Vec::new(),
    }
}
