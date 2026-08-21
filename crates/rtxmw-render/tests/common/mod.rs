//! Fixtures shared by the renderer's integration tests.
//!
//! Loading a real cell lives in `rtxmw_scene::LoadedCell`, because the engine needs it too. What is
//! left here is describing a scene without a content file, which only a test wants.

/// The bring-up both Ray Reconstruction tests share — see `NgxGpu`, and why they cannot use the
/// device the rest of the suite runs on.
#[cfg(feature = "dlss")]
pub(crate) mod ngx_gpu;

use glam::Vec3;
use rtxmw_scene::{
    Ambient, Instance, Light, Material, MaterialKind, MaterialTable, Mesh, MeshId, StaticScene,
    Submesh, TerrainLayers,
};

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
        .flat_map(|material| {
            let ground = match material.kind {
                MaterialKind::Terrain(TerrainLayers(ids)) => Some(ids),
                _ => None,
            };
            material
                .base_colour
                .into_iter()
                .chain(ground.into_iter().flatten())
        })
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
        deforming: Vec::new(),
        rigs: Vec::new(),
        materials: table,
        lights: lights.to_vec(),
        // A fixture that wants a flame builds one itself; nothing here emits anything.
        emitters: Vec::new(),
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

/// A cell holding nothing but a dark floor far enough below to be out of the way.
///
/// **Silenced deliberately**: this module is compiled into every integration binary in the
/// directory, and two of them want this. `scene_of` beside it is wanted by all of them and so needs
/// no such note.
///
/// **What every test of the sky stands under.** Two of them had built their own copy — the same
/// slab, the same near-black material, the same `ambient = None` so the dome supplies all of it —
/// and the pair is a fixture rather than a subject: what is being measured in both is what is
/// *above* it.
#[allow(dead_code)]
pub(crate) fn under_the_sky() -> StaticScene {
    let far = 200_000.0;
    let mut scene = scene_of(
        &[Mesh {
            positions: vec![
                Vec3::new(-far, -far, -50_000.0),
                Vec3::new(far, -far, -50_000.0),
                Vec3::new(far, far, -50_000.0),
                Vec3::new(-far, far, -50_000.0),
            ],
            normals: vec![Vec3::Z; 4],
            uvs: vec![glam::Vec2::ZERO; 4],
            indices: vec![0, 1, 2, 0, 2, 3],
            submeshes: vec![Submesh {
                first_index: 0,
                index_count: 6,
                material: 0,
                thin: false,
            }],
        }],
        &[Material {
            diffuse: Vec3::splat(0.02),
            ..Material::default()
        }],
        &[Instance {
            mesh: MeshId(0),
            transform: glam::Affine3A::IDENTITY,
        }],
        &[],
        Vec3::splat(0.02),
    );
    scene.ambient = None;
    scene
}
