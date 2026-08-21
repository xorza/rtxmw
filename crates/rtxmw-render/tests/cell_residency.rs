//! Cells arriving and leaving one at a time, and what they share while they are here.
//!
//! The property under test is the one the whole arrangement exists for: neighbouring cells name
//! mostly the same meshes and textures, so the second cell to name one must not upload it again.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::TestGpu;
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{
    Bone, CellId, Channel, DeformingInstance, Influence, Instance, Material, Mesh, MeshId,
    NO_PARENT, Rig, RigId, StaticScene, Submesh,
};

mod common;

const EXTENT: vk::Extent2D = vk::Extent2D {
    width: 64,
    height: 64,
};

/// A triangle, placed wherever the caller puts it. Its contents do not matter here — only that it
/// is a mesh with geometry, so it takes a slot and a structure.
fn triangle() -> Mesh {
    Mesh {
        positions: vec![
            Vec3::new(100.0, -50.0, -50.0),
            Vec3::new(100.0, 50.0, -50.0),
            Vec3::new(100.0, 0.0, 50.0),
        ],
        normals: vec![Vec3::NEG_X; 3],
        uvs: vec![Vec2::ZERO; 3],
        indices: vec![0, 1, 2],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 3,
            material: 0,
            thin: false,
        }],
    }
}

/// One cell placing `count` copies of a mesh loaded from `source`.
fn cell_of(source: &str, count: usize) -> StaticScene {
    let instances: Vec<Instance> = (0..count)
        .map(|n| Instance {
            mesh: MeshId(0),
            transform: Affine3A::from_translation(Vec3::Y * (n as f32 * 10.0)),
        })
        .collect();
    let mut scene = common::scene_of(
        &[triangle()],
        &[Material::default()],
        &instances,
        &[],
        Vec3::splat(0.2),
    );
    scene.mesh_sources = vec![source.to_owned()];
    scene
}

#[test]
fn a_mesh_two_cells_name_is_uploaded_once_and_survives_either_leaving() {
    let gpu = TestGpu::shared();
    let mut uploader = gpu.uploader();
    let limits = gpu.physical().limits();
    let mut renderer =
        SceneRenderer::new(gpu.device(), gpu.physical(), gpu.memory(), EXTENT).expect("renderer");

    let west = CellId::Exterior { x: -1, y: 0 };
    let east = CellId::Exterior { x: 0, y: 0 };
    let far = CellId::Exterior { x: 5, y: 0 };

    // Two cells placing the same rock: one placement each, and one mesh between them.
    renderer
        .add_cell(
            gpu.device(),
            &mut uploader,
            limits,
            west.clone(),
            &cell_of("rock.nif", 1),
            &[],
        )
        .expect("first cell");
    renderer
        .add_cell(
            gpu.device(),
            &mut uploader,
            limits,
            east.clone(),
            &cell_of("rock.nif", 2),
            &[],
        )
        .expect("second cell");
    renderer
        .commit(gpu.device(), &mut uploader, limits)
        .expect("commit");

    assert_eq!(
        renderer.resident_meshes(),
        1,
        "the second cell re-uploaded a mesh the first had already brought"
    );
    assert_eq!(renderer.resident_instances(), 3, "1 + 2 placements");

    // A cell naming something else adds to what is here rather than replacing it.
    renderer
        .add_cell(
            gpu.device(),
            &mut uploader,
            limits,
            far.clone(),
            &cell_of("tree.nif", 4),
            &[],
        )
        .expect("third cell");
    renderer
        .commit(gpu.device(), &mut uploader, limits)
        .expect("commit");
    assert_eq!(renderer.resident_meshes(), 2);
    assert_eq!(renderer.resident_instances(), 7, "1 + 2 + 4 placements");

    // Evicting drops placements and nothing else. The mesh the evicted cell shared stays, which is
    // what makes walking back over a boundary cost nothing — and so does the one only it named,
    // because assets are kept for the life of the renderer.
    renderer.remove_cell(&west);
    renderer.remove_cell(&far);
    renderer
        .commit(gpu.device(), &mut uploader, limits)
        .expect("commit");
    assert_eq!(renderer.resident_instances(), 2, "only the eastern cell");
    assert_eq!(
        renderer.resident_meshes(),
        2,
        "eviction must not free what a cell arriving next would ask for again"
    );

    // Re-adding a cell that is already resident replaces it rather than doubling it.
    renderer
        .add_cell(
            gpu.device(),
            &mut uploader,
            limits,
            east,
            &cell_of("rock.nif", 2),
            &[],
        )
        .expect("re-add");
    renderer
        .commit(gpu.device(), &mut uploader, limits)
        .expect("commit");
    assert_eq!(renderer.resident_instances(), 2);

    drop(uploader);
    gpu.assert_no_validation_errors();
}

#[test]
fn a_cell_reloaded_at_another_detail_gets_the_mesh_that_detail_names() {
    // **Mesh slots are grow-only and keyed by source**, which is what makes a neighbouring cell
    // nearly free — and what makes a cell that comes back *different* a trap. Terrain has one
    // heightmap per level of detail, so a cell promoted out of the distant ring must bring its own
    // mesh rather than the coarse one already sitting under its old name.
    let gpu = TestGpu::shared();
    let mut uploader = gpu.uploader();
    let limits = gpu.physical().limits();
    let mut renderer =
        SceneRenderer::new(gpu.device(), gpu.physical(), gpu.memory(), EXTENT).expect("renderer");

    let shore = CellId::Exterior { x: 2, y: -9 };
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            limits,
            shore.clone(),
            &cell_of("land:2,-9@4", 1),
            &[],
        )
        .expect("the coarse cell loads");
    let coarse = renderer.resident_meshes();

    // The same cell again, at the detail the window wants now.
    renderer
        .add_cell(
            gpu.device(),
            &mut uploader,
            limits,
            shore.clone(),
            &cell_of("land:2,-9@1", 1),
            &[],
        )
        .expect("the detailed cell loads");
    renderer
        .commit(gpu.device(), &mut uploader, limits)
        .expect("commit");

    assert_eq!(
        renderer.resident_meshes(),
        coarse + 1,
        "the cell came back at another detail and was handed the mesh it had before, so near \
         terrain is drawn with the distant tier's geometry and materials"
    );
    // One cell, one placement: the old copy went with the old cell rather than accumulating.
    assert_eq!(renderer.resident_instances(), 1);
    drop(uploader);
    gpu.assert_no_validation_errors();
}

/// A cell placing `count` copies of a mesh that has to be posed every frame.
fn moving_cell(source: &str, count: usize) -> StaticScene {
    let mut scene = cell_of(source, 0);
    let turn = |time: f32, degrees: f32| {
        let half = degrees.to_radians() / 2.0;
        rtxmw_scene::QuaternionKey {
            time,
            value: [half.cos(), 0.0, 0.0, half.sin()],
        }
    };
    scene.rigs = vec![Rig {
        mesh: MeshId(0),
        parents: vec![NO_PARENT],
        rest: vec![Affine3A::IDENTITY],
        channels: vec![Channel {
            rotations: vec![turn(0.0, 0.0), turn(1.0, 30.0)],
            ..Channel::default()
        }],
        bones: vec![Bone {
            joint: 0,
            inverse_bind: Affine3A::IDENTITY,
        }],
        influences: vec![
            Influence {
                bones: [0; 4],
                weights: [1.0, 0.0, 0.0, 0.0],
            };
            3
        ],
        groups: Vec::new(),
        playing: 0.0..1.0,
    }];
    scene.deforming = (0..count)
        .map(|n| DeformingInstance {
            instance: Instance {
                mesh: MeshId(0),
                transform: Affine3A::from_translation(Vec3::Y * (n as f32 * 10.0)),
            },
            rig: RigId(0),
            phase: 0.0,
        })
        .collect();
    scene
}

#[test]
fn what_a_frame_poses_follows_what_is_resident_rather_than_what_has_ever_been() {
    // **The bug this is here for.** A second cell arriving used to replace the placements the first
    // had left, while the structures built over them accumulated — so a frame rebuilt a bottom
    // level over a vertex region nothing had written, which on this hardware is a lost device or an
    // exploded frame rather than a missing banner. An evicted cell has the mirror of the problem:
    // its placements stayed and were posed forever.
    let gpu = TestGpu::shared();
    let limits = gpu.physical().limits();
    let mut uploader = gpu.uploader();
    let mut renderer = SceneRenderer::new(gpu.device(), gpu.physical(), gpu.memory(), EXTENT)
        .expect("renderer should build");

    let east = CellId::Exterior { x: 1, y: 0 };
    let west = CellId::Exterior { x: -1, y: 0 };
    for (id, count) in [(east.clone(), 2usize), (west.clone(), 3)] {
        renderer
            .add_cell(
                gpu.device(),
                &mut uploader,
                limits,
                id,
                &moving_cell("shared:rigged", count),
                &[],
            )
            .expect("cell should load");
    }
    renderer
        .commit(gpu.device(), &mut uploader, limits)
        .expect("commit");
    assert_eq!(
        renderer.posed_count(),
        5,
        "both cells' placements should be posed"
    );

    // One leaves. What it placed stops being posed, and what the other placed carries on.
    renderer.remove_cell(&east);
    renderer
        .commit(gpu.device(), &mut uploader, limits)
        .expect("commit");
    assert_eq!(
        renderer.posed_count(),
        3,
        "an evicted cell should stop costing a frame anything"
    );

    // And it comes back to the regions it left rather than reserving more: the pool is an asset.
    let reserved = renderer.resident_meshes();
    renderer
        .add_cell(
            gpu.device(),
            &mut uploader,
            limits,
            east,
            &moving_cell("shared:rigged", 2),
            &[],
        )
        .expect("cell should load");
    renderer
        .commit(gpu.device(), &mut uploader, limits)
        .expect("commit");
    assert_eq!(renderer.posed_count(), 5, "the cell should be posed again");
    assert_eq!(
        renderer.resident_meshes(),
        reserved,
        "a returning cell should find its vertex regions rather than reserve more"
    );

    drop(uploader);
    gpu.assert_no_validation_errors();
}
