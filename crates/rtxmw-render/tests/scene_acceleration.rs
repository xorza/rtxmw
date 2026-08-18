//! Builds real acceleration structures on the device.
//!
//! There is nothing to look at until M3e puts a ray through them, so what these assert is the shape
//! the build produced and — through the validation layer — that every address, alignment and
//! offset handed to the driver was one it accepts. A build that is silently wrong still returns
//! success, so validation is doing most of the work here.

use glam::{Affine3A, Vec2, Vec3};
use rtxmw_esm::EsmReader;
use rtxmw_gpu::TestGpu;
use rtxmw_render::{GeometryBuffers, SceneAcceleration};
use rtxmw_scene::{Instance, Material, Mesh, MeshId, ModelIndex, StaticScene, Submesh};
use rtxmw_vfs::{DATA_DIR_VAR, morrowind_archives, morrowind_data_dir};

const CELL: &str = "Seyda Neen, Census and Excise Office";

/// A quad at `offset`, two triangles, with usable normals and UVs.
fn quad(offset: f32) -> Mesh {
    Mesh {
        positions: vec![
            Vec3::new(offset, 0.0, 0.0),
            Vec3::new(offset + 10.0, 0.0, 0.0),
            Vec3::new(offset + 10.0, 10.0, 0.0),
            Vec3::new(offset, 10.0, 0.0),
        ],
        normals: vec![Vec3::Z; 4],
        uvs: vec![Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y],
        indices: vec![0, 1, 2, 0, 2, 3],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material: 0,
        }],
    }
}

#[test]
fn a_small_scene_builds_one_structure_per_mesh_and_one_instance_per_placement() {
    let gpu = TestGpu::shared();
    let meshes = [quad(0.0), quad(100.0)];
    // Three placements of two meshes: the ratio the whole mesh/instance split exists for.
    let instances = [
        Instance {
            mesh: MeshId(0),
            transform: Affine3A::IDENTITY,
        },
        Instance {
            mesh: MeshId(1),
            transform: Affine3A::from_translation(Vec3::new(0.0, 0.0, 50.0)),
        },
        Instance {
            mesh: MeshId(0),
            transform: Affine3A::from_translation(Vec3::new(0.0, 0.0, 100.0)),
        },
    ];

    let materials = [Material::default()];
    let mut uploader = gpu.uploader();
    let geometry =
        GeometryBuffers::upload(gpu.memory(), &mut uploader, &meshes).expect("upload failed");
    let acceleration = SceneAcceleration::build(
        gpu.device(),
        &mut uploader,
        gpu.physical().limits(),
        &geometry,
        &materials,
        &instances,
    )
    .expect("build failed");

    assert_eq!(acceleration.blas_count(), 2);
    assert_eq!(acceleration.instance_count(), 3);
    assert_ne!(acceleration.tlas().device_address(), 0);
    // Compaction must not grow anything; on a scene this small it may legitimately save nothing.
    assert!(
        acceleration.blas_bytes() <= acceleration.uncompacted_blas_bytes(),
        "compaction grew the bottom level"
    );

    drop(uploader);
    gpu.assert_no_validation_errors();
}

#[test]
fn a_mesh_that_flattened_to_nothing_gets_no_structure() {
    let gpu = TestGpu::shared();
    // The middle mesh has no triangles. It must not consume a BLAS slot, and it must not shift the
    // structure the third mesh's instances point at.
    let meshes = [quad(0.0), Mesh::default(), quad(100.0)];
    let instances = [Instance {
        mesh: MeshId(2),
        transform: Affine3A::IDENTITY,
    }];

    let materials = [Material::default()];
    let mut uploader = gpu.uploader();
    let geometry =
        GeometryBuffers::upload(gpu.memory(), &mut uploader, &meshes).expect("upload failed");
    let acceleration = SceneAcceleration::build(
        gpu.device(),
        &mut uploader,
        gpu.physical().limits(),
        &geometry,
        &materials,
        &instances,
    )
    .expect("build failed");

    assert_eq!(acceleration.blas_count(), 2, "the empty mesh took a slot");
    assert_eq!(acceleration.instance_count(), 1);

    drop(uploader);
    gpu.assert_no_validation_errors();
}

#[test]
fn an_empty_cell_still_produces_a_traversable_top_level() {
    let gpu = TestGpu::shared();
    let mut uploader = gpu.uploader();

    let materials: [Material; 0] = [];
    let geometry =
        GeometryBuffers::upload(gpu.memory(), &mut uploader, &[]).expect("upload failed");
    let acceleration = SceneAcceleration::build(
        gpu.device(),
        &mut uploader,
        gpu.physical().limits(),
        &geometry,
        &materials,
        &[],
    )
    .expect("build failed");

    // A ray query needs a valid structure to initialise against even where nothing is placed, or
    // every caller has to special-case the empty cell.
    assert_eq!(acceleration.blas_count(), 0);
    assert_eq!(acceleration.instance_count(), 0);
    assert_ne!(acceleration.tlas().device_address(), 0);

    drop(uploader);
    gpu.assert_no_validation_errors();
}

#[test]
fn a_real_interior_builds_and_compacts() {
    let Some(data) = morrowind_data_dir() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };
    let vfs = morrowind_archives().expect("the game is available");
    let bytes = std::fs::read(data.join("Morrowind.esm")).expect("Morrowind.esm should read");
    let esm = EsmReader::new(&bytes).expect("should parse");
    let models = ModelIndex::build(&esm).expect("model index should build");
    let scene = StaticScene::load_interior(&esm, &models, &vfs, CELL).expect("cell should load");

    let gpu = TestGpu::shared();
    let mut uploader = gpu.uploader();
    let materials = scene.materials.materials();
    let geometry =
        GeometryBuffers::upload(gpu.memory(), &mut uploader, &scene.meshes).expect("upload failed");
    let acceleration = SceneAcceleration::build(
        gpu.device(),
        &mut uploader,
        gpu.physical().limits(),
        &geometry,
        materials,
        &scene.instances,
    )
    .expect("build failed");

    // Every mesh in this cell has geometry — `StaticScene` drops placements of empty ones — so the
    // structure count and the instance count must match the scene exactly.
    assert_eq!(acceleration.blas_count(), scene.meshes.len());
    assert_eq!(
        acceleration.instance_count() as usize,
        scene.instances.len()
    );

    let compacted = acceleration.blas_bytes();
    let uncompacted = acceleration.uncompacted_blas_bytes();
    assert!(
        compacted <= uncompacted,
        "compaction grew the bottom level: {compacted} from {uncompacted}"
    );

    println!(
        "{CELL}: {} BLAS over {} triangles, {} instances\n  \
         bottom level {:.2} MiB compacted from {:.2} MiB ({:.0}% saved), top level {:.1} KiB",
        acceleration.blas_count(),
        geometry.triangle_count(),
        acceleration.instance_count(),
        compacted as f64 / (1024.0 * 1024.0),
        uncompacted as f64 / (1024.0 * 1024.0),
        (1.0 - compacted as f64 / uncompacted as f64) * 100.0,
        acceleration.tlas().size() as f64 / 1024.0,
    );

    drop(uploader);
    gpu.assert_no_validation_errors();
}
