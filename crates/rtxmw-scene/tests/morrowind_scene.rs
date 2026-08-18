//! Assembles a real interior out of the shipped content files.
//!
//! Skips when the game is not installed.

use rtxmw_esm::EsmReader;
use rtxmw_scene::{Mesh, ModelIndex, StaticScene};
use rtxmw_vfs::{DATA_DIR_VAR, morrowind_archives, morrowind_data_dir};

const CELL: &str = "Seyda Neen, Census and Excise Office";

#[test]
fn a_known_interior_assembles_into_meshes_and_instances() {
    let Some(data) = morrowind_data_dir() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };
    let vfs = morrowind_archives().expect("the game is available");
    let bytes = std::fs::read(data.join("Morrowind.esm")).expect("Morrowind.esm should read");
    let esm = EsmReader::new(&bytes).expect("should parse");

    let models = ModelIndex::build(&esm).expect("model index should build");
    assert!(models.len() > 2_000, "only {} models indexed", models.len());

    let scene = StaticScene::load_interior(&esm, &models, &vfs, CELL).expect("cell should load");

    assert!(
        !scene.instances.is_empty(),
        "a furnished interior should place something"
    );
    // Meshes are shared: an interior reuses the same crate and candlestick repeatedly, so distinct
    // meshes must come out well below placements. That ratio is the whole point of instancing.
    assert!(
        scene.meshes.len() < scene.instances.len(),
        "{} meshes for {} instances — deduplication is not working",
        scene.meshes.len(),
        scene.instances.len()
    );

    // Every instance must point at a real, non-empty mesh.
    for (index, instance) in scene.instances.iter().enumerate() {
        let mesh = scene
            .meshes
            .get(instance.mesh.0 as usize)
            .unwrap_or_else(|| panic!("instance {index} points at a missing mesh"));
        assert!(!mesh.is_empty(), "instance {index} points at an empty mesh");
        assert_eq!(
            mesh.indices.len() % 3,
            0,
            "instance {index} has a partial triangle"
        );
    }

    // Same fixture, so the placement checks belong here rather than reloading a 79 MB content
    // file and rescanning 48,000 records to assert on the same scene.
    for instance in &scene.instances {
        let mesh = &scene.meshes[instance.mesh.0 as usize];
        for &position in &mesh.positions {
            let world = instance.transform.transform_point3(position);
            assert!(
                world.is_finite(),
                "a transformed vertex is not finite: {world:?}"
            );
        }
    }

    let bounds = scene.bounds().expect("a furnished cell has geometry");
    let (min, max) = (bounds.min, bounds.max);
    let size = bounds.size();
    // One exterior cell is 8192 units. An interior room sits far inside that; anything larger means
    // a transform blew up, and anything near zero means everything collapsed onto a point.
    assert!(
        size.max_element() < 8192.0,
        "the room spans {size:?}, which is larger than an exterior cell"
    );
    assert!(
        size.min_element() > 1.0,
        "the room spans {size:?}, which is degenerate"
    );

    println!(
        "{CELL}: {} meshes, {} instances, {} unique triangles, {} placed, {} refs without a model",
        scene.meshes.len(),
        scene.instances.len(),
        scene.unique_triangle_count(),
        scene.placed_triangle_count(),
        scene.without_model.len(),
    );
    println!("  bounds {min:?} .. {max:?}, size {size:?}");
}

#[test]
fn flattening_preserves_every_visible_triangle() {
    let Some(vfs) = morrowind_archives() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };

    // Across the whole mesh library, flattening must not invent or lose triangles relative to the
    // blocks it drew from — only drop the ones behind a skipped node.
    let mut meshes = 0usize;
    let mut flattened = 0usize;
    let mut parsed = 0usize;
    // Vertices whose source block carried no normal, so a shader must derive one from the face.
    let mut degenerate_normals = 0usize;
    let mut total_vertices = 0usize;

    for path in vfs.paths().filter(|p| p.extension() == Some("nif")) {
        let Ok(bytes) = vfs.read(path.as_str()) else {
            continue;
        };
        let Ok(nif) = rtxmw_nif::NifFile::parse(&bytes) else {
            continue;
        };
        let mesh = Mesh::from_nif(&nif);

        assert_eq!(
            mesh.positions.len(),
            mesh.normals.len(),
            "{path}: normals are not parallel to positions"
        );
        assert_eq!(
            mesh.positions.len(),
            mesh.uvs.len(),
            "{path}: UVs are not parallel to positions"
        );
        for &index in &mesh.indices {
            assert!(
                (index as usize) < mesh.positions.len(),
                "{path}: index {index} with {} vertices",
                mesh.positions.len()
            );
        }

        meshes += 1;
        flattened += mesh.triangle_count();
        parsed += nif.triangle_count();
        degenerate_normals += mesh
            .normals
            .iter()
            .filter(|n| **n == glam::Vec3::ZERO)
            .count();
        total_vertices += mesh.positions.len();
    }

    assert!(meshes > 5_000, "only flattened {meshes} meshes");
    // Flattening drops collision hulls and markers, so it must be a strict subset — never more.
    assert!(
        flattened <= parsed,
        "flattening produced {flattened} triangles from {parsed}"
    );
    let kept = flattened as f64 / parsed as f64;
    assert!(
        kept > 0.5,
        "only {:.1}% of triangles survived flattening",
        kept * 100.0
    );
    println!(
        "flattened {meshes} meshes: {flattened} of {parsed} triangles kept ({:.1}%); \
         {degenerate_normals} of {total_vertices} vertices have no normal",
        kept * 100.0
    );
}
