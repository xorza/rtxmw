//! Assembles a real interior out of the shipped content files.
//!
//! Skips when the game is not installed.

use rtxmw_esm::{CellId, CellIndex, EsmReader};
use rtxmw_scene::{CellStreamer, Door, LoadedCell, MaterialTable, Mesh, ModelIndex, StaticScene};
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

    let index = CellIndex::build(&esm).expect("cell index should build");
    let scene = StaticScene::load_cell(&esm, &index, &models, &vfs, &CellId::Interior(CELL.into()))
        .expect("cell should load");

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

    // Every texture the cell's materials name must exist, or the bindless array binds nothing and
    // the surface samples whatever descriptor happened to be there.
    let mut missing = Vec::new();
    for path in scene.materials.textures() {
        if vfs.read(path).is_err() {
            missing.push(path.clone());
        }
    }
    assert!(
        missing.is_empty(),
        "{} of {} texture paths are not in the VFS, e.g. {:?}",
        missing.len(),
        scene.materials.textures().len(),
        &missing[..missing.len().min(5)]
    );

    // Submeshes must tile each mesh's index buffer exactly: a gap drops triangles from the build,
    // an overlap builds them twice.
    for (id, mesh) in scene.meshes.iter().enumerate() {
        let mut cursor = 0u32;
        for sub in &mesh.submeshes {
            assert_eq!(sub.first_index, cursor, "mesh {id} submesh gap");
            assert_eq!(sub.index_count % 3, 0, "mesh {id} partial triangle");
            assert!(
                (sub.material as usize) < scene.materials.materials().len(),
                "mesh {id} names material {} of {}",
                sub.material,
                scene.materials.materials().len()
            );
            cursor += sub.index_count;
        }
        assert_eq!(
            cursor as usize,
            mesh.indices.len(),
            "mesh {id} submesh coverage"
        );
    }

    println!(
        "  {} lights, ambient {:?}",
        scene.lights.len(),
        scene.ambient.map(|a| a.colour)
    );
    for light in scene.lights.iter().take(4) {
        println!(
            "    light at {:?} radius {} colour {:?}",
            light.position, light.radius, light.colour
        );
    }

    let textured = scene
        .materials
        .materials()
        .iter()
        .filter(|m| m.base_colour.is_some())
        .count();
    println!(
        "  {} materials ({textured} textured), {} distinct textures, {} submeshes",
        scene.materials.materials().len(),
        scene.materials.textures().len(),
        scene
            .meshes
            .iter()
            .map(|m| m.submeshes.len())
            .sum::<usize>(),
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
fn flattening_preserves_geometry_and_resolves_every_material() {
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
    let mut submeshes = 0usize;
    let mut materials = rtxmw_scene::MaterialTable::default();

    for path in vfs.paths().filter(|p| p.extension() == Some("nif")) {
        let Ok(bytes) = vfs.read(path.as_str()) else {
            continue;
        };
        let Ok(nif) = rtxmw_nif::NifFile::parse(&bytes) else {
            continue;
        };
        let mesh = Mesh::from_nif(&nif, &mut materials);

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

        // Submeshes must cover the whole index buffer for every model in the library, not just the
        // ones one cell happens to place.
        let covered: u32 = mesh.submeshes.iter().map(|s| s.index_count).sum();
        assert_eq!(
            covered as usize,
            mesh.indices.len(),
            "{path}: submeshes cover {covered} of {} indices",
            mesh.indices.len()
        );
        submeshes += mesh.submeshes.len();

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
    // Almost every texture name has to resolve, because the fixups that turn one into a path —
    // forcing `.dds`, prepending the directory — are guesses about the original data until the
    // corpus agrees with them. Not *every* one: the shipped meshes carry a tail of references to
    // textures Bethesda removed, so a rate is the honest assertion and the renderer needs a
    // fallback rather than a panic.
    let mut missing = Vec::new();
    for path in materials.textures() {
        if vfs.read(path).is_err() {
            missing.push(path.clone());
        }
    }

    println!(
        "flattened {meshes} meshes into {submeshes} submeshes: {flattened} of {parsed} triangles \
         kept ({:.1}%); {degenerate_normals} of {total_vertices} vertices have no normal",
        kept * 100.0
    );
    println!(
        "  {} materials over {} distinct textures, {} unresolved",
        materials.materials().len(),
        materials.textures().len(),
        missing.len()
    );
    for path in missing.iter().take(10) {
        println!("    missing {path}");
    }

    let unresolved = missing.len() as f64 / materials.textures().len() as f64;
    assert!(
        unresolved < 0.02,
        "{} of {} texture paths do not exist ({:.1}%), which is past what dangling references in \
         the shipped data explain — suspect the path fixups",
        missing.len(),
        materials.textures().len(),
        unresolved * 100.0
    );
}

#[test]
fn the_doors_leading_into_a_cell_land_a_traveller_inside_it() {
    let Some(data) = morrowind_data_dir() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };
    let vfs = morrowind_archives().expect("the game is available");
    let bytes = std::fs::read(data.join("Morrowind.esm")).expect("Morrowind.esm should read");
    let esm = EsmReader::new(&bytes).expect("should parse");
    let models = ModelIndex::build(&esm).expect("model index should build");

    let destination = CellId::Interior(CELL.into());
    let doors = Door::leading_to(&esm, &models, &destination).expect("scan should succeed");
    let index = CellIndex::build(&esm).expect("cell index should build");
    let scene = StaticScene::load_cell(&esm, &index, &models, &vfs, &CellId::Interior(CELL.into()))
        .expect("cell should load");
    let bounds = scene.bounds().expect("a furnished cell has geometry");
    println!("{} doors lead into {CELL}, bounds {bounds:?}", doors.len());
    for door in &doors {
        println!("  arrival {:?} facing {:?}", door.arrival, door.facing);
    }

    // Four of them, all in Seyda Neen's exterior: the customs door, the north door, and the two
    // the character-generation sequence uses. `PrisonMarker` also carries a destination into this
    // cell and is filed as a `DOOR`, but it draws with an editor marker and is excluded — a fifth
    // here would mean that filter has stopped working.
    assert_eq!(doors.len(), 4, "{doors:#?}");

    for door in &doors {
        assert_eq!(door.destination, destination);
        // Inside the cell's own geometry, which is the check that catches a destination read from
        // the wrong subrecord: the doors stand in an exterior cell whose coordinates are around
        // (-10000, -72000), so a position taken from the door rather than its `DODT` would miss
        // this by four orders of magnitude.
        assert!(
            door.arrival.cmpge(bounds.min).all() && door.arrival.cmple(bounds.max).all(),
            "arrival {:?} is outside the cell's bounds {bounds:?}",
            door.arrival
        );
        // A bearing, so it lies flat and has unit length. A zero vector would leave the camera
        // pointing at nothing in particular.
        assert_eq!(door.facing.z, 0.0);
        assert!((door.facing.length() - 1.0).abs() < 1e-5);
    }

    // The doors face different ways — they are on different walls. A constant here would mean the
    // stored yaw is being dropped rather than read.
    let facings = doors.iter().map(|d| d.facing).collect::<Vec<_>>();
    assert!(
        facings.iter().any(|f| (*f - facings[0]).length() > 0.5),
        "every door faces the same way: {facings:?}"
    );

    // Editor markers place no geometry either: they are the original engine's placement aids and
    // it never drew them. This cell places a `NorthMarker`, so the count is the assertion.
    assert!(
        models.is_editor_marker("PrisonMarker") && models.is_editor_marker("NorthMarker"),
        "the marker meshes are no longer being recognised"
    );
    assert!(!models.is_editor_marker("ex_nord_door_01"));
    // Compared against the mesh itself rather than a guess at its shape: the marker is a solid
    // arrow 160 units tall, not something a bounding box would tell apart from the furniture.
    let marker_nif = vfs
        .read("meshes/Marker_North.nif")
        .expect("the marker mesh ships");
    let marker = Mesh::from_nif(
        &rtxmw_nif::NifFile::parse(&marker_nif).expect("marker should parse"),
        &mut MaterialTable::default(),
    );
    assert!(
        !marker.is_empty(),
        "the marker draws nothing, so skipping it proves nothing"
    );
    assert!(
        !scene.meshes.iter().any(|m| m.positions == marker.positions),
        "the north marker this cell places was built into the geometry"
    );

    // Every arrival drops onto the same floor, 192 units up, which is what makes standing on it
    // give the same eye height in every cell — the authored heights themselves range over 80
    // units here and 120 across the game.
    for door in &doors {
        let ground = scene
            .ground_below(door.arrival + glam::Vec3::Z * 20.0)
            .expect("a door arrives above a floor");
        assert!(
            (ground - 192.0).abs() < 1.0,
            "arrival {:?} dropped to {ground}, not the cell's floor at 192",
            door.arrival
        );
        assert!(
            door.arrival.z - ground > 20.0,
            "the arrival is supposed to sit well above the floor, not on it"
        );
    }
}

#[test]
fn an_exterior_cell_carries_its_terrain_as_well_as_its_objects() {
    let Some(data) = morrowind_data_dir() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };
    let vfs = morrowind_archives().expect("the game is available");
    let bytes = std::fs::read(data.join("Morrowind.esm")).expect("Morrowind.esm should read");
    let esm = EsmReader::new(&bytes).expect("should parse");
    let models = ModelIndex::build(&esm).expect("model index should build");

    // Seyda Neen, where the game opens.
    let (x, y) = (-2, -9);
    let index = CellIndex::build(&esm).expect("cell index should build");
    let scene = StaticScene::load_cell(&esm, &index, &models, &vfs, &CellId::Exterior { x, y })
        .expect("cell should load");

    // The terrain is the one mesh with a vertex per point of the 65×65 grid.
    let terrain = scene
        .meshes
        .iter()
        .find(|mesh| mesh.positions.len() == rtxmw_esm::VERTICES)
        .expect("the cell has no terrain mesh");
    println!(
        "({x}, {y}): {} meshes, {} instances, terrain in {} submeshes over {} textures",
        scene.meshes.len(),
        scene.instances.len(),
        terrain.submeshes.len(),
        scene.materials.textures().len()
    );

    // Placed in the world rather than about its own origin, so its corner is the cell's corner.
    let cell = 8192.0;
    let corner = terrain.positions[0];
    assert_eq!(
        corner.truncate(),
        glam::Vec2::new(x as f32 * cell, y as f32 * cell)
    );

    // Seyda Neen is a coastal marsh: terrain either side of sea level, and nothing like the
    // thousands of units a mis-decoded delta produces.
    let (low, high) = terrain
        .positions
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), p| (l.min(p.z), h.max(p.z)));
    assert!(
        low < 0.0,
        "no ground below sea level in a coastal cell: {low}"
    );
    assert!((0.0..4_000.0).contains(&high), "highest ground at {high}");

    // Split by texture tile, so a cell of marsh, sand and rock is several submeshes rather than
    // one — and every triangle belongs to exactly one of them.
    assert!(
        terrain.submeshes.len() > 2,
        "terrain came out as {} submeshes, so the texture tiles are not being separated",
        terrain.submeshes.len()
    );
    let covered: u32 = terrain.submeshes.iter().map(|s| s.index_count).sum();
    assert_eq!(covered as usize, terrain.indices.len());

    let names: Vec<&str> = terrain
        .submeshes
        .iter()
        .map(|submesh| {
            let material = &scene.materials.materials()[submesh.material as usize];
            let texture = material.base_colour.expect("terrain draws with a texture");
            scene.materials.textures()[texture.0 as usize].as_str()
        })
        .collect();
    println!("  terrain textures: {names:?}");
    for path in &names {
        assert!(
            path.starts_with("textures/") && path.ends_with(".dds"),
            "terrain texture path {path:?} did not get the fixups every other texture gets"
        );
    }

    // **The assertion that catches an off-by-one in the palette**, and it has to be the exact set.
    // Checking the region is not enough: the palette turns out to be *grouped* by region, so
    // reading a `VTEX` index without its offset still lands on Bitter Coast art — `Tx_BC_rock_01`
    // where `Tx_BC_rock_03` belongs. Every wrong answer looks as plausible as the right one, so
    // only naming them works. The content file never changes, which is what makes that reasonable.
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        [
            "textures/Tx_BC_dirt.dds",
            "textures/Tx_BC_grass.dds",
            "textures/Tx_BC_moss.dds",
            "textures/Tx_BC_muck_01.dds",
            "textures/Tx_BC_mud.dds",
            "textures/Tx_BC_rock_03.dds",
            "textures/Tx_BC_undergrowth.dds",
            "textures/Tx_RM_grayrock_01.dds",
        ],
        "Seyda Neen's terrain is not textured with the Bitter Coast art it should be"
    );

    // And it is placed exactly once, with no transform: a heightmap belongs to one cell.
    let terrain_id = scene
        .meshes
        .iter()
        .position(|m| std::ptr::eq(m, terrain))
        .expect("terrain is in the mesh list");
    let placements: Vec<_> = scene
        .instances
        .iter()
        .filter(|i| i.mesh.0 as usize == terrain_id)
        .collect();
    assert_eq!(placements.len(), 1);
    assert_eq!(placements[0].transform, glam::Affine3A::IDENTITY);
}

#[test]
fn a_streamed_cell_is_the_same_cell_the_direct_path_loads() {
    if morrowind_data_dir().is_none() {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    }
    // Seyda Neen's shore, then a grid square far out in the ocean that has no cell record.
    let shore = CellId::Exterior { x: -2, y: -9 };
    let sea = CellId::Exterior { x: 9_999, y: 9_999 };

    let streamer = CellStreamer::spawn();
    streamer.request(shore.clone());
    streamer.request(sea.clone());

    // Answered in the order asked, which is what lets a caller send a whole window at once and
    // still get the nearest cell first.
    let first = streamer.wait_ready().expect("the worker should answer");
    assert_eq!(first.id, shore);
    let streamed = first.loaded.expect("the shore should load");

    let direct = LoadedCell::load_at(shore)
        .expect("direct load should succeed")
        .expect("the game is available");

    // The same cell, by every measure the renderer is given.
    assert_eq!(streamed.scene.meshes.len(), direct.scene.meshes.len());
    assert_eq!(streamed.scene.instances.len(), direct.scene.instances.len());
    assert_eq!(streamed.scene.mesh_sources, direct.scene.mesh_sources);
    assert_eq!(
        streamed.scene.materials.textures(),
        direct.scene.materials.textures()
    );
    assert_eq!(
        streamed.textures.len(),
        streamed.scene.materials.textures().len()
    );

    // Except its entrances, which streaming deliberately does not go looking for — that search is
    // a pass over the whole file, and a cell walked into needs no arrival point.
    assert!(streamed.entrances.is_empty());

    // A square with no cell record fails as itself rather than silently, so the caller knows which
    // request will never arrive.
    let second = streamer
        .wait_ready()
        .expect("the worker should answer again");
    assert_eq!(second.id, sea);
    assert!(second.loaded.is_err(), "open sea should not load");
}
