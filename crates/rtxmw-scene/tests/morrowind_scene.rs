//! Assembles a real interior out of the shipped content files.
//!
//! Skips when the game is not installed.

use rtxmw_esm::{CellId, CellIndex, EsmReader};
use rtxmw_scene::{
    CellDetail, CellStreamer, Door, Light, LoadedCell, MaterialKind, MaterialTable, Mesh,
    ModelIndex, StaticScene,
};
use rtxmw_vfs::{DATA_DIR_VAR, Vfs, morrowind_archives, morrowind_data_dir};

const CELL: &str = "Seyda Neen, Census and Excise Office";

/// Seyda Neen's shore, which has the trees, the boat and the silt strider on it.
const SHORE: CellId = CellId::Exterior { x: -2, y: -9 };

/// An exterior that places `LIGH` references as well as objects, which few of them do.
const LIT_EXTERIOR: CellId = CellId::Exterior { x: 0, y: -7 };

/// `Morrowind.esm`'s bytes, or nothing when the game is not installed.
///
/// Separate from [`Content`] because [`EsmReader`] borrows what it reads, so the reader and the
/// bytes it points into cannot be handed back together.
fn game_bytes() -> Option<Vec<u8>> {
    let Some(data) = morrowind_data_dir() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return None;
    };
    Some(std::fs::read(data.join("Morrowind.esm")).expect("Morrowind.esm should read"))
}

/// The shipped content, indexed and ready to load a cell out of.
#[derive(Debug)]
struct Content<'a> {
    esm: EsmReader<'a>,
    models: ModelIndex,
    index: CellIndex,
    vfs: Vfs,
}

impl<'a> Content<'a> {
    fn open(bytes: &'a [u8]) -> Self {
        let esm = EsmReader::new(bytes).expect("Morrowind.esm should parse");
        Self {
            models: ModelIndex::build(&esm).expect("model index should build"),
            index: CellIndex::build(&esm).expect("cell index should build"),
            vfs: morrowind_archives().expect("the game is available"),
            esm,
        }
    }

    fn cell(&self, id: &CellId) -> StaticScene {
        self.cell_at(id, CellDetail::Full)
    }

    fn cell_at(&self, id: &CellId, detail: CellDetail) -> StaticScene {
        StaticScene::load_cell(&self.esm, &self.index, &self.models, &self.vfs, id, detail)
            .expect("cell should load")
    }
}

#[test]
fn a_known_interior_assembles_into_meshes_and_instances() {
    let Some(bytes) = game_bytes() else { return };
    let content = Content::open(&bytes);
    assert!(
        content.models.len() > 2_000,
        "only {} models indexed",
        content.models.len()
    );
    let scene = content.cell(&CellId::Interior(CELL.into()));

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
        if content.vfs.read(path).is_err() {
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
    let Some(bytes) = game_bytes() else { return };
    let content = Content::open(&bytes);

    let destination = CellId::Interior(CELL.into());
    let doors =
        Door::leading_to(&content.esm, &content.models, &destination).expect("scan should succeed");
    let scene = content.cell(&destination);
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
        content.models.is_editor_marker("PrisonMarker")
            && content.models.is_editor_marker("NorthMarker"),
        "the marker meshes are no longer being recognised"
    );
    assert!(!content.models.is_editor_marker("ex_nord_door_01"));
    // Compared against the mesh itself rather than a guess at its shape: the marker is a solid
    // arrow 160 units tall, not something a bounding box would tell apart from the furniture.
    let marker_nif = content
        .vfs
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
    let Some(bytes) = game_bytes() else { return };
    let content = Content::open(&bytes);

    // Seyda Neen, where the game opens.
    let (x, y) = (-2, -9);
    let scene = content.cell(&CellId::Exterior { x, y });

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

    // Every layer of every patch, because the ground blends four textures at a time and a fixup
    // missed on any one of them is a magenta square somewhere on the shore.
    let names: Vec<&str> = terrain
        .submeshes
        .iter()
        .flat_map(|submesh| {
            let MaterialKind::Terrain(layers) =
                scene.materials.materials()[submesh.material as usize].kind
            else {
                panic!("ground blends between named textures")
            };
            layers
                .0
                .into_iter()
                .map(|texture| scene.materials.textures()[texture.0 as usize].as_str())
        })
        .collect();
    let mut distinct = names.clone();
    distinct.sort_unstable();
    distinct.dedup();
    println!(
        "  terrain blends {} textures across {} patches: {distinct:?}",
        distinct.len(),
        terrain.submeshes.len()
    );
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
    // Deduplicated: the ground names four textures per patch and most patches share most of them,
    // so the raw list is a thousand entries of a handful of names.
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
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

/// The terrain of `scene`, which an exterior with a heightmap has exactly one of.
fn terrain(scene: &StaticScene) -> &Mesh {
    let index = scene
        .mesh_sources
        .iter()
        .position(|source| source.starts_with("land:"))
        .expect("an exterior with a heightmap has terrain");
    &scene.meshes[index]
}

#[test]
fn distant_terrain_meets_its_own_kind_and_reports_what_it_costs_at_the_seam() {
    let Some(bytes) = game_bytes() else { return };
    let content = Content::open(&bytes);

    let side = 65 / CellDetail::DISTANT_STRIDE + 1;
    let distant = |x, y| {
        let scene = content.cell_at(&CellId::Exterior { x, y }, CellDetail::Distant);
        terrain(&scene).positions.clone()
    };
    let here = distant(-2, -9);
    assert_eq!(here.len(), side * side);

    // Two cells built at the same stride keep the same shared row of vertices, so their edges
    // coincide to the bit rather than to a tolerance. This is the property that lets the whole
    // distant world be decimated without a stitching pass between neighbours.
    let east = distant(-1, -9);
    let north = distant(-2, -8);
    for i in 0..side {
        assert_eq!(
            here[i * side + side - 1],
            east[i * side],
            "row {i} of the eastern edge does not meet the neighbour"
        );
        assert_eq!(
            here[(side - 1) * side + i],
            north[i],
            "column {i} of the northern edge does not meet the neighbour"
        );
    }

    // What it does *not* meet is a full-detail cell, and this is how far out: along a shared edge
    // the fine mesh follows every vertex and the coarse one cuts a chord between every fourth, so
    // the two part company by the relief in between. Reported rather than bounded tightly — the
    // number is what decides whether a distant ring can sit straight against the window or needs
    // something covering the seam.
    let mut worst: f32 = 0.0;
    for (x, y) in [(-2, -9), (-2, -8), (-1, -9), (-3, -10), (2, -9)] {
        let scene = content.cell_at(&CellId::Exterior { x, y }, CellDetail::Full);
        let fine = &terrain(&scene).positions;
        let coarse = distant(x, y);
        for row in 0..65 {
            let z = fine[row * 65 + 64].z;
            // The coarse chord under this row, between the two vertices the stride kept.
            let low = row / CellDetail::DISTANT_STRIDE;
            let along =
                (row % CellDetail::DISTANT_STRIDE) as f32 / CellDetail::DISTANT_STRIDE as f32;
            let below = coarse[low * side + side - 1].z;
            let above = coarse[(low + 1).min(side - 1) * side + side - 1].z;
            worst = worst.max((z - (below + (above - below) * along)).abs());
        }
    }
    println!(
        "worst gap between a full edge and a distant chord over five cells: {worst:.0} units \
         ({:.2} of a vertex spacing)",
        worst / 128.0
    );
    assert!(
        worst < 8192.0,
        "a chord across four vertices parted from the surface by {worst} units, which is more \
         relief than a cell has"
    );
}

#[test]
fn a_distant_cell_keeps_its_objects_and_drops_the_lights_they_carry() {
    let Some(bytes) = game_bytes() else { return };
    let content = Content::open(&bytes);
    // A cell with objects but no lights would prove nothing, and most of the outdoors is exactly
    // that — 229 lights across the four hundred cells around Seyda Neen. This one has six.
    let near = content.cell(&LIT_EXTERIOR);
    let far = content.cell_at(&LIT_EXTERIOR, CellDetail::Distant);

    assert!(!near.lights.is_empty(), "the fixture cell has no lights");
    assert!(
        far.lights.is_empty(),
        "a distant cell contributed {} lights; the shader walks every one of them for every pixel \
         in the frame, and none of these reaches it",
        far.lights.len()
    );
    // The objects stay, and they stay whole: dropping the lights must not drop the lamps casting
    // them, which would leave a hole in the town rather than an unlit one.
    assert_eq!(
        far.instances.len(),
        near.instances.len(),
        "a distant cell placed a different number of objects than the same cell up close"
    );
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
    streamer.request(shore.clone(), CellDetail::Full);
    streamer.request(sea.clone(), CellDetail::Full);

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

#[test]
fn the_shipped_models_that_are_cloth_are_the_ones_marked_as_sheets() {
    let Some(bytes) = game_bytes() else { return };
    let content = Content::open(&bytes);
    let scene = content.cell(&CellId::Interior(CELL.into()));

    let sheets_in = |name: &str| {
        let position = scene
            .mesh_sources
            .iter()
            .position(|source| source.to_lowercase().contains(name))
            .unwrap_or_else(|| panic!("{name} is not in this cell"));
        let mesh = &scene.meshes[position];
        mesh.submeshes.iter().filter(|run| run.thin).count()
    };

    // A tapestry and a rug are cloth, and each is one hanging run among the frame and rod that
    // carry it.
    assert_eq!(sheets_in("furn_com_tapestry_03"), 1);
    assert_eq!(sheets_in("furn_rug_big_06"), 1);

    // Neither of these is, however its runs are split — a chest and a wine bottle are closed
    // surfaces, and a room's wall shell wraps the room's air. Measured per run rather than per
    // mesh, all three come out as cloth, because splitting by material opens every one of them.
    assert_eq!(sheets_in("contain_com_chest_02"), 1, "the chest's flat lid");
    assert_eq!(sheets_in("misc_com_bottle_09"), 0);
    assert_eq!(sheets_in("in_c_plain_room_corner"), 2, "flat wall slabs");

    // And the cell as a whole is mostly solid. The exact fraction is not the property — that a
    // furnished interior is not seven-eighths cloth is.
    let (sheets, runs) = scene.meshes.iter().fold((0, 0), |(sheets, runs), mesh| {
        (
            sheets + mesh.submeshes.iter().filter(|run| run.thin).count(),
            runs + mesh.submeshes.len(),
        )
    });
    assert!(
        sheets * 4 < runs,
        "{sheets} of {runs} runs classified as cloth"
    );
}

#[test]
fn the_shipped_meshes_wind_their_triangles_to_agree_with_their_normals() {
    // **The renderer shades a hit as whichever face of the triangle the ray met**, deciding that
    // from the winding rather than from the interpolated normal — which near a silhouette can point
    // away from a face that is squarely toward the viewer. That only gives the right answer if the
    // two agree in the first place, and nothing in the format enforces it.
    //
    // So it is measured. A triangle disagrees when its plane, taken from the order of its corners,
    // points opposite to the normals its vertices carry.
    let Some(bytes) = game_bytes() else { return };
    let content = Content::open(&bytes);

    let mut disagree = 0usize;
    let mut total = 0usize;
    // An interior of furniture and cloth, and a stretch of shore with trees, rocks and a boat —
    // between them most of the shapes the game is built out of.
    for cell in [CellId::Interior(CELL.into()), SHORE] {
        let scene = content.cell(&cell);
        for mesh in &scene.meshes {
            for triangle in mesh.indices.chunks_exact(3) {
                let corner = |at: usize| mesh.positions[triangle[at] as usize];
                let plane = (corner(1) - corner(0)).cross(corner(2) - corner(0));
                let authored: glam::Vec3 =
                    (0..3).map(|at| mesh.normals[triangle[at] as usize]).sum();
                // A degenerate triangle has no plane, and a mesh with no normals carries zeroes;
                // neither is a disagreement.
                if plane.length_squared() == 0.0 || authored.length_squared() == 0.0 {
                    continue;
                }
                total += 1;
                if plane.dot(authored) < 0.0 {
                    disagree += 1;
                }
            }
        }
    }

    assert!(total > 50_000, "only {total} triangles measured");
    // A handful of backwards triangles is what the artists left behind; a large share would mean
    // the convention is the other way round and the renderer has it backwards everywhere.
    println!("{disagree} of {total} triangles are wound against their own normals");
    assert!(
        disagree * 100 < total,
        "{disagree} of {total} triangles are wound against their own normals"
    );
}

#[test]
fn a_trees_foliage_is_cloth_and_its_trunk_is_not() {
    // The case no shape test catches. A canopy is hundreds of leaf cards joined at the branches,
    // and the cupped cluster they make wraps as much air as a shell around a room — measured as
    // geometry a tree is solid, and every card facing away from the sun goes dark.
    //
    // The alpha says otherwise, and it is authored rather than inferred: Morrowind sets a cutout on
    // foliage, thatch, banners and glass, and on nothing solid.
    let Some(bytes) = game_bytes() else { return };
    let content = Content::open(&bytes);
    let scene = content.cell(&SHORE);

    let position = scene
        .mesh_sources
        .iter()
        .position(|source| source.to_lowercase().contains("flora_bc_tree_02"))
        .expect("a tree on the shore");
    let tree = &scene.meshes[position];
    let sheets = tree.submeshes.iter().filter(|run| run.thin).count();

    // Its foliage, and not the trunk and boughs the leaves hang from.
    assert_eq!(sheets, 3, "of {} runs", tree.submeshes.len());
    assert!(
        tree.submeshes.len() > sheets,
        "every run of the tree came out as cloth, trunk included"
    );

    // The trunk is the bulk of the model's triangles, so a sheet count that swallowed it would not
    // show in the run count alone.
    let solid_triangles: u32 = tree
        .submeshes
        .iter()
        .filter(|run| !run.thin)
        .map(|run| run.index_count / 3)
        .sum();
    assert!(
        solid_triangles > 300,
        "only {solid_triangles} solid triangles"
    );
}

#[test]
fn the_fireplace_opens_toward_the_fire_burning_in_it() {
    // **A hearth and the fire inside it are placed separately**, so the fire is ground truth for
    // which way the stack faces — no screenshot needed, and nothing to eyeball.
    //
    // Seyda Neen's fireplace is the model that catches a transform the original engine discards:
    // its outermost node carries a half turn about Z, and honoured, the stack presents its back to
    // the room while the fire burns behind it. Every other model in this room has an identity
    // there, which is why the room looked right while this one alone was backwards.
    let Some(bytes) = game_bytes() else { return };
    let content = Content::open(&bytes);
    let scene = content.cell(&CellId::Interior(CELL.into()));

    let placed = scene
        .instances
        .iter()
        .find(|instance| {
            scene.mesh_sources[instance.mesh.0 as usize]
                .to_lowercase()
                .contains("fireplace")
        })
        .expect("the office has a fireplace");
    let mesh = &scene.meshes[placed.mesh.0 as usize];
    let origin = glam::Vec3::from(placed.transform.translation);

    // The hearth slab is the lowest slice of the model, and it juts out on the open side.
    let low = mesh
        .positions
        .iter()
        .map(|p| p.z)
        .fold(f32::INFINITY, f32::min);
    let high = mesh
        .positions
        .iter()
        .map(|p| p.z)
        .fold(f32::NEG_INFINITY, f32::max);
    let slab: Vec<glam::Vec3> = mesh
        .positions
        .iter()
        .filter(|p| p.z <= low + (high - low) * 0.08)
        .map(|p| placed.transform.transform_point3(*p))
        .collect();
    assert!(!slab.is_empty(), "the model has no hearth slab");
    let hearth = slab.iter().sum::<glam::Vec3>() / slab.len() as f32;

    // Nearest in plan rather than in space, and for the same reason the check below is: the
    // reference's own origin sits high up the chimney while the fire is down at the hearth, a
    // couple of hundred units apart in height and a stride apart on the floor. Sorted by distance
    // in space, a lamp on a nearer wall could come first.
    let across_from = |light: &Light| (light.position - origin).truncate().length();
    let fire = scene
        .lights
        .iter()
        .min_by(|a, b| across_from(a).total_cmp(&across_from(b)))
        .expect("the office lights its fire");
    let across = across_from(fire);
    assert!(
        across < 100.0,
        "the nearest light is {across} units away in plan, too far to be this fire"
    );

    // Both measured from the fireplace's own origin, so this is about facing and not about where
    // in the room either of them sits.
    let toward_hearth = (hearth - origin).truncate().normalize();
    let toward_fire = (fire.position - origin).truncate().normalize();
    let agreement = toward_hearth.dot(toward_fire);
    assert!(
        agreement > 0.9,
        "the hearth points {toward_hearth:?} and the fire sits {toward_fire:?} — the stack is \
         turned away from its own fire"
    );
}

#[test]
fn a_book_stands_on_the_same_shelf_as_the_cups_beside_it() {
    // **Which Euler angle is applied first is not something the file says**, and it moves only the
    // references that turn about more than one axis — 22 of this cell's 268. The rest of a room is
    // unmoved either way, so the mistake hides among the props: this book ends up sunk through the
    // board it stands on and out through the side of the cupboard.
    //
    // The cups beside it turn about Z alone and so land in the same place under either order, which
    // makes them a ruler. Two things on one shelf rest on one board.
    let Some(bytes) = game_bytes() else { return };
    let content = Content::open(&bytes);
    let scene = content.cell(&CellId::Interior(CELL.into()));

    // The lowest point of every instance of a named model, near the cupboard at the far end.
    let bases_of = |needle: &str| -> Vec<f32> {
        scene
            .instances
            .iter()
            .filter(|instance| {
                scene.mesh_sources[instance.mesh.0 as usize]
                    .to_lowercase()
                    .contains(needle)
                    && (glam::Vec3::from(instance.transform.translation)
                        - glam::Vec3::new(420.0, 976.0, 300.0))
                    .length()
                        < 120.0
            })
            .map(|instance| {
                scene.meshes[instance.mesh.0 as usize]
                    .positions
                    .iter()
                    .map(|p| instance.transform.transform_point3(*p).z)
                    .fold(f32::INFINITY, f32::min)
            })
            .collect()
    };

    let books = bases_of("text_octavo_03");
    let cups = bases_of("misc_com_metal_goblet_01");
    assert_eq!(books.len(), 1, "expected the one book on this cupboard");
    assert!(cups.len() >= 4, "expected the row of cups beside it");

    let board = cups.iter().copied().fold(f32::INFINITY, f32::min);
    let book = books[0];
    assert!(
        book > board - 2.0,
        "the book's base is at {book} and the shelf it shares with the cups at {board} — it is \
         standing through the board"
    );
}
