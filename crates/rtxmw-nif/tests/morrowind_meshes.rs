//! Parses every NIF the game ships.
//!
//! This is the M2 done-condition. Blocks in this file version carry no size, so a parser that is
//! wrong by a single field desyncs and the next block type reads as garbage — over thousands of
//! meshes that is caught immediately. Skips when the game is not installed.

use std::collections::BTreeMap;

use rtxmw_nif::{Block, NifFile};
use rtxmw_vfs::{DATA_DIR_VAR, morrowind_archives};

#[test]
fn every_shipped_mesh_parses() {
    let Some(vfs) = morrowind_archives() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };

    let mut meshes: Vec<String> = vfs
        .paths()
        .filter(|p| p.extension() == Some("nif"))
        .map(|p| p.as_str().to_owned())
        .collect();
    meshes.sort();
    assert!(
        meshes.len() > 5_000,
        "expected thousands, found {}",
        meshes.len()
    );

    let mut parsed = 0usize;
    let mut triangles = 0usize;
    let mut vertices = 0usize;
    // Grouped so one missing block type reports once with a count, not thousands of times.
    let mut failures: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for path in &meshes {
        let bytes = match vfs.read(path) {
            Ok(bytes) => bytes,
            Err(e) => {
                failures
                    .entry(format!("could not read: {e}"))
                    .or_default()
                    .push(path.clone());
                continue;
            }
        };
        match NifFile::parse(&bytes) {
            Ok(nif) => {
                parsed += 1;
                triangles += nif.triangle_count();
                vertices += nif.vertex_count();
            }
            Err(e) => {
                // Collapse the variable parts so the same fault groups together.
                let key = match &e {
                    rtxmw_nif::NifError::UnknownBlock { kind, .. } => {
                        format!("unknown block type {kind:?}")
                    }
                    rtxmw_nif::NifError::UnsupportedVersion { version } => {
                        format!("unsupported version {version:#010x}")
                    }
                    rtxmw_nif::NifError::InBlock { kind, source, .. } => {
                        let what = match **source {
                            rtxmw_nif::NifError::UnexpectedEnd { .. } => "ran past the end",
                            _ => "malformed",
                        };
                        format!("{kind} {what}")
                    }
                    other => format!("{other:?}")
                        .split_whitespace()
                        .next()
                        .unwrap_or("error")
                        .to_owned(),
                };
                failures.entry(key).or_default().push(path.clone());
            }
        }
    }

    println!(
        "parsed {parsed} of {} meshes: {triangles} triangles, {vertices} vertices",
        meshes.len()
    );
    for (reason, paths) in &failures {
        println!("  {} x {reason}   e.g. {}", paths.len(), paths[0]);
    }

    assert!(
        failures.is_empty(),
        "{} of {} meshes failed to parse",
        failures.values().map(Vec::len).sum::<usize>(),
        meshes.len()
    );
}

#[test]
fn geometry_indices_stay_inside_their_vertex_buffers() {
    let Some(vfs) = morrowind_archives() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };

    let mut checked = 0usize;
    let mut with_geometry = 0usize;

    for path in vfs.paths().filter(|p| p.extension() == Some("nif")) {
        let Ok(bytes) = vfs.read(path.as_str()) else {
            continue;
        };
        let Ok(nif) = NifFile::parse(&bytes) else {
            continue;
        };

        for block in nif.blocks() {
            let Block::GeometryData(data) = block else {
                continue;
            };
            if data.triangles.is_empty() {
                continue;
            }
            with_geometry += 1;
            let limit = data.vertices.len();
            // An index past the vertex buffer is the signature of a desynced parse: the triangle
            // list was read from the wrong offset and is really some other block's bytes.
            for triangle in &data.triangles {
                for &index in triangle {
                    assert!(
                        (index as usize) < limit,
                        "{path}: index {index} with only {limit} vertices",
                    );
                }
            }
            // Every UV set must cover the same vertices.
            for set in &data.uv_sets {
                assert_eq!(
                    set.len(),
                    limit,
                    "{path}: a UV set has {} entries for {limit} vertices",
                    set.len()
                );
            }
            if !data.normals.is_empty() {
                assert_eq!(data.normals.len(), limit, "{path}: normal count mismatch");
            }
        }
        checked += 1;
    }

    assert!(checked > 5_000, "only checked {checked} meshes");
    assert!(with_geometry > 5_000, "only {with_geometry} had triangles");
    println!("validated {with_geometry} geometry blocks across {checked} meshes");
}

#[test]
fn the_transform_on_a_models_outermost_node_is_discarded() {
    // **The original engine ignores it**, and a renderer that does not turns those models against
    // everything around them. Seyda Neen's fireplace is the case that shows it: its outermost node
    // carries a half turn about Z, and honoured, the hearth faces the wall while the fire the cell
    // places burns behind the stack.
    //
    // A rig's `bip01` is the exception — that is an animation root the skeleton is posed against,
    // not a stray placement — and so is a model whose outermost block is geometry rather than a
    // node.
    let Some(vfs) = morrowind_archives() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };
    const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let root_of = |path: &str| {
        let bytes = vfs
            .read(path)
            .unwrap_or_else(|_| panic!("{path} should read"));
        let nif = NifFile::parse(&bytes).unwrap_or_else(|_| panic!("{path} should parse"));
        match nif.blocks().first().expect("a first block") {
            Block::Node(node) => (node.av.net.name.clone(), node.av.transform),
            Block::Geometry(geometry) => (geometry.av.net.name.clone(), geometry.av.transform),
            other => panic!("{path} starts with {other:?}"),
        }
    };

    // The fireplace, whose file really does carry a half turn there.
    let (name, transform) = root_of("meshes/i/in_nord_fireplace_01.nif");
    assert_eq!(name, "In_nord_fireplace_01");
    assert_eq!(
        transform.rotation, IDENTITY,
        "the outermost node's half turn survived the read"
    );

    // A rig keeps its animation root. 259 of the shipped models are rooted at `bip01` and 255 of
    // those carry a real transform there — it is what the skeleton is posed against, so discarding
    // it would flatten every piece of armour and clothing in the game.
    let (name, transform) = root_of("meshes/a/a_m_imperialchain_cuirass.nif");
    assert!(name.eq_ignore_ascii_case("bip01"));
    assert_ne!(
        transform.rotation, IDENTITY,
        "a rig's animation root was discarded along with the stray placements"
    );

    // Nothing was thrown away wholesale: across the shipped library the rule reaches only the first
    // block, so the models that carry a turn deeper down still have it.
    let mut deeper = 0;
    for path in vfs.paths().filter(|p| p.extension() == Some("nif")) {
        let name = path.as_str().to_owned();
        let Ok(bytes) = vfs.read(&name) else { continue };
        let Ok(nif) = NifFile::parse(&bytes) else {
            continue;
        };
        if nif.blocks().iter().skip(1).any(|block| match block {
            Block::Node(node) => node.av.transform.rotation != IDENTITY,
            Block::Geometry(geometry) => geometry.av.transform.rotation != IDENTITY,
            _ => false,
        }) {
            deeper += 1;
        }
    }
    assert!(
        deeper > 500,
        "only {deeper} models keep a turn below their outermost node"
    );
}
