//! Parses every NIF the game ships.
//!
//! This is the M2 done-condition. Blocks in this file version carry no size, so a parser that is
//! wrong by a single field desyncs and the next block type reads as garbage — over thousands of
//! meshes that is caught immediately. Skips when the game is not installed.

use std::collections::BTreeMap;

use rtxmw_nif::{Block, NifFile};
use rtxmw_vfs::{DATA_DIR_VAR, Vfs, morrowind_data_dir};

/// Indexes every archive the game ships, in load order.
fn game_files() -> Option<Vfs> {
    let data = morrowind_data_dir()?;
    let mut vfs = Vfs::new();
    for archive in ["Morrowind.bsa", "Tribunal.bsa", "Bloodmoon.bsa"] {
        let path = data.join(archive);
        if path.is_file() {
            vfs.add_bsa(&path)
                .unwrap_or_else(|e| panic!("could not open {archive}: {e}"));
        }
    }
    vfs.add_directory(&data).expect("loose files should index");
    Some(vfs)
}

#[test]
fn every_shipped_mesh_parses() {
    let Some(vfs) = game_files() else {
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
    let Some(vfs) = game_files() else {
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
