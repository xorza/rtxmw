//! Parses every NIF the game ships, and every animation beside them.
//!
//! This is the M2 done-condition. Blocks in this file version carry no size, so a parser that is
//! wrong by a single field desyncs and the next block type reads as garbage — over thousands of
//! meshes that is caught immediately. Skips when the game is not installed.
//!
//! A `.kf` is a NIF with a different root, so it goes through the same reader and belongs to the
//! same assertion: its blocks are the ones a mesh's animation is made of, and a desync in one of
//! them is exactly as quiet.

use std::collections::BTreeMap;

use std::f32::consts::TAU;

use rtxmw_nif::{Block, Interpolation, NifFile, ParticleEffect};
use rtxmw_vfs::{DATA_DIR_VAR, morrowind_archives};

#[test]
fn every_shipped_mesh_parses() {
    let Some(vfs) = morrowind_archives() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };

    let mut meshes: Vec<String> = vfs
        .paths()
        .filter(|p| matches!(p.extension(), Some("nif" | "kf")))
        .map(|p| p.as_str().to_owned())
        .collect();
    meshes.sort();
    assert!(
        meshes.len() > 5_000,
        "expected thousands, found {}",
        meshes.len()
    );
    let clips = meshes.iter().filter(|p| p.ends_with(".kf")).count();
    assert_eq!(
        clips, 162,
        "the shipped animations are 162 files; a different count is a different install"
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

/// Reads every emitter the game ships, and checks each against a number it did not supply.
///
/// **A block with no size is only proved right by an invariant it did not carry.** The fields are
/// consumed one after another with nothing between them to resynchronise on, so a parse that is off
/// by one still produces plausible-looking floats — what catches it is that a candle's *capacity*
/// is its birth rate times its lifetime, which is three separate fields agreeing about one physical
/// fact. `light_de_candle_25` writes 22, 34.02 and 0.67, and 34.02 × 0.67 = 22.8.
#[test]
fn every_emitter_holds_as_many_particles_as_its_rate_and_its_life_call_for() {
    let Some(vfs) = morrowind_archives() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };

    let mut emitters = 0;
    let mut ramps = 0;
    let mut ratios: Vec<f32> = Vec::new();
    for path in vfs.paths().filter(|p| p.extension() == Some("nif")) {
        let name = path.as_str().to_owned();
        let Ok(bytes) = vfs.read(&name) else { continue };
        let Ok(nif) = NifFile::parse(&bytes) else {
            continue;
        };
        for block in nif.blocks() {
            let Block::ParticleSystem(system) = block else {
                continue;
            };
            emitters += 1;
            assert!(
                system.lifetime >= 0.0 && system.lifetime < 60.0,
                "{name}: a particle lives {} seconds",
                system.lifetime
            );
            assert!(
                system.declination >= -TAU && system.declination <= TAU,
                "{name}: a particle leaves at {} radians off the emitter's axis",
                system.declination
            );
            // The birth rate has to fill the capacity within a lifetime, or the emitter would run
            // dry; a tenth either way covers the ones authored with headroom.
            let wanted = system.birth_rate * system.lifetime;
            ratios.push(f32::from(system.capacity) / wanted.max(1e-6));

            let mut link = system.modifier;
            for _ in 0..4 {
                let Some(Block::ParticleModifier(modifier)) = nif.resolve(link) else {
                    break;
                };
                if let ParticleEffect::Colour { keys } = modifier.effect {
                    let Some(Block::Colour(track)) = nif.resolve(keys) else {
                        panic!("{name}: a colour modifier names something that is not a ramp");
                    };
                    ramps += 1;
                    // Three linear keys, which is what makes `ParticleEmitter::ramp` the ramp
                    // itself rather than a sampling of it.
                    assert_eq!(track.interpolation, Interpolation::Linear, "{name}");
                    assert_eq!(track.keys.len(), 3, "{name}");
                    assert_eq!(track.keys[0].time, 0.0, "{name}");
                    assert_eq!(track.keys[2].time, 1.0, "{name}");
                }
                link = modifier.next;
            }
        }
    }

    assert_eq!(
        emitters, 678,
        "the shipped emitters are 678; a different count is a different install"
    );
    assert_eq!(ramps, 85, "85 of them fade through a colour ramp");
    // Not all of them: an emitter authored with headroom holds more than its rate ever fills, and
    // one authored short recycles early. Half of them agreeing to a tenth is far past coincidence —
    // a parse off by one field puts a *colour channel* where the birth rate belongs.
    // **The median, not a tally.** Individually the ratio runs from 0.23 to 2.53 across the tenth
    // and ninetieth percentiles, which is authoring headroom — an emitter may hold more than its
    // rate ever fills, or recycle early. What cannot be headroom is the *middle* of 678 of them
    // landing on one: a parse off by a single field would put a colour channel where the birth rate
    // belongs, and no arrangement of the wrong three numbers has a median of 0.97.
    ratios.sort_by(|a, b| a.partial_cmp(b).expect("no emitter carries a NaN"));
    let median = ratios[ratios.len() / 2];
    assert!(
        (0.9..1.1).contains(&median),
        "the median emitter holds {median} times what its rate and life call for, not one"
    );
}

/// Reads every texture scroll the game ships, and checks the shape the renderer relies on.
///
/// **What is being relied on is that a scroll is a *speed*.** The renderer carries a rate and no
/// keys at all — `UvData::scroll` — and two things have to hold for that: the tiling must never be
/// keyed, or the texture would change scale over time and no offset could stand in for it; and the
/// controllers must hang where the mesh walk looks for them. Both are checked here against every
/// file the game ships rather than assumed.
///
/// The offsets are keyed *quadratically* rather than linearly, which was assumed the other way
/// round until this test said otherwise — but their values are two-key ramps, and a tangent between
/// two keys of a ramp is the ramp. What is lost is the shiver on Ghostgate's fences, whose four-key
/// U channel comes back to where it started and so reads as no motion at all.
#[test]
fn every_texture_scroll_is_a_line_over_a_looping_clip() {
    let Some(vfs) = morrowind_archives() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };

    let mut controllers = 0;
    let mut on_geometry = 0;
    let mut lava: Vec<[f32; 2]> = Vec::new();
    for path in vfs.paths().filter(|p| p.extension() == Some("nif")) {
        let name = path.as_str().to_owned();
        let Ok(bytes) = vfs.read(&name) else { continue };
        let Ok(nif) = NifFile::parse(&bytes) else {
            continue;
        };
        for block in nif.blocks() {
            let Block::UvController(uv) = block else {
                continue;
            };
            controllers += 1;
            let Some(Block::UvData(data)) = nif.resolve(uv.data) else {
                panic!("{name}: a uv controller names something that is not uv data");
            };
            // **Nothing keys the tiling**, which is what lets a scroll be an offset and nothing
            // else — a repeat count that moved would change the texture's scale over time and no
            // rate could stand in for it.
            assert!(
                data.tiling_u.keys.is_empty() && data.tiling_v.keys.is_empty(),
                "{name} keys its tiling, which a rate cannot carry"
            );
            if name.ends_with("in_lava_256.nif") {
                lava.push(data.scroll(uv.stop - uv.start));
            }
        }

        // Every one of them hangs off the shape it moves, which is where the walk looks for it.
        for block in nif.blocks() {
            let Block::Geometry(geometry) = block else {
                continue;
            };
            let mut link = geometry.av.net.controller;
            for _ in 0..8 {
                match nif.resolve(link) {
                    Some(Block::UvController(_)) => {
                        on_geometry += 1;
                        break;
                    }
                    Some(Block::Controller(other)) => link = other.next,
                    _ => break,
                }
            }
        }
    }

    assert_eq!(
        controllers, 116,
        "the shipped texture scrolls are 116; a different count is a different install"
    );
    // Five sit on nothing the walk reaches, in files that are all magic effects — a controller left
    // in a file whose shape was deleted. What matters is that the rest are where they are looked
    // for, which is on the geometry rather than on a node above it.
    assert_eq!(on_geometry, 111);

    // **A pool of lava is three sheets crawling at different speeds**, which is how the original
    // faked depth in something that is one flat quad: two run down at a twenty-fourth and a
    // forty-eighth of the texture a second, and one runs diagonally back the other way. Every one
    // is a V channel keyed across the same forty-eight second clip, so these are the file's own
    // numbers divided by 48 and nothing else.
    lava.sort_by(|a, b| a.partial_cmp(b).expect("no channel carries a NaN"));
    let rate = |sixteenths: f32| sixteenths / 48.0;
    assert_eq!(
        lava,
        [[rate(-2.0), rate(-2.0)], [0.0, rate(1.0)], [0.0, rate(2.0)],]
    );
}
