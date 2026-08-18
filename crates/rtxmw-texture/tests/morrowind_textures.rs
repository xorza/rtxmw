//! Decodes every texture the game ships.
//!
//! The M4a done-condition. A texture decoder fails quietly by nature — a wrong offset or a
//! miscounted mip chain yields bytes that upload without complaint and render as noise — so what is
//! checked here is that every declared level is present, sized as the format demands, and packed
//! end to end. Skips when the game is not installed.

use std::collections::BTreeMap;

use rtxmw_texture::Texture;
use rtxmw_vfs::{DATA_DIR_VAR, morrowind_archives};

/// Magic plus the Direct3D 9 header, before any pixel data.
const DDS_HEADER: usize = 128;

#[test]
fn every_shipped_texture_decodes() {
    let Some(vfs) = morrowind_archives() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };

    let mut paths: Vec<String> = vfs
        .paths()
        .filter(|p| matches!(p.extension(), Some("dds" | "tga")))
        .map(|p| p.as_str().to_owned())
        .collect();
    paths.sort();
    assert!(
        paths.len() > 5_000,
        "expected thousands, found {}",
        paths.len()
    );

    let mut by_format: BTreeMap<String, usize> = BTreeMap::new();
    let mut failures: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut total_bytes = 0usize;
    let mut single_level = 0usize;

    for path in &paths {
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
        let texture = match Texture::decode(&bytes) {
            Ok(texture) => texture,
            Err(e) => {
                failures
                    .entry(format!("{e}"))
                    .or_default()
                    .push(path.clone());
                continue;
            }
        };

        *by_format
            .entry(format!("{:?}", texture.format()))
            .or_default() += 1;
        total_bytes += texture.data().len();
        if texture.levels().len() == 1 {
            single_level += 1;
        }

        // The level table must tile the buffer with no gap and no overlap, and each entry must be
        // exactly the size its dimensions imply. A chain that merely fits would still be wrong.
        let mut cursor = 0u32;
        for (index, level) in texture.levels().iter().enumerate() {
            assert_eq!(
                level.offset, cursor,
                "{path}: level {index} starts at {} not {cursor}",
                level.offset
            );
            assert_eq!(
                level.size,
                texture.format().level_size(level.width, level.height),
                "{path}: level {index} is {}x{} but sized {}",
                level.width,
                level.height,
                level.size
            );
            assert_eq!(texture.level_data(index).len(), level.size as usize);
            cursor += level.size;
        }
        assert_eq!(
            cursor as usize,
            texture.data().len(),
            "{path}: levels cover {cursor} of {} bytes",
            texture.data().len()
        );

        // Every byte of a DDS after its header belongs to the chain. Checking only that the levels
        // tile the buffer is not enough: dropping a level leaves a table that still tiles, just a
        // shorter one, and the texture renders at the wrong mip forever. This is the assertion that
        // ties the decode back to the file.
        if path.ends_with(".dds") {
            let consumed = DDS_HEADER + texture.data().len();
            assert_eq!(
                consumed,
                bytes.len(),
                "{path}: decoded {} of {} bytes across {} levels",
                consumed,
                bytes.len(),
                texture.levels().len()
            );
        }

        // The chain must descend.
        for pair in texture.levels().windows(2) {
            assert!(
                pair[1].width <= pair[0].width && pair[1].height <= pair[0].height,
                "{path}: mip chain grows"
            );
        }
    }

    println!(
        "decoded {} of {} textures, {:.1} MiB, {single_level} with a single level",
        paths.len() - failures.values().map(Vec::len).sum::<usize>(),
        paths.len(),
        total_bytes as f64 / (1024.0 * 1024.0),
    );
    for (format, count) in &by_format {
        println!("  {count:6} x {format}");
    }
    for (reason, bad) in &failures {
        println!("  {} x {reason}   e.g. {}", bad.len(), bad[0]);
    }

    assert!(
        failures.is_empty(),
        "{} of {} textures failed to decode",
        failures.values().map(Vec::len).sum::<usize>(),
        paths.len()
    );

    // The bindless array's format set is chosen from exactly this census, so it is pinned here. A
    // replacer pack introducing BC3 would otherwise be sampled as BC1 and render as noise.
    let formats: Vec<&str> = by_format.keys().map(String::as_str).collect();
    assert_eq!(
        formats,
        vec!["Bc1", "Bc2", "Bgra8", "Rgba8"],
        "the shipped format set changed"
    );
    // BC1 dominating is what makes the memory budget work, and it is the format carrying the
    // one-bit alpha the alpha test at M4d keys on.
    assert!(by_format["Bc1"] > by_format["Bc2"], "{by_format:?}");
    assert!(
        by_format["Bc1"] + by_format["Bc2"] > 20 * (by_format["Bgra8"] + by_format["Rgba8"]),
        "uncompressed textures are supposed to be a rounding error: {by_format:?}"
    );
}
