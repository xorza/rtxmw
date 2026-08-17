//! Reads the real Morrowind archives, when they are available.
//!
//! Synthetic archives prove the parser handles the format as documented; only the shipped data
//! proves it handles the format as Bethesda actually wrote it. Skips with a note when
//! `MORROWIND_DATA_DIR` is unset, so the suite still runs on a machine without the game.

use std::path::PathBuf;

use rtxmw_vfs::Vfs;

/// The `Data Files` directory, or `None` when the game is not available here.
fn data_dir() -> Option<PathBuf> {
    let raw = std::env::var_os("MORROWIND_DATA_DIR")?;
    let path = PathBuf::from(raw);
    path.is_dir().then_some(path)
}

#[test]
fn reads_the_shipped_archives() {
    let Some(data) = data_dir() else {
        eprintln!("skipping: MORROWIND_DATA_DIR is not set to a directory");
        return;
    };

    let mut vfs = Vfs::new();
    // Load order: the expansions override the base game, loose files override everything.
    for archive in ["Morrowind.bsa", "Tribunal.bsa", "Bloodmoon.bsa"] {
        let path = data.join(archive);
        if path.is_file() {
            vfs.add_bsa(&path)
                .unwrap_or_else(|e| panic!("could not open {archive}: {e}"));
        }
    }
    vfs.add_directory(&data)
        .expect("could not index loose files");

    // Morrowind.bsa alone holds thousands of entries; anything far below this means the directory
    // was misparsed rather than merely differing between installs.
    assert!(
        vfs.len() > 10_000,
        "expected well over 10k paths, indexed {}",
        vfs.len()
    );

    // A mesh every install has, exercised through a spelling no archive actually stores.
    let known = r"Meshes\Base_Anim.NIF";
    assert!(vfs.contains(known), "{known} should resolve");
    let bytes = vfs.read(known).expect("could not read the mesh");
    assert!(!bytes.is_empty());
    // Every Morrowind NIF starts with this signature line.
    assert!(
        bytes.starts_with(b"NetImmerse File Format"),
        "expected a NIF header, got {:?}",
        &bytes[..bytes.len().min(24)]
    );

    // Extension distribution is a cheap check that names decoded sensibly rather than as noise.
    let nifs = vfs.paths().filter(|p| p.extension() == Some("nif")).count();
    let textures = vfs
        .paths()
        .filter(|p| matches!(p.extension(), Some("dds") | Some("tga")))
        .count();
    assert!(nifs > 1_000, "expected thousands of meshes, found {nifs}");
    assert!(
        textures > 1_000,
        "expected thousands of textures, found {textures}"
    );

    // Directory filtering must agree with the extension scan.
    let under_meshes = vfs.paths_under("meshes").count();
    assert!(
        under_meshes >= nifs,
        "meshes/ holds {under_meshes} paths but {nifs} .nif files were found overall"
    );

    println!(
        "indexed {} paths: {nifs} meshes, {textures} textures",
        vfs.len()
    );
}

#[test]
fn every_indexed_path_in_morrowind_bsa_reads_back_at_its_stated_size() {
    let Some(data) = data_dir() else {
        eprintln!("skipping: MORROWIND_DATA_DIR is not set to a directory");
        return;
    };
    let path = data.join("Morrowind.bsa");
    if !path.is_file() {
        eprintln!("skipping: {} is absent", path.display());
        return;
    }

    let archive = rtxmw_vfs::BsaArchive::open(&path).expect("could not open Morrowind.bsa");
    assert!(!archive.is_empty());

    // Reading all 300 MB would dominate the suite, so sample across the whole directory instead —
    // a stride catches a systematically wrong data offset, which is the failure that matters.
    let stride = (archive.len() / 200).max(1);
    let mut checked = 0;
    for index in (0..archive.len()).step_by(stride) {
        let expected = archive.entry_size(index) as usize;
        let bytes = archive.read(index).unwrap_or_else(|e| {
            panic!("entry {index} ({}) failed: {e}", archive.entry_name(index))
        });
        assert_eq!(
            bytes.len(),
            expected,
            "entry {index} ({}) read back {} bytes, directory says {expected}",
            archive.entry_name(index),
            bytes.len()
        );
        checked += 1;
    }
    assert!(checked >= 100, "only sampled {checked} entries");
    println!("verified {checked} of {} entries", archive.len());
}
