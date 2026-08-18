//! Reads the real Morrowind content files and resolves what they place.
//!
//! This is the M1 done-condition: given a cell, produce its references and the mesh each one needs,
//! with every mesh actually present in the virtual file system. Skips with a note when
//! the game is not installed (see `.env`).

use std::collections::HashMap;

use rtxmw_esm::{Cell, CellId, CellIndex, EsmReader, LandRecord, ObjectRecord, RecordName};
use rtxmw_vfs::{DATA_DIR_VAR, Vfs, morrowind_data_dir};

fn morrowind_esm() -> Option<Vec<u8>> {
    let path = morrowind_data_dir()?.join("Morrowind.esm");
    std::fs::read(path).ok()
}

#[test]
fn header_names_the_expected_master_file() {
    let Some(bytes) = morrowind_esm() else {
        eprintln!("skipping: Morrowind.esm is not available");
        return;
    };
    let esm = EsmReader::new(&bytes).expect("Morrowind.esm should parse");
    let header = esm.header();

    assert_eq!(header.kind, rtxmw_esm::FileKind::Master);
    assert!(
        header.author.to_lowercase().contains("bethesda"),
        "unexpected author {:?}",
        header.author
    );
    // The base game depends on nothing.
    assert!(header.masters.is_empty());
    assert!(header.record_count > 40_000, "{}", header.record_count);

    println!(
        "version {:.2}, author {:?}, {} records",
        header.version, header.author, header.record_count
    );
}

#[test]
fn the_record_count_in_the_header_matches_what_is_there() {
    let Some(bytes) = morrowind_esm() else {
        eprintln!("skipping: Morrowind.esm is not available");
        return;
    };
    let esm = EsmReader::new(&bytes).expect("should parse");

    let mut counted = 0usize;
    for record in esm.records() {
        record.expect("every record should parse");
        counted += 1;
    }

    // The writer's own count is an independent check that traversal neither skipped a record nor
    // desynchronised and invented one.
    assert_eq!(
        counted,
        esm.header().record_count as usize,
        "walked {counted} records, header says {}",
        esm.header().record_count
    );
    println!("walked all {counted} records");
}

#[test]
fn a_known_interior_resolves_every_reference_to_a_mesh_in_the_vfs() {
    let Some(data) = morrowind_data_dir() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };
    let Some(bytes) = morrowind_esm() else {
        eprintln!("skipping: Morrowind.esm is not available");
        return;
    };

    let mut vfs = Vfs::new();
    vfs.add_bsa(&data.join("Morrowind.bsa"))
        .expect("Morrowind.bsa should open");
    vfs.add_directory(&data).expect("loose files should index");

    let esm = EsmReader::new(&bytes).expect("should parse");

    // One pass: collect every placeable record's model, and find the target cell.
    let mut models: HashMap<String, String> = HashMap::new();
    let mut target = None;
    let wanted = "Seyda Neen, Census and Excise Office";

    for record in esm.records() {
        let record = record.expect("record should parse");
        if record.is_deleted() || record.is_ignored() {
            continue;
        }
        if ObjectRecord::is_placeable(&record) {
            let object = ObjectRecord::parse(&record).expect("object should parse");
            if let Some(path) = object.model_path() {
                models.insert(object.id.to_lowercase(), path);
            }
            continue;
        }
        if record.name() == RecordName::new(b"CELL") {
            let cell = Cell::parse(&record).expect("cell should parse");
            if cell.name == wanted {
                target = Some(record);
            }
        }
    }

    assert!(
        models.len() > 2_000,
        "expected thousands of placeable records, found {}",
        models.len()
    );

    let record = target.unwrap_or_else(|| panic!("{wanted} should exist in Morrowind.esm"));
    let cell = Cell::parse(&record).unwrap();
    assert!(cell.is_interior());
    assert_eq!(cell.id(), CellId::Interior(wanted.to_owned()));

    let mut placed = 0;
    let mut resolved = 0;
    let mut missing_model = Vec::new();
    let mut missing_asset = Vec::new();

    for cell_ref in Cell::references(&record) {
        let cell_ref = cell_ref.expect("reference should parse");
        if cell_ref.deleted {
            continue;
        }
        placed += 1;
        assert!(
            !cell_ref.object_id.is_empty(),
            "reference {} has no object id",
            cell_ref.refnum
        );
        assert!(
            cell_ref.scale > 0.0,
            "reference {} has a non-positive scale {}",
            cell_ref.refnum,
            cell_ref.scale
        );

        match models.get(&cell_ref.object_id.to_lowercase()) {
            Some(path) => {
                if vfs.contains(path) {
                    resolved += 1;
                } else {
                    missing_asset.push(path.clone());
                }
            }
            // Not every placeable record has a mesh — a few carry none at all.
            None => missing_model.push(cell_ref.object_id.clone()),
        }
    }

    println!(
        "{wanted}: {placed} references, {resolved} meshes resolved, \
         {} without a model, {} model paths absent from the VFS",
        missing_model.len(),
        missing_asset.len()
    );

    assert!(placed > 20, "expected a furnished interior, got {placed}");
    assert!(
        missing_asset.is_empty(),
        "these model paths are not in the VFS: {missing_asset:?}"
    );
    assert!(
        resolved * 10 >= placed * 9,
        "only {resolved} of {placed} references resolved to a mesh"
    );
}

#[test]
fn cells_partition_into_named_interiors_and_gridded_exteriors() {
    let Some(bytes) = morrowind_esm() else {
        eprintln!("skipping: Morrowind.esm is not available");
        return;
    };
    let esm = EsmReader::new(&bytes).expect("should parse");

    let mut interiors = 0;
    let mut exteriors = 0;
    let mut total_refs = 0usize;

    for record in esm.records() {
        let record = record.expect("record should parse");
        if record.name() != RecordName::new(b"CELL") {
            continue;
        }
        let cell = Cell::parse(&record).expect("cell should parse");
        if cell.is_interior() {
            interiors += 1;
            // An interior without a name could never be addressed.
            assert!(!cell.name.is_empty(), "unnamed interior cell");
        } else {
            exteriors += 1;
        }
        for cell_ref in Cell::references(&record) {
            cell_ref.expect("reference should parse");
            total_refs += 1;
        }
    }

    assert!(
        interiors > 1_000,
        "expected many interiors, got {interiors}"
    );
    assert!(exteriors > 200, "expected many exteriors, got {exteriors}");
    println!("{interiors} interiors, {exteriors} exteriors, {total_refs} references in total");
}

#[test]
fn the_index_finds_every_cell_a_walk_finds_and_points_straight_at_it() {
    let Some(bytes) = morrowind_esm() else {
        eprintln!("skipping: Morrowind.esm is not available");
        return;
    };
    let esm = EsmReader::new(&bytes).expect("should parse");

    // What walking the file finds, which is what the index has to agree with. Later records win,
    // the same way the index resolves them.
    let mut walked: HashMap<CellId, usize> = HashMap::new();
    let mut lands: HashMap<CellId, usize> = HashMap::new();
    for record in esm.records() {
        let record = record.expect("record should parse");
        if record.name() == RecordName::new(b"CELL") {
            let cell = Cell::parse(&record).expect("cell should parse");
            let id = if cell.is_interior() {
                CellId::Interior(cell.name)
            } else {
                CellId::Exterior {
                    x: cell.grid_x,
                    y: cell.grid_y,
                }
            };
            walked.insert(id, record.offset());
        } else if record.name() == RecordName::new(b"LAND")
            && let Some(id) = LandRecord::grid_of(&record).expect("coordinates should parse")
        {
            lands.insert(id, record.offset());
        }
    }

    let index = CellIndex::build(&esm).expect("index should build");
    assert_eq!(index.len(), walked.len());
    for (id, offset) in &walked {
        let found = index.cell(id).expect("indexed cell should be found");
        assert_eq!(found.cell, *offset, "{id} is filed at the wrong offset");
        assert_eq!(
            found.land,
            lands.get(id).copied(),
            "{id} has the wrong land"
        );

        // The offset has to be usable, not merely equal: reading it back must produce the record
        // it was recorded from.
        let record = esm.record_at(found.cell).expect("record should re-read");
        assert_eq!(record.name(), RecordName::new(b"CELL"));
    }

    // Seyda Neen's shore, where the game opens, has terrain; the palette its tiles index is the
    // file's own and is shared by every cell in it.
    let shore = CellId::Exterior { x: -2, y: -9 };
    assert!(index.cell(&shore).expect("Seyda Neen").land.is_some());
    assert!(
        index.land_textures().len() > 100,
        "only {} land textures",
        index.land_textures().len()
    );
    println!(
        "{} cells indexed, {} with terrain, {} land textures",
        index.len(),
        lands.len(),
        index.land_textures().len()
    );
}
