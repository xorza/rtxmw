//! Decodes every `LAND` record in the shipped game.
//!
//! The synthetic tests round-trip the encoding against itself, which proves the decoder consistent
//! but not correct: a transposed grid or a mis-signed delta agrees with its own inverse perfectly.
//! What catches those is the shape of Vvardenfell.

use rtxmw_esm::{CellId, DEFAULT_HEIGHT, EsmReader, GRID, LandRecord, RecordName, VERTICES};
use rtxmw_vfs::{DATA_DIR_VAR, morrowind_data_dir};

#[test]
fn every_shipped_cell_decodes_into_plausible_terrain() {
    let Some(data) = morrowind_data_dir() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };
    let bytes = std::fs::read(data.join("Morrowind.esm")).expect("Morrowind.esm should read");
    let esm = EsmReader::new(&bytes).expect("should parse");
    let land_tag = RecordName::new(b"LAND");

    let mut cells = 0usize;
    let mut bare = 0usize;
    let mut with_textures = 0usize;
    let mut lowest = f32::MAX;
    let mut highest = f32::MIN;
    let mut worst_step = 0.0f32;
    let mut below_sea = 0usize;

    for record in esm.records() {
        let record = record.expect("record should parse");
        if record.name() != land_tag {
            continue;
        }
        // The cheap read the cell index uses must agree with the full parse on every record in
        // the game — it is the same eight bytes, and a cell filed under the wrong coordinates
        // would hand out the wrong terrain to whoever asked.
        let cheap = LandRecord::grid_of(&record).expect("coordinates should parse");
        let Some(land) = LandRecord::parse(&record).expect("land should parse") else {
            bare += 1;
            continue;
        };
        assert_eq!(
            cheap,
            Some(CellId::Exterior {
                x: land.grid_x,
                y: land.grid_y
            })
        );
        cells += 1;
        assert_eq!(land.heights.len(), VERTICES);
        if !land.textures.is_empty() {
            with_textures += 1;
        }

        for (index, &height) in land.heights.iter().enumerate() {
            assert!(
                height.is_finite(),
                "cell {:?} has a non-finite vertex",
                (land.grid_x, land.grid_y)
            );
            lowest = lowest.min(height);
            highest = highest.max(height);
            // Steepness between neighbours, which is what a mis-decoded delta destroys: read as
            // absolute values the surface becomes noise, and adjacent vertices leap thousands of
            // units apart.
            if index % GRID != 0 {
                worst_step = worst_step.max((height - land.heights[index - 1]).abs());
            }
        }
        if land.height(32, 32) < 0.0 {
            below_sea += 1;
        }
    }

    println!(
        "{cells} land cells ({with_textures} textured) and {bare} bare, heights {lowest:.0} to \
         {highest:.0}, worst neighbour step {worst_step:.0}, {below_sea} centred below sea level"
    );

    // Vvardenfell is about 1,300 cells of terrain, and a further hundred-odd `LAND` records carry
    // no terrain data at all — only coordinates and a flag word. Those are cells that exist for
    // their objects, and treating them as a parse failure would drop 7% of the world's records on
    // the floor without saying so.
    assert!(cells > 1_200, "only {cells} land cells");
    assert!(
        bare > 50,
        "the records carrying no terrain have stopped being recognised"
    );
    assert!(
        with_textures > cells * 9 / 10,
        "only {with_textures} of {cells} cells carry texture indices"
    );

    // Sea level is zero and Red Mountain is the island's high point. Nothing should sit below the
    // default height a cell without terrain gets, and the peak is thousands of units up but not
    // tens of thousands — those are the two ways a broken delta decode announces itself.
    assert!(lowest > DEFAULT_HEIGHT * 4.0, "lowest vertex at {lowest}");
    assert!(
        (8_000.0..40_000.0).contains(&highest),
        "highest vertex at {highest}"
    );

    // Morrowind is an island, so a good share of its cells are open water.
    assert!(
        below_sea > cells / 5,
        "only {below_sea} of {cells} cells are underwater"
    );

    // 128 units apart with the steepest cliffs in the game between them. Read as absolute heights
    // the deltas would put neighbours up to 2,000 units apart routinely.
    assert!(
        worst_step < 2_000.0,
        "neighbouring vertices {worst_step} apart"
    );
}
