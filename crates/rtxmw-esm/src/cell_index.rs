//! Where every cell in a content file is, found once so it need not be searched for again.

use std::collections::HashMap;

use crate::cell::{Cell, CellId};
use crate::error::Result;
use crate::esm_reader::EsmReader;
use crate::land_record::LandRecord;
use crate::land_texture::LandTexture;
use crate::record_name::RecordName;
use crate::region_record::RegionRecord;

/// Where one cell's records start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellOffsets {
    /// The `CELL` record, which names everything the cell places.
    pub cell: usize,
    /// The `LAND` record shaping the ground under it. Absent for an interior, and for the
    /// hundred-odd exteriors that ship without one.
    pub land: Option<usize>,
}

/// Every cell in a content file, and the land-texture palette they name.
///
/// Records are found by walking, and one cell's are scattered through a stream tens of megabytes
/// long, so finding a cell costs a pass over the whole file. **Walking once and remembering where
/// everything landed** is what makes loading a cell at a time, as the camera moves into it,
/// affordable: every load after the index is built is two reads at known offsets.
///
/// The palette has to be here for the same reason. `LTEX` records are spread through the file with
/// no relation to the cells that use them, so a cell loaded on its own would resolve its ground
/// textures against however much of the palette happened to precede it. **The regions are here on
/// exactly that argument**: a cell names one by id, `REGN` records sit wherever the file put them,
/// and a walk to find one is the pass this type exists to make unnecessary.
#[derive(Debug, Default)]
pub struct CellIndex {
    cells: HashMap<CellId, CellOffsets>,
    land_textures: Vec<String>,
    /// Keyed by the region's id lower-cased, because a cell's `RGNN` and the record's `NAME` are
    /// the same string written by different hands.
    regions: HashMap<String, RegionRecord>,
}

impl CellIndex {
    /// Walks `esm` once, recording where every cell and its terrain sit.
    pub fn build(esm: &EsmReader<'_>) -> Result<Self> {
        let cell_tag = RecordName::new(b"CELL");
        let land_tag = RecordName::new(b"LAND");
        let texture_tag = RecordName::new(b"LTEX");
        let region_tag = RecordName::new(b"REGN");

        let mut index = Self::default();
        // `LAND` and `CELL` are not adjacent and either can come first, so terrain is matched up
        // after the pass rather than during it.
        let mut lands: HashMap<CellId, usize> = HashMap::new();

        for record in esm.records() {
            let record = record?;
            if record.name() == texture_tag
                && let Some(entry) = LandTexture::parse(&record)?
            {
                let slot = entry.index as usize;
                if index.land_textures.len() <= slot {
                    index.land_textures.resize(slot + 1, String::new());
                }
                index.land_textures[slot] = entry.texture;
            } else if record.name() == cell_tag {
                let cell = Cell::parse(&record)?;
                let id = if cell.is_interior() {
                    CellId::Interior(cell.name)
                } else {
                    CellId::Exterior {
                        x: cell.grid_x,
                        y: cell.grid_y,
                    }
                };
                // Later records win, which is how a plugin overriding a cell is meant to resolve.
                index.cells.insert(
                    id,
                    CellOffsets {
                        cell: record.offset(),
                        land: None,
                    },
                );
            } else if record.name() == land_tag
                && let Some(id) = LandRecord::grid_of(&record)?
            {
                lands.insert(id, record.offset());
            } else if record.name() == region_tag
                && let Some(region) = RegionRecord::parse(&record)?
            {
                index.regions.insert(region.id.to_ascii_lowercase(), region);
            }
        }

        for (id, offset) in lands {
            if let Some(entry) = index.cells.get_mut(&id) {
                entry.land = Some(offset);
            }
        }
        Ok(index)
    }

    /// Where `id` sits, or `None` for a cell this file does not contain.
    pub fn cell(&self, id: &CellId) -> Option<CellOffsets> {
        self.cells.get(id).copied()
    }

    /// The `LTEX` palette, indexed as the records number themselves.
    ///
    /// A gap is an empty string: the palette is filled by index as records arrive, so an index no
    /// record ever claimed leaves one behind rather than shifting everything after it.
    pub fn land_textures(&self) -> &[String] {
        &self.land_textures
    }

    /// The region a cell's `RGNN` names, or `None` for one this file does not describe.
    ///
    /// Matched without regard to case, which the file needs: `Cell::region` and a `REGN` record's
    /// own `NAME` are the same string entered twice by hand.
    pub fn region(&self, id: &str) -> Option<&RegionRecord> {
        self.regions.get(&id.to_ascii_lowercase())
    }

    /// How many cells the file holds.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether the file holds no cells at all.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}
