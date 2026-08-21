//! Reader for Morrowind's ESM/ESP content files.

mod body_record;
mod cell;
mod cell_index;
mod cell_ref;
mod error;
mod esm_reader;
mod header;
mod land_record;
mod land_texture;
mod light_record;
mod npc_record;
mod object_record;
mod part_slot;
mod position;
mod race_record;
mod record_name;
mod region_record;
mod wearable_record;

pub use crate::body_record::{BodyKind, BodyPart, BodyRecord};
pub use crate::cell::{CELL_SIZE, Cell, CellAmbient, CellId, CellRefIter};
pub use crate::cell_index::{CellIndex, CellOffsets};
pub use crate::cell_ref::CellRef;
pub use crate::error::{EsmError, Result};
pub use crate::esm_reader::{EsmReader, Record, RecordIter, Subrecord, SubrecordIter};
pub use crate::header::{FileKind, Header, MasterFile};
pub use crate::land_record::{DEFAULT_HEIGHT, GRID, LandRecord, SPACING, TEXTURE_GRID, VERTICES};
pub use crate::land_texture::LandTexture;
pub use crate::light_record::LightRecord;
pub use crate::npc_record::NpcRecord;
pub use crate::object_record::ObjectRecord;
pub use crate::part_slot::PartSlot;
pub use crate::position::Position;
pub use crate::race_record::RaceRecord;
pub use crate::record_name::RecordName;
pub use crate::region_record::RegionRecord;
pub use crate::wearable_record::{PartReference, WearableKind, WearableRecord, WornSlot};

// The gate must match the module's, or the module is `pub` yet unreachable under cfg(test).
#[cfg(any(test, feature = "internals"))]
pub use crate::esm_reader::internals as esm_internals;
