//! Reader for Morrowind's ESM/ESP content files.

mod cell;
mod cell_ref;
mod error;
mod esm_reader;
mod header;
mod object_record;
mod position;
mod record_name;

pub use crate::cell::{Cell, CellAmbient, CellId, CellRefIter};
pub use crate::cell_ref::CellRef;
pub use crate::error::{EsmError, Result};
pub use crate::esm_reader::{EsmReader, Record, RecordIter, Subrecord, SubrecordIter};
pub use crate::header::{FileKind, Header, MasterFile};
pub use crate::object_record::ObjectRecord;
pub use crate::position::Position;
pub use crate::record_name::RecordName;

// The gate must match the module's, or the module is `pub` yet unreachable under cfg(test).
#[cfg(any(test, feature = "internals"))]
pub use crate::esm_reader::internals as esm_internals;
