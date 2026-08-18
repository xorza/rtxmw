//! A placed instance of an object inside a cell.

use crate::esm_reader::Subrecord;
use crate::position::Position;

/// One object placed in a cell.
///
/// A reference does not carry geometry: `object_id` names the record — `STAT`, `DOOR`, `LIGH` and
/// so on — that supplies the model. Resolving that is the caller's job.
#[derive(Debug, Clone, PartialEq)]
pub struct CellRef {
    /// Index the file uses to address this reference. Unique within its content file.
    pub refnum: u32,
    /// The id of the record this instantiates.
    pub object_id: String,
    pub position: Position,
    /// Uniform scale. Absent in the file means 1.0, never 0.
    pub scale: f32,
    /// Marked deleted by this or a later plugin.
    pub deleted: bool,
    /// For a door, the cell it leads to. Empty name means an exterior destination.
    pub destination_cell: Option<String>,
    /// For a door, where the player arrives.
    pub destination: Option<Position>,
}

impl CellRef {
    pub(crate) fn new(refnum: u32) -> Self {
        Self {
            refnum,
            object_id: String::new(),
            position: Position::default(),
            scale: 1.0,
            deleted: false,
            destination_cell: None,
            destination: None,
        }
    }

    /// Folds one of the reference's subrecords into it.
    ///
    /// `DATA` means the reference's own placement, but a preceding `DODT` claims the next `DATA`
    /// for the teleport destination — so the two are distinguished by order, not by tag.
    pub(crate) fn absorb(&mut self, sub: &Subrecord<'_>) {
        match &sub.name().0 {
            b"NAME" => self.object_id = sub.as_str().into_owned(),
            b"XSCL" => {
                if let Ok(scale) = sub.as_f32() {
                    self.scale = scale;
                }
            }
            b"DODT" => self.destination = Position::parse(sub.data()),
            b"DNAM" => self.destination_cell = Some(sub.as_str().into_owned()),
            b"DATA" => {
                if let Some(position) = Position::parse(sub.data()) {
                    self.position = position;
                }
            }
            b"DELE" => self.deleted = true,
            _ => {}
        }
    }

    /// Whether this reference teleports somewhere, i.e. is a working door.
    pub fn is_teleport(&self) -> bool {
        self.destination.is_some()
    }
}
