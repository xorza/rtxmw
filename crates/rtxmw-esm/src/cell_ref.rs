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
    /// For a door, the interior cell it leads to. `None` means the destination is an exterior
    /// cell, which is identified by [`CellRef::destination`] instead of by name.
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
    /// `DATA` is the reference's own placement and `DODT` the teleport destination; both are a
    /// whole [`Position`] in their own right, which is why neither needs the other's context to be
    /// read. OpenMW reads the same pair at `components/esm3/cellref.cpp:115`.
    pub(crate) fn absorb(&mut self, sub: &Subrecord<'_>) {
        match &sub.name().0 {
            b"NAME" => self.object_id = sub.as_str().into_owned(),
            b"XSCL" => {
                if let Ok(scale) = sub.as_f32() {
                    self.scale = scale;
                }
            }
            b"DODT" => self.destination = Position::parse(sub.data()),
            // An empty name is how a door to an exterior is written, and carries no more than an
            // absent subrecord — flattening the two here keeps every reader from having to know.
            b"DNAM" => {
                let name = sub.as_str();
                if !name.is_empty() {
                    self.destination_cell = Some(name.into_owned());
                }
            }
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
