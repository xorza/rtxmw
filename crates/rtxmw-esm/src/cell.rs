//! `CELL` records and the references they place.

use crate::cell_ref::CellRef;
use crate::error::Result;
use crate::esm_reader::{Record, SubrecordIter};

/// Cell is an interior rather than part of the exterior worldspace.
const FLAG_INTERIOR: i32 = 0x01;
/// Cell has a water plane.
const FLAG_HAS_WATER: i32 = 0x02;
/// Sleeping here without a bed is not allowed.
const FLAG_NO_SLEEP: i32 = 0x04;
/// Interior that behaves like an exterior — sky and weather. Tribunal and later.
const FLAG_QUASI_EXTERIOR: i32 = 0x80;

/// Lighting an interior cell carries, since it has no sky to light it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CellAmbient {
    /// Packed `0xAABBGGRR`, as stored.
    pub ambient: u32,
    pub sunlight: u32,
    pub fog: u32,
    /// Fog density, bit-identical to the stored float.
    pub fog_density: u32,
}

/// A `CELL` record's own fields, without its references.
///
/// References are not parsed here: a single exterior cell can hold hundreds, and most callers want
/// the cell list long before they want any cell's contents.
#[derive(Debug, Clone)]
pub struct Cell {
    /// Interior cells are named; exterior cells usually carry an empty name and are identified by
    /// their grid coordinates.
    pub name: String,
    pub flags: i32,
    /// Exterior grid coordinates. Meaningless for interiors, where both are zero.
    pub grid_x: i32,
    pub grid_y: i32,
    pub region: Option<String>,
    /// Water level in world units, when the cell has water.
    pub water_height: Option<f32>,
    pub ambient: Option<CellAmbient>,
}

impl Cell {
    /// Parses the cell's own subrecords, stopping at the first reference.
    pub fn parse(record: &Record<'_>) -> Result<Self> {
        let mut cell = Self {
            name: String::new(),
            flags: 0,
            grid_x: 0,
            grid_y: 0,
            region: None,
            water_height: None,
            ambient: None,
        };

        for sub in record.subrecords() {
            let sub = sub?;
            match &sub.name().0 {
                b"NAME" => cell.name = sub.as_str().into_owned(),
                b"DATA" => {
                    let data = sub.data();
                    if data.len() >= 12 {
                        let int =
                            |at: usize| i32::from_le_bytes(data[at..at + 4].try_into().unwrap());
                        cell.flags = int(0);
                        cell.grid_x = int(4);
                        cell.grid_y = int(8);
                    }
                }
                b"RGNN" => cell.region = Some(sub.as_str().into_owned()),
                // Water height is a float in `WHGT` and a legacy integer in `INTV`.
                b"WHGT" => cell.water_height = sub.as_f32().ok(),
                b"INTV" => cell.water_height = sub.as_i32().ok().map(|v| v as f32),
                b"AMBI" => {
                    let data = sub.data();
                    if data.len() >= 16 {
                        let word =
                            |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
                        cell.ambient = Some(CellAmbient {
                            ambient: word(0),
                            sunlight: word(4),
                            fog: word(8),
                            fog_density: word(12),
                        });
                    }
                }
                // The first reference marker ends the cell's own fields.
                b"FRMR" | b"MVRF" => break,
                _ => {}
            }
        }
        Ok(cell)
    }

    /// Whether this is an interior cell.
    pub fn is_interior(&self) -> bool {
        self.flags & FLAG_INTERIOR != 0
    }

    /// Whether the cell has a water plane.
    pub fn has_water(&self) -> bool {
        self.flags & FLAG_HAS_WATER != 0
    }

    /// Whether sleeping here without a bed is disallowed.
    pub fn no_sleep(&self) -> bool {
        self.flags & FLAG_NO_SLEEP != 0
    }

    /// Whether this interior behaves like an exterior, with sky and weather.
    pub fn is_quasi_exterior(&self) -> bool {
        self.flags & FLAG_QUASI_EXTERIOR != 0
    }

    /// How the cell is identified: by name for interiors, by grid for exteriors.
    pub fn id(&self) -> CellId {
        if self.is_interior() {
            CellId::Interior(self.name.clone())
        } else {
            CellId::Exterior {
                x: self.grid_x,
                y: self.grid_y,
            }
        }
    }

    /// Walks the references this cell places.
    pub fn references<'a>(record: &Record<'a>) -> CellRefIter<'a> {
        CellRefIter {
            subrecords: record.subrecords(),
            pending: None,
            started: false,
        }
    }
}

/// How a cell is addressed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CellId {
    /// Interiors are keyed by name.
    Interior(String),
    /// Exteriors are keyed by their place in the worldspace grid.
    Exterior { x: i32, y: i32 },
}

impl std::fmt::Display for CellId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Interior(name) => f.write_str(name),
            Self::Exterior { x, y } => write!(f, "({x}, {y})"),
        }
    }
}

/// Iterator over the references a cell places.
///
/// References are a flat run of subrecords with no length prefix: each begins at `FRMR` and
/// continues until the next `FRMR` or the end of the record, so one has to be accumulated before
/// it can be yielded.
#[derive(Debug, Clone)]
pub struct CellRefIter<'a> {
    subrecords: SubrecordIter<'a>,
    pending: Option<CellRef>,
    started: bool,
}

impl Iterator for CellRefIter<'_> {
    type Item = Result<CellRef>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let Some(sub) = self.subrecords.next() else {
                return self.pending.take().map(Ok);
            };
            let sub = match sub {
                Ok(sub) => sub,
                Err(e) => return Some(Err(e)),
            };

            match &sub.name().0 {
                b"FRMR" => {
                    self.started = true;
                    let refnum = match sub.as_u32() {
                        Ok(value) => value,
                        Err(e) => return Some(Err(e)),
                    };
                    let finished = self.pending.replace(CellRef::new(refnum));
                    if let Some(finished) = finished {
                        return Some(Ok(finished));
                    }
                }
                _ => {
                    // Everything before the first FRMR belongs to the cell, not to a reference.
                    if let Some(current) = self.pending.as_mut() {
                        current.absorb(&sub);
                    } else if !self.started {
                        continue;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esm_reader::EsmReader;
    use crate::esm_reader::internals::{SubrecordSpec, push_header, push_record};
    use crate::position::Position;

    fn position_bytes(translation: [f32; 3], rotation: [f32; 3]) -> Vec<u8> {
        translation
            .iter()
            .chain(rotation.iter())
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }

    fn interior_with_two_refs() -> Vec<u8> {
        let mut out = Vec::new();
        push_header(&mut out);

        let data: Vec<u8> = [0x01i32, 0, 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let first = position_bytes([1.0, 2.0, 3.0], [0.0, 0.0, 0.5]);
        let second = position_bytes([10.0, 20.0, 30.0], [0.0, 0.0, 0.0]);

        push_record(
            &mut out,
            b"CELL",
            0,
            &[
                SubrecordSpec {
                    name: b"NAME",
                    data: b"Test Interior\0",
                },
                SubrecordSpec {
                    name: b"DATA",
                    data: &data,
                },
                SubrecordSpec {
                    name: b"WHGT",
                    data: &(-14.0f32).to_le_bytes(),
                },
                // First reference.
                SubrecordSpec {
                    name: b"FRMR",
                    data: &1u32.to_le_bytes(),
                },
                SubrecordSpec {
                    name: b"NAME",
                    data: b"in_de_shack\0",
                },
                SubrecordSpec {
                    name: b"XSCL",
                    data: &2.5f32.to_le_bytes(),
                },
                SubrecordSpec {
                    name: b"DATA",
                    data: &first,
                },
                // Second reference, deleted.
                SubrecordSpec {
                    name: b"FRMR",
                    data: &2u32.to_le_bytes(),
                },
                SubrecordSpec {
                    name: b"NAME",
                    data: b"chest_small\0",
                },
                SubrecordSpec {
                    name: b"DATA",
                    data: &second,
                },
                SubrecordSpec {
                    name: b"DELE",
                    data: &0u32.to_le_bytes(),
                },
            ],
        );
        out
    }

    #[test]
    fn parses_cell_fields_and_stops_at_the_first_reference() {
        let file = interior_with_two_refs();
        let esm = EsmReader::new(&file).unwrap();
        let record = esm.records().next().unwrap().unwrap();
        let cell = Cell::parse(&record).unwrap();

        assert_eq!(cell.name, "Test Interior");
        assert!(cell.is_interior());
        assert_eq!(cell.water_height, Some(-14.0));
        assert_eq!(cell.id(), CellId::Interior("Test Interior".into()));

        // The cell's own NAME must not be overwritten by the first reference's NAME.
        assert_ne!(cell.name, "in_de_shack");
    }

    #[test]
    fn groups_subrecords_into_references() {
        let file = interior_with_two_refs();
        let esm = EsmReader::new(&file).unwrap();
        let record = esm.records().next().unwrap().unwrap();

        let refs: Vec<_> = Cell::references(&record).map(|r| r.unwrap()).collect();
        assert_eq!(refs.len(), 2, "both references should be yielded");

        assert_eq!(refs[0].refnum, 1);
        assert_eq!(refs[0].object_id, "in_de_shack");
        assert_eq!(refs[0].scale, 2.5);
        assert!(!refs[0].deleted);
        assert_eq!(
            refs[0].position,
            Position {
                translation: [1.0, 2.0, 3.0],
                rotation: [0.0, 0.0, 0.5],
            }
        );

        assert_eq!(refs[1].refnum, 2);
        assert_eq!(refs[1].object_id, "chest_small");
        // Absent XSCL means unscaled, not zero.
        assert_eq!(refs[1].scale, 1.0);
        assert!(refs[1].deleted);
    }

    #[test]
    fn an_exterior_cell_is_identified_by_its_grid() {
        let mut out = Vec::new();
        push_header(&mut out);
        let data: Vec<u8> = [0x02i32, -3, 7]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        push_record(
            &mut out,
            b"CELL",
            0,
            &[
                SubrecordSpec {
                    name: b"NAME",
                    data: b"\0",
                },
                SubrecordSpec {
                    name: b"DATA",
                    data: &data,
                },
            ],
        );

        let esm = EsmReader::new(&out).unwrap();
        let record = esm.records().next().unwrap().unwrap();
        let cell = Cell::parse(&record).unwrap();

        assert!(!cell.is_interior());
        assert!(cell.has_water());
        assert_eq!(cell.id(), CellId::Exterior { x: -3, y: 7 });
        assert_eq!(cell.id().to_string(), "(-3, 7)");
    }

    #[test]
    fn a_cell_with_no_references_yields_none() {
        let mut out = Vec::new();
        push_header(&mut out);
        push_record(
            &mut out,
            b"CELL",
            0,
            &[SubrecordSpec {
                name: b"NAME",
                data: b"Empty\0",
            }],
        );
        let esm = EsmReader::new(&out).unwrap();
        let record = esm.records().next().unwrap().unwrap();
        assert_eq!(Cell::references(&record).count(), 0);
    }
}
