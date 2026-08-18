//! The records a cell reference can point at.

use crate::error::Result;
use crate::esm_reader::Record;

/// A record that owns a model, reduced to the two fields a renderer needs.
///
/// Every placeable Morrowind record carries its id in `NAME` and its mesh in `MODL`, whatever else
/// it holds. Parsing only those two covers every reference a cell can make without a per-type
/// parser for each of the fifteen record types involved.
#[derive(Debug, Clone)]
pub struct ObjectRecord {
    pub id: String,
    /// Mesh path relative to `meshes/`, as stored — backslashes and original case.
    pub model: String,
    /// Display name, when the record has one.
    pub display_name: Option<String>,
}

impl ObjectRecord {
    /// Record tags that place geometry in a cell.
    ///
    /// `LEVC` and `LEVI` are deliberately absent: they are levelled lists that resolve to another
    /// record at spawn time, which is game logic rather than asset loading.
    pub const PLACEABLE: &'static [&'static [u8; 4]] = &[
        b"ACTI", b"ALCH", b"APPA", b"ARMO", b"BOOK", b"CLOT", b"CONT", b"CREA", b"DOOR", b"INGR",
        b"LIGH", b"LOCK", b"MISC", b"NPC_", b"PROB", b"REPA", b"STAT", b"WEAP",
    ];

    /// Whether `record` is one of the placeable types.
    pub fn is_placeable(record: &Record<'_>) -> bool {
        Self::PLACEABLE.contains(&&record.name().0)
    }

    /// Reads the id, model and display name, ignoring everything else.
    pub fn parse(record: &Record<'_>) -> Result<Self> {
        let mut out = Self {
            id: String::new(),
            model: String::new(),
            display_name: None,
        };
        for sub in record.subrecords() {
            let sub = sub?;
            match &sub.name().0 {
                b"NAME" => out.id = sub.as_str().into_owned(),
                b"MODL" => out.model = sub.as_str().into_owned(),
                b"FNAM" => out.display_name = Some(sub.as_str().into_owned()),
                _ => {}
            }
        }
        Ok(out)
    }

    /// The mesh's path in the virtual file system, or `None` when the record has no model.
    ///
    /// Model paths are stored relative to `meshes/`, which the file system does not know about.
    pub fn model_path(&self) -> Option<String> {
        if self.model.is_empty() {
            return None;
        }
        Some(format!("meshes/{}", self.model))
    }
}
