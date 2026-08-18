//! `LTEX` records: the palette a cell's terrain texture indices address.

use crate::error::Result;
use crate::esm_reader::Record;

/// One entry in a plugin's terrain texture palette.
#[derive(Debug, Clone)]
pub struct LandTexture {
    /// Position in the palette. **A cell's `VTEX` index is one *past* this** — zero there means the
    /// region's default rather than the first entry, so every real reference is offset by one.
    pub index: u32,
    /// Texture path as the file gives it, without a directory.
    pub texture: String,
}

impl LandTexture {
    /// Parses an `LTEX` record, or `None` if it names no texture.
    pub fn parse(record: &Record<'_>) -> Result<Option<Self>> {
        let mut index = None;
        let mut texture = String::new();
        for sub in record.subrecords() {
            let sub = sub?;
            match &sub.name().0 {
                b"INTV" => index = sub.as_u32().ok(),
                b"DATA" => texture = sub.as_str().into_owned(),
                _ => {}
            }
        }
        Ok(match index {
            Some(index) if !texture.is_empty() => Some(Self { index, texture }),
            _ => None,
        })
    }
}
