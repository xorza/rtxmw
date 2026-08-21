//! What a `RACE` record says about the shape of the people in it.

use crate::error::Result;
use crate::esm_reader::Record;

/// What a `RACE` record says about the shape of the people in it.
///
/// **Height is a multiplier, not a measurement.** It is what makes a Bosmer shorter than a Nord out
/// of one skeleton, and it is per sex.
#[derive(Debug, Clone, PartialEq)]
pub struct RaceRecord {
    pub id: String,
    /// Argonians and Khajiit, which use a skeleton of their own.
    pub beast: bool,
    pub male_height: f32,
    pub female_height: f32,
}

impl RaceRecord {
    /// The 140 bytes of `RADT`: skill bonuses, attributes, heights, weights and flags.
    const DATA_SIZE: usize = 140;
    /// Where the two heights sit in it, after fourteen skill words and sixteen attribute words.
    const HEIGHTS: usize = 14 * 4 + 16 * 4;
    /// Set on Argonian and Khajiit.
    const BEAST: u32 = 0x2;

    pub fn parse(record: &Record<'_>) -> Result<Option<Self>> {
        let mut id = String::new();
        let mut data: Option<(bool, f32, f32)> = None;
        for sub in record.subrecords() {
            let sub = sub?;
            match &sub.name().0 {
                b"NAME" => id = sub.as_str().into_owned(),
                b"RADT" if sub.data().len() >= Self::DATA_SIZE => {
                    let bytes = sub.data();
                    let word = |at: usize| {
                        f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
                    };
                    let flags = u32::from_le_bytes([
                        bytes[Self::DATA_SIZE - 4],
                        bytes[Self::DATA_SIZE - 3],
                        bytes[Self::DATA_SIZE - 2],
                        bytes[Self::DATA_SIZE - 1],
                    ]);
                    data = Some((
                        flags & Self::BEAST != 0,
                        word(Self::HEIGHTS),
                        word(Self::HEIGHTS + 4),
                    ));
                }
                _ => {}
            }
        }
        let Some((beast, male_height, female_height)) = data else {
            return Ok(None);
        };
        Ok(Some(Self {
            id,
            beast,
            male_height,
            female_height,
        }))
    }
}
