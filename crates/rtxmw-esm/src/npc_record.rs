//! What an `NPC_` record says about which body to build.

use crate::error::Result;
use crate::esm_reader::Record;

/// What an `NPC_` record says about which body to build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpcRecord {
    pub id: String,
    /// The `RACE` this one belongs to, lower-cased for lookup.
    pub race: String,
    pub female: bool,
    /// `BODY` record ids for the face and the hair, which an `NPC_` names itself rather than
    /// inheriting from its race.
    pub head: String,
    pub hair: String,
}

impl NpcRecord {
    /// Bit zero of `FLAG`.
    const FEMALE: u32 = 0x1;

    pub fn parse(record: &Record<'_>) -> Result<Self> {
        let mut out = Self {
            id: String::new(),
            race: String::new(),
            female: false,
            head: String::new(),
            hair: String::new(),
        };
        for sub in record.subrecords() {
            let sub = sub?;
            match &sub.name().0 {
                b"NAME" => out.id = sub.as_str().into_owned(),
                b"RNAM" => out.race = sub.as_str().to_lowercase(),
                b"BNAM" => out.head = sub.as_str().to_lowercase(),
                b"KNAM" => out.hair = sub.as_str().to_lowercase(),
                b"FLAG" if sub.data().len() >= 4 => {
                    let flags = u32::from_le_bytes([
                        sub.data()[0],
                        sub.data()[1],
                        sub.data()[2],
                        sub.data()[3],
                    ]);
                    out.female = flags & Self::FEMALE != 0;
                }
                _ => {}
            }
        }
        Ok(out)
    }
}
