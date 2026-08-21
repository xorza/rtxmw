//! What a piece of clothing or armour puts on a body.

use crate::error::Result;
use crate::esm_reader::Record;
use crate::part_slot::PartSlot;

/// One slot a wearable covers, and the body part it puts there.
///
/// **Two ids, because a shirt is cut differently for a woman.** Only 50 of the 707 references in
/// the master carry a female mesh; the rest dress everybody in the same one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartReference {
    pub slot: PartSlot,
    /// `BODY` record ids. `female` is empty where the item has no version of its own.
    pub male: String,
    pub female: String,
}

impl PartReference {
    /// The `BODY` record this puts on a body of that sex, lower-cased for lookup.
    pub fn worn_by(&self, female: bool) -> &str {
        match female && !self.female.is_empty() {
            true => &self.female,
            false => &self.male,
        }
    }
}

/// Whether a wearable is clothing or armour, which is what decides one over the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WearableKind {
    Clothing,
    /// Worn over clothing, and so applied after it.
    Armour,
}

/// A `CLOT` or `ARMO` record, reduced to what it covers.
///
/// The two records differ in what they are worth and what they protect, and in nothing this needs:
/// both carry the same `INDX`, `BNAM`, `CNAM` triples and both go on a body the same way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WearableRecord {
    pub id: String,
    pub kind: WearableKind,
    pub parts: Vec<PartReference>,
}

impl WearableRecord {
    /// Reads one, or `None` where it covers nothing — a ring or an amulet, which are worn and not
    /// seen.
    pub fn parse(record: &Record<'_>, kind: WearableKind) -> Result<Option<Self>> {
        let mut out = Self {
            id: String::new(),
            kind,
            parts: Vec::new(),
        };
        for sub in record.subrecords() {
            let sub = sub?;
            match &sub.name().0 {
                b"NAME" => out.id = sub.as_str().to_lowercase(),
                b"INDX" if !sub.data().is_empty() => out.parts.push(PartReference {
                    slot: PartSlot(sub.data()[0]),
                    male: String::new(),
                    female: String::new(),
                }),
                // Both follow the `INDX` they belong to, so the reference being extended is the
                // last one pushed. A file that wrote one without an index would be malformed.
                b"BNAM" => {
                    if let Some(part) = out.parts.last_mut() {
                        part.male = sub.as_str().to_lowercase();
                    }
                }
                b"CNAM" => {
                    if let Some(part) = out.parts.last_mut() {
                        part.female = sub.as_str().to_lowercase();
                    }
                }
                _ => {}
            }
        }
        out.parts
            .retain(|part| !part.male.is_empty() || !part.female.is_empty());
        Ok((!out.parts.is_empty()).then_some(out))
    }
}
