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

/// The three kinds of wearable that cover more than they name.
///
/// **A robe hides the arms and legs it never mentions**, and a skirt the legs, and a helmet the
/// hair. Nothing in the record says so — it is a rule about how the garment is cut, and the game
/// carries it in code rather than in data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WornSlot {
    Robe,
    Skirt,
    Helmet,
    Other,
}

impl WornSlot {
    /// The slots this covers over and above the ones it names, and at what priority.
    ///
    /// The numbers are the game's own — `apps/openmw/mwrender/npcanimation.cpp:590` gives a robe a
    /// base of 11 and a skirt 3, against nothing for everything else, and turns each into
    /// `((base + 1) << 1) + is_armour`. So a robe at 24 beats every other garment, a skirt at 8
    /// beats an ordinary one, armour at 3 beats clothing at 2, and skin at 1 loses to all of them.
    pub fn priority(self, kind: WearableKind) -> u8 {
        let base: u8 = match self {
            Self::Robe => 11,
            Self::Skirt => 3,
            Self::Helmet | Self::Other => 0,
        };
        ((base + 1) << 1) + u8::from(kind == WearableKind::Armour)
    }

    /// The slots it hides whether or not it puts anything in them.
    pub fn reserves(self) -> &'static [u8] {
        match self {
            // Groin, skirt, both legs, both upper arms, both knees, both forearms, and the chest.
            Self::Robe => &[4, 5, 21, 22, 13, 14, 19, 20, 11, 12, 3],
            Self::Skirt => &[4, 21, 22],
            Self::Helmet => &[1],
            Self::Other => &[],
        }
    }

    /// Which one a clothing type is: 4 is a robe and 7 a skirt.
    fn of_clothing(kind: u32) -> Self {
        match kind {
            4 => Self::Robe,
            7 => Self::Skirt,
            _ => Self::Other,
        }
    }

    /// Which one an armour type is: 0 is a helmet, and nothing else covers what it does not name.
    fn of_armour(kind: u32) -> Self {
        match kind {
            0 => Self::Helmet,
            _ => Self::Other,
        }
    }
}

/// A `CLOT` or `ARMO` record, reduced to what it covers.
///
/// The two records differ in what they are worth and what they protect, and in nothing this needs:
/// both carry the same `INDX`, `BNAM`, `CNAM` triples and both go on a body the same way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WearableRecord {
    pub id: String,
    pub kind: WearableKind,
    /// What it is, for the three that cover more than they name.
    pub worn: WornSlot,
    pub parts: Vec<PartReference>,
}

impl WearableRecord {
    /// Reads one, or `None` where it covers nothing — a ring or an amulet, which are worn and not
    /// seen.
    pub fn parse(record: &Record<'_>, kind: WearableKind) -> Result<Option<Self>> {
        let mut out = Self {
            id: String::new(),
            kind,
            worn: WornSlot::Other,
            parts: Vec::new(),
        };
        for sub in record.subrecords() {
            let sub = sub?;
            match &sub.name().0 {
                b"NAME" => out.id = sub.as_str().to_lowercase(),
                // The first word of either data block is what the thing is.
                b"CTDT" | b"AODT" if sub.data().len() >= 4 => {
                    let value = u32::from_le_bytes([
                        sub.data()[0],
                        sub.data()[1],
                        sub.data()[2],
                        sub.data()[3],
                    ]);
                    out.worn = match kind {
                        WearableKind::Clothing => WornSlot::of_clothing(value),
                        WearableKind::Armour => WornSlot::of_armour(value),
                    };
                }
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
