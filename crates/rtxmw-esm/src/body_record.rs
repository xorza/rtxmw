//! What a `BODY` record is, and where on a skeleton it hangs.
//!
//! **2,772 of the 3,049 `NPC_` records carry no model at all.** A body is put together from a base
//! skeleton chosen by the race and sex, and the `BODY` records matching that race and sex for each
//! of the fifteen parts. `CREA` is the same story for 273 of its 432 records, which name a
//! `base_anim` variant and are assembled the same way.

use crate::error::Result;
use crate::esm_reader::Record;
use crate::part_slot::PartSlot;

/// Which part of a body a `BODY` record is, as its `BYDT` names it.
///
/// The order is the format's and the numbers are what the field stores, so the enum is the table:
/// nothing else says that eight is an upper arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyPart {
    Head,
    Hair,
    Neck,
    Chest,
    Groin,
    Hand,
    Wrist,
    Forearm,
    UpperArm,
    Foot,
    Ankle,
    Knee,
    UpperLeg,
    Clavicle,
    Tail,
}

impl BodyPart {
    /// Every part, in the order `BYDT` numbers them.
    pub const ALL: [Self; 15] = [
        Self::Head,
        Self::Hair,
        Self::Neck,
        Self::Chest,
        Self::Groin,
        Self::Hand,
        Self::Wrist,
        Self::Forearm,
        Self::UpperArm,
        Self::Foot,
        Self::Ankle,
        Self::Knee,
        Self::UpperLeg,
        Self::Clavicle,
        Self::Tail,
    ];

    fn from_index(index: u8) -> Option<Self> {
        Self::ALL.get(index as usize).copied()
    }

    /// Whether the part exists once or as a left and a right.
    ///
    /// **The skeleton is what says so**: `base_anim.nif` carries one node called `Head` and two
    /// called `Left Hand` and `Right Hand`, so a hand record supplies both sides and a head does
    /// not.
    pub fn is_paired(self) -> bool {
        self.slots()[1].is_some()
    }

    /// The slot or slots this fills, right before left.
    ///
    /// **A `BODY` record has no side and a slot does**, so a hand supplies two of them out of one
    /// mesh. The pairs are the game's own — see [`PartSlot`].
    pub fn slots(self) -> [Option<PartSlot>; 2] {
        let one = |slot: u8| [Some(PartSlot(slot)), None];
        let pair = |right: u8, left: u8| [Some(PartSlot(right)), Some(PartSlot(left))];
        match self {
            Self::Head => one(0),
            Self::Hair => one(1),
            Self::Neck => one(2),
            Self::Chest => one(3),
            Self::Groin => one(4),
            Self::Tail => one(26),
            Self::Hand => pair(6, 7),
            Self::Wrist => pair(8, 9),
            Self::Forearm => pair(11, 12),
            Self::UpperArm => pair(13, 14),
            Self::Foot => pair(15, 16),
            Self::Ankle => pair(17, 18),
            Self::Knee => pair(19, 20),
            Self::UpperLeg => pair(21, 22),
            Self::Clavicle => pair(23, 24),
        }
    }
}

/// What a `BODY` record's four-byte `BYDT` says: which part, for whom, and of what.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BodyRecord {
    pub id: String,
    /// Mesh path relative to `meshes/`, as stored.
    pub model: String,
    pub part: BodyPart,
    /// Whether this is a woman's version of the part.
    pub female: bool,
    /// Skin, clothing or armour — the last two override the first when something is worn.
    pub kind: BodyKind,
}

/// What a body part is made of, from `BYDT`'s fourth byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyKind {
    Skin,
    Clothing,
    Armour,
}

impl BodyRecord {
    /// Reads one, or `None` where it names a part this does not know.
    pub fn parse(record: &Record<'_>) -> Result<Option<Self>> {
        let (mut id, mut model, mut data) = (String::new(), String::new(), None);
        for sub in record.subrecords() {
            let sub = sub?;
            match &sub.name().0 {
                b"NAME" => id = sub.as_str().into_owned(),
                b"MODL" => model = sub.as_str().into_owned(),
                b"BYDT" if sub.data().len() >= 4 => {
                    data = Some([sub.data()[0], sub.data()[2], sub.data()[3]]);
                }
                _ => {}
            }
        }
        let Some([part, flags, kind]) = data else {
            return Ok(None);
        };
        let Some(part) = BodyPart::from_index(part) else {
            return Ok(None);
        };
        Ok(Some(Self {
            id,
            model,
            part,
            // Bit zero of the flags; bit one marks a part the character generator does not offer.
            female: flags & 0x1 != 0,
            kind: match kind {
                1 => BodyKind::Clothing,
                2 => BodyKind::Armour,
                _ => BodyKind::Skin,
            },
        }))
    }
}
