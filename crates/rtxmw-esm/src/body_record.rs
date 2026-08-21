//! What a `BODY` record is, and where on a skeleton it hangs.
//!
//! **2,772 of the 3,049 `NPC_` records carry no model at all.** A body is put together from a base
//! skeleton chosen by the race and sex, and the `BODY` records matching that race and sex for each
//! of the fifteen parts. `CREA` is the same story for 273 of its 432 records, which name a
//! `base_anim` variant and are assembled the same way.

use crate::error::Result;
use crate::esm_reader::Record;

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
        !matches!(
            self,
            Self::Head | Self::Hair | Self::Neck | Self::Chest | Self::Groin | Self::Tail
        )
    }

    /// The attachment node this part hangs from, given a side.
    ///
    /// These are nodes the base skeleton already has, sitting beside the `Bip01` chain that drives
    /// them — `Head`, `Chest`, `Left Upper Arm`. Each carries a hidden placeholder of its own that
    /// the assembled part stands in for.
    pub fn bone(self, right: bool) -> &'static str {
        match self {
            Self::Head => "Head",
            // Hair sits on the head rather than on a node of its own.
            Self::Hair => "Head",
            Self::Neck => "Neck",
            Self::Chest => "Chest",
            Self::Groin => "Groin",
            Self::Tail => "Tail",
            Self::Hand if right => "Right Hand",
            Self::Hand => "Left Hand",
            Self::Wrist if right => "Right Wrist",
            Self::Wrist => "Left Wrist",
            Self::Forearm if right => "Right Forearm",
            Self::Forearm => "Left Forearm",
            Self::UpperArm if right => "Right Upper Arm",
            Self::UpperArm => "Left Upper Arm",
            Self::Foot if right => "Right Foot",
            Self::Foot => "Left Foot",
            Self::Ankle if right => "Right Ankle",
            Self::Ankle => "Left Ankle",
            Self::Knee if right => "Right Knee",
            Self::Knee => "Left Knee",
            Self::UpperLeg if right => "Right Upper Leg",
            Self::UpperLeg => "Left Upper Leg",
            Self::Clavicle if right => "Right Clavicle",
            Self::Clavicle => "Left Clavicle",
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
