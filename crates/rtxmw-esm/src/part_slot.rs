//! The twenty-seven places something can hang on a body.

/// One place on a body, as `INDX` numbers them.
///
/// **Not the same enumeration as [`crate::BodyPart`]**, which is what a `BODY` record *is*. This is
/// where a thing goes, and it names a side: a `BODY` record for a forearm supplies both arms, where
/// a cuirass names a left pauldron and a right one separately. A mesh part maps onto one slot or
/// onto a left and a right — see [`crate::BodyPart::slots`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartSlot(pub u8);

impl PartSlot {
    /// How many there are, which is what a body's parts are held in one of.
    pub const COUNT: usize = 27;

    /// The two an `NPC_` record names itself rather than inheriting from its race.
    pub const HEAD: Self = Self(0);
    pub const HAIR: Self = Self(1);

    /// The node on the skeleton this hangs from, or `None` for a slot that hangs from nothing.
    ///
    /// Straight out of the game's own rig: `base_anim.nif` carries a node for every one of these,
    /// sitting beside the `Bip01` chain that drives it. Hair hangs from the head, and a skirt from
    /// the groin, because there is no node for either.
    pub fn bone(self) -> Option<&'static str> {
        Some(match self.0 {
            0 | 1 => "Head",
            2 => "Neck",
            3 => "Chest",
            4 | 5 => "Groin",
            6 => "Right Hand",
            7 => "Left Hand",
            8 => "Right Wrist",
            9 => "Left Wrist",
            10 => "Shield Bone",
            11 => "Right Forearm",
            12 => "Left Forearm",
            13 => "Right Upper Arm",
            14 => "Left Upper Arm",
            15 => "Right Foot",
            16 => "Left Foot",
            17 => "Right Ankle",
            18 => "Left Ankle",
            19 => "Right Knee",
            20 => "Left Knee",
            21 => "Right Upper Leg",
            22 => "Left Upper Leg",
            23 => "Right Clavicle",
            24 => "Left Clavicle",
            25 => "Weapon Bone",
            26 => "Tail",
            _ => return None,
        })
    }

    /// What a skinned file calls the shapes belonging to this slot, lower-cased.
    ///
    /// **One file is the whole of a body's skin.** `b_n_dark elf_m_chest` and `b_n_dark elf_m_hand`
    /// both name `B_N_Dark Elf_M_Skins.NIF`, which holds `Tri Chest` and `Tri Left Hand 0` beside
    /// each other; a rigid part instead names its shape after the file — `Tri B_N_Dark Elf_M_Wrist`
    /// — and needs no filter, being the whole of what it is. So a skinned file has to be cut down
    /// to the slot it is being asked for, or hiding the chest under a robe leaves the naked torso
    /// showing because the hands dragged it back in.
    ///
    /// **No side in it**, deliberately: the shapes are `Tri Left Hand` and `Tri Right Hand`, and a
    /// skinned file binds by its own bone names, so one match supplies both arms at once.
    pub fn shape_name(self) -> Option<&'static str> {
        Some(match self.0 {
            0 => "head",
            1 => "hair",
            2 => "neck",
            3 => "chest",
            // A skirt hangs from the groin and is named for it.
            4 | 5 => "groin",
            6 | 7 => "hand",
            8 | 9 => "wrist",
            10 => "shield",
            11 | 12 => "forearm",
            13 | 14 => "upper arm",
            15 | 16 => "foot",
            17 | 18 => "ankle",
            19 | 20 => "knee",
            21 | 22 => "upper leg",
            23 | 24 => "clavicle",
            25 => "weapon",
            26 => "tail",
            _ => return None,
        })
    }

    /// Whether what hangs here is the reflection of the mesh as it was authored.
    ///
    /// The parts are drawn for the right side and one file supplies both, so a left slot carries a
    /// reflection — `docs/design.md` §8.95, and OpenMW decides it by the same test.
    pub fn is_reflected(self) -> bool {
        self.bone().is_some_and(|bone| bone.starts_with("Left"))
    }
}
