//! Doors that move a traveller from one cell to another.

use glam::Vec3;
use rtxmw_esm::{Cell, CellId, CellRef, EsmReader, RecordName};

use crate::error::Result;
use crate::static_scene::ModelIndex;

/// A door that moves a traveller to another cell.
///
/// **Morrowind stores the arrival point on the door you leave through, not in the cell you enter.**
/// A door reference carries `DODT`, a position in its *destination* cell, and `DNAM`, that cell's
/// name. Walking through one is therefore a local question — the reference is already in the cell
/// being played, and [`Door::from_ref`] answers it. Arriving in a cell without walking through
/// anything is not: it means searching for a door that leads there, which is
/// [`Door::leading_to`].
#[derive(Debug, Clone, PartialEq)]
pub struct Door {
    /// The cell this door opens into.
    pub destination: CellId,
    /// Where the traveller stands on arrival, in the destination cell's coordinates.
    ///
    /// An actor position, which in Morrowind is at the feet. A camera belongs a head's height
    /// above it.
    pub arrival: Vec3,
    /// Unit vector the traveller faces on arrival.
    pub facing: Vec3,
}

impl Door {
    /// The door a reference describes, or `None` if the reference does not teleport.
    ///
    /// Most doors do not: a `DOOR` record with no `DODT` is one that swings open on its hinge, and
    /// the cell is full of them.
    pub fn from_ref(cell_ref: &CellRef) -> Option<Self> {
        let dodt = cell_ref.destination?;
        let arrival = Vec3::from_array(dodt.translation);
        Some(Self {
            // A door to an exterior names no cell, because the arrival position already identifies
            // one — the worldspace grid is a function of world coordinates.
            destination: match &cell_ref.destination_cell {
                Some(name) => CellId::Interior(name.clone()),
                None => CellId::containing(arrival.x, arrival.y),
            },
            arrival,
            facing: facing_from(dodt.rotation[2]),
        })
    }

    /// Every door in `esm` a traveller could walk through to reach `destination`.
    ///
    /// One pass over every cell's references, which for `Morrowind.esm` is 316,116 of them and
    /// takes about 25 ms. There is no cheaper way round it: the answer is spread across every cell
    /// in the file precisely because a cell does not record how it is entered.
    ///
    /// Deleted references are skipped — a plugin that removes a door removes the way in through it
    /// — and so are editor markers. `PrisonMarker` is filed as a `DOOR` and carries a destination
    /// into Seyda Neen's census office, but it is the spot the character-generation script drops
    /// the player, not a door in a wall: it stands where no doorway is, and a camera put there
    /// starts inside the furniture.
    pub fn leading_to(
        esm: &EsmReader<'_>,
        models: &ModelIndex,
        destination: &CellId,
    ) -> Result<Vec<Self>> {
        let cell_tag = RecordName::new(b"CELL");
        let mut found = Vec::new();
        for record in esm.records() {
            let record = record?;
            if record.name() != cell_tag {
                continue;
            }
            for cell_ref in Cell::references(&record) {
                let cell_ref = cell_ref?;
                if cell_ref.deleted || models.is_editor_marker(&cell_ref.object_id) {
                    continue;
                }
                if let Some(door) = Self::from_ref(&cell_ref)
                    && door.destination == *destination
                {
                    found.push(door);
                }
            }
        }
        Ok(found)
    }
}

/// Turns a stored Z rotation into the direction it means.
///
/// Morrowind's yaw is measured clockwise from north about the **negated** Z axis, so zero faces
/// `+Y` and a quarter turn faces `+X` — the compass convention, and the opposite handedness to
/// what a maths library gives by default. OpenMW spells the same rotation out at
/// `apps/openmw/mwmechanics/combat.cpp:695`.
fn facing_from(yaw: f32) -> Vec3 {
    let (sin, cos) = yaw.sin_cos();
    Vec3::new(sin, cos, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rtxmw_esm::Position;

    fn door_ref(destination_cell: Option<&str>, translation: [f32; 3], yaw: f32) -> CellRef {
        CellRef {
            refnum: 1,
            object_id: "ex_nord_door_01".into(),
            position: Position::default(),
            scale: 1.0,
            deleted: false,
            destination_cell: destination_cell.map(str::to_owned),
            destination: Some(Position {
                translation,
                rotation: [0.0, 0.0, yaw],
            }),
        }
    }

    #[test]
    fn a_reference_without_a_destination_is_not_a_door_that_travels() {
        let mut plain = door_ref(None, [0.0; 3], 0.0);
        plain.destination = None;
        assert_eq!(Door::from_ref(&plain), None);
    }

    #[test]
    fn a_named_destination_is_an_interior_and_an_unnamed_one_is_the_grid_square_it_lands_in() {
        let inside = Door::from_ref(&door_ref(
            Some("Balmora, Guild of Mages"),
            [1.0, 2.0, 3.0],
            0.0,
        ))
        .expect("a reference with a destination is a door");
        assert_eq!(
            inside.destination,
            CellId::Interior("Balmora, Guild of Mages".into())
        );
        assert_eq!(inside.arrival, Vec3::new(1.0, 2.0, 3.0));

        // Seyda Neen's exterior. -10198 / 8192 floors to -2 and -71534 / 8192 to -9; truncating
        // would give -1 and -8, which is a different cell and a real bug in the making.
        let outside = Door::from_ref(&door_ref(None, [-10198.9, -71534.2, 232.0], 0.0))
            .expect("a reference with a destination is a door");
        assert_eq!(outside.destination, CellId::Exterior { x: -2, y: -9 });
    }

    #[test]
    fn the_stored_yaw_reads_as_a_compass_bearing() {
        use std::f32::consts::{FRAC_PI_2, PI};
        let facing = |yaw| {
            Door::from_ref(&door_ref(Some("anywhere"), [0.0; 3], yaw))
                .expect("a reference with a destination is a door")
                .facing
        };

        // Zero is north, and the quarter turns run clockwise through east — the opposite direction
        // to a right-handed rotation about +Z, which is why the axis is negated.
        assert!((facing(0.0) - Vec3::Y).length() < 1e-6);
        assert!((facing(FRAC_PI_2) - Vec3::X).length() < 1e-6);
        assert!((facing(PI) - Vec3::NEG_Y).length() < 1e-6);
        assert!((facing(3.0 * FRAC_PI_2) - Vec3::NEG_X).length() < 1e-6);
    }
}
