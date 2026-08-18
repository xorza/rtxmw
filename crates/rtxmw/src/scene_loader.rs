//! Which cell the engine opens in, where to stand in it, and how to report it.

use glam::Vec3;
use rtxmw_scene::LoadedCell;

use crate::camera::Camera;

/// The cell the engine opens in until there is a way to choose one.
pub(crate) const DEFAULT_CELL: &str = "Seyda Neen, Census and Excise Office";

/// A one-line summary of what was loaded, for startup output.
///
/// Takes the name rather than assuming [`DEFAULT_CELL`]: there is only one cell today, and a
/// summary that hardcoded its name would start printing the wrong one the moment there are two.
///
/// Missing textures are worth naming rather than swallowing: the shipped meshes reference art that
/// was removed, so a small count is expected and a large one means the path fixups have drifted.
pub(crate) fn describe(name: &str, cell: &LoadedCell) -> String {
    let missing = cell.missing_textures();
    format!(
        "{name}: {} meshes, {} instances, {} lights, {} textures ({missing} missing)",
        cell.scene.meshes.len(),
        cell.scene.instances.len(),
        cell.scene.lights.len(),
        cell.textures.len(),
    )
}

/// How far above a door's arrival point the traveller's eyes are, in world units.
///
/// **The arrival is not where the traveller's feet go.** Measured against the floor directly
/// beneath it, across sixteen arrivals in twelve interiors, it sits a median of 89 units up with a
/// spread of 22 to 144 — an authored marker at roughly an actor's centre, not a standing position.
/// The original engine drops the player to the ground on arrival and the height is only ever
/// approximate, which is why it varies.
///
/// Taking the median as an actor's half-height, the eyes sit about nine tenths of one above the
/// centre — the ratio a human has, and the point OpenMW measures line of sight from
/// (`mwphysics/mtphysics.cpp:767`). That puts the eye 80 units above the arrival, or 81% of the way
/// up a 194-unit `Ex_nord_door_01`, which is where a person's eyes are in a doorway.
const EYE_ABOVE_ARRIVAL: f32 = 80.0;

/// Where to stand in a freshly loaded cell, and which way to look.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Viewpoint {
    pub(crate) position: Vec3,
    pub(crate) forward: Vec3,
}

impl Viewpoint {
    /// Where the game itself would put a traveller entering `cell`.
    ///
    /// The arrival point of a door that leads here, raised to eye height. Every such door names a
    /// spot the original game teleports the player to, so it is a real standing position facing
    /// into the room rather than a guess — which the geometry centroid it replaces was not: in the
    /// default cell that landed near the ceiling and looked out through the roof.
    ///
    /// The first door in file order, because they are all equally valid and file order is at least
    /// stable. Picking between them would need a rule the data does not supply.
    pub(crate) fn entering(cell: &LoadedCell) -> Self {
        match cell.entrances.first() {
            Some(door) => Self {
                position: door.arrival + Vec3::Z * EYE_ABOVE_ARRIVAL,
                forward: door.facing,
            },
            // Nothing leads into this cell, so there is no authored answer — fall back on the
            // middle of its own geometry, which at least sits inside the cell's extent.
            None => Self {
                position: cell.scene.bounds().map_or(Vec3::ZERO, |b| b.centre()),
                forward: Vec3::X,
            },
        }
    }

    /// A camera standing here.
    pub(crate) fn camera(&self) -> Camera {
        Camera::looking(self.position, self.forward)
    }
}
