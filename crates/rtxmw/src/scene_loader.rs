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

/// How far above an actor's position its eyes are, in world units.
///
/// About 1.77 m at Morrowind's scale. A door's arrival point is where the traveller *stands*, and a
/// camera left there looks out of their ankles. OpenMW carries the same offset as
/// `MWRender::Camera::mHeight` (`apps/openmw/mwrender/camera.cpp:63`).
const EYE_HEIGHT: f32 = 124.0;

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
                position: door.arrival + Vec3::Z * EYE_HEIGHT,
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
