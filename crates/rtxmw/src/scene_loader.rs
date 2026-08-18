//! Which cell the engine opens in, where to stand in it, and how to report it.

use glam::Vec3;
use rtxmw_scene::{LoadedCell, StaticScene};

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

/// Where to put the camera in a freshly loaded cell.
///
/// The centre of the cell's own geometry, which for the default cell lands inside the larger room.
/// There is no general rule here — a real spawn point comes from the cell's own markers, which is a
/// later milestone's problem — so this is a fixture choice that happens to work.
pub(crate) fn viewpoint(scene: &StaticScene) -> Vec3 {
    scene.bounds().map_or(Vec3::ZERO, |b| b.centre())
}
