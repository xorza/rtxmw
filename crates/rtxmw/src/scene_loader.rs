//! Which cell the engine opens in, where to stand in it, and how to report it.

use glam::Vec3;
use rtxmw_scene::{CellId, LoadedCell};

use crate::camera::Camera;

/// Cells loaded either side of the one the camera is in.
///
/// Radius 3 is a 7x7 block of 49 cells — about six across, and odd so the camera's own cell is the
/// middle one rather than a corner between four. Morrowind's own active grid is radius 1, which is
/// what its draw distance could afford.
pub(crate) const GRID_RADIUS: i32 = 3;

/// How far inside a cell the camera must be before the grid recentres on it, in world units.
///
/// A twelfth of a cell. Without it the loaded block would follow every step across a boundary and
/// every step back, reloading the world each time; with it the camera has to commit.
pub(crate) const GRID_HYSTERESIS: f32 = 683.0;

/// Where the loaded block should recentre, or `None` to leave it where it is.
///
/// Separated from the loading it triggers because the two fail differently: this is a decision
/// about a position and can be checked exactly, while what follows it is thirty milliseconds of
/// file reading and an acceleration structure rebuild.
pub(crate) fn next_centre(position: Vec3, loaded: &CellId) -> Option<CellId> {
    let settled = CellId::settled_in(position.x, position.y, GRID_HYSTERESIS)?;
    // An interior never streams, and a block already centred here has nowhere to move to.
    (matches!(settled, CellId::Exterior { .. }) && settled != *loaded).then_some(settled)
}

/// The cell the engine opens in when the command line names none.
pub(crate) const DEFAULT_CELL: &str = "Seyda Neen, Census and Excise Office";

/// The cell a command-line argument names, or [`DEFAULT_CELL`] when there is none.
///
/// **A pair of integers is an exterior and anything else is an interior's name**, which is exactly
/// how Morrowind addresses the two: outdoors has coordinates, indoors has a name. So `-2,-9` is
/// Seyda Neen's shore and `"Balmora, Guild of Mages"` is a building, with no flag to say which.
///
/// Takes the absent case too, because every caller has one and each writing out the default meant
/// three copies of it.
pub(crate) fn cell_argument(argument: Option<&str>) -> CellId {
    let Some(argument) = argument else {
        return CellId::Interior(DEFAULT_CELL.to_owned());
    };
    if let Some((x, y)) = argument.split_once(',')
        && let (Ok(x), Ok(y)) = (x.trim().parse(), y.trim().parse())
    {
        return CellId::Exterior { x, y };
    }
    CellId::Interior(argument.to_owned())
}

/// A one-line summary of what was loaded, for startup output.
///
/// The name comes from the cell itself rather than from the caller, so a summary cannot announce a
/// different cell than the one it is describing — an interior is named and an exterior is a grid
/// position, and `CellId` is what knows how each is written down.
///
/// Missing textures are worth naming rather than swallowing: the shipped meshes reference art that
/// was removed, so a small count is expected and a large one means the path fixups have drifted.
pub(crate) fn describe(cell: &LoadedCell) -> String {
    let missing = cell.missing_textures();
    format!(
        "{}: {} meshes, {} instances, {} lights, {} textures ({missing} missing)",
        cell.id,
        cell.scene.meshes.len(),
        cell.scene.instances.len(),
        cell.scene.lights.len(),
        cell.textures.len(),
    )
}

/// Standing eye height above the floor, in world units.
///
/// An actor's height is twice the median height of a door's arrival point above the floor beneath
/// it — 89 units, measured across sixteen arrivals in twelve interiors — because the arrival marks
/// roughly an actor's centre. Eyes sit about nine tenths of the way up, the ratio a human has and
/// close to where OpenMW measures line of sight from (`mwphysics/mtphysics.cpp:767`). That is 160
/// units, or 83% of the way up a 194-unit `Ex_nord_door_01`, which is where a person's eyes are in
/// a doorway.
const EYE_HEIGHT: f32 = 160.0;

/// How far a traveller is lifted before being dropped onto the floor.
///
/// Straight from OpenMW's `World::adjustPosition` (`mwworld/worldimp.cpp:1207`), which raises by
/// this much and then traces down, taking whichever is lower. It matters when the authored arrival
/// sits a hair *under* the floor, where a trace from the arrival itself would fall through to the
/// storey below.
const GROUND_CLEARANCE: f32 = 20.0;

/// Where to stand in a freshly loaded cell, and which way to look.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Viewpoint {
    pub(crate) position: Vec3,
    pub(crate) forward: Vec3,
}

impl Viewpoint {
    /// Where the game itself would put a traveller entering `cell`.
    ///
    /// A door that leads here names a spot the original game teleports the player to, so it is a
    /// real position facing into the room rather than a guess — which the geometry centroid it
    /// replaces was not: in the default cell that landed near the ceiling and looked out through
    /// the roof.
    ///
    /// **Only the horizontal part of the arrival is taken at face value.** Its height is authored
    /// loosely, because the original engine drops the arriving actor to the ground and throws the
    /// stored height away — measured, it lands anywhere from 22 to 144 units above the floor. So
    /// this drops too, and stands the traveller on whatever it finds, which is what makes the
    /// camera's height above the floor the same in every cell.
    ///
    /// The first door in file order, because they are all equally valid and file order is at least
    /// stable. Picking between them would need a rule the data does not supply.
    pub(crate) fn entering(cell: &LoadedCell) -> Self {
        let door = cell.entrances.first();
        // Nothing leads into this cell, so there is no authored answer at all — the middle of its
        // own geometry at least sits inside the cell's extent. It goes through the same drop as a
        // door does, which is what stops it landing near the ceiling the way it used to.
        let above = door.map_or_else(
            || cell.scene.bounds().map_or(Vec3::ZERO, |b| b.centre()),
            |door| door.arrival,
        );
        // No ground beneath leaves the starting height as the best guess there is, which is what
        // it was before anything traced.
        let feet = cell
            .scene
            .ground_below(above + Vec3::Z * GROUND_CLEARANCE)
            .unwrap_or(above.z);
        Self {
            position: above.truncate().extend(feet + EYE_HEIGHT),
            forward: door.map_or(Vec3::X, |door| door.facing),
        }
    }

    /// A camera standing here.
    pub(crate) fn camera(&self) -> Camera {
        Camera::looking(self.position, self.forward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_block_recentres_only_once_the_camera_has_committed() {
        let here = CellId::Exterior { x: 0, y: 0 };
        let at = |x: f32, y: f32| next_centre(Vec3::new(x, y, 0.0), &here);

        // Well inside the cell it is already centred on: nothing to do.
        assert_eq!(at(4000.0, 4000.0), None);
        // Well inside the next one along: move.
        assert_eq!(at(12000.0, 4000.0), Some(CellId::Exterior { x: 1, y: 0 }));
        assert_eq!(at(-4000.0, 4000.0), Some(CellId::Exterior { x: -1, y: 0 }));
        // Diagonally, too.
        assert_eq!(
            at(-4000.0, -4000.0),
            Some(CellId::Exterior { x: -1, y: -1 })
        );

        // **Straddling the boundary commits to neither.** Without that the block would reload on
        // every step across and every step back, which at thirty milliseconds a load is a stall
        // per footfall.
        assert_eq!(at(8192.0 + 10.0, 4000.0), None);
        assert_eq!(at(8192.0 - 10.0, 4000.0), None);

        // An interior is never a destination: nothing streams into one, a door is the way in.
        assert_eq!(
            next_centre(
                Vec3::new(12000.0, 4000.0, 0.0),
                &CellId::Interior("a room".into())
            ),
            Some(CellId::Exterior { x: 1, y: 0 })
        );
    }

    #[test]
    fn a_pair_of_integers_is_an_exterior_and_anything_else_is_a_name() {
        let cell = |argument| cell_argument(Some(argument));
        assert_eq!(cell("-2,-9"), CellId::Exterior { x: -2, y: -9 });
        // Spaces around the comma are what a shell leaves behind after quoting.
        assert_eq!(cell(" 3 , 4 "), CellId::Exterior { x: 3, y: 4 });

        // Interior names contain commas — most of them do — so a comma alone cannot be the test.
        // Only a pair that *parses* as integers is a grid position.
        assert_eq!(
            cell("Balmora, Guild of Mages"),
            CellId::Interior("Balmora, Guild of Mages".into())
        );
        assert_eq!(
            cell("Seyda Neen, Census and Excise Office"),
            CellId::Interior("Seyda Neen, Census and Excise Office".into())
        );
        assert_eq!(cell("Nowhere"), CellId::Interior("Nowhere".into()));
        // Half a pair is a name too, not an error: there is a cell called that or there is not.
        assert_eq!(cell("-2,"), CellId::Interior("-2,".into()));

        // No argument at all is the cell the engine opens in.
        assert_eq!(
            cell_argument(None),
            CellId::Interior(DEFAULT_CELL.to_owned())
        );
    }
    use glam::{Affine3A, Vec2};
    use rtxmw_scene::{CellId, Door, Instance, Mesh, MeshId, StaticScene, Submesh};

    /// A cell whose only geometry is a wide floor at `floor`, with `entrances` leading into it.
    fn cell_with(floor: f32, entrances: Vec<Door>) -> LoadedCell {
        LoadedCell {
            id: CellId::Interior("a test cell".into()),
            scene: StaticScene {
                meshes: vec![Mesh {
                    positions: vec![
                        Vec3::new(-500.0, -500.0, floor),
                        Vec3::new(500.0, -500.0, floor),
                        Vec3::new(500.0, 500.0, floor),
                        Vec3::new(-500.0, 500.0, floor),
                    ],
                    normals: vec![Vec3::Z; 4],
                    uvs: vec![Vec2::ZERO; 4],
                    indices: vec![0, 1, 2, 0, 2, 3],
                    submeshes: vec![Submesh {
                        first_index: 0,
                        index_count: 6,
                        material: 0,
                    }],
                }],
                instances: vec![Instance {
                    mesh: MeshId(0),
                    transform: Affine3A::IDENTITY,
                }],
                ..StaticScene::default()
            },
            textures: Vec::new(),
            entrances,
        }
    }

    fn door_at(arrival: Vec3) -> Door {
        Door {
            destination: CellId::Interior("wherever".into()),
            arrival,
            facing: Vec3::NEG_Y,
        }
    }

    #[test]
    fn the_eye_is_a_fixed_height_above_the_floor_whatever_the_arrival_says() {
        // The two arrivals differ by 122 units of height — the spread the authored data actually
        // has — over the same floor. Standing on the floor is what makes them agree; using the
        // arrival's own height, as this did at first, would put the second camera 122 units nearer
        // the ceiling than the first.
        let low = Viewpoint::entering(&cell_with(
            192.0,
            vec![door_at(Vec3::new(10.0, 20.0, 214.0))],
        ));
        let high = Viewpoint::entering(&cell_with(
            192.0,
            vec![door_at(Vec3::new(10.0, 20.0, 336.0))],
        ));

        assert_eq!(low.position, Vec3::new(10.0, 20.0, 192.0 + EYE_HEIGHT));
        assert_eq!(high.position, low.position);
        // The horizontal part *is* taken at face value, and so is the facing.
        assert_eq!(low.position.truncate(), Vec2::new(10.0, 20.0));
        assert_eq!(low.forward, Vec3::NEG_Y);
    }

    #[test]
    fn an_arrival_just_under_the_floor_still_stands_on_it() {
        // Why the clearance exists: a marker authored a hair below the surface would otherwise
        // trace downward from beneath it and find the storey below, or nothing at all.
        let cell = cell_with(192.0, vec![door_at(Vec3::new(0.0, 0.0, 192.0 - 1.0))]);
        assert_eq!(
            Viewpoint::entering(&cell).position.z,
            192.0 + EYE_HEIGHT,
            "the arrival fell through the floor it was standing just under"
        );
    }

    #[test]
    fn a_cell_nothing_leads_into_falls_back_to_its_middle_and_still_drops() {
        // The centroid of a floor-only cell is the floor itself, so the interesting part is that it
        // goes through the same drop: the fallback used to be handed to the camera unchanged, which
        // is how the very first version ended up near the ceiling.
        let mut cell = cell_with(192.0, Vec::new());
        cell.scene.instances.push(Instance {
            mesh: MeshId(0),
            transform: Affine3A::from_translation(Vec3::Z * 400.0),
        });
        // Two floors now, at 192 and 592, so the centroid is at 392 — off the ground either way.
        let viewpoint = Viewpoint::entering(&cell);
        assert_eq!(viewpoint.position.z, 192.0 + EYE_HEIGHT);
        assert_eq!(viewpoint.forward, Vec3::X);
    }

    #[test]
    fn a_cell_with_no_ground_under_the_arrival_keeps_its_height() {
        // Nothing to stand on, so the authored height is the best answer there is rather than a
        // fall to zero.
        let mut cell = cell_with(192.0, vec![door_at(Vec3::new(10.0, 20.0, 300.0))]);
        cell.scene.instances.clear();
        assert_eq!(Viewpoint::entering(&cell).position.z, 300.0 + EYE_HEIGHT);
    }
}
