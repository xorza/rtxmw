//! Assembling one cell into meshes and placed instances.

use std::collections::HashMap;

use glam::{Affine3A, Mat3, Quat, Vec3};
use rtxmw_esm::{Cell, CellRef, EsmReader, ObjectRecord, Record, RecordName};
use rtxmw_nif::NifFile;
use rtxmw_vfs::Vfs;

use crate::error::{Result, SceneError};
use crate::material_table::MaterialTable;
use crate::mesh::{Bounds, Mesh};

/// Index of a mesh within [`StaticScene::meshes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshId(pub u32);

/// One placed copy of a mesh.
#[derive(Debug, Clone, Copy)]
pub struct Instance {
    pub mesh: MeshId,
    /// Model space to world space, in Morrowind units.
    pub transform: Affine3A,
}

/// Maps an object id to the mesh it draws with.
///
/// Built once per content file: a cell names the records it places, not the models, and the same
/// record is placed thousands of times across the world.
#[derive(Debug, Default)]
pub struct ModelIndex {
    /// Lower-cased object id to virtual file system path.
    models: HashMap<String, String>,
}

impl ModelIndex {
    /// Scans every placeable record in `esm`.
    pub fn build(esm: &EsmReader<'_>) -> Result<Self> {
        let mut models = HashMap::new();
        for record in esm.records() {
            let record = record?;
            if record.is_deleted() || record.is_ignored() || !ObjectRecord::is_placeable(&record) {
                continue;
            }
            let object = ObjectRecord::parse(&record)?;
            if let Some(path) = object.model_path() {
                models.insert(object.id.to_lowercase(), path);
            }
        }
        Ok(Self { models })
    }

    /// The mesh path for `object_id`, if it has one.
    pub fn model_of(&self, object_id: &str) -> Option<&str> {
        self.models
            .get(&object_id.to_lowercase())
            .map(String::as_str)
    }

    /// How many records carry a model.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether no record carries a model.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

/// Everything one cell places, ready to hand to a renderer.
///
/// Meshes are deduplicated: a cell that places the same crate forty times holds one mesh and forty
/// instances, which is exactly the split an acceleration structure wants.
#[derive(Debug, Default)]
pub struct StaticScene {
    pub meshes: Vec<Mesh>,
    pub instances: Vec<Instance>,
    /// Every distinct material and texture the cell's meshes name, shared across all of them.
    pub materials: MaterialTable,
    /// References skipped because their record had no model, for reporting.
    pub without_model: Vec<String>,
}

impl StaticScene {
    /// Loads the interior cell named `cell_name`.
    pub fn load_interior(
        esm: &EsmReader<'_>,
        models: &ModelIndex,
        vfs: &Vfs,
        cell_name: &str,
    ) -> Result<Self> {
        let cell_tag = RecordName::new(b"CELL");
        for record in esm.records() {
            let record = record?;
            if record.name() != cell_tag {
                continue;
            }
            let cell = Cell::parse(&record)?;
            if cell.name != cell_name || !cell.is_interior() {
                continue;
            }
            return Self::from_cell(&record, models, vfs);
        }
        Err(SceneError::NoSuchCell(cell_name.to_owned()))
    }

    /// Builds a scene from an already-located cell record.
    fn from_cell(record: &Record<'_>, models: &ModelIndex, vfs: &Vfs) -> Result<Self> {
        let mut scene = Self::default();
        let mut by_path: HashMap<String, MeshId> = HashMap::new();

        for cell_ref in Cell::references(record) {
            let cell_ref = cell_ref?;
            if cell_ref.deleted {
                continue;
            }
            let Some(path) = models.model_of(&cell_ref.object_id) else {
                scene.without_model.push(cell_ref.object_id.clone());
                continue;
            };

            let mesh = match by_path.get(path) {
                Some(&id) => id,
                None => {
                    let bytes = vfs.read(path)?;
                    let nif = NifFile::parse(&bytes).map_err(|source| SceneError::Nif {
                        path: path.to_owned(),
                        source,
                    })?;
                    let id = MeshId(scene.meshes.len() as u32);
                    scene
                        .meshes
                        .push(Mesh::from_nif(&nif, &mut scene.materials));
                    by_path.insert(path.to_owned(), id);
                    id
                }
            };

            // A mesh with nothing visible — a pure collision proxy, say — places no instance.
            if scene.meshes[mesh.0 as usize].is_empty() {
                continue;
            }
            scene.instances.push(Instance {
                mesh,
                transform: world_transform(&cell_ref),
            });
        }
        Ok(scene)
    }

    /// Triangles across every instance, counting each placement separately.
    pub fn placed_triangle_count(&self) -> usize {
        self.instances
            .iter()
            .map(|i| self.meshes[i.mesh.0 as usize].triangle_count())
            .sum()
    }

    /// Triangles across distinct meshes, which is what the acceleration structures hold.
    pub fn unique_triangle_count(&self) -> usize {
        self.meshes.iter().map(Mesh::triangle_count).sum()
    }

    /// World-space bounds of every placed vertex, or `None` when the cell places nothing.
    ///
    /// Walks the geometry rather than the mesh bounds, because an instance may rotate a mesh and
    /// the rotated box is not the box of the rotated bounds.
    pub fn bounds(&self) -> Option<Bounds> {
        let mut bounds: Option<Bounds> = None;
        for instance in &self.instances {
            for &position in &self.meshes[instance.mesh.0 as usize].positions {
                let world = instance.transform.transform_point3(position);
                bounds = Some(match bounds {
                    Some(b) => Bounds {
                        min: b.min.min(world),
                        max: b.max.max(world),
                    },
                    None => Bounds {
                        min: world,
                        max: world,
                    },
                });
            }
        }
        bounds
    }
}

/// Builds a reference's model-to-world transform.
///
/// Rotations are radians applied about the **negated** axes in Z, Y, X order — the convention the
/// original engine used, and not one any maths library will produce by default.
fn world_transform(cell_ref: &CellRef) -> Affine3A {
    let [x, y, z] = cell_ref.position.rotation;
    let rotation = Quat::from_axis_angle(Vec3::NEG_Z, z)
        * Quat::from_axis_angle(Vec3::NEG_Y, y)
        * Quat::from_axis_angle(Vec3::NEG_X, x);
    Affine3A::from_mat3_translation(
        Mat3::from_quat(rotation) * cell_ref.scale,
        Vec3::from_array(cell_ref.position.translation),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(translation: [f32; 3], rotation: [f32; 3], scale: f32) -> CellRef {
        CellRef {
            refnum: 1,
            object_id: String::new(),
            position: rtxmw_esm::Position {
                translation,
                rotation,
            },
            scale,
            deleted: false,
            destination_cell: None,
            destination: None,
        }
    }

    #[test]
    fn an_unrotated_reference_only_translates_and_scales() {
        let cell_ref = reference([10.0, 20.0, 30.0], [0.0; 3], 2.0);

        let transform = world_transform(&cell_ref);
        assert!(
            (transform.transform_point3(Vec3::ZERO) - Vec3::new(10.0, 20.0, 30.0)).length() < 1e-4
        );
        // A unit step along X lands two units away, scaled but not rotated.
        assert!(
            (transform.transform_point3(Vec3::X) - Vec3::new(12.0, 20.0, 30.0)).length() < 1e-4
        );
    }

    #[test]
    fn rotation_is_about_negated_axes() {
        // A quarter turn about Z. Negated, that takes +X to -Y rather than +Y.
        let cell_ref = reference([0.0; 3], [0.0, 0.0, std::f32::consts::FRAC_PI_2], 1.0);

        let transform = world_transform(&cell_ref);
        let turned = transform.transform_point3(Vec3::X);
        assert!(
            (turned - Vec3::NEG_Y).length() < 1e-4,
            "expected -Y, got {turned:?}"
        );
    }
}
