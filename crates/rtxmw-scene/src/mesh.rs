//! Flattening a NIF's node graph into one renderable mesh.

use glam::{Mat3, Vec2, Vec3};
use rtxmw_nif::{Block, GeometryData, Link, NifFile, Transform};

/// Nodes whose subtrees never reach the renderer.
///
/// `RootCollisionNode` holds the collision hull, which is separate geometry that would double every
/// wall if drawn. `AvoidNode` marks volumes the pathfinder steers around and has no visual form.
const NON_VISUAL_NODES: &[&str] = &["RootCollisionNode", "AvoidNode"];

/// Node-name prefixes the original editor used for its own markers.
///
/// Compared case-insensitively, because the content is inconsistent about it.
const MARKER_PREFIXES: &[&str] = &["editormarker", "tri editormarker"];

/// One model's visible geometry, with every node transform already applied.
///
/// A NIF is a hierarchy of transformed pieces, but for static scenery none of it moves relative to
/// the rest, so the hierarchy is baked away at load. That leaves one mesh per model file, which is
/// also the granularity the acceleration structure wants: one BLAS per distinct mesh, instanced by
/// the top-level structure rather than duplicated.
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    /// Always parallel to `positions`. Zero where the source block carried no normal, which a
    /// shader must detect and replace with the face normal.
    pub normals: Vec<Vec3>,
    /// Always parallel to `positions`. Zero where the source block carried no texture coordinates.
    pub uvs: Vec<Vec2>,
    pub indices: Vec<u32>,
}

impl Mesh {
    /// Flattens every visible geometry block in `nif` into one mesh.
    pub fn from_nif(nif: &NifFile) -> Self {
        let mut mesh = Self::default();
        for &root in nif.roots() {
            mesh.visit(nif, root, Placement::IDENTITY, 0);
        }
        mesh
    }

    /// Triangles in the flattened mesh.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Whether anything visible survived the walk.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Axis-aligned bounds, or `None` for an empty mesh.
    pub fn bounds(&self) -> Option<Bounds> {
        let first = *self.positions.first()?;
        let mut bounds = Bounds {
            min: first,
            max: first,
        };
        for &position in &self.positions[1..] {
            bounds.min = bounds.min.min(position);
            bounds.max = bounds.max.max(position);
        }
        Some(bounds)
    }

    /// Walks one block, accumulating `parent` into any geometry beneath it.
    ///
    /// `depth` guards against a malformed file whose links form a cycle: the format allows any
    /// block to reference any other, and nothing in it forbids a loop.
    fn visit(&mut self, nif: &NifFile, link: Link, parent: Placement, depth: u32) {
        const MAX_DEPTH: u32 = 64;
        if depth > MAX_DEPTH {
            return;
        }
        let Some(block) = nif.resolve(link) else {
            return;
        };

        match block {
            Block::Node(node) => {
                if is_skippable(&node.av.net.name) || node.av.is_hidden() {
                    return;
                }
                let here = parent.then(&node.av.transform);
                for &child in &node.children {
                    self.visit(nif, child, here, depth + 1);
                }
            }
            Block::Geometry(geometry) => {
                if is_skippable(&geometry.av.net.name) || geometry.av.is_hidden() {
                    return;
                }
                let Some(Block::GeometryData(data)) = nif.resolve(geometry.data) else {
                    return;
                };
                self.append(data, parent.then(&geometry.av.transform));
            }
            _ => {}
        }
    }

    /// Appends one geometry block, transformed into model space.
    fn append(&mut self, data: &GeometryData, placement: Placement) {
        if data.vertices.is_empty() || data.triangles.is_empty() {
            return;
        }
        let base = self.positions.len() as u32;

        self.positions.reserve(data.vertices.len());
        for &vertex in &data.vertices {
            self.positions
                .push(placement.point(Vec3::from_array(vertex)));
        }

        // Normals and UVs are optional per block, but the mesh's arrays must stay parallel to its
        // positions — so a block without them contributes placeholders rather than shortening the
        // array and desynchronising every later block's indices.
        //
        // The placeholder normal is zero, not an arbitrary axis: zero is detectably invalid, so a
        // shader can fall back to the face normal, whereas a plausible-looking `+Z` would light the
        // surface confidently and wrongly.
        self.normals.reserve(data.vertices.len());
        if data.normals.len() == data.vertices.len() {
            for &normal in &data.normals {
                self.normals
                    .push(placement.direction(Vec3::from_array(normal)));
            }
        } else {
            self.normals
                .extend(std::iter::repeat_n(Vec3::ZERO, data.vertices.len()));
        }

        self.uvs.reserve(data.vertices.len());
        match data.uv_sets.first() {
            Some(set) if set.len() == data.vertices.len() => {
                self.uvs.extend(set.iter().map(|&uv| Vec2::from_array(uv)));
            }
            _ => self
                .uvs
                .extend(std::iter::repeat_n(Vec2::ZERO, data.vertices.len())),
        }

        self.indices.reserve(data.triangles.len() * 3);
        for triangle in &data.triangles {
            for &index in triangle {
                self.indices.push(base + u32::from(index));
            }
        }
    }
}

/// Axis-aligned bounds in model space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl Bounds {
    /// The box's diagonal extent.
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }
}

/// An accumulated node transform, ready to apply to vertices.
///
/// Built once per node rather than per vertex, and stored as a combined linear part so applying it
/// is a matrix multiply and an add.
#[derive(Debug, Clone, Copy)]
struct Placement {
    linear: Mat3,
    translation: Vec3,
}

impl Placement {
    const IDENTITY: Self = Self {
        linear: Mat3::IDENTITY,
        translation: Vec3::ZERO,
    };

    /// Composes `local` onto this placement, parent-first.
    fn then(&self, local: &Transform) -> Self {
        // NIF stores the rotation row-major, and `Mat3::from_cols_array_2d` reads its argument as
        // columns — so the rows must be transposed back. Skipping this mirrors geometry in a way
        // that looks plausible until something asymmetric appears.
        let rotation = Mat3::from_cols_array_2d(&local.rotation).transpose();
        let linear = self.linear * (rotation * local.scale);
        Self {
            linear,
            translation: self.translation + self.linear * Vec3::from_array(local.translation),
        }
    }

    /// Transforms a position.
    fn point(&self, position: Vec3) -> Vec3 {
        self.linear * position + self.translation
    }

    /// Transforms a direction, discarding translation and the uniform scale.
    fn direction(&self, normal: Vec3) -> Vec3 {
        (self.linear * normal).normalize_or_zero()
    }
}

/// Whether a node's name marks it as never-rendered.
fn is_skippable(name: &str) -> bool {
    if NON_VISUAL_NODES.contains(&name) {
        return true;
    }
    let lower = name.to_ascii_lowercase();
    MARKER_PREFIXES.iter().any(|p| lower.starts_with(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_major_rotation_is_transposed_on_the_way_in() {
        // A quarter turn about Z, written the way a NIF stores it: rows, not columns.
        let local = Transform {
            translation: [0.0; 3],
            rotation: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            scale: 1.0,
        };
        let placement = Placement::IDENTITY.then(&local);

        // Read as rows, this rotation takes +X to +Y. Read as columns it would take +X to -Y — the
        // mirrored result that makes this worth pinning.
        let turned = placement.point(Vec3::X);
        assert!((turned - Vec3::Y).length() < 1e-6, "got {turned:?}");
    }

    #[test]
    fn transforms_compose_parent_first() {
        let translate = Transform {
            translation: [10.0, 0.0, 0.0],
            ..Default::default()
        };
        let scale = Transform {
            scale: 2.0,
            ..Default::default()
        };

        // Parent translates, child scales: the child's scale must not scale the parent's offset.
        let placement = Placement::IDENTITY.then(&translate).then(&scale);
        assert_eq!(placement.point(Vec3::X), Vec3::new(12.0, 0.0, 0.0));

        // Parent scales, child translates: now the offset is scaled.
        let other = Placement::IDENTITY.then(&scale).then(&translate);
        assert_eq!(other.point(Vec3::ZERO), Vec3::new(20.0, 0.0, 0.0));
    }

    #[test]
    fn scale_does_not_leak_into_normals() {
        let scale = Transform {
            scale: 5.0,
            ..Default::default()
        };
        let placement = Placement::IDENTITY.then(&scale);
        let normal = placement.direction(Vec3::X);
        assert!((normal.length() - 1.0).abs() < 1e-6, "got {normal:?}");
    }

    #[test]
    fn collision_and_marker_nodes_are_skipped_by_name() {
        assert!(is_skippable("RootCollisionNode"));
        assert!(is_skippable("AvoidNode"));
        assert!(is_skippable("EditorMarker"));
        assert!(is_skippable("Tri EditorMarker"));
        // Case is inconsistent in the shipped data.
        assert!(is_skippable("tri editormarker 01"));

        assert!(!is_skippable("Bip01"));
        assert!(!is_skippable("Tri Chest"));
        // A name merely containing the word is still real geometry.
        assert!(!is_skippable("MyEditorMarkerShelf"));
    }

    #[test]
    fn bounds_cover_every_position() {
        let mesh = Mesh {
            positions: vec![
                Vec3::new(-1.0, 2.0, 0.0),
                Vec3::new(3.0, -4.0, 5.0),
                Vec3::new(0.0, 0.0, 0.0),
            ],
            indices: vec![0, 1, 2],
            ..Default::default()
        };
        let bounds = mesh.bounds().expect("non-empty");
        assert_eq!(bounds.min, Vec3::new(-1.0, -4.0, 0.0));
        assert_eq!(bounds.max, Vec3::new(3.0, 2.0, 5.0));
        assert_eq!(bounds.size(), Vec3::new(4.0, 6.0, 5.0));

        assert!(Mesh::default().bounds().is_none());
    }
}
