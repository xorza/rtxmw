//! Flattening a NIF's node graph, or a cell's heightmap, into one renderable mesh.

use glam::{Mat3, Vec2, Vec3};
use rtxmw_esm::{CELL_SIZE, GRID, LandRecord, SPACING, TEXTURE_GRID};
use rtxmw_nif::{Block, GeometryData, Link, NifFile, Transform};

use crate::material::Properties;
use crate::material_table::MaterialTable;

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
    /// Runs of `indices` sharing one material, covering the whole buffer with no gaps.
    ///
    /// A model is one mesh but rarely one surface — a lantern is glass and metal — so the split has
    /// to survive flattening. Each of these becomes a separate geometry within the model's
    /// acceleration structure, which is what lets a hit name the material it landed on.
    pub submeshes: Vec<Submesh>,
}

/// One run of a mesh's indices drawn with a single material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Submesh {
    pub first_index: u32,
    pub index_count: u32,
    /// Index into the scene's [`MaterialTable`].
    pub material: u32,
}

impl Submesh {
    /// Triangles in this run.
    pub fn triangle_count(&self) -> usize {
        self.index_count as usize / 3
    }
}

impl Mesh {
    /// A unit quad in the XY plane facing up, which is every water surface in the game.
    ///
    /// A unit rather than a placed one, unlike terrain: water is flat, so every cell's surface is
    /// the *same* geometry moved and scaled, and one mesh with an instance per cell is what the
    /// acceleration structure wants. Terrain cannot be shared that way; water is the case where
    /// sharing is free.
    ///
    /// Two triangles. Subdividing it would only matter once the surface displaces, and the wave
    /// model planned for that perturbs the normal rather than the position.
    pub fn water_plane(material: u32) -> Self {
        Self {
            positions: vec![
                Vec3::new(-0.5, -0.5, 0.0),
                Vec3::new(0.5, -0.5, 0.0),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::new(-0.5, 0.5, 0.0),
            ],
            normals: vec![Vec3::Z; 4],
            // Nothing reads these and nothing will: water has no texture, and the wave model
            // planned for it keys off world position, which a hit already knows. Interpolated UVs
            // could not serve that anyway — the instance scale multiplies the quad, not its UVs, so
            // the same range would cover a cell and a puddle alike.
            uvs: vec![Vec2::ZERO; 4],
            indices: vec![0, 1, 2, 0, 2, 3],
            submeshes: vec![Submesh {
                first_index: 0,
                index_count: 6,
                material,
            }],
        }
    }

    /// Turns one exterior cell's heightmap into terrain, in world space.
    ///
    /// **Already placed, unlike a model.** A NIF is authored about its own origin and an instance
    /// transform puts it somewhere; a heightmap is only ever the cell it belongs to, so baking the
    /// cell's corner in costs nothing and spares the caller a transform that could only ever have
    /// one value.
    ///
    /// The grid's last row and column are shared with the neighbouring cell rather than being this
    /// one's own — which is what makes adjacent terrain meet without a seam, and why a cell spans
    /// 64 quads across 65 vertices.
    /// `tile_materials` gives the material of each of the cell's `TEXTURE_GRID` squared texture
    /// tiles, row-major. Quads are emitted grouped by it, so each distinct material comes out as
    /// one contiguous submesh — which is what the acceleration structure wants, and what lets a hit
    /// on a patch of sand resolve to sand.
    ///
    /// One texture per tile with a hard edge between them, where the original engine blended
    /// neighbouring layers across the seam. That is the next thing to fix here and needs a blend
    /// map rather than a material split.
    pub fn from_land(land: &LandRecord, tile_materials: &[u32]) -> Self {
        assert_eq!(
            tile_materials.len(),
            TEXTURE_GRID * TEXTURE_GRID,
            "a cell has one texture tile per {TEXTURE_GRID} squared"
        );
        let origin = Vec2::new(
            land.grid_x as f32 * CELL_SIZE,
            land.grid_y as f32 * CELL_SIZE,
        );

        let mut mesh = Self {
            positions: Vec::with_capacity(land.heights.len()),
            normals: Vec::with_capacity(land.heights.len()),
            uvs: Vec::with_capacity(land.heights.len()),
            indices: Vec::with_capacity((GRID - 1) * (GRID - 1) * 6),
            submeshes: Vec::new(),
        };
        for row in 0..GRID {
            for column in 0..GRID {
                let index = row * GRID + column;
                mesh.positions.push(Vec3::new(
                    origin.x + column as f32 * SPACING,
                    origin.y + row as f32 * SPACING,
                    land.height(column, row),
                ));
                // Per-vertex normals come from the file, so terrain is smooth-shaded without the
                // loader averaging anything. A cell carrying none leaves zeroes, which is the same
                // signal a NIF without normals gives.
                mesh.normals.push(
                    land.normals
                        .get(index)
                        .map_or(Vec3::ZERO, |n| Vec3::from_array(*n)),
                );
                // One texture repeat per four vertex spacings, which is the 512-unit tile the
                // texture indices are laid out on.
                mesh.uvs.push(Vec2::new(column as f32, row as f32) / 4.0);
            }
        }

        // Each tile spans four quads a side, since 16 tiles cover the 64 quads of a cell.
        const QUADS: usize = (GRID - 1) / TEXTURE_GRID;
        let mut distinct: Vec<u32> = tile_materials.to_vec();
        distinct.sort_unstable();
        distinct.dedup();

        for material in distinct {
            let first_index = mesh.indices.len() as u32;
            for (tile, &tile_material) in tile_materials.iter().enumerate() {
                if tile_material != material {
                    continue;
                }
                let (tile_x, tile_y) = (tile % TEXTURE_GRID, tile / TEXTURE_GRID);
                for quad_y in 0..QUADS {
                    for quad_x in 0..QUADS {
                        let row = tile_y * QUADS + quad_y;
                        let column = tile_x * QUADS + quad_x;
                        let corner = (row * GRID + column) as u32;
                        let above = corner + GRID as u32;
                        // Counter-clockwise seen from above, matching the upward normals.
                        mesh.indices.extend([corner, corner + 1, above + 1]);
                        mesh.indices.extend([corner, above + 1, above]);
                    }
                }
            }
            mesh.submeshes.push(Submesh {
                first_index,
                index_count: mesh.indices.len() as u32 - first_index,
                material,
            });
        }
        mesh
    }

    /// Flattens every visible geometry block in `nif` into one mesh, interning its materials.
    ///
    /// The table is shared across the whole scene rather than per model, so the indices in
    /// [`Submesh::material`] are already the ones the GPU will use.
    pub fn from_nif(nif: &NifFile, materials: &mut MaterialTable) -> Self {
        let mut mesh = Self::default();
        for &root in nif.roots() {
            mesh.visit(
                nif,
                root,
                Placement::IDENTITY,
                Properties::default(),
                materials,
                0,
            );
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
    fn visit(
        &mut self,
        nif: &NifFile,
        link: Link,
        parent: Placement,
        inherited: Properties,
        materials: &mut MaterialTable,
        depth: u32,
    ) {
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
                let properties = inherited.overridden_by(nif, &node.av.properties);
                for &child in &node.children {
                    self.visit(nif, child, here, properties, materials, depth + 1);
                }
            }
            Block::Geometry(geometry) => {
                if is_skippable(&geometry.av.net.name) || geometry.av.is_hidden() {
                    return;
                }
                let Some(Block::GeometryData(data)) = nif.resolve(geometry.data) else {
                    return;
                };
                let properties = inherited.overridden_by(nif, &geometry.av.properties);
                let resolved = properties.resolve(nif, materials);
                let material = materials.intern(resolved);
                self.append(data, parent.then(&geometry.av.transform), material);
            }
            _ => {}
        }
    }

    /// Appends one geometry block, transformed into model space and tagged with its material.
    fn append(&mut self, data: &GeometryData, placement: Placement, material: u32) {
        if data.vertices.is_empty() || data.triangles.is_empty() {
            return;
        }
        let base = self.positions.len() as u32;
        let first_index = self.indices.len() as u32;

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

        self.push_run(
            first_index,
            self.indices.len() as u32 - first_index,
            material,
        );
    }

    /// Extends the last run when it shares `material`, or starts a new one.
    ///
    /// Adjacent blocks with the same material merge; non-adjacent ones do not. Models are authored
    /// with their pieces grouped, so this collapses most of them and keeps the geometry count in
    /// the acceleration structure down — and it does so without reordering indices, which would
    /// invalidate the ranges a build reads.
    fn push_run(&mut self, first_index: u32, index_count: u32, material: u32) {
        match self.submeshes.last_mut() {
            Some(last) if last.material == material => last.index_count += index_count,
            _ => self.submeshes.push(Submesh {
                first_index,
                index_count,
                material,
            }),
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

    /// The midpoint of the box.
    pub fn centre(&self) -> Vec3 {
        (self.min + self.max) * 0.5
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
    use rtxmw_esm::VERTICES;

    /// Texture tiles in a cell, which is how many materials `from_land` wants.
    const TILES: usize = TEXTURE_GRID * TEXTURE_GRID;

    /// A heightmap sloping up to the east, on the cell at `(grid_x, grid_y)`.
    fn sloping_land(grid_x: i32, grid_y: i32) -> LandRecord {
        LandRecord {
            grid_x,
            grid_y,
            heights: (0..VERTICES).map(|i| (i % GRID) as f32 * 10.0).collect(),
            normals: vec![[0.0, 0.0, 1.0]; VERTICES],
            textures: Vec::new(),
        }
    }

    #[test]
    fn terrain_lands_where_its_cell_is() {
        // Cell (-2, -9) is Seyda Neen's, and its south-west corner is at -2 and -9 times the
        // 8,192-unit cell. A heightmap placed at the origin regardless would put every cell in the
        // world on top of every other.
        let mesh = Mesh::from_land(&sloping_land(-2, -9), &[0; TILES]);
        assert_eq!(mesh.positions.len(), VERTICES);
        assert_eq!(
            mesh.positions[0].truncate(),
            glam::Vec2::new(-16384.0, -73728.0)
        );

        // The far corner is one cell along in each direction, *not* one vertex short of it: the
        // shared edge is what makes neighbouring cells meet.
        let far = mesh.positions[VERTICES - 1];
        assert_eq!(
            far.truncate(),
            glam::Vec2::new(-16384.0 + 8192.0, -73728.0 + 8192.0)
        );

        // Height comes through unscaled — the record already decoded it into world units.
        assert_eq!(mesh.positions[0].z, 0.0);
        assert_eq!(mesh.positions[GRID - 1].z, 640.0);
    }

    #[test]
    fn every_quad_becomes_two_triangles_covering_it() {
        let mesh = Mesh::from_land(&sloping_land(0, 0), &[3; TILES]);
        // 64 quads a side, two triangles each.
        assert_eq!(mesh.triangle_count(), (GRID - 1) * (GRID - 1) * 2);
        // One material across every tile, so one submesh.
        assert_eq!(mesh.submeshes.len(), 1);
        assert_eq!(mesh.submeshes[0].material, 3);
        assert_eq!(mesh.submeshes[0].index_count as usize, mesh.indices.len());
        assert!(mesh.indices.iter().all(|i| (*i as usize) < VERTICES));

        // Wound counter-clockwise seen from above, so a face normal computed from the winding
        // points up rather than into the ground.
        let corner = |i: usize| mesh.positions[mesh.indices[i] as usize];
        let face = (corner(1) - corner(0)).cross(corner(2) - corner(0));
        assert!(face.z > 0.0, "first triangle faces {face:?}");
    }

    #[test]
    fn tiles_are_grouped_into_one_submesh_per_material() {
        // Two materials in a chequer, which is the case a single run cannot represent: each
        // submesh has to gather the quads of every tile using its material, wherever they are.
        let tiles: Vec<u32> = (0..TILES)
            .map(|t| ((t / TEXTURE_GRID + t % TEXTURE_GRID) % 2) as u32)
            .collect();
        let mesh = Mesh::from_land(&sloping_land(0, 0), &tiles);

        assert_eq!(mesh.submeshes.len(), 2);
        // Between them they cover every quad exactly once. Comparing the submeshes against the
        // buffer only proves they agree with each other — a grouping that emitted every tile under
        // *every* material would pass that and draw the cell twice.
        assert_eq!(mesh.indices.len(), (GRID - 1) * (GRID - 1) * 6);
        let covered: u32 = mesh.submeshes.iter().map(|s| s.index_count).sum();
        assert_eq!(covered as usize, mesh.indices.len());
        assert_eq!(mesh.submeshes[0].first_index, 0);
        assert_eq!(mesh.submeshes[1].first_index, mesh.submeshes[0].index_count);
        // A chequer splits them evenly.
        assert_eq!(mesh.submeshes[0].index_count, mesh.submeshes[1].index_count);
    }

    #[test]
    fn a_cell_without_normals_leaves_them_zero_rather_than_guessing() {
        let mut land = sloping_land(0, 0);
        land.normals.clear();
        let mesh = Mesh::from_land(&land, &[0; TILES]);
        // Parallel to positions either way, which is what every consumer relies on.
        assert_eq!(mesh.normals.len(), mesh.positions.len());
        assert!(mesh.normals.iter().all(|n| *n == Vec3::ZERO));
    }

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
    fn adjacent_blocks_sharing_a_material_become_one_submesh() {
        let mut mesh = Mesh::default();
        // Two runs of material 0, then material 1, then material 0 again. Only the adjacent pair
        // may merge — collapsing the other two would need reordering, which breaks the index ranges
        // an acceleration structure builds from.
        mesh.push_run(0, 3, 0);
        mesh.push_run(3, 3, 0);
        mesh.push_run(6, 3, 1);
        mesh.push_run(9, 3, 0);

        assert_eq!(
            mesh.submeshes,
            vec![
                Submesh {
                    first_index: 0,
                    index_count: 6,
                    material: 0
                },
                Submesh {
                    first_index: 6,
                    index_count: 3,
                    material: 1
                },
                Submesh {
                    first_index: 9,
                    index_count: 3,
                    material: 0
                },
            ]
        );
        assert_eq!(mesh.submeshes[0].triangle_count(), 2);
        // The runs must tile the index buffer with no gap, or geometry vanishes at build time.
        let covered: u32 = mesh.submeshes.iter().map(|s| s.index_count).sum();
        assert_eq!(covered, 12);
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
