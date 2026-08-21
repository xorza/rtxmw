//! Flattening a NIF's node graph, or a cell's heightmap, into one renderable mesh.

use std::collections::HashMap;

use glam::{Affine3A, Mat3, Vec2, Vec3};
use rtxmw_esm::{CELL_SIZE, GRID, LandRecord, SPACING, TEXTURE_GRID};
use rtxmw_nif::{Block, GeometryData, Link, NifFile, Transform};

use crate::material::{AlphaMode, Properties};
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

/// How many patches of ground a cell's side is split into for texturing.
///
/// Twice the texture grid, because the ground is blended between tile *centres* and which four
/// centres a point sits between changes every half tile. See `static_scene::terrain_materials`.
pub(crate) const TERRAIN_QUADRANTS: usize = TEXTURE_GRID * 2;

/// Quads a side in one 512-unit texture tile, at the heightmap's own resolution.
const QUADS_PER_TILE: usize = (GRID - 1) / TEXTURE_GRID;

/// Quads a side in one of those patches — half a tile, so half as many.
const QUADS_PER_PATCH: usize = QUADS_PER_TILE / 2;

/// A vertex position rounded to eighths of a unit, which is how two triangles agree that they meet.
type EdgeKey = (i32, i32, i32);

/// How far a controller chain is followed before it is called a cycle.
///
/// Nothing shipped hangs more than a handful off one node; this is a guard on a malformed file.
const CONTROLLER_CHAIN: usize = 8;

/// How much a run may enclose, by [`Mesh::run_volume`], and still be a sheet.
///
/// An order of magnitude below a cube, which scores 0.068.
const THIN_VOLUME: f32 = 0.006;

/// What fraction of a mesh's edges must be border for it to count as open.
///
/// A closed box has none at all; a sheet has its whole perimeter, which even at a fifty-by-fifty
/// tessellation is a fortieth. What the tolerance buys is room for the odd crack in a solid that
/// was never quite closed.
const BORDER_SHARE: usize = 50;

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
    /// Whether this run is a *sheet* rather than the skin of something solid.
    ///
    /// Morrowind's cloth and foliage are single layers of triangles: a sail, a tapestry, a leaf
    /// card. Light goes through them, and a renderer that lights only the face a normal happens to
    /// point at gets them wrong twice over — the unlit side is black where it should glow, and a
    /// triangle whose normal was authored the other way round to its neighbours' stands out as a
    /// patch. Knowing which runs are sheets is what lets both be answered at once.
    pub thin: bool,
}

impl Submesh {
    /// Triangles in this run.
    pub fn triangle_count(&self) -> usize {
        self.index_count as usize / 3
    }
}

/// Where one geometry block's vertices landed in the flattened mesh.
///
/// Recorded only by [`Mesh::from_nif_tracked`], for a caller that has to bind something per block
/// to a mesh that no longer has blocks in it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GeometrySpan {
    /// The `NiTriShape` or `NiTriStrips` this came from.
    pub(crate) block: Link,
    /// Its `NiSkinInstance`, or none where the block is not skinned.
    pub(crate) skin: Link,
    pub(crate) first_vertex: u32,
    pub(crate) vertex_count: u32,
    /// What was baked into those vertices on the way in.
    pub(crate) placement: Affine3A,
}

/// What a walk of the graph carries along with it, so the recursion stays under an argument count.
#[derive(Debug)]
struct Walk<'a> {
    nif: &'a NifFile,
    materials: &'a mut MaterialTable,
    spans: &'a mut Vec<GeometrySpan>,
    /// The body slot being read out of a file that holds several, or `None` to take all of it.
    slot: Option<&'a str>,
}

impl Mesh {
    /// Marks every run that is a *sheet* — cloth, a rug, a banner — rather than part of something
    /// solid, which is what lets the renderer light it from either face.
    ///
    /// **The material answers first, and answers best.** A run whose alpha is anything but opaque
    /// is a cutout, and Morrowind has no solid cutouts — the mode is set on foliage, banners,
    /// grates, thatch and glass, every one of them a single layer with nothing behind it. That is
    /// authored intent rather than a guess about shape, and it is the only thing that catches a
    /// tree: a canopy's cards join at the branches into a cupped cluster which, measured as
    /// geometry, wraps as much air as a shell around a room does.
    ///
    /// The rest is for the opaque sheets the mode says nothing about — a rug, a tapestry, a sail.
    /// **Two questions there, because either one alone gets it wrong on real data.**
    ///
    /// The first is asked of the whole mesh: has it any edge belonging to only one triangle? A
    /// closed surface has none. This cannot be asked of a run, and the temptation to is the trap —
    /// a run is a *material* boundary, so a bottle with three textures on it arrives as three open
    /// patches and every one of them looks like a sheet. Measured that way, seven eighths of a
    /// furnished interior classifies as cloth, chests and wine bottles included.
    ///
    /// The second is asked of the run: does it enclose anything? That is what separates a sail from
    /// the hull it is rigged to when both arrive in one file, and a room's wall shell — open at the
    /// ends where it meets the next section, but wrapped around the room's air — from the rug lying
    /// inside it.
    ///
    /// Neither is exact. A flat run of a solid — a floor slab modelled as one plane — passes both
    /// and is called a sheet; it is single-sided with nothing behind it, so being lit from the far
    /// side is the harmless answer anyway.
    fn classify_sheets(&mut self, materials: &MaterialTable) {
        let open = self.has_border();
        for index in 0..self.submeshes.len() {
            let run = self.submeshes[index];
            let cut_out = materials.materials()[run.material as usize].alpha != AlphaMode::Opaque;
            self.submeshes[index].thin = cut_out || (open && self.run_volume(&run) < THIN_VOLUME);
        }
    }

    /// What a run bounds, as a fraction of what a solid of the same surface area would.
    ///
    /// Zero for anything flat, whatever its size or its winding, because every tetrahedron about a
    /// plane's own centroid is degenerate. A cube scores 0.068.
    fn run_volume(&self, run: &Submesh) -> f32 {
        let span = run.first_index as usize..(run.first_index + run.index_count) as usize;
        let indices = &self.indices[span];
        if indices.is_empty() {
            return 0.0;
        }
        let mut centre = Vec3::ZERO;
        for &index in indices {
            centre += self.positions[index as usize];
        }
        centre /= indices.len() as f32;

        let mut volume = 0.0;
        let mut area = 0.0;
        for triangle in indices.as_chunks::<3>().0 {
            let a = self.positions[triangle[0] as usize] - centre;
            let b = self.positions[triangle[1] as usize] - centre;
            let c = self.positions[triangle[2] as usize] - centre;
            volume += a.dot(b.cross(c)) / 6.0;
            area += (b - a).cross(c - a).length() / 2.0;
        }
        if area <= 0.0 {
            return 0.0;
        }
        volume.abs() / area.powf(1.5)
    }

    /// Whether the mesh has a border — an edge with a triangle on one side and nothing on the other.
    ///
    /// Edges are keyed by the positions they join rather than by vertex index. Morrowind splits
    /// vertices wherever a texture seam or a hard crease needs two normals at one corner, so two
    /// triangles sharing an edge routinely name four different vertices for it; counted by index,
    /// every seam would read as a border and every solid as a sheet.
    ///
    /// [`BORDER_SHARE`] is where the line falls.
    fn has_border(&self) -> bool {
        let mut edges: HashMap<(EdgeKey, EdgeKey), u32> = HashMap::new();
        for triangle in self.indices.as_chunks::<3>().0 {
            for corner in 0..3 {
                let a = self.quantised(triangle[corner]);
                let b = self.quantised(triangle[(corner + 1) % 3]);
                *edges
                    .entry(if a <= b { (a, b) } else { (b, a) })
                    .or_default() += 1;
            }
        }
        let boundary = edges.values().filter(|&&shared| shared == 1).count();
        boundary * BORDER_SHARE > edges.len()
    }

    /// A vertex position as a key, rounded fine enough to separate real corners and coarse enough
    /// that two triangles meeting along an edge agree about where it is.
    fn quantised(&self, index: u32) -> EdgeKey {
        let p = self.positions[index as usize] * 8.0;
        (p.x.round() as i32, p.y.round() as i32, p.z.round() as i32)
    }

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
                // A water surface is the thinnest thing there is, but it shades through its own
                // path and never asks.
                thin: false,
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
    ///
    /// `quadrant_materials` gives the material of each of the 32 by 32 patches a cell's ground is
    /// split into, row-major. Quads are emitted grouped by it, so each distinct material comes out
    /// as one contiguous submesh — which is what the acceleration structure wants, and what lets a
    /// hit on a patch of sand resolve to sand.
    ///
    /// `stride` takes one vertex in that many on each axis, so a cell too far away to read detail
    /// from costs a fraction of the triangles. It must divide the 64 quads a cell spans, and it
    /// keeps the shared last row and column whatever it is — so two cells built at the *same*
    /// stride still meet exactly. Two at different strides do not, and the seam between them is
    /// what a caller mixing them has to answer for.
    pub fn from_land(land: &LandRecord, quadrant_materials: &[u32], stride: usize) -> Self {
        assert_eq!(
            quadrant_materials.len(),
            TERRAIN_QUADRANTS * TERRAIN_QUADRANTS,
            "a cell is split into {TERRAIN_QUADRANTS} squared patches of ground"
        );
        // Quads a side in a cell, which is one fewer than its vertices because the last row is
        // shared with the neighbour.
        const SPAN: usize = GRID - 1;
        assert!(
            stride.is_power_of_two() && stride < GRID,
            "a stride of {stride} does not divide the {SPAN} quads a cell spans"
        );
        let origin = Vec2::new(
            land.grid_x as f32 * CELL_SIZE,
            land.grid_y as f32 * CELL_SIZE,
        );
        // Quads a side at this stride, and the vertices that span them — one more, because the last
        // row and column belong to the neighbour as much as to this cell.
        let quads = SPAN / stride;
        let side = quads + 1;

        let mut mesh = Self {
            positions: Vec::with_capacity(side * side),
            normals: Vec::with_capacity(side * side),
            uvs: Vec::with_capacity(side * side),
            indices: Vec::with_capacity(quads * quads * 6),
            submeshes: Vec::new(),
        };
        for row in 0..side {
            for column in 0..side {
                let (x, y) = (column * stride, row * stride);
                mesh.positions.push(Vec3::new(
                    origin.x + x as f32 * SPACING,
                    origin.y + y as f32 * SPACING,
                    land.height(x, y),
                ));
                // Per-vertex normals come from the file, so terrain is smooth-shaded without the
                // loader averaging anything. A cell carrying none leaves zeroes, which is the same
                // signal a NIF without normals gives.
                mesh.normals.push(
                    land.normals
                        .get(y * GRID + x)
                        .map_or(Vec3::ZERO, |n| Vec3::from_array(*n)),
                );
                // One texture repeat per tile, which is what the texture indices are laid out on.
                mesh.uvs
                    .push(Vec2::new(x as f32, y as f32) / QUADS_PER_TILE as f32);
            }
        }

        // Which patch of ground a quad takes its material from: the one its *centre* falls in, so a
        // quad coarse enough to span several picks the middle rather than a corner. At stride 1 the
        // half-quad offset rounds away and each pair of quads takes the patch they sit in.
        let patch_of = |column: usize, row: usize| {
            let centre = |quad: usize| (quad * stride + stride / 2) / QUADS_PER_PATCH;
            quadrant_materials[centre(row) * TERRAIN_QUADRANTS + centre(column)]
        };

        // Only the materials some quad actually uses. A coarse stride skips whole patches, and a
        // submesh covering none of the cell would be an empty range in the geometry table.
        let mut distinct: Vec<u32> = (0..quads * quads)
            .map(|quad| patch_of(quad % quads, quad / quads))
            .collect();
        distinct.sort_unstable();
        distinct.dedup();

        for material in distinct {
            let first_index = mesh.indices.len() as u32;
            for row in 0..quads {
                for column in 0..quads {
                    if patch_of(column, row) != material {
                        continue;
                    }
                    let corner = (row * side + column) as u32;
                    let above = corner + side as u32;
                    // Counter-clockwise seen from above, matching the upward normals.
                    mesh.indices.extend([corner, corner + 1, above + 1]);
                    mesh.indices.extend([corner, above + 1, above]);
                }
            }
            mesh.submeshes.push(Submesh {
                first_index,
                index_count: mesh.indices.len() as u32 - first_index,
                material,
                // Terrain is a sheet by construction, and the one sheet nothing shines through.
                thin: false,
            });
        }
        mesh
    }

    /// Flattens every visible geometry block in `nif` into one mesh, interning its materials.
    ///
    /// The table is shared across the whole scene rather than per model, so the indices in
    /// [`Submesh::material`] are already the ones the GPU will use.
    pub fn from_nif(nif: &NifFile, materials: &mut MaterialTable) -> Self {
        Self::from_nif_tracked(nif, materials, &mut Vec::new(), None)
    }

    /// As [`Mesh::from_nif`], recording where each geometry block landed, and taking only one body
    /// slot out of a file that holds several.
    ///
    /// **What binds a rig to the mesh it poses.** Skinning weights are per vertex *of a geometry
    /// block*, and the flattening puts every block's vertices somewhere in one array; without the
    /// spans a rig would have to replay this walk to find them, and two walks that must agree are
    /// two walks that will not.
    ///
    /// **And what cuts a body's skin down to the part being asked for.** A file that skins anything
    /// is a whole region of a body: `B_N_Dark Elf_M_Skins.NIF` holds `Tri Chest` beside
    /// `Tri Left Hand 0`, and both the chest record and the hand record name it. Given a `slot` —
    /// [`rtxmw_esm::PartSlot::shape_name`] — such a file gives up only the shapes named for it, and
    /// its *unskinned* blocks with them: those are authoring leftovers, and `c_m_shirt_extrav_1_c`
    /// carries a stray pair of trousers that would otherwise be worn along with the shirt.
    ///
    /// A file that skins nothing is the whole of one part and is taken entire, however it names its
    /// shapes — the rigid parts are named after their own file, not after the slot.
    pub(crate) fn from_nif_tracked(
        nif: &NifFile,
        materials: &mut MaterialTable,
        spans: &mut Vec<GeometrySpan>,
        slot: Option<&str>,
    ) -> Self {
        let mut mesh = Self::default();
        spans.clear();
        let mut walk = Walk {
            nif,
            materials,
            spans,
            slot: slot.filter(|_| nif.blocks().iter().any(|b| matches!(b, Block::Skin(_)))),
        };
        for &root in nif.roots() {
            mesh.visit(
                &mut walk,
                root,
                Placement::IDENTITY,
                Properties::default(),
                0,
            );
        }
        // Only once every block has been flattened: a border is a property of the whole model, and
        // a mesh half-built has borders where the rest of it has yet to arrive.
        mesh.classify_sheets(materials);
        mesh
    }

    /// Appends `other` whole, rebasing its indices and runs onto what is already here.
    ///
    /// **What an assembled body is built by**: a person is a dozen files, each flattened on its own
    /// and then stacked into one mesh so the whole of them is one acceleration structure. Material
    /// ids are already the shared table's, so nothing is remapped.
    pub(crate) fn absorb(&mut self, other: &Self) {
        let vertex_base = self.positions.len() as u32;
        let index_base = self.indices.len() as u32;
        self.positions.extend_from_slice(&other.positions);
        self.normals.extend_from_slice(&other.normals);
        self.uvs.extend_from_slice(&other.uvs);
        self.indices
            .extend(other.indices.iter().map(|index| index + vertex_base));
        self.submeshes
            .extend(other.submeshes.iter().map(|run| Submesh {
                first_index: run.first_index + index_base,
                ..*run
            }));
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
        walk: &mut Walk<'_>,
        link: Link,
        parent: Placement,
        inherited: Properties,
        depth: u32,
    ) {
        const MAX_DEPTH: u32 = 64;
        if depth > MAX_DEPTH {
            return;
        }
        let Some(block) = walk.nif.resolve(link) else {
            return;
        };

        match block {
            Block::Node(node) => {
                if is_skippable(&node.av.net.name) || node.av.is_hidden() {
                    return;
                }
                let here = parent.then(&node.av.transform);
                let properties = inherited.overridden_by(walk.nif, &node.av.properties);
                for &child in &node.children {
                    self.visit(walk, child, here, properties, depth + 1);
                }
            }
            Block::Geometry(geometry) => {
                if is_skippable(&geometry.av.net.name) || geometry.av.is_hidden() {
                    return;
                }
                if walk
                    .slot
                    .is_some_and(|slot| !names_slot(&geometry.av.net.name, slot))
                {
                    return;
                }
                let Some(Block::GeometryData(data)) = walk.nif.resolve(geometry.data) else {
                    return;
                };
                let properties = inherited.overridden_by(walk.nif, &geometry.av.properties);
                let mut resolved = properties.resolve(walk.nif, walk.materials);
                resolved.scroll = sliding(walk.nif, geometry.av.net.controller);
                let material = walk.materials.intern(resolved);
                let placement = parent.then(&geometry.av.transform);
                let first_vertex = self.positions.len() as u32;
                self.append(data, placement, material);
                if self.positions.len() as u32 != first_vertex {
                    walk.spans.push(GeometrySpan {
                        block: link,
                        skin: geometry.skin,
                        first_vertex,
                        vertex_count: self.positions.len() as u32 - first_vertex,
                        placement: placement.to_affine(),
                    });
                }
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
                // Decided in `classify_sheets` once the whole model is flattened: a run still
                // being extended has not shown what it encloses yet.
                thin: false,
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

    /// The same placement as an affine, for a caller that has to invert it.
    fn to_affine(self) -> Affine3A {
        Affine3A::from_mat3_translation(self.linear, self.translation)
    }

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

/// How fast a `NiUVController` on this block's chain slides its texture, in coordinates a second.
///
/// **Reached from the node rather than through the controller's own target**, which is the rule
/// every other controller here follows — a target points *back* at what it drives, and following it
/// instead works for a mesh and breaks for a `.kf` file.
///
/// Zero where there is none, which is all but fifty of the game's files.
fn sliding(nif: &NifFile, controller: Link) -> Vec2 {
    let mut link = controller;
    for _ in 0..CONTROLLER_CHAIN {
        match nif.resolve(link) {
            Some(Block::UvController(uv)) => {
                let Some(Block::UvData(data)) = nif.resolve(uv.data) else {
                    return Vec2::ZERO;
                };
                return Vec2::from_array(data.scroll(uv.stop - uv.start));
            }
            Some(Block::Controller(other)) => link = other.next,
            _ => return Vec2::ZERO,
        }
    }
    Vec2::ZERO
}

/// Whether a shape's name says it belongs to `slot`.
///
/// A substring rather than a prefix, and case-insensitive, because that is how the shapes read:
/// `Tri Left Hand 0` and `Tri Right Hand 2` are both the hand slot, and the robes spell theirs
/// `Tri chest 0` where the skins spell it `Tri Chest`.
fn names_slot(name: &str, slot: &str) -> bool {
    name.as_bytes()
        .windows(slot.len())
        .any(|window| window.eq_ignore_ascii_case(slot.as_bytes()))
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
    use crate::material::Material;
    use rtxmw_esm::VERTICES;

    /// Adds a run covering the indices appended since `first_index`.
    fn close_run(mesh: &mut Mesh, first_index: u32) {
        mesh.submeshes.push(Submesh {
            first_index,
            index_count: mesh.indices.len() as u32 - first_index,
            material: 0,
            thin: false,
        });
    }

    /// A closed box of side `size` about the origin, as twelve triangles in one run.
    fn box_mesh(size: f32) -> Mesh {
        let h = size / 2.0;
        let mut mesh = Mesh {
            positions: [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ]
            .iter()
            .map(|c| Vec3::from_array(*c) * h)
            .collect(),
            normals: vec![Vec3::Z; 8],
            uvs: vec![Vec2::ZERO; 8],
            indices: vec![
                0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2,
                7, 6, 3, 0, 4, 3, 4, 7,
            ],
            submeshes: Vec::new(),
        };
        close_run(&mut mesh, 0);
        mesh
    }

    /// A flat quad of side `size` in the XY plane, as one run.
    fn quad_mesh(size: f32) -> Mesh {
        let h = size / 2.0;
        let mut mesh = Mesh {
            positions: vec![
                Vec3::new(-h, -h, 0.0),
                Vec3::new(h, -h, 0.0),
                Vec3::new(h, h, 0.0),
                Vec3::new(-h, h, 0.0),
            ],
            normals: vec![Vec3::Z; 4],
            uvs: vec![Vec2::ZERO; 4],
            indices: vec![0, 1, 2, 0, 2, 3],
            submeshes: Vec::new(),
        };
        close_run(&mut mesh, 0);
        mesh
    }

    /// Appends a cylinder wall of `sides` facets — open at both ends, like a mast or a room's wall
    /// shell — and closes a run over it.
    fn add_tube(mesh: &mut Mesh, radius: f32, height: f32, sides: u32) {
        let first_index = mesh.indices.len() as u32;
        let base = mesh.positions.len() as u32;
        for step in 0..sides {
            let angle = std::f32::consts::TAU * step as f32 / sides as f32;
            for &z in &[-height / 2.0, height / 2.0] {
                mesh.positions
                    .push(Vec3::new(radius * angle.cos(), radius * angle.sin(), z));
                mesh.normals.push(Vec3::new(angle.cos(), angle.sin(), 0.0));
                mesh.uvs.push(Vec2::ZERO);
            }
        }
        for step in 0..sides {
            let here = base + step * 2;
            let next = base + ((step + 1) % sides) * 2;
            mesh.indices
                .extend_from_slice(&[here, next, next + 1, here, next + 1, here + 1]);
        }
        close_run(mesh, first_index);
    }

    /// A table holding one opaque material, so the geometry half of the test is what answers.
    fn opaque_table() -> MaterialTable {
        let mut table = MaterialTable::default();
        table.intern(Material::default());
        table
    }

    #[test]
    fn a_sheet_is_told_from_a_solid_by_its_border_and_what_it_wraps() {
        // **What separates a sail from a crate**, on data where nothing says which is which.
        //
        // A closed box has no border at all, so nothing in it is cloth however its runs are split.
        let mut solid = box_mesh(100.0);
        solid.classify_sheets(&opaque_table());
        assert!(!solid.submeshes[0].thin, "a closed box is not cloth");

        // A flat quad is all border and wraps nothing.
        let mut flat = quad_mesh(100.0);
        flat.classify_sheets(&opaque_table());
        assert!(flat.submeshes[0].thin);

        // A tube — a mast, or the wall shell of a room — has a border at each end but wraps its own
        // air, and that is the case the border test alone gets wrong. Morrowind's interiors are
        // built from these, and calling one cloth would let a lamp in the next room shine through
        // the wall between them.
        let mut tube = Mesh::default();
        add_tube(&mut tube, 60.0, 200.0, 12);
        tube.classify_sheets(&opaque_table());
        assert!(!tube.submeshes[0].thin, "a wall shell is not cloth");

        // And the case the *volume* test alone gets wrong: a bottle textured in three parts arrives
        // as three open patches, each of which encloses nothing by itself. Asking the mesh rather
        // than the run is what keeps them solid — here, a box whose top is a run of its own.
        let mut split = box_mesh(100.0);
        let top = split.submeshes[0];
        split.submeshes[0] = Submesh {
            index_count: 6,
            ..top
        };
        split.submeshes.push(Submesh {
            first_index: 6,
            index_count: top.index_count - 6,
            ..top
        });
        split.classify_sheets(&opaque_table());
        assert!(
            split.submeshes.iter().all(|run| !run.thin),
            "splitting a solid by material turned it into cloth"
        );

        // The two signals together are still per run, so a sail rigged to a hull in one file is
        // answered separately from the hull.
        let mut boat = Mesh::default();
        add_tube(&mut boat, 60.0, 200.0, 12);
        let sail_first = boat.indices.len() as u32;
        let base = boat.positions.len() as u32;
        let sail = quad_mesh(100.0);
        boat.positions.extend_from_slice(&sail.positions);
        boat.normals.extend_from_slice(&sail.normals);
        boat.uvs.extend_from_slice(&sail.uvs);
        boat.indices.extend(sail.indices.iter().map(|i| i + base));
        close_run(&mut boat, sail_first);
        boat.classify_sheets(&opaque_table());
        assert!(!boat.submeshes[0].thin, "the hull");
        assert!(boat.submeshes[1].thin, "the sail");

        // And the material has the last word, over a shape that says the opposite. A canopy's cards
        // join at the branches into a cupped cluster that wraps as much air as a closed box does,
        // so nothing about its geometry marks it out — but its alpha does, and Morrowind has no
        // solid cutouts.
        let mut cutout = MaterialTable::default();
        cutout.intern(Material {
            alpha: AlphaMode::Blend,
            ..Material::default()
        });
        let mut foliage = box_mesh(100.0);
        foliage.classify_sheets(&cutout);
        assert!(
            foliage.submeshes[0].thin,
            "a cutout is a sheet whatever shape its triangles make"
        );
    }

    /// Patches of ground in a cell, which is how many materials `from_land` wants.
    const PATCHES: usize = TERRAIN_QUADRANTS * TERRAIN_QUADRANTS;

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
        let mesh = Mesh::from_land(&sloping_land(-2, -9), &[0; PATCHES], 1);
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
        let mesh = Mesh::from_land(&sloping_land(0, 0), &[3; PATCHES], 1);
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
    fn a_stride_samples_the_heightmap_rather_than_averaging_it() {
        // Stride 4 keeps one vertex in four on each axis: 17 a side, 16 quads, 512 triangles
        // against 8,192. The count is arithmetic; what the test is really for is the *values* —
        // every vertex it keeps has to be a vertex the full-detail mesh has, in the same place,
        // because a heightmap that was smoothed on the way down would no longer meet its
        // neighbours at the shared edge.
        let land = sloping_land(3, -4);
        let fine = Mesh::from_land(&land, &[0; PATCHES], 1);
        let coarse = Mesh::from_land(&land, &[0; PATCHES], 4);

        assert_eq!(coarse.positions.len(), 17 * 17);
        assert_eq!(coarse.triangle_count(), 16 * 16 * 2);
        assert_eq!(fine.triangle_count() / coarse.triangle_count(), 16);

        for row in 0..17 {
            for column in 0..17 {
                assert_eq!(
                    coarse.positions[row * 17 + column],
                    fine.positions[row * 4 * GRID + column * 4],
                    "the vertex at ({column}, {row}) of the coarse grid moved"
                );
            }
        }
        // And the shared edge is still shared: the last column is the neighbour's first, whatever
        // the stride, which is the one thing decimation must not drop.
        assert_eq!(
            coarse.positions[16].truncate(),
            fine.positions[GRID - 1].truncate()
        );
        assert_eq!(
            *coarse.positions.last().unwrap(),
            *fine.positions.last().unwrap()
        );
    }

    #[test]
    fn a_coarse_quad_takes_the_material_of_the_patch_under_its_middle() {
        // Patches run 32 to a side and a stride-4 quad spans four of them, so it cannot have all
        // their materials and has to choose. Half the cell one material and half the other, split
        // down the middle: at any stride the boundary lands between quads, so both meshes are two
        // submeshes of equal size — and a quad taking a *corner* patch instead of the middle one
        // would put the boundary half a quad out at stride 4 and nowhere at stride 1.
        let patches: Vec<u32> = (0..PATCHES)
            .map(|p| u32::from(p % TERRAIN_QUADRANTS >= TERRAIN_QUADRANTS / 2))
            .collect();
        for stride in [1, 2, 4, 8] {
            let mesh = Mesh::from_land(&sloping_land(0, 0), &patches, stride);
            assert_eq!(mesh.submeshes.len(), 2, "at stride {stride}");
            let west = mesh.submeshes[0].index_count;
            let east = mesh.submeshes[1].index_count;
            assert_eq!(
                west, east,
                "the split is not down the middle at stride {stride}"
            );
            let quads = (GRID - 1) / stride;
            assert_eq!(
                (west + east) as usize,
                quads * quads * 6,
                "at stride {stride}"
            );
        }

        // A patch the coarse grid never samples must not leave an empty submesh behind. One patch
        // in the far corner, at a stride whose quads all sample the middles of four-patch blocks.
        let mut lone = vec![0u32; PATCHES];
        lone[0] = 7;
        let mesh = Mesh::from_land(&sloping_land(0, 0), &lone, 8);
        assert_eq!(
            mesh.submeshes.len(),
            1,
            "a material no quad uses became a submesh covering nothing"
        );
        assert!(mesh.submeshes.iter().all(|s| s.index_count > 0));
    }

    #[test]
    fn patches_are_grouped_into_one_submesh_per_material() {
        // Two materials in a chequer, which is the case a single run cannot represent: each
        // submesh has to gather the quads of every patch using its material, wherever they are.
        let patches: Vec<u32> = (0..PATCHES)
            .map(|p| ((p / TERRAIN_QUADRANTS + p % TERRAIN_QUADRANTS) % 2) as u32)
            .collect();
        let mesh = Mesh::from_land(&sloping_land(0, 0), &patches, 1);

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
        let mesh = Mesh::from_land(&land, &[0; PATCHES], 1);
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
                    material: 0,
                    thin: false,
                },
                Submesh {
                    first_index: 6,
                    index_count: 3,
                    material: 1,
                    thin: false,
                },
                Submesh {
                    first_index: 9,
                    index_count: 3,
                    material: 0,
                    thin: false,
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
