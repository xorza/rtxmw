//! Assembling one cell into meshes and placed instances.

use std::collections::HashMap;

use glam::{Affine3A, Mat3, Quat, Vec2, Vec3};
use rtxmw_esm::{
    Cell, CellId, CellRef, EsmReader, LandRecord, LandTexture, LightRecord, ObjectRecord, Record,
    RecordName, TEXTURE_GRID,
};
use rtxmw_nif::NifFile;
use rtxmw_vfs::Vfs;

use crate::error::{Result, SceneError};
use crate::light::{Ambient, Light};
use crate::material::{self, Material};
use crate::material_table::MaterialTable;
use crate::mesh::{Bounds, Mesh};
use crate::srgb;
use crate::sun::Sun;

/// What a terrain tile draws with when its index is zero, or names a palette entry that is absent.
///
/// Zero is not "the first texture" but "the region's default", which the shipped art supplies under
/// this name; OpenMW resolves it identically at `components/esmterrain/storage.cpp:376`.
const DEFAULT_TERRAIN_TEXTURE: &str = "_land_default.dds";

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
    /// Lower-cased object id to light data, for the `LIGH` records among them.
    lights: HashMap<String, LightRecord>,
}

impl ModelIndex {
    /// Scans every placeable record in `esm`.
    pub fn build(esm: &EsmReader<'_>) -> Result<Self> {
        let light_tag = RecordName::new(b"LIGH");
        let mut models = HashMap::new();
        let mut lights = HashMap::new();
        for record in esm.records() {
            let record = record?;
            if record.is_deleted() || record.is_ignored() || !ObjectRecord::is_placeable(&record) {
                continue;
            }
            let object = ObjectRecord::parse(&record)?;
            let id = object.id.to_lowercase();
            if let Some(path) = object.model_path() {
                models.insert(id.clone(), path);
            }
            // A light is a placeable object *and* a light: the same record carries the lamp you see
            // and the illumination it casts, and a cell reference to it places both.
            if record.name() == light_tag
                && let Some(light) = LightRecord::parse(&record)?
            {
                lights.insert(id, light);
            }
        }
        Ok(Self { models, lights })
    }

    /// The light data for `object_id`, if the record is a `LIGH`.
    pub fn light_of(&self, object_id: &str) -> Option<LightRecord> {
        self.lights.get(&object_id.to_lowercase()).copied()
    }

    /// The mesh path for `object_id`, if it has one.
    pub fn model_of(&self, object_id: &str) -> Option<&str> {
        self.models
            .get(&object_id.to_lowercase())
            .map(String::as_str)
    }

    /// Whether `object_id` draws with an editor marker rather than real art.
    ///
    /// Morrowind names them `meshes/Marker_*.nif` and there are six in the shipped game — the north
    /// arrow, the door and travel arrows, the temple and divine intervention targets, and the
    /// prison marker. They are placement aids the original engine never drew, and one of them is
    /// filed as a `DOOR`, which makes the distinction matter to more than rendering: `PrisonMarker`
    /// carries a teleport destination and is not a door anybody walks through.
    pub fn is_editor_marker(&self, object_id: &str) -> bool {
        self.model_of(object_id).is_some_and(|path| {
            // Both separators: a `MODL` string is Morrowind's own, so it uses backslashes, and only
            // the `meshes/` this crate prepends is a forward slash. `str::get` rather than an index
            // because a byte offset that lands mid-character panics, and nothing promises these
            // paths are ASCII.
            path.rsplit(['/', '\\'])
                .next()
                .and_then(|file| file.get(..7))
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("marker_"))
        })
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
    /// Point lights placed in the cell, in world space.
    pub lights: Vec<Light>,
    /// The cell's fixed lighting. Absent for a cell that declares none.
    pub ambient: Option<Ambient>,
    /// The sun, for a cell that has a sky. Absent indoors, where there is none.
    pub sun: Option<Sun>,
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
            let mut scene = Self::from_cell(&record, models, vfs)?;
            scene.ambient = cell.ambient.map(|a| Ambient {
                colour: srgb::to_linear(a.ambient),
                sunlight: srgb::to_linear(a.sunlight),
                fog: srgb::to_linear(a.fog),
            });
            return Ok(scene);
        }
        Err(SceneError::NoSuchCell(cell_name.to_owned()))
    }

    /// Loads the exterior cell at grid position `(x, y)`, terrain included.
    ///
    /// Two records make a cell out of doors: the `CELL` that places its objects and the `LAND` that
    /// shapes the ground under them. They are not adjacent in the file, so this walks it once and
    /// takes whichever it meets.
    ///
    /// A cell with no `LAND`, or one whose record carries no heightmap, is legal — a hundred of
    /// them ship — and comes back as its objects over flat ground at the default height.
    pub fn load_exterior(
        esm: &EsmReader<'_>,
        models: &ModelIndex,
        vfs: &Vfs,
        x: i32,
        y: i32,
    ) -> Result<Self> {
        let cell_tag = RecordName::new(b"CELL");
        let land_tag = RecordName::new(b"LAND");
        let texture_tag = RecordName::new(b"LTEX");
        let mut scene: Option<Self> = None;
        let mut land: Option<LandRecord> = None;
        let mut palette: Vec<String> = Vec::new();

        for record in esm.records() {
            let record = record?;
            if record.name() == texture_tag
                && let Some(entry) = LandTexture::parse(&record)?
            {
                let index = entry.index as usize;
                if palette.len() <= index {
                    palette.resize(index + 1, String::new());
                }
                palette[index] = entry.texture;
            } else if record.name() == cell_tag {
                let cell = Cell::parse(&record)?;
                if !cell.is_interior() && cell.grid_x == x && cell.grid_y == y {
                    scene = Some(Self::from_cell(&record, models, vfs)?);
                }
            } else if record.name() == land_tag
                && let Some(parsed) = LandRecord::parse(&record)?
                && (parsed.grid_x, parsed.grid_y) == (x, y)
            {
                land = Some(parsed);
            }
        }

        let mut scene =
            scene.ok_or_else(|| SceneError::NoSuchCell(CellId::Exterior { x, y }.to_string()))?;
        // An exterior carries no `AMBI`: out of doors the ambient *is* the sky, which the original
        // engine drove from weather and time of day. Until that exists this is a fixed overcast
        // daylight, and it is the only thing lighting an exterior at all — there is no sun yet
        // either. Blue-shifted because sky light is, and bright enough that auto-exposure has
        // something to work from.
        scene.ambient = Some(Ambient {
            colour: Vec3::new(0.35, 0.42, 0.55),
            ..Ambient::default()
        });
        // Fixed for now: the orbit is a function of time of day, and nothing tracks that yet.
        scene.sun = Some(Sun::default_daylight());
        if let Some(land) = land {
            let tile_materials = terrain_materials(&land, &palette, &mut scene.materials);
            let mesh = MeshId(scene.meshes.len() as u32);
            scene.meshes.push(Mesh::from_land(&land, &tile_materials));
            // No transform: `Mesh::from_land` places its vertices in the world already, because a
            // heightmap belongs to exactly one cell and could never be instanced elsewhere.
            scene.instances.push(Instance {
                mesh,
                transform: Affine3A::IDENTITY,
            });
        }
        Ok(scene)
    }

    /// Builds a scene from an already-located cell record.
    fn from_cell(record: &Record<'_>, models: &ModelIndex, vfs: &Vfs) -> Result<Self> {
        let mut scene = Self::default();
        let mut by_path: HashMap<String, MeshId> = HashMap::new();

        for cell_ref in Cell::references(record) {
            let cell_ref = cell_ref?;
            // An editor marker has a model and is deliberately not drawn with it: these are the
            // original engine's placement aids, and it never rendered them. 1,145 of them are
            // placed across the shipped game, so leaving them in puts a floating arrow or an
            // obelisk in a great many cells.
            if cell_ref.deleted || models.is_editor_marker(&cell_ref.object_id) {
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

            // A `LIGH` reference places both a mesh and the light it casts, and the two are
            // independent: a lamp whose mesh failed to load should still light the room.
            if let Some(light) = models.light_of(&cell_ref.object_id)
                && light.is_placeable()
            {
                scene.lights.push(Light {
                    position: Vec3::from_array(cell_ref.position.translation),
                    colour: srgb::to_linear(light.colour),
                    radius: light.radius as f32,
                });
            }

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

    /// Height of the highest surface directly below `from`, or `None` if nothing is under it.
    ///
    /// A downward ray against the cell's *visible* geometry, which is an approximation twice over:
    /// Morrowind ships separate collision meshes that this does not read, and a query that walks
    /// every placed triangle is linear in the cell. Both are fine for the once-per-cell question it
    /// answers — where the floor is under a spawn point — and neither survives an actor that has to
    /// ask every frame. That wants the collision meshes and an acceleration structure over them.
    ///
    /// Furniture counts as floor, because to a falling body it is.
    pub fn ground_below(&self, from: Vec3) -> Option<f32> {
        let mut highest: Option<f32> = None;
        for instance in &self.instances {
            let mesh = &self.meshes[instance.mesh.0 as usize];
            for triangle in mesh.indices.chunks_exact(3) {
                let [a, b, c] = [0, 1, 2].map(|corner| {
                    instance
                        .transform
                        .transform_point3(mesh.positions[triangle[corner] as usize])
                });
                let Some(z) = height_at(a, b, c, from.truncate()) else {
                    continue;
                };
                if z <= from.z && highest.is_none_or(|best| z > best) {
                    highest = Some(z);
                }
            }
        }
        highest
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

/// Where a triangle sits directly above or below `at`, or `None` if it does not cover it.
///
/// Barycentric coordinates of the point in the triangle's *plan view*, which is what makes this a
/// vertical query: a triangle covers `at` when both are in `0..1` and sum to no more than one, and
/// the same weights then interpolate the height. A triangle seen edge-on from above covers no
/// ground at all, which is the degenerate denominator.
fn height_at(a: Vec3, b: Vec3, c: Vec3, at: Vec2) -> Option<f32> {
    let (edge_u, edge_v, offset) = (
        b.truncate() - a.truncate(),
        c.truncate() - a.truncate(),
        at - a.truncate(),
    );
    let area = edge_u.x * edge_v.y - edge_v.x * edge_u.y;
    let u = (offset.x * edge_v.y - edge_v.x * offset.y) / area;
    let v = (edge_u.x * offset.y - offset.x * edge_u.y) / area;
    // Negated rather than `u < 0.0 || ...` so that a NaN fails it. A triangle standing exactly on
    // its edge — a wall, of which every cell is full — has zero area in plan and divides by it, and
    // every plain comparison against the NaN that produces would be false, letting it through as a
    // covered triangle with a NaN height.
    if !(u >= 0.0 && v >= 0.0 && u + v <= 1.0) {
        return None;
    }
    Some(a.z + u * (b.z - a.z) + v * (c.z - a.z))
}

/// A material for every terrain tile, interned into the scene's table.
///
/// **A `VTEX` index is one past its `LTEX` index.** Zero is not the palette's first entry but the
/// default texture every cell falls back to, so everything real is offset by one — reading them as
/// the same numbering textures the whole world one slot out, and plausibly, because the palette is
/// grouped by region and the neighbouring entry is usually more of the same terrain.
fn terrain_materials(
    land: &LandRecord,
    palette: &[String],
    materials: &mut MaterialTable,
) -> Vec<u32> {
    (0..TEXTURE_GRID * TEXTURE_GRID)
        .map(|tile| {
            let name = match land.textures.get(tile).copied().unwrap_or(0) {
                0 => DEFAULT_TERRAIN_TEXTURE,
                index => palette
                    .get(index as usize - 1)
                    // A hole rather than a name: the palette is filled by index as records arrive,
                    // so an index no `LTEX` record ever claimed leaves an empty string behind, and
                    // `get` hands that back as happily as a real one.
                    .filter(|name| !name.is_empty())
                    .map_or(DEFAULT_TERRAIN_TEXTURE, String::as_str),
            };
            let texture = materials.intern_texture(&material::texture_path(name));
            materials.intern(Material {
                base_colour: Some(texture),
                ..Material::default()
            })
        })
        .collect()
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
    use rtxmw_esm::LandRecord;

    fn land_with(textures: Vec<u16>) -> LandRecord {
        LandRecord {
            grid_x: 0,
            grid_y: 0,
            heights: vec![0.0; rtxmw_esm::VERTICES],
            normals: Vec::new(),
            textures,
        }
    }

    /// The texture path a tile's material ends up naming.
    fn tile_texture(materials: &MaterialTable, tile_material: u32) -> &str {
        let material = &materials.materials()[tile_material as usize];
        let id = material.base_colour.expect("terrain draws with a texture");
        &materials.textures()[id.0 as usize]
    }

    #[test]
    fn a_texture_index_addresses_one_past_its_palette_entry() {
        let palette = ["sand.tga".to_owned(), "rock.tga".to_owned()];
        // Index 1 is the *first* palette entry, not the second; index 0 is not an entry at all.
        let land = land_with(vec![0, 1, 2]);
        let mut materials = MaterialTable::default();
        let tiles = terrain_materials(&land, &palette, &mut materials);

        assert_eq!(tiles.len(), TEXTURE_GRID * TEXTURE_GRID);
        assert_eq!(
            tile_texture(&materials, tiles[0]),
            "textures/_land_default.dds"
        );
        assert_eq!(tile_texture(&materials, tiles[1]), "textures/sand.dds");
        assert_eq!(tile_texture(&materials, tiles[2]), "textures/rock.dds");
        // Tiles the record says nothing about fall back the same way the zero index does.
        assert_eq!(
            tile_texture(&materials, tiles[3]),
            "textures/_land_default.dds"
        );
        // One material per distinct texture, not one per tile.
        assert_eq!(materials.textures().len(), 3);
    }

    #[test]
    fn an_index_no_palette_entry_claimed_falls_back_rather_than_naming_nothing() {
        // The palette is filled by index as records arrive, so a gap is an empty string rather
        // than a missing element — and `get` returns it as readily as a real name. Left alone it
        // becomes the path "textures/.dds", which resolves to nothing and silently swaps a
        // texture for the renderer's magenta fallback.
        let palette = ["sand.tga".to_owned(), String::new(), "rock.tga".to_owned()];
        let land = land_with(vec![2, 3]);
        let mut materials = MaterialTable::default();
        let tiles = terrain_materials(&land, &palette, &mut materials);

        assert_eq!(
            tile_texture(&materials, tiles[0]),
            "textures/_land_default.dds"
        );
        assert_eq!(tile_texture(&materials, tiles[1]), "textures/rock.dds");
    }

    use crate::mesh::Submesh;

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

    /// A horizontal square floor at height `z`, `size` units on a side from the origin.
    fn slab(z: f32, size: f32) -> Mesh {
        Mesh {
            positions: vec![
                Vec3::new(0.0, 0.0, z),
                Vec3::new(size, 0.0, z),
                Vec3::new(size, size, z),
                Vec3::new(0.0, size, z),
            ],
            normals: vec![Vec3::Z; 4],
            uvs: vec![Vec2::ZERO; 4],
            indices: vec![0, 1, 2, 0, 2, 3],
            submeshes: vec![Submesh {
                first_index: 0,
                index_count: 6,
                material: 0,
            }],
        }
    }

    #[test]
    fn the_ground_below_a_point_is_the_highest_surface_under_it() {
        // Three floors stacked at 0, 100 and 250, the top one covering only a quarter of the plan
        // so a query can fall past it.
        let mut scene = StaticScene {
            meshes: vec![slab(0.0, 100.0), slab(100.0, 100.0), slab(250.0, 25.0)],
            ..StaticScene::default()
        };
        for mesh in 0..3u32 {
            scene.instances.push(Instance {
                mesh: MeshId(mesh),
                transform: Affine3A::IDENTITY,
            });
        }

        // Standing at (50, 20) there is no third floor overhead, so a drop from 300 lands on the
        // second at 100 — the highest surface *at or below* the start, not the highest anywhere.
        assert_eq!(
            scene.ground_below(Vec3::new(50.0, 20.0, 300.0)),
            Some(100.0)
        );
        // From below that floor, the one under it.
        assert_eq!(scene.ground_below(Vec3::new(50.0, 20.0, 60.0)), Some(0.0));
        // Exactly on a surface counts as standing on it rather than falling through.
        assert_eq!(
            scene.ground_below(Vec3::new(50.0, 20.0, 100.0)),
            Some(100.0)
        );
        // At (10, 10) the small top slab does cover the point, and a drop from above finds it.
        assert_eq!(
            scene.ground_below(Vec3::new(10.0, 10.0, 300.0)),
            Some(250.0)
        );
        // Below everything, and outside the plan entirely.
        assert_eq!(scene.ground_below(Vec3::new(50.0, 20.0, -1.0)), None);
        assert_eq!(scene.ground_below(Vec3::new(500.0, 500.0, 300.0)), None);

        // A wall covers no ground: seen from above it is a line, so its plan area is zero and the
        // coverage test divides by it. Every cell is full of them, and one leaking a NaN height
        // into the answer would take the floor with it.
        scene.meshes.push(Mesh {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 400.0),
                Vec3::new(100.0, 0.0, 400.0),
            ],
            normals: vec![Vec3::Y; 3],
            uvs: vec![Vec2::ZERO; 3],
            indices: vec![0, 1, 2],
            submeshes: vec![Submesh {
                first_index: 0,
                index_count: 3,
                material: 0,
            }],
        });
        scene.instances.push(Instance {
            mesh: MeshId(3),
            transform: Affine3A::IDENTITY,
        });
        assert_eq!(
            scene.ground_below(Vec3::new(50.0, 20.0, 300.0)),
            Some(100.0)
        );
    }

    #[test]
    fn a_triangle_seen_edge_on_from_above_covers_no_ground() {
        // Three points on one line in plan: a wall. Its plan area is zero, so the coverage test
        // divides by zero and both weights come back NaN. `ground_below` would discard the answer
        // anyway when it compared the height, but this is the function's own contract — a triangle
        // that covers nothing reports nothing, rather than leaving a NaN for a caller to catch.
        let vertical = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 400.0),
            Vec3::new(100.0, 0.0, 400.0),
        ];
        assert_eq!(
            height_at(vertical[0], vertical[1], vertical[2], Vec2::new(50.0, 20.0)),
            None
        );
        // On the line it lies along, too, where the point genuinely is within the triangle's plan.
        assert_eq!(
            height_at(vertical[0], vertical[1], vertical[2], Vec2::new(50.0, 0.0)),
            None
        );

        // A triangle that does cover the point reports the height interpolated across it: the
        // midpoint of a ramp from 0 to 90 over the same span is 45.
        let ramp = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(90.0, 0.0, 90.0),
            Vec3::new(0.0, 90.0, 0.0),
        ];
        assert_eq!(
            height_at(ramp[0], ramp[1], ramp[2], Vec2::new(45.0, 10.0)),
            Some(45.0)
        );
    }

    #[test]
    fn an_instance_transform_moves_the_ground_with_it() {
        // The query has to work in world space: a floor raised by its instance must answer at its
        // placed height, not its authored one.
        let scene = StaticScene {
            meshes: vec![slab(0.0, 100.0)],
            instances: vec![Instance {
                mesh: MeshId(0),
                transform: Affine3A::from_translation(Vec3::new(0.0, 0.0, 40.0)),
            }],
            ..StaticScene::default()
        };
        assert_eq!(scene.ground_below(Vec3::new(50.0, 20.0, 300.0)), Some(40.0));
    }

    fn index_of(models: &[(&str, &str)]) -> ModelIndex {
        ModelIndex {
            models: models
                .iter()
                .map(|(id, path)| (id.to_lowercase(), (*path).to_owned()))
                .collect(),
            lights: HashMap::new(),
        }
    }

    #[test]
    fn a_marker_mesh_is_recognised_whichever_separator_precedes_it() {
        let index = index_of(&[
            ("NorthMarker", "meshes/Marker_North.nif"),
            // A `MODL` string is Morrowind's own and uses backslashes; only the `meshes/` this
            // crate prepends is a forward slash, so a marker in a subdirectory has both.
            ("Nested", "meshes/x\\Marker_Travel.nif"),
            ("Shouty", "meshes/MARKER_PRISON.NIF"),
            ("ex_nord_door_01", "meshes/d\\Ex_nord_door_01.NIF"),
            // Substring, not prefix: the rule is about the filename beginning with it.
            ("Decoy", "meshes/x\\not_marker_at_all.nif"),
            // Shorter than the prefix, and one whose seventh byte falls inside a character:
            // "marker" is six bytes, so the accent spans the boundary a fixed-width slice would
            // cut at, and an unchecked `&file[..7]` panics rather than answering.
            ("Tiny", "meshes/m.nif"),
            ("Accented", "meshes/markeré.nif"),
        ]);

        for id in ["NorthMarker", "Nested", "Shouty"] {
            assert!(index.is_editor_marker(id), "{id} should be a marker");
        }
        for id in ["ex_nord_door_01", "Decoy", "Tiny", "Accented", "Absent"] {
            assert!(!index.is_editor_marker(id), "{id} should not be a marker");
        }
    }
}
