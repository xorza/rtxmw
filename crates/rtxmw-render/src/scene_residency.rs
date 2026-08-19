//! Which cells are on the device, and the tables built from what they place.

use std::collections::HashMap;

use glam::Vec3;
use rtxmw_gpu::{Buffer, Device, RayTracingLimits, Uploader};
use rtxmw_scene::{
    CellId, Instance, Light, MaterialKind, MaterialTable, Mesh, MeshId, StaticScene, TerrainLayers,
    TextureId,
};
use rtxmw_texture::Texture;

use crate::geometry_buffers::GeometryBuffers;
use crate::gpu_light::GpuLight;
use crate::light_grid::{LightGrid, LightGridExtent};
use crate::material_buffers::MaterialBuffers;
use crate::scene_acceleration::SceneAcceleration;
use crate::texture_array::TextureArray;
use crate::visibility_pass::Lighting;

/// One cell's placements, held for as long as it is resident.
///
/// How thick an exterior's fog is, until the weather system says.
///
/// The record carries fog only for interiors; an exterior's belongs to the weather, which is per
/// region and per weather out of the original engine's ini fallbacks rather than the ESM. This is
/// the number to replace when that arrives rather than one to tune for its own sake.
const OUTDOOR_FOG_DENSITY: f32 = 0.75;

/// What an interior's recorded density is worth against an exterior's.
///
/// **The same dial does not mean the same thing indoors.** The original engine set fog by a start
/// and an end distance scaled to the view range, so a density of 0.75 filled a room and the same
/// 0.75 filled a valley — very different amounts of air. This renderer's extinction is absolute, so
/// the conversion has to be explicit: without it a room reads as a smoke-filled cellar at the
/// setting that gives a landscape its haze.
const INDOOR_FOG_SCALE: f32 = 0.035;

/// Placements only. Everything a cell *names* — its meshes, its textures, its materials — belongs
/// to the renderer rather than to the cell, because the cell next door names most of the same
/// things and uploading them twice is what made a block reload cost eighty milliseconds.
#[derive(Debug)]
struct ResidentCell {
    id: CellId,
    /// Instances whose `MeshId` addresses the renderer's mesh slots, not the cell's own numbering.
    instances: Vec<Instance>,
    lights: Vec<Light>,
}

/// Every cell on the device, and the assets they share.
///
/// The split is by lifetime, and it is what makes streaming affordable:
///
/// - **Assets are grow-only.** Mesh data, acceleration structures, textures and materials are
///   keyed by where they came from and kept for the life of the renderer. A neighbouring cell
///   names the same rocks and the same ground textures, so it arrives having to upload almost
///   nothing. The ceiling is the shipped library — 4,311 textures and some three thousand
///   meshes — which fits the device with room to spare, so nothing is ever freed and no slot is
///   ever renumbered.
/// - **Cells come and go.** A resident cell owns its instances and its lights and nothing else, so
///   evicting one is dropping a list.
///
/// What a change costs, then, is [`SceneResidency::commit`]: the small tables and one top-level
/// build over what is currently placed.
pub(crate) struct SceneResidency {
    geometry: GeometryBuffers,
    textures: TextureArray,
    /// Every material and texture path any cell has named, in the numbering the device sees.
    materials: MaterialTable,
    /// Which mesh slot each source path was appended into.
    by_source: HashMap<String, u32>,
    /// How many textures have been uploaded, which is also the next id [`Self::materials`] will
    /// hand out for a path it has not seen.
    uploaded_textures: u32,
    cells: Vec<ResidentCell>,
    acceleration: SceneAcceleration,
    tables: MaterialBuffers,
    /// Every light every resident cell places, as the shader indexes them.
    lights: Buffer,
    /// Which of those lights reach where, so a shading point walks a handful rather than all.
    light_grid: LightGrid,
    lighting: Lighting,
}

// The GPU-side members hold `ash` function tables, which implement no `Debug`.
impl std::fmt::Debug for SceneResidency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SceneResidency")
            .field("cells", &self.cells.len())
            .field("meshes", &self.by_source.len())
            .field("textures", &self.uploaded_textures)
            .finish_non_exhaustive()
    }
}

impl SceneResidency {
    /// A renderer with nothing resident.
    pub(crate) fn new(
        device: &Device,
        uploader: &mut Uploader,
        limits: RayTracingLimits,
    ) -> rtxmw_gpu::Result<Self> {
        let memory = uploader.memory().clone();
        let geometry = GeometryBuffers::new(&memory)?;
        let textures = TextureArray::new(device, uploader)?;
        let acceleration = SceneAcceleration::new(device, uploader, limits)?;
        let materials = MaterialTable::default();
        let tables = MaterialBuffers::upload(uploader, &geometry, materials.materials())?;
        let lights = Buffer::storage_of(uploader, "scene lights", &[])?;
        let light_grid = LightGrid::build(uploader, &[])?;
        Ok(Self {
            geometry,
            textures,
            materials,
            by_source: HashMap::new(),
            uploaded_textures: 0,
            cells: Vec::new(),
            acceleration,
            tables,
            lights,
            light_grid,
            lighting: Lighting {
                ambient: Vec3::ZERO,
                light_grid: LightGridExtent::default(),
                sun: None,
                water_level: None,
                fog: Vec3::ZERO,
                fog_density: 0.0,
            },
        })
    }

    /// Makes `id` resident, uploading only what no resident cell already named.
    ///
    /// Replaces the cell if it is already here. Takes effect at the next [`Self::commit`], which is
    /// what a caller adding several cells at once wants — the top level is rebuilt once for the
    /// batch rather than once per cell.
    pub(crate) fn add(
        &mut self,
        device: &Device,
        uploader: &mut Uploader,
        limits: RayTracingLimits,
        id: CellId,
        scene: &StaticScene,
        textures: &[Option<Texture>],
    ) -> rtxmw_gpu::Result<()> {
        debug_assert_eq!(
            scene.meshes.len(),
            scene.mesh_sources.len(),
            "every mesh needs the source it is shared by"
        );
        let texture_remap = self.intern_textures(uploader, scene, textures)?;
        let material_remap = self.intern_materials(scene, &texture_remap);
        let mesh_remap = self.append_meshes(device, uploader, limits, scene, &material_remap)?;

        let instances = scene
            .instances
            .iter()
            .map(|instance| Instance {
                mesh: MeshId(mesh_remap[instance.mesh.0 as usize]),
                transform: instance.transform,
            })
            .collect();
        self.cells.retain(|cell| cell.id != id);
        self.cells.push(ResidentCell {
            id,
            instances,
            lights: scene.lights.clone(),
        });

        // The newest cell's, because every exterior cell carries the same sky and a caller loading
        // an interior loads exactly one. Per-cell ambient is what a portal-aware lighting pass
        // would need, and nothing here is one yet.
        self.lighting.ambient = scene.ambient.map_or(Vec3::ZERO, |ambient| ambient.colour);
        self.lighting.sun = scene.sun;
        self.lighting.water_level = scene.water_level;
        // **Only interiors carry fog in the record.** An exterior's comes from the weather system —
        // per region and per weather, out of the original engine's ini fallbacks rather than the
        // ESM — and none of that is read yet. Until it is, the sky is the honest stand-in: fog is
        // lit by it, so a fog the colour of the sky is what aerial perspective looks like anyway.
        let ambient = scene.ambient;
        self.lighting.fog = match ambient {
            Some(ambient) if ambient.fog_density > 0.0 => ambient.fog,
            _ => self.lighting.ambient,
        };
        self.lighting.fog_density = ambient.map_or(OUTDOOR_FOG_DENSITY, |ambient| {
            if ambient.fog_density > 0.0 {
                ambient.fog_density * INDOOR_FOG_SCALE
            } else {
                OUTDOOR_FOG_DENSITY
            }
        });
        Ok(())
    }

    /// Drops a cell's placements, if it was resident. Takes effect at the next [`Self::commit`].
    pub(crate) fn remove(&mut self, id: &CellId) {
        self.cells.retain(|cell| cell.id != *id);
    }

    /// Keeps only the cell added most recently, dropping every other cell's placements.
    ///
    /// What "this cell and nothing else" means once assets are shared: the meshes and textures the
    /// dropped cells brought stay resident, because whatever is loaded next is likely to name them
    /// again, and re-reading a file to upload bytes the device already holds is the cost this whole
    /// arrangement exists to avoid.
    pub(crate) fn retain_newest(&mut self) {
        if self.cells.len() > 1 {
            self.cells.drain(..self.cells.len() - 1);
        }
    }

    /// Rebuilds what the resident set determines: the tables, the lights and the top level.
    ///
    /// Everything here is proportional to what is *placed* rather than to what is loaded, which is
    /// why it is cheap enough to run whenever a cell arrives or leaves.
    pub(crate) fn commit(
        &mut self,
        device: &Device,
        uploader: &mut Uploader,
        limits: RayTracingLimits,
    ) -> rtxmw_gpu::Result<()> {
        self.tables =
            MaterialBuffers::upload(uploader, &self.geometry, self.materials.materials())?;

        let table: Vec<GpuLight> = self
            .cells
            .iter()
            .flat_map(|cell| cell.lights.iter())
            .map(|light| GpuLight::new(*light))
            .collect();
        self.lights = Buffer::storage_of(uploader, "scene lights", bytemuck::cast_slice(&table))?;
        self.light_grid = LightGrid::build(uploader, &table)?;
        self.lighting.light_grid = self.light_grid.extent();

        let instances: Vec<Instance> = self
            .cells
            .iter()
            .flat_map(|cell| cell.instances.iter())
            .copied()
            .collect();
        self.acceleration
            .rebuild_top(device, uploader, limits, &self.geometry, &instances)
    }

    /// Meshes uploaded, across every cell that has been resident.
    ///
    /// Counted from the arena rather than from the paths they were keyed by: what matters is how
    /// many were *uploaded*, and a lookup that failed to recognise a mesh it already held would
    /// leave the key count right and put a second copy on the device.
    pub(crate) fn mesh_count(&self) -> usize {
        self.geometry.ranges().len()
    }

    pub(crate) fn geometry(&self) -> &GeometryBuffers {
        &self.geometry
    }

    pub(crate) fn tables(&self) -> &MaterialBuffers {
        &self.tables
    }

    pub(crate) fn light_grid(&self) -> &LightGrid {
        &self.light_grid
    }

    pub(crate) fn lights(&self) -> &Buffer {
        &self.lights
    }

    pub(crate) fn textures(&self) -> &TextureArray {
        &self.textures
    }

    pub(crate) fn acceleration(&self) -> &SceneAcceleration {
        &self.acceleration
    }

    pub(crate) fn lighting(&self) -> Lighting {
        self.lighting
    }

    /// Uploads whichever of the cell's textures are new, and maps its ids onto the shared ones.
    fn intern_textures(
        &mut self,
        uploader: &mut Uploader,
        scene: &StaticScene,
        textures: &[Option<Texture>],
    ) -> rtxmw_gpu::Result<Vec<u32>> {
        let paths = scene.materials.textures();
        debug_assert_eq!(
            paths.len(),
            textures.len(),
            "a cell's textures are decoded from its own path list, one for one"
        );
        let mut remap = Vec::with_capacity(paths.len());
        for (local, path) in paths.iter().enumerate() {
            let shared = self.materials.intern_texture(path).0;
            // Interning hands out ids in order, so one past what has been uploaded is one this
            // renderer has never seen — and the only case where the file has to be read.
            if shared >= self.uploaded_textures {
                // **Each texture is followed by the shading map estimated from it**, so a material's
                // id addresses a pair rather than a slot. One array rather than two because Vulkan
                // allows a variable descriptor count only on a set's final binding, and there is
                // therefore no second one to put the maps in — `surface.glsl` does the arithmetic.
                let texture = textures[local].as_ref();
                let colour = self.textures.insert(uploader, texture)?;
                // A neutral map rather than nothing where the texture is missing: an empty slot
                // holds the array's magenta stand-in, which would divide the surface by two.
                let map = texture.map_or_else(Texture::neutral_shading, Texture::shading_map);
                let shading = self.textures.insert(uploader, Some(&map))?;
                assert_eq!(colour, shared * 2, "a texture id addresses its own pair");
                assert_eq!(
                    shading,
                    colour + 1,
                    "a shading map follows the texture it is of"
                );
                self.uploaded_textures = shared + 1;
            }
            remap.push(shared);
        }
        Ok(remap)
    }

    /// Adds the cell's materials to the shared table, and maps its ids onto the shared ones.
    fn intern_materials(&mut self, scene: &StaticScene, texture_remap: &[u32]) -> Vec<u32> {
        scene
            .materials
            .materials()
            .iter()
            .map(|material| {
                let mut shared = *material;
                shared.base_colour = shared
                    .base_colour
                    .map(|id| TextureId(texture_remap[id.0 as usize]));
                // Ground names four textures rather than one, and every one of them has to be
                // moved onto the shared slots or a patch blends whatever happens to sit at that
                // index in the array this renderer built.
                if let MaterialKind::Terrain(TerrainLayers(ids)) = &mut shared.kind {
                    *ids = ids.map(|id| TextureId(texture_remap[id.0 as usize]));
                }
                self.materials.intern(shared)
            })
            .collect()
    }

    /// Uploads the meshes this cell is the first to name, and maps its ids onto the shared slots.
    fn append_meshes(
        &mut self,
        device: &Device,
        uploader: &mut Uploader,
        limits: RayTracingLimits,
        scene: &StaticScene,
        material_remap: &[u32],
    ) -> rtxmw_gpu::Result<Vec<u32>> {
        let base = self.geometry.ranges().len() as u32;
        let mut pending: Vec<&Mesh> = Vec::new();
        let mut remap = Vec::with_capacity(scene.meshes.len());
        for (index, mesh) in scene.meshes.iter().enumerate() {
            let source = scene.mesh_sources[index].as_str();
            let slot = match self.by_source.get(source) {
                Some(&slot) => slot,
                None => {
                    let slot = base + pending.len() as u32;
                    pending.push(mesh);
                    self.by_source.insert(source.to_owned(), slot);
                    slot
                }
            };
            remap.push(slot);
        }

        let appended = self.geometry.append(uploader, &pending, material_remap)?;
        self.acceleration.append_meshes(
            device,
            uploader,
            limits,
            &self.geometry,
            self.materials.materials(),
            appended,
        )?;
        Ok(remap)
    }
}
