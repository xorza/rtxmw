//! The per-geometry and per-material tables a hit reads.

use bytemuck::{Pod, Zeroable};
use rtxmw_gpu::{Buffer, Uploader};
use rtxmw_scene::{AlphaMode, Material, MaterialKind, TerrainLayers};

use crate::geometry_buffers::GeometryBuffers;

/// The cutoff a blended surface is treated with until real transparency exists.
///
/// Morrowind does its foliage, grates and banners with `NiAlphaProperty` *blending* over a texture
/// whose alpha is very nearly binary — 539 of the shipped library's 4,593 materials are blended
/// against only 72 explicitly masked. Committing those as opaque draws every tree as a rectangle,
/// so they run the same cutout path until ordered transparency arrives to replace it.
const BLEND_CUTOFF: f32 = 0.5;

/// Stands in for a material with no base colour texture.
///
/// A sentinel rather than a separate flag bit: the shader has to branch on it either way, and a
/// value that indexes nothing is harder to mistake for a valid slot than a zero would be.
///
/// Declared again as `NO_TEXTURE` in `primary_visibility.comp`, because a GLSL shader cannot see a
/// Rust constant. The test below pins the literal so the two cannot drift apart silently.
pub(crate) const NO_TEXTURE: u32 = u32::MAX;

/// A surface lit by what reaches it, which is everything a NIF describes.
///
/// Declared again in `primary_visibility.comp`; the test below pins both literals, because a shader
/// cannot see a Rust constant and a silent disagreement would shade every surface as water or none
/// of them.
const KIND_DIFFUSE: u32 = 0;

/// A water surface, which the shader reflects, refracts and attenuates through rather than lighting.
const KIND_WATER: u32 = 1;

/// Ground, blended across the four terrain textures in [`GpuMaterial::terrain_layers`].
const KIND_TERRAIN: u32 = 2;

/// Set on a run whose triangles are a sheet rather than the skin of something solid, which is what
/// lets the shader light a sail or a rug from whichever side the sun is on.
///
/// Declared again in `primary_visibility.comp`.
pub(crate) const GEOMETRY_THIN: u32 = 1;

/// One acceleration structure geometry, indexed by `instance_custom_index + geometry_index`.
///
/// That sum is the whole reason the geometry table is flat: a hit reports the two halves separately
/// and adding them lands on this entry with no per-mesh indirection to chase first.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct GpuGeometry {
    /// Where this run starts in the shared index buffer.
    pub(crate) first_index: u32,
    /// Added to each index value to reach the shared vertex streams.
    pub(crate) first_vertex: u32,
    pub(crate) material: u32,
    /// Bit flags describing the run itself rather than its material, currently only [`GEOMETRY_THIN`].
    pub(crate) flags: u32,
}

/// One surface description as the shader reads it.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuMaterial {
    pub(crate) diffuse: [f32; 3],
    pub(crate) opacity: f32,
    pub(crate) emissive: [f32; 3],
    /// Texels below this are absent. Zero when the surface is not alpha tested.
    pub(crate) alpha_cutoff: f32,
    /// Index into the bindless texture array, or [`NO_TEXTURE`].
    pub(crate) base_colour: u32,
    /// Which shading model the hit runs, matching [`MaterialKind`]'s discriminants.
    ///
    /// A number rather than a flag bit because it selects between models rather than modifying one,
    /// and Morrowind's lava and slime will each want a value of their own.
    pub(crate) kind: u32,
    /// The four textures a [`KIND_TERRAIN`] surface blends, packed sixteen bits apiece.
    ///
    /// Packed rather than given four words of their own because the block had exactly two spare and
    /// a texture id has never needed more than sixteen bits — the whole shipped library is under
    /// four thousand. `GpuMaterial::new` asserts that rather than trusting it.
    pub(crate) terrain_layers: [u32; 2],
    /// How far the texture slides across the surface each second — see [`rtxmw_scene::Material`].
    pub(crate) scroll: [f32; 2],
}

/// The four ground textures as two words, sixteen bits each.
fn pack_layers(TerrainLayers(ids): TerrainLayers) -> [u32; 2] {
    let ids = ids.map(|id| {
        assert!(
            id.0 <= u32::from(u16::MAX),
            "texture {} is past what a packed layer can name",
            id.0
        );
        id.0
    });
    [ids[0] | (ids[1] << 16), ids[2] | (ids[3] << 16)]
}

impl GpuMaterial {
    /// Flattens a scene material into its GPU form.
    fn new(material: Material) -> Self {
        Self {
            diffuse: material.diffuse.to_array(),
            opacity: material.opacity,
            emissive: material.emissive.to_array(),
            // Zero means "keep every texel", which is what an opaque surface wants and what makes
            // the shader's test a single comparison with no mode to branch on.
            alpha_cutoff: match material.alpha {
                AlphaMode::Mask(threshold) => threshold,
                AlphaMode::Blend => BLEND_CUTOFF,
                AlphaMode::Opaque => 0.0,
            },
            base_colour: material.base_colour.map_or(NO_TEXTURE, |id| id.0),
            kind: match material.kind {
                MaterialKind::Diffuse => KIND_DIFFUSE,
                MaterialKind::Water => KIND_WATER,
                MaterialKind::Terrain(_) => KIND_TERRAIN,
            },
            // Zeroes for anything but ground, which never reads them: the shader branches on
            // `kind` first.
            terrain_layers: match material.kind {
                MaterialKind::Terrain(layers) => pack_layers(layers),
                _ => [0; 2],
            },
            scroll: material.scroll.to_array(),
        }
    }
}

/// The two tables that turn a hit into a surface.
#[derive(Debug)]
pub(crate) struct MaterialBuffers {
    geometries: Buffer,
    materials: Buffer,
}

impl MaterialBuffers {
    /// Builds and uploads both tables for an already-packed scene.
    pub(crate) fn upload(
        uploader: &mut Uploader,
        geometry: &GeometryBuffers,
        materials: &[Material],
    ) -> rtxmw_gpu::Result<Self> {
        let mut geometry_table = Vec::with_capacity(geometry.submeshes().len());
        for range in geometry.ranges() {
            let span =
                range.first_submesh as usize..(range.first_submesh + range.submesh_count) as usize;
            for submesh in &geometry.submeshes()[span] {
                geometry_table.push(GpuGeometry {
                    first_index: submesh.first_index,
                    first_vertex: range.first_vertex,
                    material: submesh.material,
                    flags: if submesh.thin { GEOMETRY_THIN } else { 0 },
                });
            }
        }

        let material_table: Vec<GpuMaterial> =
            materials.iter().copied().map(GpuMaterial::new).collect();

        let geometry_bytes: &[u8] = bytemuck::cast_slice(&geometry_table);
        let material_bytes: &[u8] = bytemuck::cast_slice(&material_table);

        Ok(Self {
            geometries: Buffer::storage_of(uploader, "scene geometry table", geometry_bytes)?,
            materials: Buffer::storage_of(uploader, "scene material table", material_bytes)?,
        })
    }

    /// One entry per acceleration structure geometry.
    pub(crate) fn geometries(&self) -> &Buffer {
        &self.geometries
    }

    /// One entry per distinct material.
    pub(crate) fn materials(&self) -> &Buffer {
        &self.materials
    }
}

/// Read-only in a shader and written by a staging copy.
#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use rtxmw_scene::TextureId;

    #[test]
    fn the_tables_match_the_layouts_the_shader_declares() {
        // Both are read with a hardcoded stride; a field added without updating the shader would
        // shift every entry after the first.
        assert_eq!(size_of::<GpuGeometry>(), 16);
        assert_eq!(size_of::<GpuMaterial>(), 56);
        // The shader spells this out as `0xFFFFFFFFu`; changing it here alone would leave every
        // untextured surface sampling slot zero of the array instead of taking the fallback branch.
        assert_eq!(NO_TEXTURE, 0xFFFF_FFFF);
        // Likewise the shading models, which the shader compares against literals of its own.
        assert_eq!(KIND_DIFFUSE, 0);
        assert_eq!(KIND_WATER, 1);
        // And the geometry flag, which the shader tests with a literal mask.
        assert_eq!(GEOMETRY_THIN, 1);
    }

    #[test]
    fn water_is_the_only_material_that_reaches_the_shader_as_water() {
        assert_eq!(GpuMaterial::new(Material::default()).kind, KIND_DIFFUSE);
        assert_eq!(
            GpuMaterial::new(Material {
                kind: MaterialKind::Water,
                ..Material::default()
            })
            .kind,
            KIND_WATER
        );
    }

    #[test]
    fn an_untextured_material_carries_the_sentinel_rather_than_slot_zero() {
        let plain = GpuMaterial::new(Material::default());
        assert_eq!(plain.base_colour, NO_TEXTURE);
        // Slot zero is a real texture, so it must be distinguishable from having none.
        let textured = GpuMaterial::new(Material {
            base_colour: Some(TextureId(0)),
            ..Material::default()
        });
        assert_eq!(textured.base_colour, 0);
    }

    #[test]
    fn every_non_opaque_material_carries_a_cutoff() {
        // A masked surface keeps the threshold the NIF gave it.
        let masked = GpuMaterial::new(Material {
            alpha: AlphaMode::Mask(0.25),
            ..Material::default()
        });
        assert_eq!(masked.alpha_cutoff, 0.25);

        // A blended one gets the stand-in, because the geometry it describes is foliage far more
        // often than it is glass, and drawing foliage solid is the worse of the two errors.
        let blended = GpuMaterial::new(Material {
            alpha: AlphaMode::Blend,
            ..Material::default()
        });
        assert_eq!(blended.alpha_cutoff, BLEND_CUTOFF);

        // Only an opaque surface keeps every texel unconditionally.
        let opaque = GpuMaterial::new(Material {
            alpha: AlphaMode::Opaque,
            ..Material::default()
        });
        assert_eq!(opaque.alpha_cutoff, 0.0);

        // The colours survive the flattening.
        let lit = GpuMaterial::new(Material {
            emissive: Vec3::new(0.1, 0.2, 0.3),
            opacity: 0.5,
            ..Material::default()
        });
        assert_eq!(lit.emissive, [0.1, 0.2, 0.3]);
        assert_eq!(lit.opacity, 0.5);
    }
}
