//! The per-geometry and per-material tables a hit reads.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use rtxmw_gpu::{Buffer, BufferMemory, Uploader};
use rtxmw_scene::{AlphaMode, Material};

use crate::geometry_buffers::GeometryBuffers;

/// Stands in for a material with no base colour texture.
///
/// A sentinel rather than a separate flag bit: the shader has to branch on it either way, and a
/// value that indexes nothing is harder to mistake for a valid slot than a zero would be.
///
/// Declared again as `NO_TEXTURE` in `primary_visibility.comp`, because a GLSL shader cannot see a
/// Rust constant. The test below pins the literal so the two cannot drift apart silently.
pub const NO_TEXTURE: u32 = u32::MAX;

/// One acceleration structure geometry, indexed by `instance_custom_index + geometry_index`.
///
/// That sum is the whole reason the geometry table is flat: a hit reports the two halves separately
/// and adding them lands on this entry with no per-mesh indirection to chase first.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuGeometry {
    /// Where this run starts in the shared index buffer.
    pub first_index: u32,
    /// Added to each index value to reach the shared vertex streams.
    pub first_vertex: u32,
    pub material: u32,
    pub padding: u32,
}

/// One surface description as the shader reads it.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Pod, Zeroable)]
pub struct GpuMaterial {
    pub diffuse: [f32; 3],
    pub opacity: f32,
    pub emissive: [f32; 3],
    /// Texels below this are absent. Zero when the surface is not alpha tested.
    pub alpha_cutoff: f32,
    /// Index into the bindless texture array, or [`NO_TEXTURE`].
    pub base_colour: u32,
    pub padding: [u32; 3],
}

impl GpuMaterial {
    /// Flattens a scene material into its GPU form.
    fn new(material: Material) -> Self {
        Self {
            diffuse: material.diffuse.to_array(),
            opacity: material.opacity,
            emissive: material.emissive.to_array(),
            // Blended surfaces have no cutout, so zero means "keep every texel" rather than being
            // an absent value the shader has to special-case.
            alpha_cutoff: match material.alpha {
                AlphaMode::Mask(threshold) => threshold,
                AlphaMode::Opaque | AlphaMode::Blend => 0.0,
            },
            base_colour: material.base_colour.map_or(NO_TEXTURE, |id| id.0),
            padding: [0; 3],
        }
    }
}

/// The two tables that turn a hit into a surface.
#[derive(Debug)]
pub struct MaterialBuffers {
    geometries: Buffer,
    materials: Buffer,
}

impl MaterialBuffers {
    /// Builds and uploads both tables for an already-packed scene.
    pub fn upload(
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
                    padding: 0,
                });
            }
        }

        let material_table: Vec<GpuMaterial> =
            materials.iter().copied().map(GpuMaterial::new).collect();

        let geometry_bytes: &[u8] = bytemuck::cast_slice(&geometry_table);
        let material_bytes: &[u8] = bytemuck::cast_slice(&material_table);

        let geometries = Buffer::new(
            uploader.memory(),
            "scene geometry table",
            (geometry_bytes.len() as vk::DeviceSize).max(Buffer::MIN_SIZE),
            usage(),
            BufferMemory::Device,
        )?;
        let materials_buffer = Buffer::new(
            uploader.memory(),
            "scene material table",
            (material_bytes.len() as vk::DeviceSize).max(Buffer::MIN_SIZE),
            usage(),
            BufferMemory::Device,
        )?;
        uploader.upload(&geometries, geometry_bytes)?;
        uploader.upload(&materials_buffer, material_bytes)?;

        Ok(Self {
            geometries,
            materials: materials_buffer,
        })
    }

    /// One entry per acceleration structure geometry.
    pub fn geometries(&self) -> &Buffer {
        &self.geometries
    }

    /// One entry per distinct material.
    pub fn materials(&self) -> &Buffer {
        &self.materials
    }
}

/// Read-only in a shader and written by a staging copy.
fn usage() -> vk::BufferUsageFlags {
    vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC
}

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
        assert_eq!(size_of::<GpuMaterial>(), 48);
        // The shader spells this out as `0xFFFFFFFFu`; changing it here alone would leave every
        // untextured surface sampling slot zero of the array instead of taking the fallback branch.
        assert_eq!(NO_TEXTURE, 0xFFFF_FFFF);
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
    fn only_a_masked_material_carries_a_cutoff() {
        let masked = GpuMaterial::new(Material {
            alpha: AlphaMode::Mask(0.25),
            ..Material::default()
        });
        assert_eq!(masked.alpha_cutoff, 0.25);

        // Blended and opaque surfaces keep every texel, so the cutoff is zero rather than absent.
        for alpha in [AlphaMode::Opaque, AlphaMode::Blend] {
            let other = GpuMaterial::new(Material {
                alpha,
                ..Material::default()
            });
            assert_eq!(other.alpha_cutoff, 0.0, "{alpha:?}");
        }

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
