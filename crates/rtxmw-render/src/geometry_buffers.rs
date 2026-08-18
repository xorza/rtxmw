//! A scene's geometry, packed into the buffers an acceleration structure build reads.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use rtxmw_gpu::{Buffer, BufferMemory, Memory, Uploader};
use rtxmw_scene::Mesh;

/// Per-vertex shading data, parallel to the position stream.
///
/// Kept out of the position buffer because an acceleration structure build reads positions with its
/// own stride and would have to step over anything interleaved with them. Stored unpacked:
/// octahedral normals and half-precision UVs would more than halve this, but Morrowind tiles
/// textures with UVs well outside [0, 1], where `f16` loses visible precision — so packing waits
/// for a measured reason.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct VertexAttributes {
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

/// Where one mesh's vertices and indices sit inside the shared buffers.
///
/// Indices stay mesh-local, so `first_vertex` is both what an acceleration structure build passes
/// as its `firstVertex` and what a shader adds before reading attributes. Local indices mean a
/// mesh's index data does not depend on where in the buffer it landed, which is what keeps meshes
/// relocatable once cells start streaming.
///
/// Host-side only. The per-geometry struct a hit shader reads is `GeometryRef` at M4, which carries
/// a material alongside these offsets; nothing uploads this one, so its layout is not pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshRange {
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_index: u32,
    pub index_count: u32,
}

impl MeshRange {
    /// Triangles in this mesh.
    pub fn triangle_count(&self) -> u32 {
        self.index_count / 3
    }
}

/// Every mesh of a scene, concatenated into one set of device-local buffers.
///
/// One buffer per stream rather than one per mesh: a cell holds a hundred-odd meshes, and a
/// separate allocation for each would cost a descriptor and a bind per mesh at trace time for
/// nothing. [`MeshRange`] carries the offsets that make the split recoverable.
#[derive(Debug)]
pub struct GeometryBuffers {
    positions: Buffer,
    attributes: Buffer,
    indices: Buffer,
    ranges: Vec<MeshRange>,
}

impl GeometryBuffers {
    /// Stride of the position stream, which an acceleration structure build is told explicitly.
    pub const POSITION_STRIDE: vk::DeviceSize = size_of::<[f32; 3]>() as vk::DeviceSize;

    /// Packs `meshes` and uploads them, one [`MeshRange`] per mesh in the order given.
    ///
    /// A mesh with no visible geometry keeps its slot as a zero-length range, so a `MeshId` from
    /// the scene stays a direct index into [`GeometryBuffers::ranges`].
    pub fn upload(
        memory: &Memory,
        uploader: &mut Uploader,
        meshes: &[Mesh],
    ) -> rtxmw_gpu::Result<Self> {
        let packed = pack(meshes);
        let position_bytes: &[u8] = bytemuck::cast_slice(&packed.positions);
        let attribute_bytes: &[u8] = bytemuck::cast_slice(&packed.attributes);
        let index_bytes: &[u8] = bytemuck::cast_slice(&packed.indices);

        let positions = Buffer::new(
            memory,
            "scene positions",
            buffer_size(position_bytes.len()),
            build_input_usage(),
            BufferMemory::Device,
        )?;
        let attributes = Buffer::new(
            memory,
            "scene vertex attributes",
            buffer_size(attribute_bytes.len()),
            // Never read by a build, only at a hit, so no build-input usage.
            geometry_usage(),
            BufferMemory::Device,
        )?;
        let indices = Buffer::new(
            memory,
            "scene indices",
            buffer_size(index_bytes.len()),
            build_input_usage(),
            BufferMemory::Device,
        )?;

        uploader.upload(&positions, position_bytes)?;
        uploader.upload(&attributes, attribute_bytes)?;
        uploader.upload(&indices, index_bytes)?;

        Ok(Self {
            positions,
            attributes,
            indices,
            ranges: packed.ranges,
        })
    }

    /// Tightly packed `float3` positions, the vertex stream a build reads.
    pub fn positions(&self) -> &Buffer {
        &self.positions
    }

    /// [`VertexAttributes`] parallel to the positions.
    pub fn attributes(&self) -> &Buffer {
        &self.attributes
    }

    /// `u32` triangle indices, local to each mesh.
    pub fn indices(&self) -> &Buffer {
        &self.indices
    }

    /// Where each mesh sits, in the order the meshes were given.
    pub fn ranges(&self) -> &[MeshRange] {
        &self.ranges
    }

    /// Vertices across every mesh.
    pub fn vertex_count(&self) -> usize {
        self.ranges.iter().map(|r| r.vertex_count as usize).sum()
    }

    /// Triangles across every mesh.
    pub fn triangle_count(&self) -> usize {
        self.ranges
            .iter()
            .map(|r| r.triangle_count() as usize)
            .sum()
    }

    /// Device memory held by the three buffers.
    pub fn byte_size(&self) -> vk::DeviceSize {
        self.positions.size() + self.attributes.size() + self.indices.size()
    }
}

/// Usage shared by every geometry buffer.
///
/// `TRANSFER_SRC` is for readback. Copying a buffer back is the only way to check that what reached
/// the device is what was packed, and a usage flag on a buffer costs neither memory nor bandwidth —
/// unlike on an image, where it can change the layout the driver picks.
fn geometry_usage() -> vk::BufferUsageFlags {
    vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC
}

/// Additionally, what an acceleration structure build reads directly.
fn build_input_usage() -> vk::BufferUsageFlags {
    geometry_usage() | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
}

/// Vulkan rejects a zero-sized buffer, and a cell with no geometry still needs valid handles.
fn buffer_size(bytes: usize) -> vk::DeviceSize {
    bytes.max(4) as vk::DeviceSize
}

/// The host-side streams, before they become buffers.
#[derive(Debug)]
struct Packed {
    positions: Vec<[f32; 3]>,
    attributes: Vec<VertexAttributes>,
    indices: Vec<u32>,
    ranges: Vec<MeshRange>,
}

/// Concatenates every mesh into the three parallel streams, recording where each landed.
fn pack(meshes: &[Mesh]) -> Packed {
    let vertices: usize = meshes.iter().map(|m| m.positions.len()).sum();
    let index_total: usize = meshes.iter().map(|m| m.indices.len()).sum();

    let mut positions = Vec::with_capacity(vertices);
    let mut attributes = Vec::with_capacity(vertices);
    let mut indices = Vec::with_capacity(index_total);
    let mut ranges = Vec::with_capacity(meshes.len());

    for mesh in meshes {
        debug_assert_eq!(
            mesh.positions.len(),
            mesh.normals.len(),
            "flattening should leave normals parallel to positions"
        );
        debug_assert_eq!(
            mesh.positions.len(),
            mesh.uvs.len(),
            "flattening should leave UVs parallel to positions"
        );

        ranges.push(MeshRange {
            first_vertex: positions.len() as u32,
            vertex_count: mesh.positions.len() as u32,
            first_index: indices.len() as u32,
            index_count: mesh.indices.len() as u32,
        });

        positions.extend(mesh.positions.iter().map(|p| p.to_array()));
        attributes.extend(mesh.normals.iter().zip(&mesh.uvs).map(|(normal, uv)| {
            VertexAttributes {
                normal: normal.to_array(),
                uv: uv.to_array(),
            }
        }));
        indices.extend_from_slice(&mesh.indices);
    }

    Packed {
        positions,
        attributes,
        indices,
        ranges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec3};

    /// A mesh whose vertices and normals are distinguishable by index.
    fn mesh(vertices: u32, indices: &[u32]) -> Mesh {
        Mesh {
            positions: (0..vertices).map(|i| Vec3::splat(i as f32)).collect(),
            normals: (0..vertices).map(|i| Vec3::X * i as f32).collect(),
            uvs: (0..vertices).map(|i| Vec2::splat(i as f32 * 0.5)).collect(),
            indices: indices.to_vec(),
        }
    }

    #[test]
    fn packing_concatenates_meshes_and_leaves_indices_mesh_local() {
        // 3 vertices / 1 triangle, then 4 vertices / 2 triangles.
        let meshes = [mesh(3, &[0, 1, 2]), mesh(4, &[0, 1, 2, 1, 2, 3])];
        let Packed {
            positions,
            attributes,
            indices,
            ranges,
        } = pack(&meshes);

        assert_eq!(
            ranges,
            vec![
                MeshRange {
                    first_vertex: 0,
                    vertex_count: 3,
                    first_index: 0,
                    index_count: 3,
                },
                MeshRange {
                    first_vertex: 3,
                    vertex_count: 4,
                    first_index: 3,
                    index_count: 6,
                },
            ]
        );
        assert_eq!(positions.len(), 7);
        assert_eq!(attributes.len(), 7);
        assert_eq!(indices.len(), 9);
        assert_eq!(ranges[1].triangle_count(), 2);

        // The second mesh starts at vertex 3, but its indices are unchanged — 0..3, not 3..6. A
        // rebased buffer would read the first mesh's vertices instead.
        assert_eq!(&indices[3..], &[0, 1, 2, 1, 2, 3]);
        // Its vertices did move, so position 3 is the second mesh's vertex 0.
        assert_eq!(positions[3], [0.0, 0.0, 0.0]);
        assert_eq!(positions[6], [3.0, 3.0, 3.0]);
    }

    #[test]
    fn attributes_stay_aligned_with_the_positions_they_belong_to() {
        let meshes = [mesh(2, &[0, 1, 1]), mesh(2, &[0, 1, 1])];
        let packed = pack(&meshes);
        let (positions, attributes) = (packed.positions, packed.attributes);

        // Vertex 1 of the second mesh is the fourth entry overall, and carries that mesh's own
        // second normal and UV, not a continuation of the first mesh's numbering.
        assert_eq!(positions[3], [1.0, 1.0, 1.0]);
        assert_eq!(
            attributes[3],
            VertexAttributes {
                normal: [1.0, 0.0, 0.0],
                uv: [0.5, 0.5],
            }
        );
    }

    #[test]
    fn an_empty_mesh_keeps_its_slot_as_a_zero_length_range() {
        // The scene indexes meshes by position, so a mesh that flattened to nothing must not shift
        // the ones after it.
        let meshes = [mesh(3, &[0, 1, 2]), mesh(0, &[]), mesh(3, &[0, 1, 2])];
        let ranges = pack(&meshes).ranges;

        assert_eq!(ranges.len(), 3);
        assert_eq!(
            ranges[1],
            MeshRange {
                first_vertex: 3,
                vertex_count: 0,
                first_index: 3,
                index_count: 0,
            }
        );
        assert_eq!(ranges[2].first_vertex, 3);
        assert_eq!(ranges[2].first_index, 3);
    }

    #[test]
    fn the_position_stream_is_tightly_packed() {
        // A build is told this stride explicitly; if the type ever grew padding, the build would
        // read every position from the wrong offset.
        assert_eq!(GeometryBuffers::POSITION_STRIDE, 12);
        assert_eq!(size_of::<VertexAttributes>(), 20);
    }
}
