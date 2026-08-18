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
/// Host-side only. The per-geometry struct a hit shader reads is `GpuGeometry`, which
/// carries a material alongside its offsets; nothing uploads this one, so its layout is not pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshRange {
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_index: u32,
    pub index_count: u32,
    /// Range into [`GeometryBuffers::submeshes`] — one entry per acceleration structure geometry.
    pub first_submesh: u32,
    pub submesh_count: u32,
}

impl MeshRange {
    /// Triangles in this mesh.
    pub fn triangle_count(&self) -> u32 {
        self.index_count / 3
    }
}

/// One material-uniform run of a mesh, addressed against the shared index buffer.
///
/// The scene's [`rtxmw_scene::Submesh`] counts from the start of its own mesh; this counts from the
/// start of the buffer every mesh was packed into, which is what a build range and a shader both
/// want. Kept flat across all meshes so a hit can index it directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubmeshRange {
    pub first_index: u32,
    pub index_count: u32,
    pub material: u32,
}

impl SubmeshRange {
    /// Triangles in this run.
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
    submeshes: Vec<SubmeshRange>,
    /// Vertices and indices written so far, which is where the next append starts.
    vertices: u32,
    index_total: u32,
}

impl GeometryBuffers {
    /// Stride of the position stream, which an acceleration structure build is told explicitly.
    pub const POSITION_STRIDE: vk::DeviceSize = size_of::<[f32; 3]>() as vk::DeviceSize;

    /// An arena holding nothing.
    ///
    /// Meshes are appended as cells arrive and **never removed**: the whole shipped library is
    /// about three thousand meshes, so a session that walked every cell would hold a bounded set
    /// small enough to keep. That is what makes a mesh's slot, and with it the `first_submesh` an
    /// instance carries, stable for as long as the renderer lives — nothing renumbers, so nothing
    /// has to be rewritten when a cell goes away.
    pub fn new(memory: &Memory) -> rtxmw_gpu::Result<Self> {
        Ok(Self {
            positions: Buffer::new(
                memory,
                "scene positions",
                Buffer::MIN_SIZE,
                build_input_usage(),
                BufferMemory::Device,
            )?,
            attributes: Buffer::new(
                memory,
                "scene vertex attributes",
                Buffer::MIN_SIZE,
                // Never read by a build, only at a hit, so no build-input usage.
                geometry_usage(),
                BufferMemory::Device,
            )?,
            indices: Buffer::new(
                memory,
                "scene indices",
                Buffer::MIN_SIZE,
                build_input_usage(),
                BufferMemory::Device,
            )?,
            ranges: Vec::new(),
            submeshes: Vec::new(),
            vertices: 0,
            index_total: 0,
        })
    }

    /// Appends `meshes` and uploads them, returning the slots they landed in.
    ///
    /// `material_remap` turns the cell-local material each submesh names into the index it has in
    /// the renderer's own table, because a mesh shared by two cells is described once here and must
    /// not carry either cell's numbering.
    ///
    /// A mesh with no visible geometry keeps its slot as a zero-length range, so the caller's mesh
    /// order stays a direct index into [`GeometryBuffers::ranges`].
    pub fn append(
        &mut self,
        uploader: &mut Uploader,
        meshes: &[&Mesh],
        material_remap: &[u32],
    ) -> rtxmw_gpu::Result<std::ops::Range<u32>> {
        let first_slot = self.ranges.len() as u32;
        let packed = pack(
            meshes,
            self.vertices,
            self.index_total,
            self.submeshes.len() as u32,
        );

        let position_bytes: &[u8] = bytemuck::cast_slice(&packed.positions);
        let attribute_bytes: &[u8] = bytemuck::cast_slice(&packed.attributes);
        let index_bytes: &[u8] = bytemuck::cast_slice(&packed.indices);
        let position_offset = self.vertices as vk::DeviceSize * Self::POSITION_STRIDE;
        let attribute_offset =
            self.vertices as vk::DeviceSize * size_of::<VertexAttributes>() as vk::DeviceSize;
        let index_offset = self.index_total as vk::DeviceSize * size_of::<u32>() as vk::DeviceSize;

        grow(
            uploader,
            &mut self.positions,
            position_offset,
            position_offset + position_bytes.len() as vk::DeviceSize,
            "scene positions",
            build_input_usage(),
        )?;
        grow(
            uploader,
            &mut self.attributes,
            attribute_offset,
            attribute_offset + attribute_bytes.len() as vk::DeviceSize,
            "scene vertex attributes",
            geometry_usage(),
        )?;
        grow(
            uploader,
            &mut self.indices,
            index_offset,
            index_offset + index_bytes.len() as vk::DeviceSize,
            "scene indices",
            build_input_usage(),
        )?;

        uploader.upload_at(&self.positions, position_offset, position_bytes)?;
        uploader.upload_at(&self.attributes, attribute_offset, attribute_bytes)?;
        uploader.upload_at(&self.indices, index_offset, index_bytes)?;

        self.vertices += packed.positions.len() as u32;
        self.index_total += packed.indices.len() as u32;
        self.ranges.extend_from_slice(&packed.ranges);
        self.submeshes
            .extend(packed.submeshes.into_iter().map(|submesh| SubmeshRange {
                material: material_remap[submesh.material as usize],
                ..submesh
            }));

        Ok(first_slot..self.ranges.len() as u32)
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

    /// The runs belonging to one mesh.
    pub fn submeshes_of(&self, range: &MeshRange) -> &[SubmeshRange] {
        let start = range.first_submesh as usize;
        &self.submeshes[start..start + range.submesh_count as usize]
    }

    /// Every material-uniform run across every mesh, in mesh order.
    ///
    /// One acceleration structure geometry is built per entry, so a hit's `geometry_index` added to
    /// its mesh's `first_submesh` lands back here — which is how a surface finds its material.
    pub fn submeshes(&self) -> &[SubmeshRange] {
        &self.submeshes
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

/// The host-side streams, before they become buffers.
#[derive(Debug)]
struct Packed {
    positions: Vec<[f32; 3]>,
    attributes: Vec<VertexAttributes>,
    indices: Vec<u32>,
    ranges: Vec<MeshRange>,
    submeshes: Vec<SubmeshRange>,
}

/// Moves a buffer's contents into a larger one when what is needed no longer fits.
///
/// Doubling rather than fitting exactly: an arena grown one cell at a time would otherwise copy
/// everything it holds on every append, which is quadratic in the number of cells streamed.
fn grow(
    uploader: &mut Uploader,
    buffer: &mut Buffer,
    used: vk::DeviceSize,
    needed: vk::DeviceSize,
    name: &str,
    usage: vk::BufferUsageFlags,
) -> rtxmw_gpu::Result<()> {
    if needed <= buffer.size() {
        return Ok(());
    }
    let memory = uploader.memory().clone();
    let bigger = Buffer::new(
        &memory,
        name,
        needed.max(buffer.size() * 2),
        usage,
        BufferMemory::Device,
    )?;
    uploader.copy_buffer(buffer, &bigger, used)?;
    *buffer = bigger;
    Ok(())
}

/// Concatenates every mesh into the three parallel streams, recording where each landed.
fn pack(meshes: &[&Mesh], first_vertex: u32, first_index: u32, first_submesh: u32) -> Packed {
    let vertices: usize = meshes.iter().map(|m| m.positions.len()).sum();
    let index_total: usize = meshes.iter().map(|m| m.indices.len()).sum();

    let mut positions = Vec::with_capacity(vertices);
    let mut attributes = Vec::with_capacity(vertices);
    let mut indices = Vec::with_capacity(index_total);
    let mut ranges = Vec::with_capacity(meshes.len());
    let mut submeshes = Vec::with_capacity(meshes.iter().map(|m| m.submeshes.len()).sum());

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

        let base_index = first_index + indices.len() as u32;
        ranges.push(MeshRange {
            first_vertex: first_vertex + positions.len() as u32,
            vertex_count: mesh.positions.len() as u32,
            first_index: base_index,
            index_count: mesh.indices.len() as u32,
            first_submesh: first_submesh + submeshes.len() as u32,
            submesh_count: mesh.submeshes.len() as u32,
        });
        // Rebased onto the shared buffer. The scene counts a submesh from the start of its own
        // mesh, which is the right thing there and the wrong thing for a build range.
        submeshes.extend(mesh.submeshes.iter().map(|sub| SubmeshRange {
            first_index: base_index + sub.first_index,
            index_count: sub.index_count,
            material: sub.material,
        }));

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
        submeshes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec3};
    use rtxmw_scene::Submesh;

    /// Packs into an empty arena, which is what most of these check.
    fn packed(meshes: &[Mesh]) -> Packed {
        pack(&meshes.iter().collect::<Vec<_>>(), 0, 0, 0)
    }

    /// A mesh whose vertices and normals are distinguishable by index.
    fn mesh(vertices: u32, indices: &[u32]) -> Mesh {
        Mesh {
            positions: (0..vertices).map(|i| Vec3::splat(i as f32)).collect(),
            normals: (0..vertices).map(|i| Vec3::X * i as f32).collect(),
            uvs: (0..vertices).map(|i| Vec2::splat(i as f32 * 0.5)).collect(),
            indices: indices.to_vec(),
            // Empty in, empty out: flattening only opens a run when a block contributes triangles,
            // so a mesh with no geometry has no submeshes rather than one covering nothing.
            submeshes: if indices.is_empty() {
                Vec::new()
            } else {
                vec![Submesh {
                    first_index: 0,
                    index_count: indices.len() as u32,
                    material: 0,
                }]
            },
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
            submeshes,
        } = packed(&meshes);

        assert_eq!(
            ranges,
            vec![
                MeshRange {
                    first_vertex: 0,
                    vertex_count: 3,
                    first_index: 0,
                    index_count: 3,
                    first_submesh: 0,
                    submesh_count: 1,
                },
                MeshRange {
                    first_vertex: 3,
                    vertex_count: 4,
                    first_index: 3,
                    index_count: 6,
                    first_submesh: 1,
                    submesh_count: 1,
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
        // Submesh offsets, unlike the indices, *are* rebased onto the shared buffer — a build range
        // addresses the whole buffer, so the second mesh's run starts at 3 rather than 0.
        assert_eq!(
            submeshes,
            vec![
                SubmeshRange {
                    first_index: 0,
                    index_count: 3,
                    material: 0
                },
                SubmeshRange {
                    first_index: 3,
                    index_count: 6,
                    material: 0
                },
            ]
        );
        // Its vertices did move, so position 3 is the second mesh's vertex 0.
        assert_eq!(positions[3], [0.0, 0.0, 0.0]);
        assert_eq!(positions[6], [3.0, 3.0, 3.0]);
    }

    #[test]
    fn an_append_lands_past_what_the_arena_already_holds() {
        // An arena holding 10 vertices, 30 indices and 2 submeshes takes a 3-vertex, 1-triangle
        // mesh: it starts at vertex 10 and index 30, and its submesh is the third.
        let meshes = [mesh(3, &[0, 1, 2])];
        let out = pack(&meshes.iter().collect::<Vec<_>>(), 10, 30, 2);

        assert_eq!(
            out.ranges[0],
            MeshRange {
                first_vertex: 10,
                vertex_count: 3,
                first_index: 30,
                index_count: 3,
                first_submesh: 2,
                submesh_count: 1,
            }
        );
        // The submesh counts from the start of the whole buffer, not of its mesh.
        assert_eq!(out.submeshes[0].first_index, 30);
        // Index *values* stay mesh-local, which is what lets a mesh be appended anywhere without
        // rewriting its indices — the build adds `first_vertex` instead.
        assert_eq!(out.indices, vec![0, 1, 2]);
    }

    #[test]
    fn attributes_stay_aligned_with_the_positions_they_belong_to() {
        let meshes = [mesh(2, &[0, 1, 1]), mesh(2, &[0, 1, 1])];
        let out = packed(&meshes);
        let (positions, attributes) = (out.positions, out.attributes);

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
        let ranges = packed(&meshes).ranges;

        assert_eq!(ranges.len(), 3);
        assert_eq!(
            ranges[1],
            MeshRange {
                first_vertex: 3,
                vertex_count: 0,
                first_index: 3,
                index_count: 0,
                first_submesh: 1,
                submesh_count: 0,
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
