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
    /// Whether this run is a sheet — see [`rtxmw_scene::Submesh::thin`].
    pub thin: bool,
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

    /// Reserves `copies` blank vertex regions shaped like the mesh in `source`.
    ///
    /// **What makes a deforming placement need no new architecture.** A skinned instance cannot
    /// share its neighbour's vertices — the two are in different poses — so it takes a slice of
    /// these same buffers of its own, and a mesh slot of its own pointing at it. Every index a
    /// shader or a build reads is already rebased by `first_vertex`, so nothing downstream can tell
    /// the difference between a region a cell uploaded and one a compute pass fills.
    ///
    /// **Indices and submeshes are shared with the source rather than copied.** A `MeshRange` names
    /// an index range it does not own, and the indices are mesh-local, so every copy can point at
    /// the same triangles. The submesh *entries* are duplicated because `first_submesh` is what an
    /// instance carries as its custom index and each copy needs its own, but they describe the same
    /// runs of the same materials.
    ///
    /// Nothing is uploaded: the regions hold whatever the allocator left there until the first
    /// deform writes them, which is why the caller must not build an acceleration structure over
    /// one before then.
    pub fn reserve_deformed(
        &mut self,
        uploader: &mut Uploader,
        source: u32,
        copies: u32,
    ) -> rtxmw_gpu::Result<std::ops::Range<u32>> {
        let first_slot = self.ranges.len() as u32;
        let shape = self.ranges[source as usize];
        let vertices = shape.vertex_count as vk::DeviceSize * copies as vk::DeviceSize;
        let position_offset = self.vertices as vk::DeviceSize * Self::POSITION_STRIDE;
        let attribute_offset =
            self.vertices as vk::DeviceSize * size_of::<VertexAttributes>() as vk::DeviceSize;
        grow(
            uploader,
            &mut self.positions,
            position_offset,
            position_offset + vertices * Self::POSITION_STRIDE,
            "scene positions",
            build_input_usage(),
        )?;
        grow(
            uploader,
            &mut self.attributes,
            attribute_offset,
            attribute_offset + vertices * size_of::<VertexAttributes>() as vk::DeviceSize,
            "scene vertex attributes",
            geometry_usage(),
        )?;

        let runs = self.submeshes_of(&shape).to_vec();
        for _ in 0..copies {
            self.ranges.push(MeshRange {
                first_vertex: self.vertices,
                first_submesh: self.submeshes.len() as u32,
                ..shape
            });
            self.submeshes.extend_from_slice(&runs);
            self.vertices += shape.vertex_count;
        }
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
            thin: sub.thin,
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
mod tests;
