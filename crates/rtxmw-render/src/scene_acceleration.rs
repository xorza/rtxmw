//! Building a cell's acceleration structures: one BLAS per mesh, one TLAS over the placements.

use ash::vk;
use rtxmw_gpu::{Buffer, BufferMemory, Device, RayTracingLimits, Uploader, memory_barrier};
use rtxmw_scene::Instance;

use crate::acceleration_structure::AccelerationStructure;
use crate::geometry_buffers::GeometryBuffers;

/// Everything the ray tracer traverses for one cell.
///
/// One BLAS per distinct mesh and one instance per placement is the whole point of the mesh/instance
/// split: Seyda Neen's Census and Excise Office places 261 objects drawn from 104 meshes, so a
/// per-placement structure would build and store the same geometry two and a half times over.
#[derive(Debug)]
pub struct SceneAcceleration {
    /// One per mesh that had triangles, in mesh order but skipping the empty ones.
    blas: Vec<AccelerationStructure>,
    tlas: AccelerationStructure,
    /// Owned so it outlives the top-level build, and because a refit will rewrite it in place.
    instances: Buffer,
    instance_count: u32,
    /// What the uncompacted structures measured, kept so the saving can be reported.
    uncompacted_blas_bytes: vk::DeviceSize,
}

impl SceneAcceleration {
    /// Builds every structure and compacts the bottom level.
    ///
    /// Costs four queue submissions — build and measure, compact, upload instances, build top level
    /// — each waiting on a fence. That is load-time work; overlapping it belongs with cell streaming
    /// at M9, where the latency is actually visible.
    pub fn build(
        device: &Device,
        uploader: &mut Uploader,
        limits: RayTracingLimits,
        geometry: &GeometryBuffers,
        instances: &[Instance],
    ) -> rtxmw_gpu::Result<Self> {
        let BuiltBottomLevel {
            structures,
            by_mesh,
            uncompacted_bytes,
            compacted_sizes,
        } = build_bottom_level(device, uploader, limits, geometry)?;
        let blas = compact(device, uploader, structures, &compacted_sizes)?;

        let instance_data = describe_instances(&by_mesh, &blas, instances);
        let instance_buffer = Buffer::with_alignment(
            uploader.memory(),
            "tlas instances",
            instance_buffer_size(instance_data.len()),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            BufferMemory::Device,
            INSTANCE_ADDRESS_ALIGNMENT,
        )?;
        uploader.upload(&instance_buffer, instance_bytes_of(&instance_data))?;

        let instance_count = instance_data.len() as u32;
        let tlas = build_top_level(device, uploader, limits, &instance_buffer, instance_count)?;

        Ok(Self {
            blas,
            tlas,
            instances: instance_buffer,
            instance_count,
            uncompacted_blas_bytes: uncompacted_bytes,
        })
    }

    /// The structure a ray query is initialised against.
    pub fn tlas(&self) -> &AccelerationStructure {
        &self.tlas
    }

    /// How many bottom-level structures were built.
    pub fn blas_count(&self) -> usize {
        self.blas.len()
    }

    /// How many placements the top level holds.
    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }

    /// Device memory across the bottom level, after compaction.
    pub fn blas_bytes(&self) -> vk::DeviceSize {
        self.blas.iter().map(AccelerationStructure::size).sum()
    }

    /// What the bottom level measured before compaction.
    pub fn uncompacted_blas_bytes(&self) -> vk::DeviceSize {
        self.uncompacted_blas_bytes
    }

    /// Every byte the scene's traversal data occupies, instance buffer included.
    pub fn byte_size(&self) -> vk::DeviceSize {
        self.blas_bytes() + self.tlas.size() + self.instances.size()
    }
}

/// Alignment a top-level build requires of its instance buffer's address.
const INSTANCE_ADDRESS_ALIGNMENT: vk::DeviceSize = 16;

/// Bottom-level structures fresh from a build, before compaction.
struct BuiltBottomLevel {
    structures: Vec<AccelerationStructure>,
    /// Mesh index to position in `structures`, with `NO_BLAS` where the mesh had no triangles.
    by_mesh: Vec<u32>,
    uncompacted_bytes: vk::DeviceSize,
    /// What each structure reports it needs once compacted, queried in the build's own submission.
    compacted_sizes: Vec<vk::DeviceSize>,
}

/// Marks a mesh that flattened to nothing, so no structure was built for it.
const NO_BLAS: u32 = u32::MAX;

/// Builds one structure per mesh that has triangles, and measures what each would compact to.
fn build_bottom_level(
    device: &Device,
    uploader: &mut Uploader,
    limits: RayTracingLimits,
    geometry: &GeometryBuffers,
) -> rtxmw_gpu::Result<BuiltBottomLevel> {
    let loader = device.acceleration_structure();
    let positions = geometry.positions().device_address();
    let indices = geometry.indices().device_address();

    // Kept in one flat vector rather than one per structure: every build info points into it, so it
    // must not move while the infos are alive.
    let mut geometries = Vec::with_capacity(geometry.ranges().len());
    let mut ranges = Vec::with_capacity(geometry.ranges().len());
    let mut primitive_counts = Vec::with_capacity(geometry.ranges().len());
    let mut by_mesh = Vec::with_capacity(geometry.ranges().len());

    for range in geometry.ranges() {
        if range.index_count == 0 {
            by_mesh.push(NO_BLAS);
            continue;
        }
        by_mesh.push(geometries.len() as u32);

        let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: positions,
            })
            .vertex_stride(GeometryBuffers::POSITION_STRIDE)
            // The highest vertex this build will address. Indices are mesh-local and `first_vertex`
            // is added to them, so the ceiling is where this mesh ends, not where it starts.
            .max_vertex(range.first_vertex + range.vertex_count - 1)
            .index_type(vk::IndexType::UINT32)
            .index_data(vk::DeviceOrHostAddressConstKHR {
                device_address: indices,
            });
        geometries.push(
            vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
                // Pairs with `gl_RayFlagsOpaqueEXT` in the visibility shader. Both change together
                // at M4, when alpha-tested foliage needs the any-hit path.
                .flags(vk::GeometryFlagsKHR::OPAQUE),
        );
        ranges.push(
            vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(range.triangle_count())
                // Byte offset into the index buffer, against `first_vertex` added to each index.
                .primitive_offset(range.first_index * size_of::<u32>() as u32)
                .first_vertex(range.first_vertex),
        );
        primitive_counts.push(range.triangle_count());
    }

    if geometries.is_empty() {
        return Ok(BuiltBottomLevel {
            structures: Vec::new(),
            by_mesh,
            uncompacted_bytes: 0,
            compacted_sizes: Vec::new(),
        });
    }

    // Sizes first: the structures cannot be created until the driver has been asked how large they
    // need to be, and scratch cannot be allocated until every build's share is known.
    let mut structures = Vec::with_capacity(geometries.len());
    let mut scratch_offsets = Vec::with_capacity(geometries.len());
    let mut scratch_total = 0;
    let mut uncompacted_bytes = 0;

    for (index, count) in primitive_counts.iter().enumerate() {
        let sizing = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(blas_flags())
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries[index..=index]);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: `sizing` is fully initialised and the counts array matches its geometry count.
        unsafe {
            loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &sizing,
                &[*count],
                &mut sizes,
            )
        };

        structures.push(AccelerationStructure::create(
            uploader.memory(),
            loader,
            "blas",
            sizes.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        )?);
        uncompacted_bytes += sizes.acceleration_structure_size;

        scratch_offsets.push(scratch_total);
        // Each build needs scratch nothing else is touching, so they get separate regions rather
        // than one shared block plus barriers — that way the whole set builds in one call.
        scratch_total += align_up(
            sizes.build_scratch_size,
            limits.min_scratch_offset_alignment as vk::DeviceSize,
        );
    }

    let scratch = Buffer::with_alignment(
        uploader.memory(),
        "blas build scratch",
        scratch_total,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        BufferMemory::Device,
        limits.min_scratch_offset_alignment as vk::DeviceSize,
    )?;
    let scratch_base = scratch.device_address();

    let infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHR<'_>> = (0..geometries.len())
        .map(|index| {
            vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(blas_flags())
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .dst_acceleration_structure(structures[index].raw())
                .geometries(&geometries[index..=index])
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_base + scratch_offsets[index],
                })
        })
        .collect();
    let range_slices: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> =
        ranges.iter().map(std::slice::from_ref).collect();

    // The compacted-size query rides along in the build's own submission, behind a barrier. Split
    // across two submissions it would depend on ordering that a fence wait does not establish for
    // the device, and cost a round trip for nothing.
    let count = structures.len() as u32;
    let handles: Vec<vk::AccelerationStructureKHR> =
        structures.iter().map(AccelerationStructure::raw).collect();
    let raw_device = device.raw();
    let pool_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
        .query_count(count);
    // SAFETY: `pool_info` is fully initialised and the device is alive.
    let pool = unsafe { raw_device.create_query_pool(&pool_info, None)? };

    let mut compacted_sizes = vec![0u64; count as usize];
    let measured = uploader
        .submit_and_wait(|raw, cmd| {
            // SAFETY: every structure, buffer and scratch region is alive and distinct, and the
            // query pool was created on this device.
            unsafe {
                raw.cmd_reset_query_pool(cmd, pool, 0, count);
                loader.cmd_build_acceleration_structures(cmd, &infos, &range_slices);
                memory_barrier::full(raw, cmd);
                loader.cmd_write_acceleration_structures_properties(
                    cmd,
                    &handles,
                    vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                    pool,
                    0,
                );
            }
        })
        .and_then(|()| {
            // SAFETY: the pool holds `count` results and the submission that wrote them completed.
            unsafe {
                raw_device.get_query_pool_results(
                    pool,
                    0,
                    &mut compacted_sizes,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
            }
            .map_err(Into::into)
        });
    // SAFETY: the submission that used the pool has completed.
    unsafe { raw_device.destroy_query_pool(pool, None) };
    measured?;

    Ok(BuiltBottomLevel {
        structures,
        by_mesh,
        uncompacted_bytes,
        compacted_sizes,
    })
}

/// `PREFER_FAST_TRACE` because static scenery is built once and traversed forever; `ALLOW_COMPACTION`
/// because the driver's conservative first estimate is routinely half again what the result needs.
fn blas_flags() -> vk::BuildAccelerationStructureFlagsKHR {
    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
        | vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION
}

/// Static scenery is traversed far more often than it is built, and nothing refits yet.
fn tlas_flags() -> vk::BuildAccelerationStructureFlagsKHR {
    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
}

/// Copies every structure down to the size measured during its build.
fn compact(
    device: &Device,
    uploader: &mut Uploader,
    originals: Vec<AccelerationStructure>,
    compacted_sizes: &[vk::DeviceSize],
) -> rtxmw_gpu::Result<Vec<AccelerationStructure>> {
    if originals.is_empty() {
        return Ok(originals);
    }
    debug_assert_eq!(originals.len(), compacted_sizes.len());
    let loader = device.acceleration_structure();

    let mut compacted = Vec::with_capacity(originals.len());
    for size in compacted_sizes {
        compacted.push(AccelerationStructure::create(
            uploader.memory(),
            loader,
            "blas compacted",
            *size,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        )?);
    }

    let copies: Vec<vk::CopyAccelerationStructureInfoKHR> = originals
        .iter()
        .zip(&compacted)
        .map(|(from, to)| {
            vk::CopyAccelerationStructureInfoKHR::default()
                .src(from.raw())
                .dst(to.raw())
                .mode(vk::CopyAccelerationStructureModeKHR::COMPACT)
        })
        .collect();

    uploader.submit_and_wait(|raw, cmd| {
        for copy in &copies {
            // SAFETY: source and destination are alive, and the source's build completed on the
            // previous submission.
            unsafe { loader.cmd_copy_acceleration_structure(cmd, copy) };
        }
        // SAFETY: the command buffer is recording.
        unsafe { memory_barrier::full(raw, cmd) };
    })?;

    // `originals` is dropped on return, after the copies completed — that is what the fence wait in
    // `submit_and_wait` buys.
    Ok(compacted)
}

/// Turns scene placements into the 64-byte records a top-level build reads.
fn describe_instances(
    by_mesh: &[u32],
    blas: &[AccelerationStructure],
    instances: &[Instance],
) -> Vec<vk::AccelerationStructureInstanceKHR> {
    let mut out = Vec::with_capacity(instances.len());
    for instance in instances {
        let slot = by_mesh[instance.mesh.0 as usize];
        // `StaticScene` never places a mesh that flattened to nothing, so this is caller error
        // rather than bad data — and skipping the placement would silently lose geometry.
        assert_ne!(
            slot, NO_BLAS,
            "instance references mesh {} which has no geometry",
            instance.mesh.0
        );

        out.push(vk::AccelerationStructureInstanceKHR {
            transform: row_major_transform(&instance.transform),
            // The custom index is how a hit finds its mesh again, which is what the material lookup
            // at M4 is built on.
            instance_custom_index_and_mask: vk::Packed24_8::new(instance.mesh.0, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                // Morrowind authors single-sided planes and relies on them being visible from both
                // faces, and winding is inconsistent across the mesh library. Culling would drop
                // geometry that the original engine draws.
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas[slot as usize].device_address(),
            },
        });
    }
    out
}

/// Converts a scene transform into the row-major 3x4 a top-level instance stores.
///
/// `glam` is column-major, so the rows here are read across the columns — the same transpose the
/// NIF loader does, and the same silent mirroring if it is skipped.
fn row_major_transform(transform: &glam::Affine3A) -> vk::TransformMatrixKHR {
    let m = transform.matrix3;
    let t = transform.translation;
    vk::TransformMatrixKHR {
        matrix: [
            m.x_axis.x, m.y_axis.x, m.z_axis.x, t.x, //
            m.x_axis.y, m.y_axis.y, m.z_axis.y, t.y, //
            m.x_axis.z, m.y_axis.z, m.z_axis.z, t.z,
        ],
    }
}

/// Builds the top level over an already-uploaded instance buffer.
fn build_top_level(
    device: &Device,
    uploader: &mut Uploader,
    limits: RayTracingLimits,
    instances: &Buffer,
    count: u32,
) -> rtxmw_gpu::Result<AccelerationStructure> {
    let loader = device.acceleration_structure();

    let data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
        .array_of_pointers(false)
        .data(vk::DeviceOrHostAddressConstKHR {
            device_address: instances.device_address(),
        });
    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { instances: data })
        .flags(vk::GeometryFlagsKHR::OPAQUE);
    let geometries = [geometry];

    let sizing = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(tlas_flags())
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geometries);
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: `sizing` is fully initialised and the counts array matches its geometry count.
    unsafe {
        loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &sizing,
            &[count],
            &mut sizes,
        )
    };

    let tlas = AccelerationStructure::create(
        uploader.memory(),
        loader,
        "tlas",
        sizes.acceleration_structure_size,
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
    )?;
    let scratch = Buffer::with_alignment(
        uploader.memory(),
        "tlas build scratch",
        sizes.build_scratch_size.max(1),
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        BufferMemory::Device,
        limits.min_scratch_offset_alignment as vk::DeviceSize,
    )?;

    let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(tlas_flags())
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .dst_acceleration_structure(tlas.raw())
        .geometries(&geometries)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch.device_address(),
        });
    let range = vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(count);

    uploader.submit_and_wait(|_device, cmd| {
        // SAFETY: the structure, its scratch and the instance buffer are alive, and the bottom-level
        // builds this references completed on earlier submissions.
        unsafe {
            loader.cmd_build_acceleration_structures(
                cmd,
                std::slice::from_ref(&info),
                &[std::slice::from_ref(&range)],
            )
        };
    })?;

    Ok(tlas)
}

/// Size of an instance buffer holding `count` records, never zero.
fn instance_buffer_size(count: usize) -> vk::DeviceSize {
    (count * size_of::<vk::AccelerationStructureInstanceKHR>())
        .max(size_of::<vk::AccelerationStructureInstanceKHR>()) as vk::DeviceSize
}

/// Reinterprets the instance records as the bytes an upload copies.
///
/// `VkAccelerationStructureInstanceKHR` is `repr(C)` and 64 bytes with no padding, but it holds a
/// union, so `bytemuck` cannot see that it is plain data.
fn instance_bytes_of(instances: &[vk::AccelerationStructureInstanceKHR]) -> &[u8] {
    const {
        assert!(size_of::<vk::AccelerationStructureInstanceKHR>() == 64);
    }
    // SAFETY: the type is `repr(C)` with no padding and no pointers, so every byte is initialised,
    // and the slice borrow keeps it alive for the returned lifetime.
    unsafe {
        std::slice::from_raw_parts(
            instances.as_ptr().cast::<u8>(),
            std::mem::size_of_val(instances),
        )
    }
}

/// Rounds `value` up to a multiple of `alignment`.
fn align_up(value: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize {
    debug_assert!(
        alignment.is_power_of_two(),
        "alignment {alignment} is not a power of two"
    );
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Affine3A, Mat3, Vec3};

    #[test]
    fn a_column_major_transform_is_written_out_by_rows() {
        // Deliberately asymmetric: a quarter turn about Z followed by a translation. Read back the
        // wrong way round this mirrors, and a symmetric matrix would not show it.
        let rotation = Mat3::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let affine = Affine3A::from_mat3_translation(rotation, Vec3::new(1.0, 2.0, 3.0));

        let written = row_major_transform(&affine).matrix;

        // The rotation takes +X to +Y, so column 0 is (0, 1, 0) and it lands in the *first entry of
        // each row*: row 0 starts 0, row 1 starts 1, row 2 starts 0.
        let expect = [
            0.0, -1.0, 0.0, 1.0, //
            1.0, 0.0, 0.0, 2.0, //
            0.0, 0.0, 1.0, 3.0,
        ];
        for (index, (actual, wanted)) in written.iter().zip(expect).enumerate() {
            assert!(
                (actual - wanted).abs() < 1e-6,
                "entry {index}: got {actual}, wanted {wanted}"
            );
        }

        // The translation must be the last column, not the last row — the classic way to get this
        // wrong is to write the matrix out as 4x3.
        assert_eq!([written[3], written[7], written[11]], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn scratch_offsets_round_up_to_the_required_alignment() {
        assert_eq!(align_up(0, 128), 0);
        assert_eq!(align_up(1, 128), 128);
        assert_eq!(align_up(128, 128), 128);
        assert_eq!(align_up(129, 128), 256);
    }

    #[test]
    fn an_instance_record_is_the_size_the_build_expects() {
        // The build reads a tightly packed array with a hardcoded 64-byte stride.
        assert_eq!(size_of::<vk::AccelerationStructureInstanceKHR>(), 64);
        assert_eq!(instance_buffer_size(3), 192);
        // Never zero, or the buffer creation would be rejected for an empty cell.
        assert_eq!(instance_buffer_size(0), 64);
    }
}
