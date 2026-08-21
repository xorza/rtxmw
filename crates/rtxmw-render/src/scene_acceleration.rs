//! The structures a ray traverses: one BLAS per mesh, one TLAS over every resident placement.

use ash::vk;
use rtxmw_gpu::{Buffer, BufferMemory, Device, RayTracingLimits, Uploader, memory_barrier};
use rtxmw_scene::{AlphaMode, Instance, Material, MaterialKind};

use crate::acceleration_structure::AccelerationStructure;
use crate::geometry_buffers::GeometryBuffers;

/// Everything the ray tracer traverses, across every cell that is resident.
///
/// One BLAS per distinct mesh and one instance per placement is the whole point of the
/// mesh/instance split: Seyda Neen's Census and Excise Office places 261 objects drawn from 104
/// meshes, so a per-placement structure would build and store the same geometry two and a half
/// times over. The saving compounds across cells, where a rock is shared by the whole window.
///
/// The two levels have different lifetimes and that is the streaming story in one line: bottom
/// levels are built once per mesh and kept, the top level is rebuilt whenever a cell arrives or
/// leaves.
pub struct SceneAcceleration {
    /// A handle copy, not an owner: the real `Device` outlives this by construction, and the frame
    /// path records builds without one to hand.
    loader: ash::khr::acceleration_structure::Device,
    /// One per mesh that had triangles, in mesh order but skipping the empty ones.
    blas: Vec<AccelerationStructure>,
    /// Which entry of `blas` each mesh slot built into, or [`NO_BLAS`] for one with no triangles.
    by_mesh: Vec<u32>,
    /// Whether each mesh slot is water, which decides the mask its instances carry.
    is_water: Vec<bool>,
    tlas: AccelerationStructure,
    /// Scratch the top level is rebuilt through, kept because the frame rebuilds it every time
    /// anything below it moved and must not allocate to do so.
    tlas_scratch: Buffer,
    /// Owned so it outlives the top-level build, and because a refit will rewrite it in place.
    instances: Buffer,
    instance_count: u32,
    /// The bottom levels rebuilt inside the frame, or `None` while nothing moves.
    deforming: Option<Deforming>,
    /// What the uncompacted structures measured, kept so the saving can be reported.
    uncompacted_blas_bytes: vk::DeviceSize,
}

// The extension loader is a table of function pointers and implements no `Debug`.
impl std::fmt::Debug for SceneAcceleration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SceneAcceleration")
            .field("blas", &self.blas.len())
            .field("instances", &self.instance_count)
            .field("deforming", &self.deforming)
            .finish_non_exhaustive()
    }
}

impl SceneAcceleration {
    /// An empty scene: no bottom levels, and a top level over no instances.
    ///
    /// A structure rather than `None`, so the descriptor a pass binds is valid from the moment the
    /// renderer exists — a ray query against an empty top level misses, which is the right answer
    /// before any cell is resident.
    pub fn new(
        device: &Device,
        uploader: &mut Uploader,
        limits: RayTracingLimits,
    ) -> rtxmw_gpu::Result<Self> {
        let instances = instance_buffer(uploader, 0)?;
        let built = build_top_level(device, uploader, limits, &instances, 0)?;
        Ok(Self {
            loader: device.acceleration_structure().clone(),
            blas: Vec::new(),
            by_mesh: Vec::new(),
            is_water: Vec::new(),
            tlas: built.structure,
            tlas_scratch: built.scratch,
            instances,
            instance_count: 0,
            deforming: None,
            uncompacted_blas_bytes: 0,
        })
    }

    /// Builds and compacts bottom levels for the meshes occupying `slots`.
    ///
    /// Called once per mesh, when the first cell to name it arrives — a bottom level costs more to
    /// build than everything else about a cell put together, and the cell next door places the
    /// same rocks. Costs two queue submissions for the whole batch, not per mesh.
    pub fn append_meshes(
        &mut self,
        device: &Device,
        uploader: &mut Uploader,
        limits: RayTracingLimits,
        geometry: &GeometryBuffers,
        materials: &[Material],
        slots: std::ops::Range<u32>,
    ) -> rtxmw_gpu::Result<()> {
        let BuiltBottomLevel {
            structures,
            by_mesh,
            uncompacted_bytes,
            compacted_sizes,
        } = build_bottom_level(device, uploader, limits, geometry, materials, slots.clone())?;
        let base = self.blas.len() as u32;
        self.blas
            .extend(compact(device, uploader, structures, &compacted_sizes)?);
        self.by_mesh.extend(by_mesh.into_iter().map(|slot| {
            if slot == NO_BLAS {
                NO_BLAS
            } else {
                slot + base
            }
        }));
        // A mesh is water when its geometry says so, which for the shared water quad is all of it.
        for range in &geometry.ranges()[slots.start as usize..slots.end as usize] {
            let water = geometry
                .submeshes_of(range)
                .iter()
                .all(|submesh| materials[submesh.material as usize].kind == MaterialKind::Water);
            self.is_water.push(water && range.submesh_count > 0);
        }
        self.uncompacted_blas_bytes += uncompacted_bytes;
        Ok(())
    }

    /// Rebuilds the top level over `instances`, replacing whatever it held.
    ///
    /// The whole cost of a cell arriving or leaving, once its meshes are resident: the bottom
    /// levels are untouched and this is one build over what is placed.
    pub fn rebuild_top(
        &mut self,
        device: &Device,
        uploader: &mut Uploader,
        limits: RayTracingLimits,
        geometry: &GeometryBuffers,
        instances: &[Instance],
    ) -> rtxmw_gpu::Result<()> {
        let first_submesh: Vec<u32> = geometry.ranges().iter().map(|r| r.first_submesh).collect();
        let records = describe_instances(
            &self.by_mesh,
            &self.blas,
            &first_submesh,
            &self.is_water,
            instances,
        );
        if records.len() as vk::DeviceSize * INSTANCE_SIZE > self.instances.size() {
            self.instances = instance_buffer(uploader, records.len())?;
        }
        uploader.upload(&self.instances, instance_bytes_of(&records))?;

        self.instance_count = records.len() as u32;
        let built = build_top_level(
            device,
            uploader,
            limits,
            &self.instances,
            self.instance_count,
        )?;
        self.tlas = built.structure;
        self.tlas_scratch = built.scratch;
        Ok(())
    }

    /// Creates a bottom level for each of the mesh slots in `slots`, to be rebuilt every frame.
    ///
    /// **No build happens here**, and the caller owes one before anything references them. The
    /// regions these describe hold whatever the allocator left there until a deform pass has
    /// written them — see [`GeometryBuffers::reserve_deformed`] — so a top level built over them
    /// meanwhile is undefined, which on this hardware is a lost device rather than a wrong
    /// picture. `SceneResidency::commit` pays that debt with one submission of the same two passes
    /// a frame uses; §8.91 is what it cost to find out. What this does is reserve the structures,
    /// at the size the driver says the geometry needs, and the scratch a build will run through.
    ///
    /// `refit` chooses between rebuilding the structure every frame and updating it in place.
    /// Which is better is a measurement rather than a fact — `docs/design.md` M12 — so it is a
    /// parameter and `tests/deforming.rs` is where the two are compared.
    pub fn append_deforming(
        &mut self,
        uploader: &mut Uploader,
        limits: RayTracingLimits,
        geometry: &GeometryBuffers,
        materials: &[Material],
        set: Deformable,
    ) -> rtxmw_gpu::Result<()> {
        let Deformable { slots, refit } = set;
        assert!(
            self.deforming.is_none(),
            "a second deforming set would need its own scratch; there is one set today"
        );
        assert_eq!(
            self.by_mesh.len() as u32,
            slots.start,
            "deforming slots must follow the meshes already registered"
        );
        let loader = &self.loader;
        let described = describe_geometry(geometry, materials, slots.clone());

        let first = self.blas.len() as u32;
        let mut scratch_offsets = Vec::with_capacity(described.spans.len());
        let mut scratch_total = 0;
        for (index, span) in described.spans.iter().enumerate() {
            let sizes = build_sizes(
                loader,
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                refit_flags(refit),
                &described.geometries[span.start as usize..span.end as usize],
                &described.primitive_counts[span.start as usize..span.end as usize],
            );
            self.blas.push(AccelerationStructure::create(
                uploader.memory(),
                loader,
                "deforming blas",
                sizes.acceleration_structure_size,
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            )?);
            self.by_mesh.push(first + index as u32);
            self.is_water.push(false);
            scratch_offsets.push(scratch_total);
            // The larger of the two: the set can be rebuilt as well as refitted — the first frame
            // has to be — and a region sized for an update is not enough for a build.
            scratch_total += align_up(
                sizes.build_scratch_size.max(sizes.update_scratch_size),
                limits.min_scratch_offset_alignment as vk::DeviceSize,
            );
        }
        assert_eq!(
            self.by_mesh.len() as u32,
            slots.end,
            "every reserved slot must have got a structure; one that flattened to nothing did not"
        );

        let scratch = Buffer::with_alignment(
            uploader.memory(),
            "deforming blas scratch",
            scratch_total.max(1),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            BufferMemory::Device,
            limits.min_scratch_offset_alignment as vk::DeviceSize,
        )?;
        self.deforming = Some(Deforming {
            slots,
            first,
            geometries: described.geometries,
            ranges: described.ranges,
            spans: described.spans,
            scratch,
            scratch_offsets,
            refit,
            built: false,
        });
        Ok(())
    }

    /// Re-derives what the per-frame build reads, after anything that could have moved the buffers.
    ///
    /// **A bottom level built once keeps its own copy of the geometry; one rebuilt every frame does
    /// not.** `GeometryBuffers` reallocates when it grows, so every device address baked into a
    /// build description goes stale the moment another cell arrives. Called from the commit that
    /// follows any such change, which is the same place the top level is rebuilt and for the same
    /// reason.
    pub fn refresh_deforming(&mut self, geometry: &GeometryBuffers, materials: &[Material]) {
        let Some(deforming) = &mut self.deforming else {
            return;
        };
        let described = describe_geometry(geometry, materials, deforming.slots.clone());
        deforming.geometries = described.geometries;
        deforming.ranges = described.ranges;
        deforming.spans = described.spans;
        // A rebuilt buffer is a different source, so an update against the last pose is no longer
        // meaningful — the next frame builds from scratch whatever mode it is in.
        deforming.built = false;
    }

    /// Records a build of every bottom level that moves, over the vertices just written.
    ///
    /// **Also what makes the structures valid in the first place.** They are created unbuilt, over
    /// regions holding whatever the allocator left there, so the load calls this once before it
    /// builds a top level that would otherwise reference garbage — see `SceneResidency::commit`.
    ///
    /// Allocates nothing: every structure, scratch region and description was made at load time,
    /// which is what `tests/frame_allocations.rs` holds this to.
    ///
    /// # Safety
    /// `command_buffer` must be recording, and whatever writes the deforming vertex regions must
    /// already be recorded into it.
    pub unsafe fn record_deforming(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
    ) {
        let loader = &self.loader;
        if let Some(deforming) = &mut self.deforming {
            // SAFETY: the caller guarantees recording; the vertices are ordered against this by the
            // barrier below, which the deform pass's writes are on the far side of.
            unsafe {
                memory_barrier::scoped(
                    device,
                    command_buffer,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                    vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                )
            };
            let update = deforming.refit && deforming.built;
            let mode = if update {
                vk::BuildAccelerationStructureModeKHR::UPDATE
            } else {
                vk::BuildAccelerationStructureModeKHR::BUILD
            };
            let scratch_base = deforming.scratch.device_address();
            let infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHR<'_>> = deforming
                .spans
                .iter()
                .enumerate()
                .map(|(index, span)| {
                    let structure = self.blas[deforming.first as usize + index].raw();
                    vk::AccelerationStructureBuildGeometryInfoKHR::default()
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                        .flags(refit_flags(deforming.refit))
                        .mode(mode)
                        // An update reads the structure it writes: refitting moves the vertices a
                        // tree already has rather than choosing a tree, which is the whole
                        // difference between the two modes.
                        .src_acceleration_structure(if update {
                            structure
                        } else {
                            vk::AccelerationStructureKHR::null()
                        })
                        .dst_acceleration_structure(structure)
                        .geometries(&deforming.geometries[span.start as usize..span.end as usize])
                        .scratch_data(vk::DeviceOrHostAddressKHR {
                            device_address: scratch_base + deforming.scratch_offsets[index],
                        })
                })
                .collect();
            let range_slices: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> = deforming
                .spans
                .iter()
                .map(|span| &deforming.ranges[span.start as usize..span.end as usize])
                .collect();
            // SAFETY: every structure, scratch region and geometry buffer is alive, and the
            // descriptions were re-derived at the last commit.
            unsafe {
                loader.cmd_build_acceleration_structures(command_buffer, &infos, &range_slices)
            };
            deforming.built = true;

            // SAFETY: recording, and this is the dependency a top-level build has on the
            // structures just written.
            unsafe {
                memory_barrier::scoped(
                    device,
                    command_buffer,
                    vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                    vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                    vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                    vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
                )
            };
        }
    }

    /// Rebuilds the top level over whatever the bottom levels have become.
    ///
    /// **Necessary whether or not any instance moved**: a top level holds the bounds its bottom
    /// levels had when it was built, so one of them being rebuilt in place leaves it stale.
    ///
    /// # Safety
    /// `command_buffer` must be recording, and [`SceneAcceleration::record_deforming`] must already
    /// be recorded into it.
    pub unsafe fn record_top(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        let loader = &self.loader;
        // **Into the structure that is already there**, never a fresh one: the descriptor a pass
        // binds is written when a cell arrives, and a new handle every frame would mean rewriting
        // it every frame. The instance buffer has not changed either — what makes this necessary is
        // the bottom levels underneath it having moved.
        let data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: self.instances.device_address(),
            });
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { instances: data })
            .flags(vk::GeometryFlagsKHR::OPAQUE);
        let geometries = [geometry];
        let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(tlas_flags())
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(self.tlas.raw())
            .geometries(&geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: self.tlas_scratch.device_address(),
            });
        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(self.instance_count);
        // SAFETY: the structure, its scratch and the instance buffer are alive, and the bottom
        // levels this reads were ordered against it above.
        unsafe {
            loader.cmd_build_acceleration_structures(
                command_buffer,
                std::slice::from_ref(&info),
                &[std::slice::from_ref(&range)],
            );
            memory_barrier::scoped(
                device,
                command_buffer,
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
                    | vk::AccessFlags2::SHADER_STORAGE_READ,
            );
        }
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
}

/// The mask bit ordinary geometry carries, and the one a shadow ray tests against.
///
/// **Water is deliberately absent from it.** Sunlight reaching a seabed has passed *through* the
/// surface, so water must not occlude — and the mask is how traversal is told that, at no cost.
/// Doing it in the any-hit loop instead, by building water non-opaque, measured at half the frame
/// rate: every shadow ray crossing the sea would invoke a shader where hardware had been enough.
const MASK_SOLID: u8 = 0x01;

/// The mask bit water carries. Visible to a ray that asks for everything, invisible to one that
/// asks only for [`MASK_SOLID`].
const MASK_WATER: u8 = 0x02;

/// Alignment a top-level build requires of its instance buffer's address.
const INSTANCE_ADDRESS_ALIGNMENT: vk::DeviceSize = 16;

/// What a set of deforming placements is registered with.
///
/// A pair rather than two arguments because they are one decision: which slots move, and what the
/// frame does with the structures over them.
#[derive(Debug, Clone)]
pub struct Deformable {
    /// The mesh slots reserved for it — see [`GeometryBuffers::reserve_deformed`].
    pub slots: std::ops::Range<u32>,
    /// Whether a frame updates its structures in place rather than rebuilding them.
    pub refit: bool,
}

/// The bottom levels a frame rebuilds, and everything that rebuild needs.
///
/// One set rather than one per cell: the whole point of the per-frame path is that it is a single
/// build call and a single scratch allocation however many placements are in it.
#[derive(Debug)]
struct Deforming {
    /// Mesh slots these were reserved for, which is also where their descriptions come from.
    slots: std::ops::Range<u32>,
    /// Where in `SceneAcceleration::blas` the first of these structures sits; they are contiguous.
    first: u32,
    /// Kept in one flat vector rather than one per structure: every build info points into it, so
    /// it must not move while the infos are alive.
    geometries: Vec<vk::AccelerationStructureGeometryKHR<'static>>,
    ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
    /// Which slice of `geometries` and `ranges` each structure is built from.
    spans: Vec<std::ops::Range<u32>>,
    scratch: Buffer,
    scratch_offsets: Vec<vk::DeviceSize>,
    /// Whether a frame updates these in place rather than rebuilding them.
    refit: bool,
    /// Whether anything has been built into them yet. An update has to have something to update.
    built: bool,
}

/// What a build reads, derived from a run of mesh slots.
#[derive(Debug)]
struct Described {
    geometries: Vec<vk::AccelerationStructureGeometryKHR<'static>>,
    ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
    primitive_counts: Vec<u32>,
    /// Which slice of the three above each structure is built from.
    spans: Vec<std::ops::Range<u32>>,
    /// Mesh index to position in `spans`, with [`NO_BLAS`] where the mesh had no triangles.
    by_mesh: Vec<u32>,
}

/// Describes the meshes occupying `slots` as acceleration structure geometry.
///
/// Shared by the build that happens once when a cell arrives and the one that happens every frame,
/// because the two describe the same thing and a second copy of this is a second set of device
/// addresses to keep in step.
fn describe_geometry(
    geometry: &GeometryBuffers,
    materials: &[Material],
    slots: std::ops::Range<u32>,
) -> Described {
    let positions = geometry.positions().device_address();
    let indices = geometry.indices().device_address();
    let building = &geometry.ranges()[slots.start as usize..slots.end as usize];
    let mut described = Described {
        geometries: Vec::with_capacity(building.len()),
        ranges: Vec::with_capacity(building.len()),
        primitive_counts: Vec::with_capacity(building.len()),
        spans: Vec::new(),
        by_mesh: Vec::with_capacity(building.len()),
    };

    for range in building {
        if range.submesh_count == 0 {
            described.by_mesh.push(NO_BLAS);
            continue;
        }
        described.by_mesh.push(described.spans.len() as u32);

        // One geometry per material-uniform run rather than one per mesh. A hit reports which it
        // landed on as `geometry_index`, and that is the only thing that lets a lantern's glass
        // shade differently from its metal.
        let first_geometry = described.geometries.len() as u32;
        for submesh in geometry.submeshes_of(range) {
            let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: positions,
                })
                .vertex_stride(GeometryBuffers::POSITION_STRIDE)
                // The highest vertex this build will address. Indices are mesh-local and
                // `first_vertex` is added to them, so the ceiling is where the mesh ends.
                .max_vertex(range.first_vertex + range.vertex_count - 1)
                .index_type(vk::IndexType::UINT32)
                .index_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: indices,
                });
            described.geometries.push(
                vk::AccelerationStructureGeometryKHR::default()
                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                    .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
                    .flags(geometry_flags(materials[submesh.material as usize])),
            );
            described.ranges.push(
                vk::AccelerationStructureBuildRangeInfoKHR::default()
                    .primitive_count(submesh.triangle_count())
                    // Byte offset into the shared index buffer, against `first_vertex` added to
                    // each index value.
                    .primitive_offset(submesh.first_index * size_of::<u32>() as u32)
                    .first_vertex(range.first_vertex),
            );
            described.primitive_counts.push(submesh.triangle_count());
        }
        described
            .spans
            .push(first_geometry..described.geometries.len() as u32);
    }
    described
}

/// Asks the driver how large a structure and its scratch have to be.
fn build_sizes(
    loader: &ash::khr::acceleration_structure::Device,
    kind: vk::AccelerationStructureTypeKHR,
    flags: vk::BuildAccelerationStructureFlagsKHR,
    geometries: &[vk::AccelerationStructureGeometryKHR<'_>],
    primitive_counts: &[u32],
) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
    let sizing = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(kind)
        .flags(flags)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(geometries);
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: `sizing` is fully initialised and the counts array matches its geometry count.
    unsafe {
        loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &sizing,
            primitive_counts,
            &mut sizes,
        )
    };
    sizes
}

/// What a bottom level rebuilt inside the frame is built with.
///
/// **`ALLOW_UPDATE` only where updates are actually taken.** It is not free: a structure that has to
/// stay refittable cannot be organised as freely as one that does not, and these are traversed by
/// every ray in the frame and built once. `PREFER_FAST_BUILD` for the rebuilding case, because a
/// structure thrown away at the end of the frame has no traversals to amortise a slow build over —
/// though which of the two wins overall is what `tests/deforming.rs` measures rather than assumes.
///
/// `ALLOW_COMPACTION` is absent from both: compaction costs a second submission and a copy, and
/// there is nothing to reclaim from a structure that is overwritten sixty times a second.
fn refit_flags(refit: bool) -> vk::BuildAccelerationStructureFlagsKHR {
    if refit {
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
            | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE
    } else {
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD
    }
}

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
    materials: &[Material],
    slots: std::ops::Range<u32>,
) -> rtxmw_gpu::Result<BuiltBottomLevel> {
    if let Some(worst) = geometry.submeshes().iter().map(|s| s.material).max() {
        assert!(
            (worst as usize) < materials.len(),
            "geometry names material {worst} but the table holds {}",
            materials.len()
        );
    }

    let loader = device.acceleration_structure();
    let Described {
        geometries,
        ranges,
        primitive_counts,
        spans: geometry_spans,
        by_mesh,
    } = describe_geometry(geometry, materials, slots);

    if geometry_spans.is_empty() {
        return Ok(BuiltBottomLevel {
            structures: Vec::new(),
            by_mesh,
            uncompacted_bytes: 0,
            compacted_sizes: Vec::new(),
        });
    }

    // Sizes first: the structures cannot be created until the driver has been asked how large they
    // need to be, and scratch cannot be allocated until every build's share is known.
    let mut structures = Vec::with_capacity(geometry_spans.len());
    let mut scratch_offsets = Vec::with_capacity(geometry_spans.len());
    let mut scratch_total = 0;
    let mut uncompacted_bytes = 0;

    for slice in &geometry_spans {
        let span = slice.start as usize..slice.end as usize;
        let sizes = build_sizes(
            loader,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            blas_flags(),
            &geometries[span.clone()],
            &primitive_counts[span],
        );

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

    let infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHR<'_>> = geometry_spans
        .iter()
        .enumerate()
        .map(|(index, slice)| {
            vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(blas_flags())
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .dst_acceleration_structure(structures[index].raw())
                .geometries(&geometries[slice.start as usize..slice.end as usize])
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_base + scratch_offsets[index],
                })
        })
        .collect();
    let range_slices: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> = geometry_spans
        .iter()
        .map(|slice| &ranges[slice.start as usize..slice.end as usize])
        .collect();

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

/// Whether a geometry can skip the any-hit path.
///
/// `OPAQUE` lets traversal commit the first hit without ever invoking a shader. Marking an
/// alpha-tested surface opaque is what turns Morrowind's foliage and grates into solid rectangles,
/// so the flag has to follow the material rather than being set once for everything.
fn geometry_flags(material: Material) -> vk::GeometryFlagsKHR {
    match material.alpha {
        AlphaMode::Opaque => vk::GeometryFlagsKHR::OPAQUE,
        AlphaMode::Mask(_) | AlphaMode::Blend => vk::GeometryFlagsKHR::empty(),
    }
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
    first_submesh: &[u32],
    is_water: &[bool],
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
            // Where this mesh's geometries start in the flat submesh table. A hit adds its own
            // `geometry_index` to it and lands exactly on its own entry, which is what makes the
            // material lookup a single indexed read with no per-mesh indirection.
            instance_custom_index_and_mask: vk::Packed24_8::new(
                first_submesh[instance.mesh.0 as usize],
                if is_water[instance.mesh.0 as usize] {
                    MASK_WATER
                } else {
                    MASK_SOLID
                },
            ),
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

/// A top level and the scratch it was built through, which the frame keeps to rebuild it.
#[derive(Debug)]
struct BuiltTopLevel {
    structure: AccelerationStructure,
    scratch: Buffer,
}

/// Builds the top level over an already-uploaded instance buffer.
fn build_top_level(
    device: &Device,
    uploader: &mut Uploader,
    limits: RayTracingLimits,
    instances: &Buffer,
    count: u32,
) -> rtxmw_gpu::Result<BuiltTopLevel> {
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

    let sizes = build_sizes(
        loader,
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        tlas_flags(),
        &geometries,
        &[count],
    );

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

    Ok(BuiltTopLevel {
        structure: tlas,
        scratch,
    })
}

/// A buffer sized for `count` instance records, aligned to what a top-level build requires.
fn instance_buffer(uploader: &mut Uploader, count: usize) -> rtxmw_gpu::Result<Buffer> {
    let memory = uploader.memory().clone();
    Buffer::with_alignment(
        &memory,
        "tlas instances",
        instance_buffer_size(count),
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST,
        BufferMemory::Device,
        INSTANCE_ADDRESS_ALIGNMENT,
    )
}

/// What one instance record occupies.
const INSTANCE_SIZE: vk::DeviceSize =
    size_of::<vk::AccelerationStructureInstanceKHR>() as vk::DeviceSize;

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
    fn a_shadow_ray_asking_for_solid_geometry_cannot_see_water() {
        // `occluded` spells `MASK_SOLID` out as `0x01u`, because a shader cannot see a Rust
        // constant. What matters is not either value but that they do not overlap: an overlapping
        // bit would put the whole seabed back in the shadow of its own sea.
        assert_eq!(MASK_SOLID, 0x01);
        assert_eq!(MASK_WATER, 0x02);
        assert_eq!(MASK_SOLID & MASK_WATER, 0);
    }

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
