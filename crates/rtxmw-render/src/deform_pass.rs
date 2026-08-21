//! Rewriting the vertices of every placement that moves, ahead of the build over them.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use rtxmw_gpu::{Binding, ComputePipeline, Device};

use crate::geometry_buffers::GeometryBuffers;
use crate::shaders;

/// Matches `local_size_x` in `deform.comp`.
const WORKGROUP: u32 = 64;

/// How far a vertex travels at most, in world units, and how many waves span one unit of x.
///
/// Both belong to the stand-in deformation `deform.comp` describes, and both go when skinning
/// replaces it. Chosen so the movement is unmistakable at the scale of the shipped meshes rather
/// than so it looks like anything: what this is for is proving the vertices reach the structure.
const AMPLITUDE: f32 = 8.0;
const WAVENUMBER: f32 = 0.05;

/// One placement, as the residency describes it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Placement {
    /// Mesh slot the pose is read from — the bind pose, as the cell uploaded it.
    pub(crate) source: u32,
    /// Mesh slot the result is written into, reserved by [`GeometryBuffers::reserve_deformed`].
    pub(crate) destination: u32,
    /// How far into its own cycle this placement stands, in seconds.
    pub(crate) phase: f32,
}

/// One placement's vertices, as the shader is handed them.
///
/// **Read and written in the same two buffers.** A deforming placement takes a slice of the shared
/// streams of its own, so the pose stays where the cell uploaded it and the result lands somewhere
/// nothing else addresses — see [`GeometryBuffers::reserve_deformed`].
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Push {
    source: u32,
    destination: u32,
    count: u32,
    time: f32,
    phase: f32,
    amplitude: f32,
    wavenumber: f32,
}

/// The compute pass that moves them, and the list of what it moves.
#[derive(Debug)]
pub(crate) struct DeformPass {
    pass: ComputePipeline,
    /// One entry per placement, in the order its mesh slots were reserved.
    placements: Vec<Push>,
}

impl DeformPass {
    pub(crate) fn new(device: &Device) -> rtxmw_gpu::Result<Self> {
        Ok(Self {
            pass: ComputePipeline::new(
                device,
                &[Binding::storage_buffer(0), Binding::storage_buffer(1)],
                size_of::<Push>() as u32,
                shaders::deform(),
            )?,
            placements: Vec::new(),
        })
    }

    /// Records what each placement reads from and writes to.
    ///
    /// Resolved to vertex offsets here rather than at record time: the slots do not move between
    /// commits, and the frame path must not go looking anything up.
    pub(crate) fn place(&mut self, geometry: &GeometryBuffers, placements: &[Placement]) {
        self.placements.clear();
        self.placements.reserve_exact(placements.len());
        for placement in placements {
            let pose = geometry.ranges()[placement.source as usize];
            let region = geometry.ranges()[placement.destination as usize];
            assert_eq!(
                pose.vertex_count, region.vertex_count,
                "a deforming region has to be shaped like the pose it is written from"
            );
            self.placements.push(Push {
                source: pose.first_vertex,
                destination: region.first_vertex,
                count: region.vertex_count,
                time: 0.0,
                phase: placement.phase,
                amplitude: AMPLITUDE,
                wavenumber: WAVENUMBER,
            });
        }
    }

    /// Points the pass at the streams it rewrites. Re-run whenever they are reallocated.
    pub(crate) fn bind(&mut self, geometry: &GeometryBuffers) {
        self.pass
            .bind_storage_buffers(0, &[geometry.positions(), geometry.attributes()]);
    }

    /// Whether anything has been placed, and so whether a frame has any of this to do.
    pub(crate) fn is_empty(&self) -> bool {
        self.placements.is_empty()
    }

    /// # Safety
    /// `command_buffer` must be recording and [`DeformPass::bind`] must have run.
    ///
    /// One dispatch per placement rather than one over all of them: a placement is a few hundred
    /// vertices and the worst cell in the game holds twenty-two of them, so the dispatch overhead
    /// is smaller than the indirection a single dispatch would need to find its way back to which
    /// placement an invocation belongs to.
    pub(crate) unsafe fn record(&self, command_buffer: vk::CommandBuffer, time: f32) {
        for placement in &self.placements {
            let push = Push { time, ..*placement };
            // SAFETY: the caller guarantees recording and that the set names live buffers.
            unsafe {
                self.pass.dispatch(
                    command_buffer,
                    [placement.count.div_ceil(WORKGROUP), 1, 1],
                    bytemuck::bytes_of(&push),
                )
            };
        }
    }
}
