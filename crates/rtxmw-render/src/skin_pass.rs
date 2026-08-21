//! Posing every placement that moves, ahead of the build over it.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Affine3A;
use rtxmw_gpu::{Binding, Buffer, BufferMemory, ComputePipeline, Device, Uploader};
use rtxmw_scene::{INFLUENCES, Influence, Pose, Rig};

use crate::geometry_buffers::GeometryBuffers;
use crate::shaders;

/// Matches `local_size_x` in `skin.comp`.
const WORKGROUP: u32 = 64;

/// One vertex's influences as the shader reads them.
///
/// Bone indices packed into one word rather than four bytes of their own: a skin names at most
/// sixty-four bones, and scalar layout would otherwise pad the array of four to sixteen.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuInfluence {
    bones: u32,
    weights: [f32; INFLUENCES],
}

impl GpuInfluence {
    fn new(influence: &Influence) -> Self {
        let bones = influence
            .bones
            .iter()
            .enumerate()
            .fold(0u32, |packed, (slot, bone)| {
                packed | (u32::from(*bone) << (slot * 8))
            });
        Self {
            bones,
            weights: influence.weights,
        }
    }
}

/// One bone's pose, as three rows of an affine transform.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuBone {
    rows: [[f32; 4]; 3],
}

impl GpuBone {
    fn new(transform: Affine3A) -> Self {
        let m = transform.matrix3;
        let t = transform.translation;
        Self {
            rows: [
                [m.x_axis.x, m.y_axis.x, m.z_axis.x, t.x],
                [m.x_axis.y, m.y_axis.y, m.z_axis.y, t.y],
                [m.x_axis.z, m.y_axis.z, m.z_axis.z, t.z],
            ],
        }
    }
}

/// What one placement's dispatch is told.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Push {
    source: u32,
    destination: u32,
    count: u32,
    influence_base: u32,
    bone_base: u32,
}

/// One placement, as the residency describes it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Placement {
    /// Mesh slot the bind pose is read from.
    pub(crate) source: u32,
    /// Mesh slot the posed result is written into.
    pub(crate) destination: u32,
    /// Which of the scene's rigs poses it.
    pub(crate) rig: u32,
    /// How far into its own clip it stands, in seconds.
    pub(crate) phase: f32,
}

/// One placement, resolved to the offsets a dispatch and a pose need.
#[derive(Debug, Clone, Copy)]
struct Posed {
    push: Push,
    rig: u32,
    phase: f32,
    bones: u32,
}

/// The compute pass that poses them, the rigs it poses through, and what it poses per frame.
pub(crate) struct SkinPass {
    pass: ComputePipeline,
    /// Every rig any resident cell named, in the numbering `Placement::rig` uses.
    rigs: Vec<Rig>,
    /// Where each rig's influences start in the buffer below.
    influence_bases: Vec<u32>,
    influences: Buffer,
    /// One matrix per bone per placement, rewritten every frame through a mapped pointer.
    bones: Buffer,
    placements: Vec<Posed>,
    /// Scratch the poses are worked out through, so a frame allocates nothing.
    pose: Pose,
}

// The pipeline holds `ash` function tables, which implement no `Debug`.
impl std::fmt::Debug for SkinPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkinPass")
            .field("rigs", &self.rigs.len())
            .field("placements", &self.placements.len())
            .finish_non_exhaustive()
    }
}

impl SkinPass {
    pub(crate) fn new(device: &Device, uploader: &mut Uploader) -> rtxmw_gpu::Result<Self> {
        Ok(Self {
            pass: ComputePipeline::new(
                device,
                &[
                    Binding::storage_buffer(0),
                    Binding::storage_buffer(1),
                    Binding::storage_buffer(2),
                    Binding::storage_buffer(3),
                ],
                size_of::<Push>() as u32,
                shaders::skin(),
            )?,
            rigs: Vec::new(),
            influence_bases: Vec::new(),
            influences: Buffer::storage_of(uploader, "skin influences", &[])?,
            bones: Buffer::new(
                uploader.memory(),
                "skin bones",
                Buffer::MIN_SIZE,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                // Written straight through a mapped pointer every frame, like the frame constants:
                // the renderer keeps one frame in flight and has already waited for it.
                BufferMemory::Upload,
            )?,
            placements: Vec::new(),
            pose: Pose::default(),
        })
    }

    /// Adds `rigs` to what this can pose, returning where they landed.
    ///
    /// The influences of every rig go into one buffer, uploaded whole: a cell arriving is rare and
    /// the whole set is a few hundred kilobytes at the game's own scale.
    pub(crate) fn add_rigs(
        &mut self,
        uploader: &mut Uploader,
        rigs: &[Rig],
    ) -> rtxmw_gpu::Result<u32> {
        let first = self.rigs.len() as u32;
        self.rigs.extend_from_slice(rigs);
        let mut packed: Vec<GpuInfluence> = Vec::new();
        self.influence_bases.clear();
        for rig in &self.rigs {
            self.influence_bases.push(packed.len() as u32);
            packed.extend(rig.influences.iter().map(GpuInfluence::new));
        }
        self.influences =
            Buffer::storage_of(uploader, "skin influences", bytemuck::cast_slice(&packed))?;
        Ok(first)
    }

    /// Records what each placement reads from, writes to and is posed by.
    pub(crate) fn place(
        &mut self,
        uploader: &mut Uploader,
        geometry: &GeometryBuffers,
        placements: &[Placement],
    ) -> rtxmw_gpu::Result<()> {
        self.placements.clear();
        self.placements.reserve_exact(placements.len());
        let mut bones = 0u32;
        for placement in placements {
            let pose = geometry.ranges()[placement.source as usize];
            let region = geometry.ranges()[placement.destination as usize];
            let rig = &self.rigs[placement.rig as usize];
            assert_eq!(
                rig.influences.len() as u32,
                pose.vertex_count,
                "a rig has one influence per vertex of the mesh it poses"
            );
            self.placements.push(Posed {
                push: Push {
                    source: pose.first_vertex,
                    destination: region.first_vertex,
                    count: region.vertex_count,
                    influence_base: self.influence_bases[placement.rig as usize],
                    bone_base: bones,
                },
                rig: placement.rig,
                phase: placement.phase,
                bones,
            });
            bones += rig.bones.len() as u32;
        }
        let wanted = vk::DeviceSize::from(bones.max(1)) * size_of::<GpuBone>() as vk::DeviceSize;
        if wanted > self.bones.size() {
            self.bones = Buffer::new(
                uploader.memory(),
                "skin bones",
                wanted,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                BufferMemory::Upload,
            )?;
        }
        Ok(())
    }

    /// Points the pass at the streams it rewrites. Re-run whenever they are reallocated.
    pub(crate) fn bind(&mut self, geometry: &GeometryBuffers) {
        self.pass.bind_storage_buffers(
            0,
            &[
                geometry.positions(),
                geometry.attributes(),
                &self.influences,
                &self.bones,
            ],
        );
    }

    /// Whether anything has been placed, and so whether a frame has any of this to do.
    pub(crate) fn is_empty(&self) -> bool {
        self.placements.is_empty()
    }

    /// Poses every placement at `time` and writes the matrices the dispatch will read.
    ///
    /// Host work, and the only part of this that is: a skeleton is tens of joints and a frame holds
    /// tens of placements, so what the GPU is handed is a few hundred matrices worked out in
    /// microseconds. Allocates nothing — `Pose` is scratch this owns, and the write goes straight
    /// into mapped memory.
    pub(crate) fn pose(&mut self, time: f32) {
        let mapped = self
            .bones
            .mapped_mut()
            .expect("skin bones are host visible");
        for placement in &self.placements {
            let rig = &self.rigs[placement.rig as usize];
            rig.pose(time + placement.phase, &mut self.pose);
            let at = placement.bones as usize * size_of::<GpuBone>();
            for (index, bone) in self.pose.bones().iter().enumerate() {
                let packed = GpuBone::new(*bone);
                let offset = at + index * size_of::<GpuBone>();
                mapped[offset..offset + size_of::<GpuBone>()]
                    .copy_from_slice(bytemuck::bytes_of(&packed));
            }
        }
    }

    /// # Safety
    /// `command_buffer` must be recording, [`SkinPass::bind`] must have run, and [`SkinPass::pose`]
    /// must have written this frame's matrices.
    ///
    /// One dispatch per placement rather than one over all of them: a placement is a few hundred
    /// vertices and the busiest cell in the game holds twenty-two, so the dispatch overhead is
    /// smaller than the indirection a single dispatch would need to find its way back to which
    /// placement an invocation belongs to.
    pub(crate) unsafe fn record(&self, command_buffer: vk::CommandBuffer) {
        for placement in &self.placements {
            // SAFETY: the caller guarantees recording and that the set names live buffers.
            unsafe {
                self.pass.dispatch(
                    command_buffer,
                    [placement.push.count.div_ceil(WORKGROUP), 1, 1],
                    bytemuck::bytes_of(&placement.push),
                )
            };
        }
    }
}
