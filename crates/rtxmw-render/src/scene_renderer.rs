//! Everything needed to trace one cell, with no window involved.
//!
//! The split that matters is here: this owns the scene, the pass and the image traced into, and
//! knows nothing about surfaces, swapchains or presentation. That is what lets a test drive the
//! *same* code the engine runs rather than a replica of it — and what keeps the renderer's shape
//! independent of how its output reaches a screen.

use ash::vk;
use glam::Vec3;
use rtxmw_gpu::{
    Device, Image, Memory, PhysicalDevice, RayTracingLimits, Timestamps, Uploader, image_barrier,
    memory_barrier,
};
use rtxmw_scene::{CellId, StaticScene};
use rtxmw_texture::Texture;

use crate::auto_exposure::AutoExposure;
use crate::composite::Composite;
use crate::denoiser::{DEFAULT_PASSES, Denoiser};
use crate::gbuffer::GBuffer;
use crate::light_grid::LightGridExtent;
use crate::scene_residency::SceneResidency;
use crate::tonemap::Tonemap;
use crate::visibility_pass::{FrameConstants, Lighting, SceneBindings, VisibilityPass};

/// Half-float rather than 8-bit: the trace writes linear radiance that tone mapping consumes at M8,
/// and an 8-bit intermediate would clip highlights before anything got the chance to.
pub const TARGET_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Ceiling on the bindless array, fixed because the descriptor set layout is.
///
/// A cell uses far fewer — Seyda Neen's office needs 118 — and the whole shipped library holds
/// 4,311 distinct textures, so this leaves room for a cell far larger than any interior.
const MAX_TEXTURES: u32 = 8192;

/// Diffuse bounce rays per pixel unless a caller says otherwise.
///
/// Four rather than the one the frame budget eventually wants, because nothing accumulates or
/// denoises yet: at one sample the indirect term is a dither pattern rather than an image. It drops
/// to one at M7, where the denoiser is what turns a single sample into a smooth field.
const DEFAULT_BOUNCE_SAMPLES: u32 = 4;

/// How long the device spent on each stage of a frame, in milliseconds.
///
/// Device time, not wall clock: what the GPU was busy for, which is the number `docs/design.md`
/// §5.3's budget is written in. Every field is zero on a device whose queue cannot write
/// timestamps.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct FrameTimings {
    /// The ray traced pass: primary visibility, shadow rays and the diffuse bounce.
    pub trace: f32,
    /// Every à-trous pass together.
    pub denoise: f32,
    pub composite: f32,
    /// Histogram and the reduction over it.
    pub exposure: f32,
    /// Tone curve and sRGB encoding.
    pub tonemap: f32,
}

impl FrameTimings {
    /// How many stages a frame is measured in, and so how many timestamps it writes.
    const STAGES: usize = 5;

    /// Everything the device spent on the frame.
    pub fn total(&self) -> f32 {
        self.trace + self.denoise + self.composite + self.exposure + self.tonemap
    }

    /// Reads the durations back in the order [`SceneRenderer::record`] wrote them.
    fn from_durations(durations: [f32; Self::STAGES]) -> Self {
        Self {
            trace: durations[0],
            denoise: durations[1],
            composite: durations[2],
            exposure: durations[3],
            tonemap: durations[4],
        }
    }
}

impl std::fmt::Display for FrameTimings {
    /// One diagnostic line, totals first.
    ///
    /// Here rather than at the caller so that adding a stage does not silently stop being reported:
    /// a new field would otherwise have to be remembered in every place that prints one.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "device time {:.2} ms: trace {:.2}, denoise {:.2}, composite {:.2}, exposure {:.2}, \
             tonemap {:.2}",
            self.total(),
            self.trace,
            self.denoise,
            self.composite,
            self.exposure,
            self.tonemap,
        )
    }
}

/// A loaded cell and the passes that turn it into a picture.
pub struct SceneRenderer {
    pass: VisibilityPass,
    gbuffer: GBuffer,
    denoiser: Denoiser,
    composite: Composite,
    exposure: AutoExposure,
    tonemap: Tonemap,
    target: Image,
    scene: Option<SceneResidency>,
    bounce_samples: u32,
    denoise_passes: u32,
    time: f32,
    timestamps: Timestamps,
}

impl std::fmt::Debug for SceneRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SceneRenderer")
            .field("target", &self.target.extent())
            .field("loaded", &self.scene.is_some())
            .finish_non_exhaustive()
    }
}

/// The image the trace writes its radiance into.
fn target_image(memory: &Memory, extent: vk::Extent2D) -> rtxmw_gpu::Result<Image> {
    Image::new(
        memory,
        "primary visibility target",
        extent,
        TARGET_FORMAT,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
    )
}

impl SceneRenderer {
    /// Creates the pass and the image it traces into.
    /// The uploader is borrowed rather than owned, here and in every other method: it wraps one
    /// command pool on one queue, and a queue may not be submitted to from two places at once. A
    /// renderer that kept its own would make two renderers on a device a data race.
    pub fn new(
        device: &Device,
        physical: &PhysicalDevice,
        memory: &Memory,
        extent: vk::Extent2D,
    ) -> rtxmw_gpu::Result<Self> {
        let mut renderer = Self {
            pass: VisibilityPass::new(device, memory, MAX_TEXTURES)?,
            gbuffer: GBuffer::new(memory, extent)?,
            denoiser: Denoiser::new(device)?,
            composite: Composite::new(device)?,
            exposure: AutoExposure::new(device, memory)?,
            tonemap: Tonemap::new(device, memory, extent)?,
            target: target_image(memory, extent)?,
            scene: None,
            bounce_samples: DEFAULT_BOUNCE_SAMPLES,
            denoise_passes: DEFAULT_PASSES,
            time: 0.0,
            // One more than the stages, since a duration needs a timestamp either side of it.
            timestamps: Timestamps::new(device, physical, FrameTimings::STAGES as u32 + 1)?,
        };
        renderer.bind_targets();
        Ok(renderer)
    }

    /// Points the post passes at the images they read and write.
    ///
    /// Separate from [`SceneRenderer::bind_scene`] because the two go stale for different reasons:
    /// these follow the images, which only a resize replaces, and that one follows the cell.
    fn bind_targets(&mut self) {
        self.exposure.bind(&self.target);
        self.tonemap.bind(&self.target, self.exposure.buffer());
        self.denoiser.bind(&self.gbuffer);
        self.composite.bind(&self.target, &self.gbuffer);
    }

    /// Sets how many à-trous passes smooth the lighting. Zero leaves it as traced.
    ///
    /// Must be even: the filter ping-pongs between two images and an odd count would finish in the
    /// one nothing reads.
    pub fn set_denoise_passes(&mut self, passes: u32) {
        assert_eq!(
            passes % 2,
            0,
            "the ping-pong has to land back on the first image"
        );
        self.denoise_passes = passes;
    }

    /// Sets how many diffuse bounce rays each pixel casts.
    ///
    /// Zero leaves the cell's ambient as a flat unoccluded fill, which is what the renderer did
    /// before indirect light existed — the two are the same estimator at its endpoints, so this is
    /// the honest A/B for what a bounce actually contributes.
    /// Sets the clock the water's waves move against, in seconds.
    ///
    /// Zero unless something sets it, which is what keeps a screenshot and a test reproducible: the
    /// surface at time zero is one definite shape rather than whenever the frame happened to run.
    pub fn set_time(&mut self, seconds: f32) {
        self.time = seconds;
    }

    pub fn set_bounce_samples(&mut self, samples: u32) {
        self.bounce_samples = samples;
    }

    /// Makes `scene` the only resident cell, replacing whatever was loaded.
    ///
    /// The caller must have waited for device idle: this frees structures a queued frame could
    /// still be reading.
    pub fn load_scene(
        &mut self,
        device: &Device,
        uploader: &mut Uploader,
        limits: RayTracingLimits,
        id: CellId,
        scene: &StaticScene,
        textures: &[Option<Texture>],
    ) -> rtxmw_gpu::Result<()> {
        self.add_cell(device, uploader, limits, id, scene, textures)?;
        let residency = self.scene.as_mut().expect("`add_cell` leaves one in place");
        // Retaining only what was just added, rather than clearing first: an add uploads nothing a
        // resident cell already named, and clearing would throw that away for no gain.
        residency.retain_newest();
        self.commit(device, uploader, limits)
    }

    /// Makes one more cell resident, sharing everything it has in common with those already here.
    ///
    /// Nothing is visible until [`SceneRenderer::commit`], so a caller bringing in several cells
    /// pays for one top-level build rather than one per cell.
    ///
    /// The caller must have waited for device idle.
    pub fn add_cell(
        &mut self,
        device: &Device,
        uploader: &mut Uploader,
        limits: RayTracingLimits,
        id: CellId,
        scene: &StaticScene,
        textures: &[Option<Texture>],
    ) -> rtxmw_gpu::Result<()> {
        let residency = match self.scene.as_mut() {
            Some(residency) => residency,
            None => self
                .scene
                .insert(SceneResidency::new(device, uploader, limits)?),
        };
        residency.add(device, uploader, limits, id, scene, textures)?;
        assert!(
            residency.textures().len() <= MAX_TEXTURES,
            "cells need {} texture slots but the array is built for {MAX_TEXTURES}",
            residency.textures().len()
        );
        Ok(())
    }

    /// Drops a cell, if it was resident. Takes effect at the next [`SceneRenderer::commit`].
    pub fn remove_cell(&mut self, id: &CellId) {
        if let Some(residency) = self.scene.as_mut() {
            residency.remove(id);
        }
    }

    /// How many distinct meshes are on the device, across every cell that has been resident.
    ///
    /// Grow-only, so this counts what has ever been loaded rather than what is placed now — which
    /// is the number that says whether cells are sharing what they name.
    pub fn resident_meshes(&self) -> usize {
        self.scene.as_ref().map_or(0, SceneResidency::mesh_count)
    }

    /// How many instances the top level holds, which is what resident cells actually place.
    pub fn resident_instances(&self) -> u32 {
        self.scene
            .as_ref()
            .map_or(0, |scene| scene.acceleration().instance_count())
    }

    /// Rebuilds what the resident set determines, and points the trace at the result.
    ///
    /// The caller must have waited for device idle.
    pub fn commit(
        &mut self,
        device: &Device,
        uploader: &mut Uploader,
        limits: RayTracingLimits,
    ) -> rtxmw_gpu::Result<()> {
        if let Some(residency) = self.scene.as_mut() {
            residency.commit(device, uploader, limits)?;
        }
        self.bind_scene();
        Ok(())
    }

    /// Points the trace at the loaded cell and the images it writes.
    ///
    /// Called after a cell loads and again after a resize, because both replace one side of that
    /// pairing: the scene changes, or the images it writes into do.
    fn bind_scene(&mut self) {
        let Some(scene) = &self.scene else {
            return;
        };
        self.pass.bind(
            SceneBindings {
                structure: scene.acceleration().tlas(),
                geometry: scene.geometry(),
                tables: scene.tables(),
                lights: scene.lights(),
                light_grid: scene.light_grid(),
                textures: scene.textures(),
            },
            &self.target,
            &self.gbuffer,
        );
    }

    /// Rebuilds every image at `extent`, keeping the loaded cell.
    ///
    /// The pipelines and descriptor sets survive — only what they point at changes — so this is
    /// several image allocations and a round of descriptor writes, not a rebuild.
    ///
    /// The caller must have waited for device idle: these are the images a queued frame reads.
    pub fn resize(&mut self, memory: &Memory, extent: vk::Extent2D) -> rtxmw_gpu::Result<()> {
        if self.target.extent() == extent {
            return Ok(());
        }
        self.target = target_image(memory, extent)?;
        self.gbuffer = GBuffer::new(memory, extent)?;
        self.tonemap.resize(memory, extent)?;
        self.bind_targets();
        self.bind_scene();
        Ok(())
    }

    /// The linear radiance the trace produced, in `TRANSFER_SRC_OPTIMAL` once
    /// [`SceneRenderer::record`] has run.
    ///
    /// Scene-referred and unbounded, which is what a test asserting on hand-computed radiance wants
    /// and what M7's denoiser will consume. Anything going to a screen wants
    /// [`SceneRenderer::output`] instead.
    pub fn target(&self) -> &Image {
        &self.target
    }

    /// The exposed, tone-mapped, sRGB-encoded image, ready to blit to a swapchain or write to a
    /// PNG unchanged.
    pub fn output(&self) -> &Image {
        self.tonemap.output()
    }

    /// Whether a cell is loaded. With none, [`SceneRenderer::record`] does nothing.
    pub fn has_scene(&self) -> bool {
        self.scene.is_some()
    }

    /// The frame constants for a camera, filled in with the loaded cell's lighting.
    pub fn frame_constants(
        &self,
        view: glam::Mat4,
        projection: glam::Mat4,
        camera_position: Vec3,
    ) -> FrameConstants {
        let lighting = self.scene.as_ref().map_or(
            Lighting {
                ambient: Vec3::ZERO,
                light_grid: LightGridExtent::default(),
                sun: None,
                water_level: None,
            },
            SceneResidency::lighting,
        );
        FrameConstants::new(
            view,
            projection,
            camera_position,
            lighting,
            // From the renderer's own target height, so the mip a surface samples follows the
            // resolution it is being traced at.
            FrameConstants::cone_spread_from(projection, self.target.extent().height),
            self.bounce_samples,
            self.time,
        )
    }

    /// What the device spent on the last recorded frame.
    ///
    /// Blocks until the queries resolve, so the frame must already have been submitted. All zeroes
    /// on a device whose queue cannot write timestamps.
    pub fn timings(&self) -> rtxmw_gpu::Result<FrameTimings> {
        // A stack array, so asking what a frame cost is not itself something a frame pays for.
        let mut durations = [0.0; FrameTimings::STAGES];
        self.timestamps.read(&mut durations)?;
        Ok(FrameTimings::from_durations(durations))
    }

    /// Records the trace into `command_buffer`, leaving the target ready to copy from.
    ///
    /// Does nothing without a loaded cell, so a caller need not branch on it.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state, and `device` must own this renderer.
    pub unsafe fn record(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        constants: &FrameConstants,
    ) {
        if !self.has_scene() {
            return;
        }
        let extent = self.target.extent();
        // SAFETY: the caller guarantees the command buffer is recording, and every object below
        // belongs to this device and outlives the submission.
        unsafe {
            // Every image a frame writes starts from `UNDEFINED`, which discards whatever the last
            // one left there. That is wanted rather than tolerated: each is rewritten in full, so
            // preserving the old contents would cost a transition and buy nothing.
            for image in [&self.target]
                .into_iter()
                .chain(self.gbuffer.images())
                .chain([self.tonemap.output()])
            {
                image_barrier::transition(
                    device,
                    command_buffer,
                    image.raw(),
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                );
            }

            // A timestamp between every stage. They mean something here only because the stages
            // are separated by full barriers anyway — without those the device would overlap them
            // and a per-stage figure would be invented.
            self.timestamps.reset(command_buffer);
            self.timestamps.write(command_buffer, 0);

            self.pass.record(command_buffer, extent, constants);
            memory_barrier::full(device, command_buffer);
            self.timestamps.write(command_buffer, 1);

            // The filter leaves the lighting back in the G-buffer's first illumination image, which
            // is the one the composite is bound to.
            self.denoiser
                .record(device, command_buffer, extent, self.denoise_passes);
            self.timestamps.write(command_buffer, 2);

            self.composite.record(command_buffer, extent);
            memory_barrier::full(device, command_buffer);
            self.timestamps.write(command_buffer, 3);

            // Exposure is measured on the composed frame, so it has to follow the composite rather
            // than run alongside the trace.
            self.exposure.record(device, command_buffer, extent);
            self.timestamps.write(command_buffer, 4);

            self.tonemap.record(command_buffer, extent);
            memory_barrier::full(device, command_buffer);
            self.timestamps.write(command_buffer, 5);

            for image in [self.target.raw(), self.tonemap.output().raw()] {
                image_barrier::transition(
                    device,
                    command_buffer,
                    image,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
            }
        }
    }

    /// Records one trace and waits for it, for a caller with no frame loop of its own.
    pub fn render_once(
        &mut self,
        uploader: &mut Uploader,
        constants: &FrameConstants,
    ) -> rtxmw_gpu::Result<()> {
        uploader.submit_and_wait(|device, cmd| {
            // SAFETY: the command buffer is recording and every object is alive.
            unsafe { self.record(device, cmd, constants) };
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_device_that_wrote_no_timestamps_reports_nothing_rather_than_zeroes_it_invented() {
        // `Timestamps::read` returns an empty slice on a queue that cannot time anything, and the
        // caller has to be able to tell that from a frame that genuinely took no time.
        assert_eq!(
            FrameTimings::from_durations([0.0; FrameTimings::STAGES]),
            FrameTimings::default()
        );

        // In the order `record` writes them. The array's width is the type's guarantee that a
        // caller cannot hand over a stage count that disagrees with the pool's.
        let timings = FrameTimings::from_durations([1.0, 2.0, 4.0, 8.0, 16.0]);
        assert_eq!(timings.trace, 1.0);
        assert_eq!(timings.denoise, 2.0);
        assert_eq!(timings.tonemap, 16.0);
        assert_eq!(timings.total(), 31.0);
    }
}
