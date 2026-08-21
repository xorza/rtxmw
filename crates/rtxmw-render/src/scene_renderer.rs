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
use rtxmw_scene::{CellId, Sky, SkyTextures, StaticScene};
use rtxmw_texture::Texture;

use crate::auto_exposure::AutoExposure;
use crate::composite::Composite;
use crate::denoiser::{DEFAULT_PASSES, Denoiser};
use crate::gbuffer::GBuffer;
use crate::scene_residency::SceneResidency;
use crate::tonemap::Tonemap;
use crate::visibility_pass::{
    FrameConstants, Lighting, Sampling, SceneBindings, Viewpoint, VisibilityPass,
};

/// Half-float rather than 8-bit: the trace writes linear radiance that tone mapping consumes at M8,
/// and an 8-bit intermediate would clip highlights before anything got the chance to.
pub const TARGET_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Ceiling on the bindless array, fixed because the descriptor set layout is.
///
/// **Slots, not textures**, and each texture takes two of them: the array interleaves every texture
/// with the shading map estimated from it, because Vulkan allows a variable descriptor count only
/// on a set's final binding and there is therefore no second array to put the maps in. A cell uses
/// far fewer — Seyda Neen's office needs 118 textures — and the whole shipped library holds 4,311
/// distinct ones, which is 8,622 slots.
const MAX_TEXTURES: u32 = 16384;

/// Diffuse bounce rays per pixel unless a caller says otherwise.
///
/// Four rather than the one the frame budget eventually wants, because nothing accumulates or
/// denoises yet: at one sample the indirect term is a dither pattern rather than an image. It drops
/// to one at M7, where the denoiser is what turns a single sample into a smooth field.
const DEFAULT_BOUNCE_SAMPLES: u32 = 4;

/// How much baked lighting a frame divides out unless a caller says otherwise.
///
/// **The whole estimate.** Vanilla albedo has shading painted into it for a renderer with no
/// lighting of its own, and tracing over that lights every surface twice — leaving it in is the
/// wrong default for a renderer whose whole point is that it lights things. `--delight 0` is the
/// A/B, and §5.1's warning about over-correction is what that switch is for.
const DEFAULT_DELIGHT: f32 = 1.0;

/// How far a texture's painted relief tilts the normal unless a caller says otherwise.
///
/// **All of it.** Vanilla meshes carry no normal map and the format cannot hold one, so without
/// this every wall in the game is lit as the flat triangle it is — `--relief 0` is the A/B, and
/// `relief.glsl` is what it switches off.
const DEFAULT_RELIEF: f32 = 1.0;

/// Whether the structures over everything that moves are refitted rather than rebuilt.
///
/// **Measured, not assumed** — `docs/design.md` M12. Twenty-two placements of 1,682 triangles, the
/// busiest cell in the game: refitting builds them in 0.108 ms against 0.242, and the frame traces
/// them in the same 0.085 either way, with no traversal penalty even where the pose has moved a
/// third of the mesh's own size from the one the tree was built for.
const DEFAULT_REFIT: bool = true;

/// How long the device spent on each stage of a frame, in milliseconds, and at what size.
///
/// Device time, not wall clock: what the GPU was busy for, which is the number `docs/design.md`
/// §5.3's budget is written in. Every field is zero on a device whose queue cannot write
/// timestamps.
///
/// **The sizes travel with the durations because a duration on its own is not a measurement.** A
/// trace time means nothing without the resolution it traced at, and that resolution is *not* the
/// one a caller asked for: `--screenshot 1920x1080` names the output, and an upscaler set to quality
/// silently traces at 1280x720 — two and a quarter times fewer pixels. A figure copied out of this
/// line and written down as "at 1920x1080" was wrong by that factor once already, so the line now
/// carries what it was measured at and there is nothing left to assume.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct FrameTimings {
    /// What the trace ran at, which is what every duration below is a duration *for*.
    pub traced: vk::Extent2D,
    /// What came out, which differs from [`Self::traced`] exactly when an upscaler is attached.
    pub displayed: vk::Extent2D,
    /// Moving what moves and rebuilding what a ray traverses over it — see
    /// `SceneResidency::record_animation`. Zero for a frame in which nothing does.
    pub animate: f32,
    /// The ray traced pass: primary visibility, shadow rays and the diffuse bounce.
    pub trace: f32,
    /// Every à-trous pass together.
    pub denoise: f32,
    pub composite: f32,
    /// The upscaler, where one is attached; an empty window otherwise. Its own stage rather than
    /// part of the composite because it is the most expensive thing in a frame that has one, and
    /// measuring it as part of the cheapest reported 0.65 ms of compositing that cost 0.01.
    pub upscale: f32,
    /// Histogram and the reduction over it.
    pub exposure: f32,
    /// Tone curve and sRGB encoding.
    pub tonemap: f32,
}

impl FrameTimings {
    /// How many stages a frame is measured in, and so how many timestamps it writes.
    const STAGES: usize = 7;

    /// Everything the device spent on the frame.
    pub fn total(&self) -> f32 {
        self.animate
            + self.trace
            + self.denoise
            + self.composite
            + self.upscale
            + self.exposure
            + self.tonemap
    }

    /// Reads the durations back in the order [`SceneRenderer::record`] wrote them.
    fn from_durations(
        durations: [f32; Self::STAGES],
        traced: vk::Extent2D,
        displayed: vk::Extent2D,
    ) -> Self {
        Self {
            traced,
            displayed,
            animate: durations[0],
            trace: durations[1],
            denoise: durations[2],
            composite: durations[3],
            upscale: durations[4],
            exposure: durations[5],
            tonemap: durations[6],
        }
    }
}

impl std::fmt::Display for FrameTimings {
    /// One diagnostic line, totals first.
    ///
    /// Here rather than at the caller so that adding a stage does not silently stop being reported:
    /// a new field would otherwise have to be remembered in every place that prints one.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "device time {:.2} ms ", self.total())?;
        // Both sizes, never one: the mistake this exists to stop is taking the size a caller asked
        // for as the size the trace ran at, and printing only one of them is what allowed it.
        match self.traced == self.displayed {
            true => write!(f, "at {}x{}", self.traced.width, self.traced.height)?,
            false => write!(
                f,
                "tracing {}x{} to {}x{}",
                self.traced.width, self.traced.height, self.displayed.width, self.displayed.height
            )?,
        }
        write!(
            f,
            ": animate {:.2}, trace {:.2}, denoise {:.2}, composite {:.2}, upscale {:.2}, \
             exposure {:.2}, tonemap {:.2}",
            self.animate,
            self.trace,
            self.denoise,
            self.composite,
            self.upscale,
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
    /// The clock the weather's own events run on, which is the frame's with the speed key left out.
    ///
    /// **A flash is an event, not a rate** — see `WorldClock::weather_seconds`. Driven by [`Self::time`]
    /// like everything else, a quarter-second flash lasted a quarter second divided by whatever the
    /// world's speed was set to, so asking for a bolt did nothing at any setting but the slowest.
    storm: f32,
    /// How far the lightning clock is nudged from the frame's, so a key can bring a flash forward
    /// without anything having to remember that it was pressed — see [`Self::strike`].
    lightning_offset: f32,
    /// When the last flash a key asked for began, on the storm's own clock.
    ///
    /// The one thing here that *is* remembered, and only so [`Self::restrike`] can go back to it.
    /// Everything the flash is made of is still drawn from that moment rather than stored.
    last_strike: Option<f32>,
    /// Where the camera stood when [`Self::frame_constants`] was last asked, or `None` before the
    /// first frame. What this frame's motion vectors are measured against.
    previous_view: Option<Viewpoint>,
    /// Whether to offset each frame's rays inside their pixel. Off until an upscaler asks.
    jitter: bool,
    /// How much of the lighting painted into a texture to divide back out. See
    /// [`SceneRenderer::set_delight`].
    delight: f32,
    /// How far a texture's painted relief tilts the normal. See [`SceneRenderer::set_relief`].
    relief: f32,
    /// What a residency built later should do with the structures over what moves — see
    /// [`SceneRenderer::set_refit_deforming`]. Held because the residency is built lazily.
    pending_refit: bool,
    /// How much of the cell's fog to apply. See [`SceneRenderer::set_fog`].
    fog: f32,
    /// The sky every resident exterior stands under. See [`SceneRenderer::set_sky`].
    sky: Sky,
    /// The upscaler, when one was handed over. Absent, the denoiser and the render-resolution tone
    /// curve carry the frame as they always have.
    #[cfg(feature = "dlss")]
    upscaler: Option<crate::dlss::Upscaler>,
    /// Frames asked for so far, which is what indexes the jitter sequence.
    frames: u32,
    /// This frame's jitter and whether it had a previous frame, kept because the upscaler is told
    /// them at *record* time while they are decided when the constants are built.
    #[cfg(feature = "dlss")]
    last_jitter: glam::Vec2,
    #[cfg(feature = "dlss")]
    had_history: bool,
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
///
/// `SAMPLED` because this is the colour an upscaler reads, and DLSS reads through a sampler — see
/// [`crate::gbuffer::GBuffer::new`] for what its absence costs.
fn target_image(memory: &Memory, extent: vk::Extent2D) -> rtxmw_gpu::Result<Image> {
    Image::new(
        memory,
        "primary visibility target",
        extent,
        TARGET_FORMAT,
        vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC,
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
            storm: 0.0,
            lightning_offset: 0.0,
            last_strike: None,
            previous_view: None,
            jitter: false,
            delight: DEFAULT_DELIGHT,
            relief: DEFAULT_RELIEF,
            pending_refit: DEFAULT_REFIT,
            fog: 1.0,
            sky: Sky::default(),
            #[cfg(feature = "dlss")]
            upscaler: None,
            frames: 0,
            #[cfg(feature = "dlss")]
            last_jitter: glam::Vec2::ZERO,
            #[cfg(feature = "dlss")]
            had_history: false,
            // One more than the stages, since a duration needs a timestamp either side of it.
            timestamps: Timestamps::new(device, physical, FrameTimings::STAGES as u32 + 1)?,
        };
        renderer.bind_targets();
        // Every pipeline this run needs now exists, so what the driver compiled can go back to disk
        // for the next one — see `Device::store_pipeline_cache`. Three seconds a process, and the
        // test suite starts fourteen of them.
        device.store_pipeline_cache();
        Ok(renderer)
    }

    /// Points the post passes at the images they read and write.
    ///
    /// Separate from [`SceneRenderer::bind_scene`] because the two go stale for different reasons:
    /// these follow the images, which only a resize replaces, and that one follows the cell.
    fn bind_targets(&mut self) {
        self.denoiser.bind(&self.gbuffer);
        self.composite.bind(&self.target, &self.gbuffer);
        // Read as fields rather than through a method: the two passes below are borrowed mutably
        // here, and asking `self` for the source would borrow the whole of it.
        #[cfg(feature = "dlss")]
        let source = self
            .upscaler
            .as_ref()
            .map_or(&self.target, crate::dlss::Upscaler::output);
        #[cfg(not(feature = "dlss"))]
        let source = &self.target;
        // **The same image the tone curve maps**, not the render-resolution one it read before. The
        // histogram bins `log2(luminance)` per pixel and the mean of a log sits below the log of the
        // mean, so a noisy frame measures darker than it is and the curve opens to compensate —
        // which under an upscaler, where the à-trous passes are off, is a single sample per pixel.
        // See `docs/design.md` §8.28 for what that cost.
        self.exposure.bind(source);
        self.tonemap.bind(source, self.exposure.buffer());
    }

    /// Hands this renderer an upscaler, which replaces the denoiser and moves the tone curve to the
    /// upscaled resolution — and taking one away restores both.
    ///
    /// A caller wanting some other pass count says so afterwards, through
    /// [`SceneRenderer::set_denoise_passes`].
    ///
    /// **Built by the caller**, because NGX comes up on a Vulkan instance and this does not own one.
    /// Passing one also turns jitter on: DLSS resolves detail across frames and cannot do that if
    /// every frame samples the same point inside its pixel.
    #[cfg(feature = "dlss")]
    pub fn set_upscaler(
        &mut self,
        memory: &Memory,
        upscaler: Option<crate::dlss::Upscaler>,
    ) -> rtxmw_gpu::Result<()> {
        let output = match &upscaler {
            Some(upscaler) => upscaler.output().extent(),
            None => self.target.extent(),
        };
        self.jitter = upscaler.is_some();
        // **Ray Reconstruction stands in for the à-trous filter rather than beside it**, so taking
        // one away has to put the filter back. Leaving that to the caller is what left a renderer
        // whose upscaler failed to rebuild running with neither.
        self.denoise_passes = if upscaler.is_some() {
            0
        } else {
            crate::denoiser::DEFAULT_PASSES
        };
        self.upscaler = upscaler;
        // The tone curve now runs at the upscaled size, so its own output follows.
        self.tonemap.resize(memory, output)?;
        self.bind_targets();
        Ok(())
    }

    /// Sets how much of the lighting painted into a texture is divided back out.
    ///
    /// Zero is the texture as shipped and one the whole estimate; anything between fades between
    /// them, which is what makes the A/B `docs/design.md` §5.1 asks for a slider rather than a
    /// rebuild.
    pub fn set_delight(&mut self, strength: f32) {
        assert!(
            (0.0..=1.0).contains(&strength),
            "de-lighting runs from none to the whole estimate, not {strength}"
        );
        self.delight = strength;
    }

    /// Chooses between rebuilding and refitting the structures over everything that moves.
    ///
    /// **Set before the scene that moves is loaded**: the choice is baked into the structures when
    /// they are created, because a refittable one is sized and organised differently. Refitting is
    /// the default because it measured better on both counts — `docs/design.md` M12 — and this is
    /// the A/B that says so, kept because the answer is content-dependent.
    pub fn set_refit_deforming(&mut self, refit: bool) {
        self.pending_refit = refit;
        if let Some(scene) = self.scene.as_mut() {
            scene.set_refit_deforming(refit);
        }
    }

    /// Sets how far a texture's painted relief tilts the normal it is shaded by.
    ///
    /// Zero is the vertex normal alone — the surface as the mesh describes it — and one the whole
    /// of what the texture says about its own shape. See `relief.glsl` for what it reads.
    pub fn set_relief(&mut self, strength: f32) {
        assert!(
            (0.0..=1.0).contains(&strength),
            "relief runs from none to the whole of it, not {strength}"
        );
        self.relief = strength;
    }

    /// Moves the sky: the sun in it, and the light it casts on everything out of doors.
    ///
    /// **Not part of a cell**, and that is the whole reason this is here: an hour later the same
    /// geometry is lit by a different sun, and nothing about the cell has changed. An interior keeps
    /// the lighting its own record carries and ignores this entirely.
    pub fn set_sky(&mut self, sky: Sky) {
        self.sky = sky;
        if let Some(residency) = self.scene.as_mut() {
            residency.set_sky(sky);
        }
    }

    /// Sets how much of the cell's fog is applied.
    ///
    /// Zero is the frame with none of it, whatever the cell records, which is the A/B — the density
    /// itself belongs to the cell rather than to a caller.
    pub fn set_fog(&mut self, strength: f32) {
        assert!(
            (0.0..=1.0).contains(&strength),
            "fog runs from none to the cell's own, not {strength}"
        );
        self.fog = strength;
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

    /// Sets the clock the weather's own events run on, in seconds the speed key has not touched.
    ///
    /// Separate from [`Self::set_time`] because the two answer different questions: that one is how
    /// fast the world turns, which a swell and a fog bank and a day all ride on, and this is how long
    /// a thing that happens takes. A flash is two hundred milliseconds however fast the day is going.
    pub fn set_storm(&mut self, seconds: f32) {
        self.storm = seconds;
    }

    /// Brings the next flash forward to now.
    ///
    /// **By moving the clock rather than by setting a flag**, which is what keeps the whole thing a
    /// function of time: `Lightning::flash` reads a schedule and nothing anywhere remembers that a
    /// key was pressed, so a screenshot taken a frame later draws exactly what the window did. The
    /// offset nudges the lightning clock onto a flash boundary and stays there, so the storm carries
    /// on from the moment it was asked for rather than snapping back.
    ///
    /// Brings forward a flash that can be *seen* from `eye` looking `facing` — see
    /// [`rtxmw_scene::Lightning::staged`], which is why this needs a camera at all.
    ///
    /// Does nothing under a weather with no lightning, which is nine of the ten.
    pub fn strike(&mut self, eye: Vec3, facing: Vec3) {
        let now = self.storm + self.lightning_offset;
        let altitude = self.sky.clouds.altitude;
        self.lightning_offset += self.sky.lightning.staged(now, eye, altitude, facing);
        self.last_strike = Some(self.storm + self.lightning_offset);
    }

    /// Sends the last flash again, down the same channel.
    ///
    /// **Which is what a restrike is**, and why it costs nothing to offer: a real flash is a train of
    /// return strokes re-lighting one path, and this is that a moment later. The clock goes back to
    /// where the last one began, so the shape, the shore it stands on and the number of times it
    /// stutters are all drawn again from the same second — see [`rtxmw_scene::Lightning::flash`],
    /// which reads them rather than storing them.
    ///
    /// Does nothing before [`Self::strike`] has been asked for one.
    pub fn restrike(&mut self) {
        if let Some(at) = self.last_strike {
            self.lightning_offset = at - self.storm;
        }
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
        let residency = self.residency(device, uploader, limits)?;
        residency.add(device, uploader, limits, id, scene, textures)?;
        assert!(
            residency.textures().len() <= MAX_TEXTURES,
            "cells need {} texture slots but the array is built for {MAX_TEXTURES}",
            residency.textures().len()
        );
        Ok(())
    }

    /// The residency, building one if this renderer has never held a cell.
    ///
    /// **Lazy because a renderer is useful before it has a scene** — it can be sized, given a sky
    /// and handed an upscaler — but everything resident shares one set of buffers, so the first
    /// caller to need them is the one that pays for them. Which caller that is stopped being
    /// `add_cell` alone when the moons' portraits arrived, and this exists so the two do not each
    /// carry their own copy of the same three lines.
    fn residency(
        &mut self,
        device: &Device,
        uploader: &mut Uploader,
        limits: RayTracingLimits,
    ) -> rtxmw_gpu::Result<&mut SceneResidency> {
        if self.scene.is_none() {
            let residency = self
                .scene
                .insert(SceneResidency::new(device, uploader, limits)?);
            // A fresh residency starts under the default sky and rebuilding what moves; this one
            // is standing under whatever the caller last set.
            residency.set_sky(self.sky);
            residency.set_refit_deforming(self.pending_refit);
        }
        Ok(self.scene.as_mut().expect("just built if it was absent"))
    }

    /// Hands the sky its own pictures — the moons' portraits and the weather's cloud sheet.
    ///
    /// Order-free: their slots exist from the moment a residency does, so this may be called before
    /// or after any cell. Without it the moons are flat discs of their measured colour and there is
    /// no cloud layer, which is what the headless tests draw.
    ///
    /// The caller must have waited for device idle.
    pub fn set_sky_textures(
        &mut self,
        device: &Device,
        uploader: &mut Uploader,
        limits: RayTracingLimits,
        textures: &SkyTextures,
    ) -> rtxmw_gpu::Result<()> {
        self.residency(device, uploader, limits)?
            .set_sky_textures(uploader, textures)?;
        self.bind_scene();
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

    /// How many placements a frame poses, which is what resident cells put in front of the camera.
    ///
    /// **Not what has ever been placed.** The vertex regions and the structures over them are
    /// grow-only like a mesh, but the posing is not: an evicted cell stops being posed, and this is
    /// the number that says so.
    pub fn posed_count(&self) -> usize {
        self.scene.as_ref().map_or(0, SceneResidency::posed_count)
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

    /// Whether each frame's rays are offset inside their pixel, and whether this frame has history.
    ///
    /// **Off by default, because on its own jitter is only shimmer.** It pays for itself when
    /// something accumulates across frames and can resolve detail no single frame holds; until then
    /// it moves every edge a fraction of a pixel a frame and resolves nothing. DLSS Ray
    /// Reconstruction is what turns it on.
    pub fn set_jitter(&mut self, enabled: bool) {
        self.jitter = enabled;
    }

    /// Whether the next frame has a previous one to reuse, or is starting from nothing.
    ///
    /// False before the first frame and after [`SceneRenderer::forget_history`] — the reset an
    /// upscaler needs when the camera has jumped somewhere its motion vectors cannot describe, such
    /// as a walk through a door.
    pub fn has_history(&self) -> bool {
        self.previous_view.is_some()
    }

    /// Declares the previous frame unusable, so the next one reprojects onto itself.
    pub fn forget_history(&mut self) {
        self.previous_view = None;
    }

    /// The specular albedo and roughness of each pixel's surface, in `rgb` and `a`.
    ///
    /// In `GENERAL` once [`SceneRenderer::record`] has run. The specular *distance* that goes with
    /// them rides in the alpha of [`SceneRenderer::target`]'s companion albedo image, which the
    /// composite does not read.
    pub fn material(&self) -> &Image {
        self.gbuffer.material()
    }

    /// World normal in `rgb`, roughness in `a` — the layout DLSS Ray Reconstruction reads.
    ///
    /// In `GENERAL` once [`SceneRenderer::record`] has run.
    pub fn normal_roughness(&self) -> &Image {
        self.gbuffer.normal_roughness()
    }

    /// Where each pixel's surface was on the previous frame's screen, as a displacement in pixels.
    ///
    /// In `GENERAL` once [`SceneRenderer::record`] has run. Written every frame and read by nothing
    /// yet — DLSS Ray Reconstruction is what will consume it, and temporal reuse of the sun's
    /// shadow samples after that.
    pub fn motion(&self) -> &Image {
        self.gbuffer.motion()
    }

    /// What Ray Reconstruction reconstructed, at the output size and still linear.
    ///
    /// `None` without an upscaler, where [`SceneRenderer::target`] is already the finished frame.
    /// In `GENERAL` once [`SceneRenderer::record`] has run — the tone curve reads it there, and
    /// nothing transitions it afterwards.
    ///
    /// **Scene-referred, because Ray Reconstruction does not do exposure**, so this and `target`
    /// are the same quantity at two resolutions and one is directly comparable to the other. That
    /// comparison is the whole of `tests/reconstruction.rs`.
    #[cfg(feature = "dlss")]
    pub fn upscaled(&self) -> Option<&Image> {
        self.upscaler.as_ref().map(crate::dlss::Upscaler::output)
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
    /// The block for this frame, and the camera it was built from is remembered for the next.
    ///
    /// **Takes `&mut self` for that memory, and expects one call per frame.** A motion vector is the
    /// difference between two consecutive frames' cameras, and nothing else in the renderer knows
    /// where the camera was — asking a caller to hand back its own previous matrices would put the
    /// one piece of state that must not drift outside the thing that reads it.
    pub fn frame_constants(
        &mut self,
        view: glam::Mat4,
        projection: glam::Mat4,
        camera_position: Vec3,
    ) -> FrameConstants {
        let lighting = self
            .scene
            .as_ref()
            .map_or(Lighting::default(), SceneResidency::lighting);
        let now = Viewpoint {
            view,
            projection,
            position: camera_position,
        };
        // The first frame has no previous one, and is its own: the surfaces then reproject onto
        // themselves and every motion vector is zero, which is what a temporal filter with no
        // history should be told.
        #[cfg(feature = "dlss")]
        let had_history = self.previous_view.is_some();
        let previous = self.previous_view.replace(now).unwrap_or(now);
        // Zero unless something is accumulating over frames, which nothing is until DLSS Ray
        // Reconstruction arrives — see `FrameConstants::jitter`.
        let jitter = if self.jitter {
            FrameConstants::jitter_at(self.frames)
        } else {
            glam::Vec2::ZERO
        };
        self.frames = self.frames.wrapping_add(1);
        #[cfg(feature = "dlss")]
        {
            self.last_jitter = jitter;
            self.had_history = had_history;
        }
        FrameConstants::new(
            now,
            previous,
            lighting,
            Sampling {
                jitter,
                bounce_samples: self.bounce_samples,
                sequence: self.frames,
                delight: self.delight,
                relief: self.relief,
                fog: self.fog,
            },
            // From the renderer's own target height, so the mip a surface samples follows the
            // resolution it is being traced at.
            FrameConstants::cone_spread_from(projection, self.target.extent().height),
            self.time,
            self.storm + self.lightning_offset,
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
        Ok(FrameTimings::from_durations(
            durations,
            self.target.extent(),
            self.output().extent(),
        ))
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
            #[cfg(feature = "dlss")]
            let upscaled = self.upscaler.as_ref().map(crate::dlss::Upscaler::output);
            #[cfg(not(feature = "dlss"))]
            let upscaled: Option<&Image> = None;
            for image in [&self.target]
                .into_iter()
                .chain(self.gbuffer.images())
                .chain([self.tonemap.output()])
                .chain(upscaled)
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

            // **Before anything reads the scene**, and skipped entirely where nothing moves: a
            // frame over static geometry must cost exactly what it did before this existed.
            if let Some(scene) = self.scene.as_mut()
                && scene.has_deforming()
            {
                scene.record_animation(device, command_buffer, constants.seconds());
            }
            self.timestamps.write(command_buffer, 1);

            self.pass.record(command_buffer, extent, constants);
            memory_barrier::full(device, command_buffer);
            self.timestamps.write(command_buffer, 2);

            // The filter leaves the lighting back in the G-buffer's first illumination image, which
            // is the one the composite is bound to.
            self.denoiser
                .record(device, command_buffer, extent, self.denoise_passes);
            self.timestamps.write(command_buffer, 3);

            // **Whether the composite still has to put the rain in**, which is exactly when there
            // is no upscaler: Ray Reconstruction composites `DLSS.TransparencyLayer` itself. Here
            // rather than after the tone curve because the exposure pass meters what this leaves.
            #[cfg(feature = "dlss")]
            let overlay = self.upscaler.is_none();
            #[cfg(not(feature = "dlss"))]
            let overlay = true;
            self.composite.record(command_buffer, extent, overlay);
            // The composed frame is what the upscaler reads, so it has to have been written.
            memory_barrier::full(device, command_buffer);
            self.timestamps.write(command_buffer, 4);

            // **Ray Reconstruction stands in for the filter, not beside it** — it denoises,
            // antialiases and upscales in one pass, so the à-trous passes above are set to zero by
            // whoever turned it on, and what it reads is the composed frame as traced.
            #[cfg(feature = "dlss")]
            if let Some(upscaler) = &self.upscaler
                && let Err(failed) = upscaler.record(
                    command_buffer,
                    &self.target,
                    &self.gbuffer,
                    self.last_jitter,
                    !self.had_history,
                )
            {
                // Reported rather than propagated: the frame is already half recorded, and a caller
                // that cannot draw one has nothing better to do with the news than this.
                eprintln!("DLSS did not run: {failed}");
            }
            memory_barrier::full(device, command_buffer);
            self.timestamps.write(command_buffer, 5);

            // The size of the frame the tone curve maps, which is the upscaled one where there is
            // an upscaler. Its *output's* size, because the curve is per pixel and the two are
            // therefore the same — and dispatching either pass over the render size instead would
            // leave three quarters of a 4K frame as the allocator left it.
            let displayed = self.tonemap.output().extent();

            // After the upscaler as well as the composite: what exposure measures has to be what
            // the tone curve maps.
            // The hour's own bias, which an interior does not have: a room's brightness is its
            // own business and its record says what it is, and a scale of zero is what says so.
            let outdoors = self
                .scene
                .as_ref()
                .is_some_and(|residency| residency.lighting().sky_scale > 0.0);
            let bias = if outdoors {
                self.sky.exposure_bias
            } else {
                1.0
            };
            self.exposure
                .record(device, command_buffer, displayed, bias);
            self.timestamps.write(command_buffer, 6);

            self.tonemap.record(command_buffer, displayed);
            memory_barrier::full(device, command_buffer);
            self.timestamps.write(command_buffer, 7);

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

    /// An extent, as these tests write one.
    fn extent(width: u32, height: u32) -> vk::Extent2D {
        vk::Extent2D { width, height }
    }

    #[test]
    fn a_device_that_wrote_no_timestamps_reports_nothing_rather_than_zeroes_it_invented() {
        // `Timestamps::read` returns an empty slice on a queue that cannot time anything, and the
        // caller has to be able to tell that from a frame that genuinely took no time.
        let nothing = extent(0, 0);
        assert_eq!(
            FrameTimings::from_durations([0.0; FrameTimings::STAGES], nothing, nothing),
            FrameTimings::default()
        );

        // In the order `record` writes them. The array's width is the type's guarantee that a
        // caller cannot hand over a stage count that disagrees with the pool's.
        let timings = FrameTimings::from_durations(
            [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            extent(1920, 1080),
            extent(3840, 2160),
        );
        assert_eq!(timings.animate, 0.5);
        assert_eq!(timings.trace, 1.0);
        assert_eq!(timings.denoise, 2.0);
        assert_eq!(timings.upscale, 8.0);
        assert_eq!(timings.tonemap, 32.0);
        assert_eq!(timings.total(), 63.5);
    }

    #[test]
    fn a_reported_duration_carries_the_size_it_was_measured_at() {
        // **The mistake this exists to stop.** A trace time copied out of this line and written down
        // against the resolution the *caller* asked for was wrong by 2.25x, because `--screenshot
        // 1920x1080` names the output and an upscaler on quality traces at 1280x720. Both sizes are
        // in the line, so there is nothing left to assume about which one a number belongs to.
        let upscaled = FrameTimings::from_durations(
            [0.0, 1.0, 0.0, 0.0, 8.0, 0.0, 0.0],
            extent(1280, 720),
            extent(1920, 1080),
        );
        let line = upscaled.to_string();
        assert!(line.contains("tracing 1280x720 to 1920x1080"), "{line}");
        assert!(line.contains("trace 1.00"), "{line}");

        // And where nothing upscales, one size rather than the same one twice.
        let native = FrameTimings::from_durations(
            [1.0; FrameTimings::STAGES],
            extent(1920, 1080),
            extent(1920, 1080),
        );
        assert!(native.to_string().contains("at 1920x1080"), "{native}");
        assert!(!native.to_string().contains("tracing"), "{native}");
    }
}
