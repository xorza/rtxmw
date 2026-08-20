//! The primary visibility pass: one ray query per pixel, into an offscreen HDR image.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};
use rtxmw_gpu::{Binding, Buffer, BufferMemory, ComputePipeline, Device, Image, Memory};
use rtxmw_scene::{Clouds, Moon, Sun, Veil};

use crate::acceleration_structure::AccelerationStructure;
use crate::gbuffer::GBuffer;
use crate::geometry_buffers::GeometryBuffers;
use crate::light_grid::{LightGrid, LightGridExtent};
use crate::material_buffers::MaterialBuffers;
use crate::shaders;
use crate::texture_array::TextureArray;
use crate::wave_spectrum::{GpuWave, SeaState, WAVE_COUNT};

/// Everything lighting a cell, which arrives together and goes stale together.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Lighting {
    /// Sky light outdoors, the cell's own fixed term indoors.
    pub(crate) ambient: Vec3,
    /// Where the light grid sits, which is how a shading point finds the lights that reach it.
    pub(crate) light_grid: LightGridExtent,
    /// The sun, for a cell with a sky.
    pub(crate) sun: Option<Sun>,
    /// Where the water surface sits, for shading what is under it. Absent for a dry cell.
    pub(crate) water_level: Option<f32>,
    /// The colour the cell's fog scatters, and how thickly it sits.
    ///
    /// Read by the composite rather than by the trace: fog attenuates the whole frame, and the
    /// trace has only its two halves.
    pub(crate) fog: Vec3,
    pub(crate) fog_density: f32,
    /// The warm end of every tint on the sky's dome — see [`rtxmw_scene::Sky::shape`].
    pub(crate) sky_warm: Vec3,
    /// How far the sunward side of it has gone toward that: nothing at noon, nearly all at dusk.
    pub(crate) sky_warmth: f32,
    /// What the dome's shape is multiplied by. **Zero is an interior**, which has no dome.
    pub(crate) sky_scale: f32,
    /// What every direction gets on top of the shape: the night floor out of doors, and an
    /// interior's own recorded colour indoors, which is the whole of its sky.
    pub(crate) sky_floor: Vec3,
    /// How much of the star field is out. Zero by day and indoors.
    pub(crate) sky_stars: f32,
    /// The weather's own medium, over everything the sky has. [`Veil::NONE`] under clear and
    /// indoors.
    pub(crate) sky_veil: Veil,
    /// Whether the fog forms banks. Weather does that to a landscape; a room's air is still.
    pub(crate) fog_banked: bool,
    /// Which way the air moves and how hard, out of the weather's `Wind Speed`. Zero indoors.
    pub(crate) fog_wind: Vec2,
    /// How deep the fog layer stands, against clear weather's in still air. One indoors.
    pub(crate) fog_lift: f32,
    /// The larger moon. [`Moon::NONE`] for a cell with no sky, which draws and lights nothing
    /// without a branch anywhere to say so.
    pub(crate) masser: Moon,
    /// The smaller one, on the same terms.
    pub(crate) secunda: Moon,
    /// Where Masser's vanilla portrait sits in the bindless array.
    ///
    /// **Zero is none**, which is the array's own magenta fallback and so the one slot that can
    /// never hold a real portrait — the shader draws a flat disc of the moon's own colour instead.
    /// That is what a renderer nobody handed the faces to gets, which is every test that does not
    /// need them.
    pub(crate) masser_face: u32,
    /// Where Secunda's sits, on the same terms.
    pub(crate) secunda_face: u32,
    /// The cloud layer over the cell. [`Clouds::NONE`] where there is no sky.
    pub(crate) clouds: Clouds,
    /// Where the weather's painted sky sheet sits in the bindless array, or zero for none — in
    /// which case no layer is drawn at all, since its shape is the whole of what the sheet is for.
    pub(crate) cloud_sheet: u32,
}

impl Default for Lighting {
    /// Nothing resident, so nothing lights anything.
    fn default() -> Self {
        Self {
            ambient: Vec3::ZERO,
            light_grid: LightGridExtent::default(),
            sun: None,
            water_level: None,
            fog: Vec3::ZERO,
            fog_density: 0.0,
            // Unobservable at zero density, and this is what a cell gets until one records its own.
            fog_banked: true,
            fog_wind: Vec2::ZERO,
            fog_lift: 1.0,
            // Unobservable at zero scale, the same as `fog_banked` is at zero density: with no
            // dome to shape there is nothing for a tint to tint.
            sky_warm: Vec3::ONE / 3.0,
            sky_warmth: 0.0,
            sky_scale: 0.0,
            sky_floor: Vec3::ZERO,
            sky_stars: 0.0,
            sky_veil: Veil::NONE,
            masser: Moon::NONE,
            secunda: Moon::NONE,
            masser_face: 0,
            secunda_face: 0,
            clouds: Clouds::NONE,
            cloud_sheet: 0,
        }
    }
}

/// One moon as the shader reads it — see `struct Moon` in `bindings.glsl`.
///
/// Sixteen floats packed tightly, which under `scalar` layout is exactly what the shader expects.
/// `face` is a slot in the bindless array or **zero for none**, which is the array's own fallback
/// and so the one index that can never name a real portrait.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuMoon {
    direction: [f32; 3],
    cos_radius: f32,
    colour: [f32; 3],
    face: u32,
    light: [f32; 3],
    face_mean: f32,
    pole: [f32; 3],
    lunar_lambert: f32,
}

impl GpuMoon {
    /// `moon` as the shader reads it, showing the portrait at `face`.
    fn new(moon: Moon, face: u32) -> Self {
        Self {
            direction: moon.direction.to_array(),
            cos_radius: moon.angular_radius.cos(),
            colour: moon.colour.to_array(),
            face,
            light: moon.light.to_array(),
            face_mean: moon.face_mean,
            pole: moon.pole.to_array(),
            lunar_lambert: moon.lunar_lambert,
        }
    }
}

/// What the shader needs to turn a pixel into a ray.
///
/// **A buffer, not push constants.** It was the latter until it reached exactly the 128 bytes
/// Vulkan guarantees and waves needed a clock as well; M7's motion vectors would have forced the
/// same move. Read with `scalar` layout, which packs a `vec3` tightly at four-byte alignment and so
/// matches this `repr(C)` struct field for field — under std430's sixteen-byte vector alignment the
/// two would disagree from the first `vec3` onward. Every offset is pinned by a test below.
///
/// The combined matrix rather than the two separately. That began as a way to fit the old 128-byte
/// block and is kept because it is simply less to send and less to get wrong.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FrameConstants {
    /// A pixel's clip coordinates to its offset from the eye, in world axes.
    ///
    /// **The eye is taken out of the view before the inverse, and that is the whole point.** The
    /// obvious matrix is the inverse view-projection, which unprojects a pixel to a *world point* on
    /// the near plane — and the shader then has to subtract the camera position from it to get a
    /// direction. Both are the size of the world and the difference is the near distance, 0.05
    /// units: at Seyda Neen's 75,000 an `f32` step is 0.024 units, so that subtraction throws away
    /// almost every bit of the answer. Measured against a double-precision reference, the ray
    /// through a pixel came out **127 pixels** from where it belonged, and 377 at the far corner of
    /// Vvardenfell — near zero at the origin, which is why it looked like a jitter that grew as the
    /// camera travelled rather than like a bug.
    ///
    /// Inverting [`Viewpoint::clip_from_offset`] instead lands the unprojection in a space centred
    /// on the camera, where the near plane is at 0.05 and an `f32` step is 6e-9. There is then
    /// nothing to cancel, and the error is **0.0001 pixels wherever the camera stands**. It also
    /// leaves a far better conditioned matrix to invert, since no entry is the size of the world any
    /// more.
    ///
    /// [`Self::camera_position`] is still sent — a ray needs an origin — but it is used only as one,
    /// never differenced against anything.
    ndc_to_world_offset: [f32; 16],
    /// The previous frame's `projection * rotation`, taking an offset from *that* frame's eye to
    /// its clip coordinates. See [`Viewpoint::clip_from_offset`].
    previous_clip_from_offset: [f32; 16],
    /// This frame's, for the clip depth an upscaler reprojects with.
    clip_from_offset: [f32; 16],
    camera_position: [f32; 3],
    /// Sub-pixel offset added to the pixel centre this frame, in pixels.
    ///
    /// **Zero unless an upscaler asked for it.** Jitter exists so that successive frames sample
    /// different points inside a pixel and a temporal filter can resolve detail no single frame
    /// holds; on its own it is just shimmer, so nothing turns it on until something is accumulating.
    /// It moves the ray and needs no unpicking afterwards: applied to the coordinate rather than to
    /// the matrix, it cancels out of a motion vector measured against that same coordinate.
    jitter: [f32; 2],
    /// How far the eye moved since the previous frame, `now - before`.
    ///
    /// A difference of two positions the size of the world, and exact all the same: subtracting two
    /// `f32`s within a factor of two of each other is exact, and a camera does not cross half the
    /// world in a frame. Sending the delta rather than the previous position is what keeps the
    /// shader from having to make that subtraction itself at world scale — the mistake `§8.7`
    /// records.
    camera_motion: [f32; 3],
    /// Reciprocal of the light grid's cell size, so a lookup multiplies rather than divides.
    light_grid_scale: f32,
    /// The corner the light grid is addressed from, and how many cells it spans.
    ///
    /// **Zero dimensions is a scene with no lights**, and needs no flag of its own: the shader's
    /// bounds test rejects every lookup against an empty grid, which is the same code path as a
    /// point standing outside a grid that does have lights.
    light_grid_origin: [f32; 3],
    light_grid_dimensions: [u32; 3],
    /// The cell's fixed lighting, standing in for every bounce the original engine could not
    /// compute. Interiors are mostly this.
    ambient: [f32; 3],
    /// How fast a pixel's ray cone widens with distance, in world units per unit travelled.
    ///
    /// A rasterizer gets its texture level from screen-space derivatives, which a compute shader
    /// does not have. This is what replaces them: the cone's width at the hit is its footprint on
    /// the surface, and the mip that matches that footprint is the one to sample.
    cone_spread: f32,
    /// Unit vector the sun's light travels along, from the sky toward the world.
    sun_direction: [f32; 3],
    /// Cosine of the sun's angular radius, which is what a shadow ray tests against.
    sun_cos_radius: f32,
    /// The sun's radiance. **Zero means no sun**, which is how an interior says it has no sky.
    sun_colour: [f32; 3],
    /// How many diffuse bounce rays each pixel casts to gather indirect light.
    ///
    /// Zero is not "no indirect light" but "every bounce ray escapes", which is the limit in which
    /// the estimator collapses back to a flat `albedo * ambient` fill — so it is exactly the
    /// lighting model this had before one bounce existed, and the A/B against it is honest.
    bounce_samples: u32,
    /// Where the water surface sits, or **negative infinity for a cell with no water**.
    ///
    /// A sentinel rather than a flag: every use is `water_level - z > 0`, and negative infinity
    /// makes that false everywhere without a branch of its own to get wrong.
    water_level: f32,
    /// Seconds since the engine started, which is what makes water move.
    ///
    /// Zero unless a caller sets one, so a screenshot and a test are reproducible: the surface at
    /// time zero is a definite shape rather than whatever the clock happened to say.
    time: f32,
    /// Which frame this is, counted from the renderer's first.
    ///
    /// **It moves the sampler's hash streams**, and without it a still camera redraws bit-identical
    /// noise every frame — the estimator's error becomes a fixed pattern rather than something
    /// averaging away. A spatial filter hides that; a temporal one reads it as detail and keeps it.
    sequence: u32,
    /// How much of the lighting painted into a texture to divide back out.
    ///
    /// **Zero is the texture as Bethesda shipped it**, and one the whole estimate. Vanilla assets
    /// have shading and ambient occlusion painted into their albedo for a renderer with no lighting
    /// of its own, so tracing over them lights everything twice — `docs/design.md` §5.1.
    delight: f32,
    /// The radiance the cell's fog scatters, how thickly it sits, and how much of it to apply.
    ///
    /// Read by the trace rather than by the composite, which is where the lights are — see
    /// `fog.glsl` for why fogging both halves of the split is the same as fogging their sum.
    fog: [f32; 3],
    fog_density: f32,
    fog_strength: f32,
    /// The sky dome's shape, which `lighting.glsl` draws per pixel from these four and
    /// `tests/sky_dome.rs` cross-checks against [`rtxmw_scene::Sky::shape`]. See [`Lighting`] for
    /// what each is.
    sky_warm: [f32; 3],
    sky_warmth: f32,
    sky_scale: f32,
    sky_floor: [f32; 3],
    sky_stars: f32,
    /// The weather's medium and how much of the sky it has: [`rtxmw_scene::Veil`], flattened.
    sky_veil: [f32; 3],
    sky_veiled: f32,
    /// How far the fog is an even haze rather than banks: one in a room, and the weather's own
    /// wind out of doors.
    fog_uniform: f32,
    /// The weather's wind: a unit bearing times `Wind Speed`, out of [`Lighting::fog_wind`].
    fog_wind: [f32; 2],
    /// How deep the layer stands — see [`rtxmw_scene::Sky::fog_lift`].
    fog_lift: f32,
    /// The two moons — see [`GpuMoon`].
    masser: GpuMoon,
    secunda: GpuMoon,
    /// The cloud layer: how high it sits, how far the world curves under it, how much of it one
    /// tile of the sheet spans,
    /// how far the wind has carried it, what a lit and a shadowed cloud radiate, the sheet's own
    /// mean luminance, how much sky it covers, and where the sheet sits in the array.
    ///
    /// Zero cover or a zero slot is no layer, which is what an interior and a renderer nobody
    /// handed a sheet to both get.
    cloud_altitude: f32,
    cloud_world_radius: f32,
    cloud_tile: f32,
    cloud_drift: [f32; 2],
    cloud_lit: [f32; 3],
    cloud_shadowed: [f32; 3],
    cloud_mean: f32,
    cloud_cover: f32,
    cloud_sheet: u32,
    /// The sinusoids the water surface is summed from — see [`crate::wave_spectrum`].
    ///
    /// Static for the life of a sea state, and carried here rather than in a buffer of its own
    /// because it is six hundred bytes beside a block that is already uploaded every frame, and a
    /// second binding would cost more to explain than the copy costs to make.
    waves: [GpuWave; WAVE_COUNT],
}

/// How one frame draws its samples.
///
/// The three travel together because they are one decision: where in the pixel this frame looks,
/// how many bounce rays it casts from there, and which hash streams those rays draw from.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Sampling {
    /// Sub-pixel offset from the pixel centre, in pixels. Zero unless an upscaler asked for it.
    pub(crate) jitter: Vec2,
    pub(crate) bounce_samples: u32,
    /// How much of the baked lighting to divide out of every texture.
    pub(crate) delight: f32,
    /// How much of the cell's fog to apply, from none to all of it.
    pub(crate) fog: f32,
    /// Which frame this is, counted from the renderer's first.
    pub(crate) sequence: u32,
}

impl FrameConstants {
    /// Builds the block from a camera's matrices.
    ///
    /// `projection` must be the reverse-Z Vulkan projection the shader assumes — it unprojects the
    /// near plane at depth 1, which is the only plane an infinite projection leaves invertible.
    pub(crate) fn new(
        now: Viewpoint,
        previous: Viewpoint,
        lighting: Lighting,
        sampling: Sampling,
        cone_spread: f32,
        time: f32,
    ) -> Self {
        // No sun is a black one: every term it feeds is a multiplication, so the shader needs no
        // flag to branch on and an interior costs nothing for having no sky.
        let sun = lighting.sun.unwrap_or(Sun {
            direction: Vec3::NEG_Z,
            colour: Vec3::ZERO,
            angular_radius: 0.0,
        });
        Self {
            ndc_to_world_offset: now.clip_from_offset().inverse().to_cols_array(),
            previous_clip_from_offset: previous.clip_from_offset().to_cols_array(),
            clip_from_offset: now.clip_from_offset().to_cols_array(),
            camera_position: now.position.to_array(),
            jitter: sampling.jitter.to_array(),
            camera_motion: (now.position - previous.position).to_array(),
            light_grid_scale: lighting.light_grid.scale,
            light_grid_origin: lighting.light_grid.origin.to_array(),
            light_grid_dimensions: lighting.light_grid.dimensions,
            ambient: lighting.ambient.to_array(),
            cone_spread,
            sun_direction: sun.direction.to_array(),
            sun_cos_radius: sun.angular_radius.cos(),
            sun_colour: sun.colour.to_array(),
            bounce_samples: sampling.bounce_samples,
            water_level: lighting.water_level.unwrap_or(f32::NEG_INFINITY),
            time,
            sequence: sampling.sequence,
            delight: sampling.delight,
            fog: lighting.fog.to_array(),
            fog_density: lighting.fog_density,
            fog_strength: sampling.fog,
            sky_warm: lighting.sky_warm.to_array(),
            sky_warmth: lighting.sky_warmth,
            sky_scale: lighting.sky_scale,
            sky_floor: lighting.sky_floor.to_array(),
            sky_stars: lighting.sky_stars,
            sky_veil: lighting.sky_veil.hue.to_array(),
            sky_veiled: lighting.sky_veil.amount,
            // **Indoors is even whatever the weather, and out of doors the wind says how far.**
            // The two are different arguments for the same number: a room is smaller than one bank
            // would be, so its air has no shape to take, while a landscape's banks are stirred out
            // by exactly the turbulence that lifts the layer. Settled here rather than in the
            // shader because it is one number for the whole frame and the march would recompute it
            // at every one of twenty-four steps a ray.
            fog_uniform: match lighting.fog_banked {
                true => lighting.fog_wind.length().min(1.0),
                false => 1.0,
            },
            fog_wind: lighting.fog_wind.to_array(),
            fog_lift: lighting.fog_lift,
            masser: GpuMoon::new(lighting.masser, lighting.masser_face),
            secunda: GpuMoon::new(lighting.secunda, lighting.secunda_face),
            cloud_altitude: lighting.clouds.altitude,
            cloud_world_radius: lighting.clouds.world_radius,
            cloud_tile: lighting.clouds.tile,
            cloud_drift: lighting.clouds.drift.to_array(),
            cloud_lit: lighting.clouds.lit.to_array(),
            cloud_shadowed: lighting.clouds.shadowed.to_array(),
            cloud_mean: lighting.clouds.sheet_mean,
            cloud_cover: lighting.clouds.cover,
            cloud_sheet: lighting.cloud_sheet,
            // Rebuilt each frame rather than cached: it is a few hundred floats of arithmetic once
            // per frame against a million rays that read it, and a cache would need invalidating
            // the moment the sea state becomes something a cell can set.
            waves: SeaState::default().waves(),
        }
    }

    /// The `index`th term of the Halton sequence in `base`, on `-0.5..0.5`.
    ///
    /// **A low-discrepancy sequence rather than a random one**, which is the whole point: over any
    /// prefix of it the samples are spread evenly inside the pixel, where random offsets clump and
    /// leave gaps that a temporal filter resolves as neither. Bases 2 and 3 are the usual pair, and
    /// are coprime so the two axes do not walk in step.
    fn halton(mut index: u32, base: u32) -> f32 {
        let mut result = 0.0;
        let mut fraction = 1.0 / base as f32;
        while index > 0 {
            result += (index % base) as f32 * fraction;
            index /= base;
            fraction /= base as f32;
        }
        result - 0.5
    }

    /// Where inside its pixel frame `index` samples, in pixels.
    ///
    /// One-based, because Halton's zeroth term is zero on every axis and a first frame that sampled
    /// the exact pixel centre would repeat whatever an un-jittered frame already had.
    pub(crate) fn jitter_at(index: u32) -> Vec2 {
        Vec2::new(Self::halton(index + 1, 2), Self::halton(index + 1, 3))
    }

    /// How wide one pixel's ray cone grows per unit of distance.
    ///
    /// Two pixels' worth of vertical field of view divided by the height, which is the angle one
    /// pixel subtends. Derived from the projection rather than passed separately so it cannot
    /// disagree with the matrix the same frame is unprojected with.
    pub(crate) fn cone_spread_from(projection: Mat4, height: u32) -> f32 {
        // The projection's [1][1] is `1 / tan(fov_y / 2)`, negated by the Vulkan Y flip.
        let cot_half_fov = projection.y_axis.y.abs();
        2.0 / (cot_half_fov * height as f32)
    }
}

/// A camera as one thing: its matrices and the eye they were built from.
///
/// **Bundled because the three have to agree.** The position must be the point the view looks from,
/// and passed separately nothing says so — a frame whose rays start somewhere its matrices do not
/// would render a plausible picture of the wrong place. It is also what a motion vector is measured
/// against, so the previous frame's camera is this same type rather than a second one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Viewpoint {
    pub(crate) view: Mat4,
    pub(crate) projection: Mat4,
    pub(crate) position: Vec3,
}

impl Viewpoint {
    /// The view with its translation dropped, which takes an offset from the eye to view space.
    ///
    /// A look-at view is `rotation * translate(-eye)`, so its fourth column *is* the eye's
    /// contribution and clearing it leaves the rotation alone.
    fn rotation(&self) -> Mat4 {
        let mut rotation = self.view;
        rotation.w_axis = glam::Vec4::W;
        rotation
    }

    /// Clip coordinates from an offset to the eye — the forward half of
    /// [`FrameConstants::ndc_to_world_offset`].
    ///
    /// A world point would have to be built and then differenced against the eye, and at
    /// Vvardenfell's scale that costs the same precision §8.7 is about; the offset never becomes a
    /// world position at all.
    fn clip_from_offset(&self) -> Mat4 {
        self.projection * self.rotation()
    }
}

/// Everything a trace reads that changes when the cell does.
///
/// A bundle rather than seven more parameters: they arrive and go stale together — one cell's worth
/// of device data — and as a positional list of references, four of which are buffers, transposing
/// two would compile and then read the wrong table.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SceneBindings<'a> {
    pub(crate) structure: &'a AccelerationStructure,
    pub(crate) geometry: &'a GeometryBuffers,
    pub(crate) tables: &'a MaterialBuffers,
    pub(crate) lights: &'a Buffer,
    pub(crate) light_grid: &'a LightGrid,
    pub(crate) textures: &'a TextureArray,
}

/// Descriptors for tracing primary rays, over a compute pipeline.
///
/// A compute shader with inline ray queries rather than a ray tracing pipeline: there is one
/// material path and no recursion, so a shader binding table would add a build step and a dispatch
/// indirection for nothing. Revisit when a hit needs to launch further rays it cannot inline.
#[derive(Debug)]
pub(crate) struct VisibilityPass {
    pipeline: ComputePipeline,
    /// The frame's constants, mapped and rewritten each frame.
    constants: Buffer,
}

/// Matches `local_size_x`/`local_size_y` in `primary_visibility.comp`.
///
/// Declared in the shader and needed here to size the dispatch, so the two have to agree; a
/// mismatch under-covers the image and leaves a band of it never written.
const WORKGROUP: u32 = 8;

impl VisibilityPass {
    /// Creates the descriptor set and pipeline.
    ///
    /// `max_textures` sizes the bindless array's binding. It is a ceiling rather than a count: the
    /// set is allocated with room for it and each scene writes as many slots as it actually has.
    pub(crate) fn new(
        device: &Device,
        memory: &Memory,
        max_textures: u32,
    ) -> rtxmw_gpu::Result<Self> {
        // Host-visible and mapped for the life of the pass: the constants change every frame and
        // are a hundred-odd bytes, so staging them through a transfer would cost a submission to
        // move less than a cache line.
        let constants = Buffer::new(
            memory,
            "frame constants",
            size_of::<FrameConstants>() as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            BufferMemory::Upload,
        )?;
        Ok(Self {
            constants,
            pipeline: ComputePipeline::new(
                device,
                &[
                    Binding::acceleration_structure(0),
                    Binding::storage_image(1),
                    Binding::storage_buffer(2),
                    Binding::storage_buffer(3),
                    Binding::storage_buffer(4),
                    Binding::storage_buffer(5),
                    Binding::storage_buffer(6),
                    Binding::storage_buffer(7),
                    // The G-buffer the trace splits itself into.
                    Binding::storage_image(8),
                    Binding::storage_image(9),
                    Binding::storage_image(10),
                    Binding::storage_image(11),
                    Binding::storage_image(12),
                    Binding::storage_image(13),
                    // The frame's own constants. They were push constants until waves needed a
                    // clock and the block was already exactly the 128 bytes Vulkan guarantees —
                    // see `FrameConstants`. Motion vectors at M7 would have forced the same move.
                    Binding::storage_buffer(14),
                    // The light grid: prefix offsets, then the light indices they address.
                    Binding::storage_buffer(15),
                    Binding::storage_buffer(16),
                    // Last, because Vulkan allows a variable descriptor count only on a set's final
                    // binding. Adding anything after this one moves it — validation rejects the set
                    // outright, which is how this was caught.
                    Binding::variable_samplers(17, max_textures),
                ],
                0,
                shaders::primary_visibility(),
            )?,
        })
    }

    /// Points the pass at the scene it traces and the image it writes.
    ///
    /// Takes `&mut self` because it rewrites the one descriptor set the pass owns, which must not
    /// happen while a dispatch using it is in flight. One set is deliberate: the scene changes per
    /// cell, not per frame.
    pub(crate) fn bind(&mut self, scene: SceneBindings<'_>, target: &Image, gbuffer: &GBuffer) {
        let structures = [scene.structure.raw()];
        let mut acceleration = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&structures);
        // Set by hand because the count lives on the write while the handles live on the chained
        // struct, and no builder method bridges the two. It must be non-zero: the write is rejected
        // outright otherwise, rather than binding nothing.
        let mut scene_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut acceleration);
        scene_write.descriptor_count = 1;

        let frame_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.constants.raw())
            .range(vk::WHOLE_SIZE)];
        let frame_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(14)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&frame_info);

        let geometry_info = [vk::DescriptorBufferInfo::default()
            .buffer(scene.tables.geometries().raw())
            .range(vk::WHOLE_SIZE)];
        let geometry_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&geometry_info);

        let material_info = [vk::DescriptorBufferInfo::default()
            .buffer(scene.tables.materials().raw())
            .range(vk::WHOLE_SIZE)];
        let material_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&material_info);

        let texture_infos = scene.textures.descriptors();
        let texture_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(17)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&texture_infos);

        // The storage images go through the pipeline's own helper: the emissive-and-sky target at
        // binding one, and the G-buffer as a run from eight.
        self.pipeline.bind_storage_images(1, &[target]);
        self.pipeline.bind_storage_images(
            8,
            &[
                gbuffer.albedo(),
                gbuffer.normal_roughness(),
                gbuffer.illumination(),
                gbuffer.motion(),
                gbuffer.material(),
                gbuffer.depth(),
            ],
        );

        let position_info = [vk::DescriptorBufferInfo::default()
            .buffer(scene.geometry.positions().raw())
            .range(vk::WHOLE_SIZE)];
        let position_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(7)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&position_info);

        let light_info = [vk::DescriptorBufferInfo::default()
            .buffer(scene.lights.raw())
            .range(vk::WHOLE_SIZE)];
        let light_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&light_info);

        let grid_offset_info = [vk::DescriptorBufferInfo::default()
            .buffer(scene.light_grid.offsets().raw())
            .range(vk::WHOLE_SIZE)];
        let grid_offset_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(15)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&grid_offset_info);

        let grid_index_info = [vk::DescriptorBufferInfo::default()
            .buffer(scene.light_grid.indices().raw())
            .range(vk::WHOLE_SIZE)];
        let grid_index_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(16)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&grid_index_info);

        let index_info = [vk::DescriptorBufferInfo::default()
            .buffer(scene.geometry.indices().raw())
            .range(vk::WHOLE_SIZE)];
        let index_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&index_info);

        let attribute_info = [vk::DescriptorBufferInfo::default()
            .buffer(scene.geometry.attributes().raw())
            .range(vk::WHOLE_SIZE)];
        let attribute_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(5)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&attribute_info);

        // SAFETY: every write names this pass's own set, and no dispatch using it is in flight.
        unsafe {
            self.pipeline.device().update_descriptor_sets(
                &[
                    scene_write,
                    frame_write,
                    geometry_write,
                    material_write,
                    texture_write,
                    index_write,
                    attribute_write,
                    light_write,
                    grid_offset_write,
                    grid_index_write,
                    position_write,
                ],
                &[],
            )
        };
    }

    /// Records one dispatch covering `extent`.
    ///
    /// The target must already be in `GENERAL`, and the caller is responsible for the barrier
    /// afterwards — it knows whether the image is next blitted, read back or sampled.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state, and [`VisibilityPass::bind`] must have run
    /// for the scene and target in use.
    pub(crate) unsafe fn record(
        &mut self,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        constants: &FrameConstants,
    ) {
        // Written straight into mapped memory rather than staged. The renderer keeps one frame in
        // flight and has already waited for it, so nothing is reading these while they change.
        self.constants
            .mapped_mut()
            .expect("frame constants are host-visible by construction")
            [..size_of::<FrameConstants>()]
            .copy_from_slice(bytemuck::bytes_of(constants));

        // SAFETY: the caller guarantees the command buffer is recording and the set is bound.
        unsafe {
            self.pipeline.dispatch(
                command_buffer,
                [
                    extent.width.div_ceil(WORKGROUP),
                    extent.height.div_ceil(WORKGROUP),
                    1,
                ],
                &[],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    #[test]
    fn the_frame_block_matches_the_scalar_layout_the_shader_reads() {
        // Scalar layout puts every field at its natural alignment — a `vec3` is twelve bytes
        // aligned to four — so the block the shader reads is this struct packed tightly, with no
        // padding anywhere. A field inserted in the middle, or one Rust chose to align, shifts
        // everything after it and the shader reads the whole frame displaced.
        assert_eq!(offset_of!(FrameConstants, ndc_to_world_offset), 0);
        assert_eq!(offset_of!(FrameConstants, previous_clip_from_offset), 64);
        assert_eq!(offset_of!(FrameConstants, clip_from_offset), 128);
        assert_eq!(offset_of!(FrameConstants, camera_position), 192);
        assert_eq!(offset_of!(FrameConstants, jitter), 204);
        assert_eq!(offset_of!(FrameConstants, camera_motion), 212);
        assert_eq!(offset_of!(FrameConstants, light_grid_scale), 224);
        assert_eq!(offset_of!(FrameConstants, light_grid_origin), 228);
        assert_eq!(offset_of!(FrameConstants, light_grid_dimensions), 240);
        assert_eq!(offset_of!(FrameConstants, ambient), 252);
        assert_eq!(offset_of!(FrameConstants, cone_spread), 264);
        assert_eq!(offset_of!(FrameConstants, sun_direction), 268);
        assert_eq!(offset_of!(FrameConstants, sun_cos_radius), 280);
        assert_eq!(offset_of!(FrameConstants, sun_colour), 284);
        assert_eq!(offset_of!(FrameConstants, bounce_samples), 296);
        assert_eq!(offset_of!(FrameConstants, water_level), 300);
        assert_eq!(offset_of!(FrameConstants, time), 304);
        assert_eq!(offset_of!(FrameConstants, sequence), 308);
        assert_eq!(offset_of!(FrameConstants, delight), 312);
        assert_eq!(offset_of!(FrameConstants, fog), 316);
        assert_eq!(offset_of!(FrameConstants, fog_density), 328);
        assert_eq!(offset_of!(FrameConstants, fog_strength), 332);
        assert_eq!(offset_of!(FrameConstants, sky_warm), 336);
        assert_eq!(offset_of!(FrameConstants, sky_warmth), 348);
        assert_eq!(offset_of!(FrameConstants, sky_scale), 352);
        assert_eq!(offset_of!(FrameConstants, sky_floor), 356);
        assert_eq!(offset_of!(FrameConstants, sky_stars), 368);
        assert_eq!(offset_of!(FrameConstants, sky_veil), 372);
        assert_eq!(offset_of!(FrameConstants, sky_veiled), 384);
        assert_eq!(offset_of!(FrameConstants, fog_uniform), 388);
        assert_eq!(offset_of!(FrameConstants, fog_wind), 392);
        assert_eq!(offset_of!(FrameConstants, fog_lift), 400);
        // Two moons of sixteen tightly packed floats apiece.
        assert_eq!(offset_of!(FrameConstants, masser), 404);
        assert_eq!(offset_of!(FrameConstants, secunda), 468);
        // Then the cloud layer, fourteen floats of it.
        assert_eq!(offset_of!(FrameConstants, cloud_altitude), 532);
        assert_eq!(offset_of!(FrameConstants, cloud_sheet), 584);
        // The wave table follows, twenty tightly packed bytes apiece.
        assert_eq!(offset_of!(FrameConstants, waves), 588);
        assert_eq!(size_of::<FrameConstants>(), 588 + 20 * WAVE_COUNT);
    }

    #[test]
    fn a_pixels_cone_widens_with_the_angle_it_subtends() {
        // One pixel of a 90-degree, 100-pixel-tall view subtends 2*tan(45)/100 = 0.02 units per
        // unit of distance.
        let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
            std::f32::consts::FRAC_PI_2,
            1.0,
            0.05,
        );
        let spread = FrameConstants::cone_spread_from(projection, 100);
        assert!((spread - 0.02).abs() < 1e-4, "got {spread}");
        // Twice the pixels, half the angle each.
        let finer = FrameConstants::cone_spread_from(projection, 200);
        assert!((finer - spread / 2.0).abs() < 1e-5);
    }

    /// A camera, as these tests describe one.
    #[derive(Debug, Clone, Copy)]
    struct Shot {
        eye: Vec3,
        forward: Vec3,
    }

    impl Shot {
        fn new(eye: Vec3, forward: Vec3) -> Self {
            Self {
                eye,
                forward: forward.normalize(),
            }
        }

        fn view(self) -> Mat4 {
            glam::camera::rh::view::look_to_mat4(self.eye, self.forward, Vec3::Z)
        }

        fn viewpoint(self, projection: Mat4) -> Viewpoint {
            Viewpoint {
                view: self.view(),
                projection,
                position: self.eye,
            }
        }
    }

    /// A frame built the way the renderer builds one, for a camera that moved `before` to `now`.
    fn frame_between(before: Shot, now: Shot, projection: Mat4) -> FrameConstants {
        FrameConstants::new(
            now.viewpoint(projection),
            before.viewpoint(projection),
            Lighting::default(),
            Sampling::default(),
            0.0,
            0.0,
        )
    }

    /// A frame for a camera that has not moved, which is what the first frame gets.
    fn frame_at(eye: Vec3, forward: Vec3, projection: Mat4) -> FrameConstants {
        let shot = Shot::new(eye, forward);
        frame_between(shot, shot, projection)
    }

    /// The direction the shader would trace for the pixel at `ndc`.
    ///
    /// The three lines of `primary_visibility.comp` that read the matrix, so what these tests
    /// assert is what the frame actually produces rather than an arrangement of it they agree with
    /// among themselves.
    fn aim(frame: &FrameConstants, ndc: glam::Vec2) -> Vec3 {
        let near = Mat4::from_cols_array(&frame.ndc_to_world_offset)
            * glam::Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
        (near.truncate() / near.w).normalize()
    }

    #[test]
    fn a_pixel_aims_the_same_way_wherever_in_the_world_the_camera_stands() {
        // **The bug this exists for**: the matrix used to be the inverse view-projection, which
        // unprojects a pixel to a world-space point on the near plane, and the shader subtracted the
        // camera position from it. Both operands are the size of the world; their difference is the
        // near distance. At Seyda Neen that lost all but a couple of bits of the direction, and the
        // aim drifted as the camera travelled — invisible at the origin and 127 pixels out at
        // 75,000, which is what made it read as jitter rather than as a wrong projection.
        //
        // Every direction is checked against the same unprojection carried out in double precision,
        // which is the only reference that does not share the fault being measured.
        const FOV: f32 = 60.0;
        let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
            FOV.to_radians(),
            16.0 / 9.0,
            0.05,
        );
        let forward = Vec3::new(0.3, 0.9, -0.2).normalize();
        // What one pixel of a 1080-tall view of that field subtends.
        let pixel = f64::from(FOV.to_radians()) / 1080.0;

        // **Built once, outside the loop, because it does not depend on where the camera is** — that
        // is the property under test. Its own view, its own product and its own inverse, all in
        // double precision, so it shares nothing with the matrix it judges but the inputs.
        let wide = |m: Mat4| glam::DMat4::from_cols_array(&m.to_cols_array().map(f64::from));
        let reference = (wide(projection)
            * wide(glam::camera::rh::view::look_to_mat4(
                Vec3::ZERO,
                forward,
                Vec3::Z,
            )))
        .inverse();

        let mut worst_by_eye = Vec::new();
        for eye in [
            Vec3::ZERO,
            Vec3::new(1_000.0, 1_000.0, 100.0),
            // Seyda Neen, and then the far corner of Vvardenfell.
            Vec3::new(-16_384.0, -73_728.0, 700.0),
            Vec3::new(200_000.0, 150_000.0, 900.0),
        ] {
            // Through the whole builder rather than the one function under test: the matrix has to
            // be the one a frame carries, or `new` could go back to the old formula unnoticed.
            let frame = frame_at(eye, forward, projection);

            let mut worst = 0.0f64;
            for i in 0..=8 {
                for j in 0..=8 {
                    let ndc = glam::Vec2::new(-1.0 + i as f32 * 0.25, -1.0 + j as f32 * 0.25);
                    let aimed = aim(&frame, ndc);
                    let aimed = glam::DVec3::new(aimed.x.into(), aimed.y.into(), aimed.z.into());

                    let want = reference * glam::DVec4::new(ndc.x.into(), ndc.y.into(), 1.0, 1.0);
                    let want = (want.truncate() / want.w).normalize();
                    worst = worst.max((aimed - want).length());
                }
            }
            worst_by_eye.push((eye, worst / pixel));
        }

        for (eye, pixels) in &worst_by_eye {
            println!(
                "eye {:?}: worst aim error {pixels:.5} pixels",
                eye.to_array()
            );
        }
        // A hundredth of a pixel, at every one of them. The old matrix passes this at the origin
        // and fails it by four orders of magnitude at Seyda Neen, which is exactly the shape of the
        // fault: the check has to be made away from the origin to mean anything.
        for (eye, pixels) in worst_by_eye {
            assert!(
                pixels < 0.01,
                "a pixel's ray at {:?} aims {pixels} pixels away from where double precision puts \
                 it; the unprojection is losing the camera's position",
                eye.to_array()
            );
        }
    }

    /// Where the surface at `ndc`, `offset` from this frame's eye, was on the previous screen.
    ///
    /// The lines `primary_visibility.comp` writes into the motion target, over a 1000x1000 frame so
    /// a pixel is a thousandth of the screen on each axis.
    fn motion(frame: &FrameConstants, offset: Vec3, ndc: glam::Vec2) -> glam::Vec2 {
        let was = offset + Vec3::from_array(frame.camera_motion);
        let before = Mat4::from_cols_array(&frame.previous_clip_from_offset)
            * glam::Vec4::new(was.x, was.y, was.z, 1.0);
        assert!(before.w > 0.0, "the point was behind the previous eye");
        ((before.truncate().truncate() / before.w) - ndc) * 0.5 * 1000.0
    }

    /// The pixel a surface `offset` from `shot`'s eye lands on, in the same units [`motion`] uses.
    ///
    /// **The naive formulation, in double precision**: build the world point, run it through the
    /// full view-projection, divide. That is the calculation the shader must not make in `f32` — at
    /// Vvardenfell's corner it is the one §8.7 is about — and doing it exactly here is what makes
    /// it an independent answer rather than the same arithmetic agreeing with itself.
    fn projects_to(shot: Shot, projection: Mat4, offset: Vec3) -> glam::DVec2 {
        let wide = |m: Mat4| glam::DMat4::from_cols_array(&m.to_cols_array().map(f64::from));
        let eye = shot.eye.as_dvec3();
        let mut rotation = shot.view();
        rotation.w_axis = glam::Vec4::W;
        let view = wide(rotation) * glam::DMat4::from_translation(-eye);
        let clip = wide(projection) * view * (eye + offset.as_dvec3()).extend(1.0);
        clip.truncate().truncate() / clip.w * 0.5 * 1000.0
    }

    /// Where the surface the pixel at `ndc` sees sits, relative to the eye, at distance `t`.
    ///
    /// Derived through the frame's own aim, exactly as the shader derives it — the offset it
    /// reprojects is `direction * t`, and a test that built the same point another way would
    /// measure the difference between the two constructions rather than the reprojection.
    fn seen_at(frame: &FrameConstants, ndc: glam::Vec2, t: f32) -> Vec3 {
        aim(frame, ndc) * t
    }

    #[test]
    fn a_still_camera_moves_nothing_and_a_turning_one_moves_everything_alike() {
        // Two properties that pin the reprojection between them without needing a reference to
        // compare against: a camera that did not move must leave every pixel where it is, and a
        // camera that only *turned* must move every surface by the same amount whatever its
        // distance, because rotation has no parallax.
        //
        // Not *exactly* zero, and it cannot be: the shader reprojects `direction * t`, and the
        // direction came from unprojecting the same pixel — so a still frame is an unproject
        // followed by a project, and `f32` rounds in between. What is left is a hundred-thousandth
        // of a pixel, or sixty thousand still frames before a temporal filter's history has crept
        // one pixel sideways.
        let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
            60f32.to_radians(),
            1.0,
            0.05,
        );
        let eye = Vec3::new(-16_384.0, -73_728.0, 700.0);
        let ahead = Vec3::new(1.0, 0.0, 0.0);
        let still = frame_at(eye, ahead, projection);

        // A pixel off centre, and depths from arm's length to most of a cell.
        let ndc = glam::Vec2::new(0.3, -0.45);
        let depths = [40.0f32, 300.0, 5_000.0];
        for t in depths {
            let moved = motion(&still, seen_at(&still, ndc, t), ndc);
            assert!(
                moved.length() < 1.0e-3,
                "a camera that did not move reported {moved:?} for a surface {t} away"
            );
        }

        // Now the same camera, turned in place. Every surface must move together.
        let turned = Shot::new(eye, Vec3::new(1.0, 0.03, 0.0));
        let rotating = frame_between(Shot::new(eye, ahead), turned, projection);
        let mut moves = Vec::new();
        for t in depths {
            moves.push(motion(&rotating, seen_at(&rotating, ndc, t), ndc));
        }
        println!("turning in place moved: {moves:?}");
        for pair in moves.windows(2) {
            assert!(
                (pair[0] - pair[1]).length() < 0.01,
                "a pure rotation moved a near surface by {:?} and a far one by {:?}; motion is \
                 picking up a parallax that is not there",
                pair[0],
                pair[1]
            );
        }
        // And it moved them at all, or the assertion above holds for a reprojection that does
        // nothing. Turning 0.03 radians of a 60-degree view is about 28 pixels of a thousand.
        assert!(
            (moves[0].length() - 28.0).abs() < 2.0,
            "a 0.03 radian turn moved the frame by {:?}",
            moves[0]
        );
    }

    #[test]
    fn a_walking_camera_moves_near_surfaces_further_than_far_ones() {
        // Parallax, which is the half a rotation cannot show and the half that carries depth into a
        // temporal filter. Checked against the projection itself: where a surface *was* on the
        // previous screen is something the previous camera can simply be asked, and the motion
        // vector has to be the difference between that and where it is now.
        let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
            60f32.to_radians(),
            1.0,
            0.05,
        );
        // At Vvardenfell's far corner, walking a stride north — the precision case as well as the
        // parallax one, since both eyes are world-scale and their difference is three units.
        let before = Shot::new(Vec3::new(200_000.0, 150_000.0, 900.0), Vec3::X);
        let now = Shot::new(before.eye + Vec3::new(0.0, 3.0, 0.0), Vec3::X);
        let frame = frame_between(before, now, projection);

        let ndc = glam::Vec2::new(0.2, 0.1);
        let mut moved = Vec::new();
        for depth in [40.0f32, 300.0, 5_000.0] {
            let offset = seen_at(&frame, ndc, depth);
            // The same surface seen from the previous eye is that much further away from it.
            let here = projects_to(now, projection, offset);
            let there = projects_to(before, projection, offset + (now.eye - before.eye));
            let want = there - here;

            let got = motion(&frame, offset, ndc);
            let got = glam::DVec2::new(got.x.into(), got.y.into());
            println!("depth {depth}: motion {got:?}, projection says {want:?}");
            assert!(
                (got - want).length() < 0.02,
                "at depth {depth} the motion vector is {got:?} where the previous projection puts \
                 the surface at {want:?}"
            );
            moved.push(got.length());
        }

        // Nearer is further: three units of parallax is a big move at forty units of depth and
        // almost none at five thousand.
        assert!(
            moved[0] > moved[1] * 5.0 && moved[1] > moved[2] * 5.0,
            "parallax did not fall off with depth: {moved:?}"
        );
    }

    #[test]
    fn the_jitter_covers_the_pixel_evenly_and_never_repeats_a_place() {
        // **The property a random offset would not have.** Jitter earns its cost by letting
        // successive frames resolve detail no single frame holds, and that needs the samples spread
        // across the pixel rather than merely unpredictable: random offsets clump, leaving parts of
        // the pixel unvisited for runs of frames and others visited twice over.
        let samples: Vec<Vec2> = (0..64).map(FrameConstants::jitter_at).collect();

        // Inside the pixel, and centred on it: a sequence biased to one side would drag every edge
        // in the frame that way.
        for (index, sample) in samples.iter().enumerate() {
            assert!(
                sample.x.abs() <= 0.5 && sample.y.abs() <= 0.5,
                "sample {index} is at {sample:?}, outside its own pixel"
            );
        }
        let mean = samples.iter().sum::<Vec2>() / samples.len() as f32;
        assert!(mean.length() < 0.02, "the sequence leans toward {mean:?}");

        // **Evenly spread, checked as coverage rather than as a mean** — which a sequence that
        // alternated between two opposite corners would also pass. Sixteen frames must touch every
        // quadrant of the pixel, and sixty-four must touch all sixteen sixteenths.
        let cell = |sample: &Vec2, side: f32| {
            let at = |v: f32| ((v + 0.5) * side).floor().min(side - 1.0) as u32;
            (at(sample.y) * side as u32) + at(sample.x)
        };
        for (frames, side) in [(16usize, 2.0f32), (64, 4.0)] {
            let mut seen: Vec<u32> = samples[..frames].iter().map(|s| cell(s, side)).collect();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(
                seen.len(),
                (side * side) as usize,
                "{frames} frames reached {} of the pixel's {} parts",
                seen.len(),
                side * side
            );
        }

        // And the first frame is not the pixel centre, which an un-jittered frame already sampled.
        assert!(
            samples[0].length() > 0.1,
            "the first sample is {:?}",
            samples[0]
        );
    }

    #[test]
    fn the_matrix_reaches_the_shader_the_way_it_was_built() {
        // Transposition and column order survive any amount of multiplication and show up only
        // against a known answer. Straight ahead from the origin is that answer: the centre pixel
        // of a view looking along +X must aim along +X, and the corners must splay off it by half
        // the field of view.
        let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
            90f32.to_radians(),
            1.0,
            0.05,
        );
        let frame = frame_at(Vec3::ZERO, Vec3::X, projection);
        let centre = aim(&frame, glam::Vec2::ZERO);
        assert!(
            (centre - Vec3::X).length() < 1e-5,
            "the centre pixel aims {centre:?}"
        );
        // At 90 degrees the frustum's edge is 45 degrees off centre, so a corner ray makes an angle
        // whose cosine is 1/sqrt(3) with the forward axis.
        let corner = aim(&frame, glam::Vec2::NEG_ONE);
        assert!(
            (corner.dot(Vec3::X) - 1.0 / 3f32.sqrt()).abs() < 1e-4,
            "a corner ray aims {corner:?}"
        );
        // Vulkan's NDC runs Y *down*, so (-1, -1) is the image's top-left rather than its bottom
        // left — and looking east, left is north. The ray there must go north and up; a transposed
        // matrix, or the DirectX projection in place of the Vulkan one, reverses one of the two.
        assert!(
            corner.y > 0.0 && corner.z > 0.0,
            "the top-left ray aims {corner:?}, which is not north and up"
        );
    }
}
