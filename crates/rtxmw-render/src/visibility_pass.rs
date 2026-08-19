//! The primary visibility pass: one ray query per pixel, into an offscreen HDR image.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use rtxmw_gpu::{Binding, Buffer, BufferMemory, ComputePipeline, Device, Image, Memory};
use rtxmw_scene::Sun;

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
    ndc_to_world_offset: [f32; 16],
    camera_position: [f32; 3],
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
    /// The sinusoids the water surface is summed from — see [`crate::wave_spectrum`].
    ///
    /// Static for the life of a sea state, and carried here rather than in a buffer of its own
    /// because it is six hundred bytes beside a block that is already uploaded every frame, and a
    /// second binding would cost more to explain than the copy costs to make.
    waves: [GpuWave; WAVE_COUNT],
}

impl FrameConstants {
    /// Builds the block from a camera's matrices.
    ///
    /// `projection` must be the reverse-Z Vulkan projection the shader assumes — it unprojects the
    /// near plane at depth 1, which is the only plane an infinite projection leaves invertible.
    pub(crate) fn new(
        view: Mat4,
        projection: Mat4,
        camera_position: Vec3,
        lighting: Lighting,
        cone_spread: f32,
        bounce_samples: u32,
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
            ndc_to_world_offset: Self::ndc_to_world_offset(view, projection),
            camera_position: camera_position.to_array(),
            light_grid_scale: lighting.light_grid.scale,
            light_grid_origin: lighting.light_grid.origin.to_array(),
            light_grid_dimensions: lighting.light_grid.dimensions,
            ambient: lighting.ambient.to_array(),
            cone_spread,
            sun_direction: sun.direction.to_array(),
            sun_cos_radius: sun.angular_radius.cos(),
            sun_colour: sun.colour.to_array(),
            bounce_samples,
            water_level: lighting.water_level.unwrap_or(f32::NEG_INFINITY),
            time,
            // Rebuilt each frame rather than cached: it is a few hundred floats of arithmetic once
            // per frame against a million rays that read it, and a cache would need invalidating
            // the moment the sea state becomes something a cell can set.
            waves: SeaState::default().waves(),
        }
    }

    /// The matrix taking a pixel's clip coordinates to its offset from the eye, in world axes.
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
    /// Dropping the view's translation makes the unprojection land in a space centred on the camera,
    /// where the near plane is at 0.05 and an `f32` step is 6e-9. There is then nothing to cancel,
    /// and the error is **0.0001 pixels wherever the camera stands**. It also leaves a far better
    /// conditioned matrix to invert, since no entry is the size of the world any more.
    ///
    /// The camera's position is still sent — a ray needs an origin — but it is used only as one,
    /// never differenced against anything.
    fn ndc_to_world_offset(view: Mat4, projection: Mat4) -> [f32; 16] {
        let mut rotation = view;
        // A look-at view is `rotation * translate(-eye)`, so its fourth column *is* the eye's
        // contribution and clearing it leaves the rotation alone.
        rotation.w_axis = glam::Vec4::W;
        (projection * rotation).inverse().to_cols_array()
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
                    // The frame's own constants. They were push constants until waves needed a
                    // clock and the block was already exactly the 128 bytes Vulkan guarantees —
                    // see `FrameConstants`. Motion vectors at M7 would have forced the same move.
                    Binding::storage_buffer(11),
                    // The light grid: prefix offsets, then the light indices they address.
                    Binding::storage_buffer(12),
                    Binding::storage_buffer(13),
                    // Last, because Vulkan allows a variable descriptor count only on a set's final
                    // binding. Adding anything after this one moves it — validation rejects the set
                    // outright, which is how this was caught.
                    Binding::variable_samplers(14, max_textures),
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
            .dst_binding(11)
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
            .dst_binding(14)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&texture_infos);

        // The storage images go through the pipeline's own helper: the emissive-and-sky target at
        // binding one, and the G-buffer as a run from eight.
        self.pipeline.bind_storage_images(1, &[target]);
        self.pipeline.bind_storage_images(
            8,
            &[
                gbuffer.albedo(),
                gbuffer.normal_depth(),
                gbuffer.illumination(),
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
            .dst_binding(12)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&grid_offset_info);

        let grid_index_info = [vk::DescriptorBufferInfo::default()
            .buffer(scene.light_grid.indices().raw())
            .range(vk::WHOLE_SIZE)];
        let grid_index_write = vk::WriteDescriptorSet::default()
            .dst_set(self.pipeline.set())
            .dst_binding(13)
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
        assert_eq!(offset_of!(FrameConstants, camera_position), 64);
        assert_eq!(offset_of!(FrameConstants, light_grid_scale), 76);
        assert_eq!(offset_of!(FrameConstants, light_grid_origin), 80);
        assert_eq!(offset_of!(FrameConstants, light_grid_dimensions), 92);
        assert_eq!(offset_of!(FrameConstants, ambient), 104);
        assert_eq!(offset_of!(FrameConstants, cone_spread), 116);
        assert_eq!(offset_of!(FrameConstants, sun_direction), 120);
        assert_eq!(offset_of!(FrameConstants, sun_cos_radius), 132);
        assert_eq!(offset_of!(FrameConstants, sun_colour), 136);
        assert_eq!(offset_of!(FrameConstants, bounce_samples), 148);
        assert_eq!(offset_of!(FrameConstants, water_level), 152);
        assert_eq!(offset_of!(FrameConstants, time), 156);
        // The wave table follows, twenty tightly packed bytes apiece.
        assert_eq!(offset_of!(FrameConstants, waves), 160);
        assert_eq!(size_of::<FrameConstants>(), 160 + 20 * WAVE_COUNT);
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

    /// A frame built the way the renderer builds one, for a camera at `eye` looking `forward`.
    fn frame_at(eye: Vec3, forward: Vec3, projection: Mat4) -> FrameConstants {
        FrameConstants::new(
            glam::camera::rh::view::look_to_mat4(eye, forward, Vec3::Z),
            projection,
            eye,
            Lighting {
                ambient: Vec3::ZERO,
                light_grid: LightGridExtent::default(),
                sun: None,
                water_level: None,
            },
            0.0,
            0,
            0.0,
        )
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
