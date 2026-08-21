//! What a texture's painted relief does to the normal a surface is shaded by.
//!
//! **Measured as a ratio between two traces of the same wall**, one with relief and one without.
//! Everything the two frames do not disagree about — the albedo, the mip, the shadow ray's own
//! sampling — cancels exactly, so what is left is the cosine the perturbed normal makes with the
//! light and nothing else. That is the whole of what `relief.glsl` claims to change.
//!
//! Lit by a sun rather than by a lamp, because a sun is the only light in this renderer whose
//! direction is the same at every pixel: a lamp's varies across the wall, and then the ratio at one
//! pixel is no longer something that can be worked out on paper.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{CellId, Instance, Material, Mesh, MeshId, StaticScene, Submesh, Sun, TextureId};
use rtxmw_texture::{Texture, TextureFormat, channel_to_linear};

mod common;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;

/// How far ahead of the camera the wall stands, and how far its edges reach.
///
/// Wide enough to cover the frame: at 75 degrees the view is 76.7 units tall at this distance, so
/// every ray lands on wall and a miss cannot be mistaken for a flat hit.
const DISTANCE: f32 = 100.0;
const REACH: f32 = 120.0;

/// `RELIEF_SLOPE`, `RELIEF_MAX_SLOPE` and `RELIEF_BLACK` from `relief.glsl`, which a Rust test
/// cannot see.
///
/// Written out again rather than derived, and the duplication is the point: these are tuned
/// constants, and moving one should fail here so that it is a decision rather than a side effect.
const SLOPE: f32 = 0.7;
const MAX_SLOPE: f32 = 1.2;
const BLACK: f32 = 1.0 / 255.0;

/// How far a measured ratio may fall short of the boundary's own.
///
/// A ray lands where the projection puts it, not on a texel boundary, and the nearest one to a step
/// misses it by up to half a pixel. The wall is 240 units across wearing eight texels, so a texel
/// is 30 units and a pixel about 0.6 — the taps of the nearest ray therefore carry at most a
/// hundredth of the neighbouring texel, and report that much less tilt.
const NEAREST_RAY: f32 = 0.02;

/// The wall every test here looks at: `REACH` square, facing the camera, uvs running twice across.
///
/// `u` runs along world `+Y` and `v` along `+Z`, so a texture varying only across its columns
/// varies only along `+Y` in the world. Twice rather than once so that several texel boundaries
/// fall inside the view and both directions of step are seen.
fn wall() -> Mesh {
    Mesh {
        positions: vec![
            Vec3::new(DISTANCE, -REACH, -REACH),
            Vec3::new(DISTANCE, REACH, -REACH),
            Vec3::new(DISTANCE, REACH, REACH),
            Vec3::new(DISTANCE, -REACH, REACH),
        ],
        normals: vec![Vec3::NEG_X; 4],
        uvs: vec![
            Vec2::ZERO,
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ],
        // Wound so the triangles' own plane comes out along -X, which is the side the camera is on.
        indices: vec![0, 2, 1, 0, 3, 2],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material: 0,
            thin: false,
        }],
    }
}

/// A four-square grey texture whose columns are `left`, `left`, `right`, `right`.
///
/// **Two texels of each**, so a step falls on a texel boundary rather than between two centres: the
/// four taps `relief_gradient` takes sit half a texel out, and at a boundary that puts each of them
/// exactly on a neighbouring texel's centre. The gradient there is then the difference of two texel
/// values and nothing else — no bilinear mixing to account for.
///
/// Every row identical, which makes the vertical gradient exactly zero: the two taps above and the
/// two below straddle the same rows, so whatever they return, they return it twice.
fn columns(left: u8, right: u8) -> Texture {
    let mut data = Vec::with_capacity(4 * 4 * 4);
    for _ in 0..4 {
        for value in [left, left, right, right] {
            data.extend([value, value, value, 255]);
        }
    }
    Texture::from_pixels(TextureFormat::Rgba8, 4, 4, data)
}

/// The wall wearing texture zero, under a sun travelling along `sunward`.
fn lit_wall(sunward: Vec3) -> StaticScene {
    let mut scene = common::scene_of(
        &[wall()],
        &[Material {
            base_colour: Some(TextureId(0)),
            ..Material::default()
        }],
        &[Instance {
            mesh: MeshId(0),
            transform: Affine3A::IDENTITY,
        }],
        &[],
        // Nothing but the sun: an ambient term would add a part of the frame that the tilt does not
        // change, and dilute the ratio being measured by however large it was.
        Vec3::ZERO,
    );
    scene.sun = Some(Sun {
        direction: sunward.normalize(),
        colour: Vec3::splat(2.0),
        // A point sun, so the shadow ray has one direction rather than a disc to sample. Nothing
        // occludes this wall, but a sampled disc would still be one more thing between the
        // assertion and the cosine.
        angular_radius: 0.0,
    });
    scene
}

/// What a trace of `scene` reads back, either the radiance it wrote or the normal it guided with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Read {
    Radiance,
    GuideNormal,
}

fn trace(scene: &StaticScene, texture: &Texture, relief: f32, read: Read) -> Vec<Vec3> {
    let gpu = TestGpu::shared();
    let mut renderer = SceneRenderer::new(
        gpu.device(),
        gpu.physical(),
        gpu.memory(),
        vk::Extent2D {
            width: WIDTH,
            height: HEIGHT,
        },
    )
    .expect("renderer should build");
    // No bounce, no filter, no fog: all three would put light on the wall that the tilt does not
    // account for, and the ratio measures the tilt.
    renderer.set_bounce_samples(0);
    renderer.set_denoise_passes(0);
    renderer.set_fog(0.0);
    renderer.set_relief(relief);

    let mut uploader = gpu.uploader();
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            CellId::Interior("relief".to_owned()),
            scene,
            std::slice::from_ref(&Some(texture.clone())),
        )
        .expect("scene should load");

    let view = glam::camera::rh::view::look_to_mat4(Vec3::ZERO, Vec3::X, Vec3::Z);
    let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
        75f32.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
        0.05,
    );
    let constants = renderer.frame_constants(view, projection, Vec3::ZERO);
    renderer
        .render_once(&mut uploader, &constants)
        .expect("trace should run");
    let channels = match read {
        Read::Radiance => readback::image_to_f32(
            &mut uploader,
            renderer.target(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ),
        Read::GuideNormal => readback::image_to_f32(
            &mut uploader,
            renderer.normal_roughness(),
            vk::ImageLayout::GENERAL,
        ),
    }
    .expect("readback should succeed");
    drop(uploader);
    gpu.assert_no_validation_errors();
    channels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect()
}

/// How much brighter relief made each pixel of `texture`'s wall under a sun along `sunward`.
///
/// The albedo, the mip and the shadow ray are the same in both traces, so what is left in the
/// quotient is the ratio of the two cosines.
fn brightening(texture: &Texture, sunward: Vec3, relief: f32) -> Vec<f32> {
    let scene = lit_wall(sunward);
    let flat = trace(&scene, texture, 0.0, Read::Radiance);
    let tilted = trace(&scene, texture, relief, Read::Radiance);
    flat.iter()
        .zip(&tilted)
        .map(|(flat, tilted)| {
            assert!(flat.x > 0.05, "the fixture went dark: {flat:?}");
            tilted.x / flat.x
        })
        .collect()
}

/// The tangent-space slope `relief.glsl` gives a step between two grey bytes, at `strength`.
///
/// The shader's arithmetic written out from the two published pieces it is built on — the sRGB
/// transfer, and the fact that Rec. 709's weights sum to one so a grey's luminance is its own
/// value. A step from `left` to `right` has a log-luminance gradient of
/// `ln(right + BLACK) - ln(left + BLACK)` across one texel, and the normal leans *away* from the
/// brighter side because brighter is higher.
fn slope_across(left: u8, right: u8, strength: f32) -> f32 {
    let gradient = (channel_to_linear(right) + BLACK).ln() - (channel_to_linear(left) + BLACK).ln();
    MAX_SLOPE * (-strength * SLOPE * gradient / MAX_SLOPE).tanh()
}

/// What a slope of `slope` about the wall's `v` axis does to the sun's cosine.
///
/// The wall faces `-X` and `u` runs along `+Y`, so a tilt of `s` makes the normal
/// `normalize((-1, s, 0))`. A sun whose light travels along `(1, -1, 0)` arrives from
/// `(-1, 1, 0)/sqrt(2)`, where the flat wall's cosine is `1/sqrt(2)` and the tilted wall's is
/// `(1 + s)/(sqrt(2) * sqrt(1 + s*s))` — so the quotient is what this returns.
fn across_the_tilt(slope: f32) -> f32 {
    (1.0 + slope) / (1.0 + slope * slope).sqrt()
}

/// The same quotient for a sun in the `x`-`z` plane, which the tilt does not lean toward.
///
/// Only the normalisation is left: the cosine falls by `1/sqrt(1 + s*s)` because the normal grew a
/// component the light has none of. A tilt that had leaked into `v` would move this linearly, where
/// the term that belongs here is quadratic — which is what makes it a test of the axis.
fn along_the_tilt(slope: f32) -> f32 {
    1.0 / (1.0 + slope * slope).sqrt()
}

/// Where the sun stands for each of the two measurements.
const ACROSS: Vec3 = Vec3::new(1.0, -1.0, 0.0);
const ALONG: Vec3 = Vec3::new(1.0, 0.0, -1.0);

#[test]
fn a_flat_texture_leaves_the_shading_exactly_as_the_mesh_gave_it() {
    // Nothing to read: a texture of one value has no gradient at all, at any offset and any level,
    // so this is exact rather than approximate. It is the case that decides whether relief is a
    // reading of the texture or a noise floor added to every surface in the world.
    for ratio in brightening(&columns(150, 150), ACROSS, 1.0) {
        assert!(
            (ratio - 1.0).abs() < 1.0e-3,
            "a uniform texture changed the shading by {ratio}"
        );
    }
}

#[test]
fn a_step_tilts_the_normal_by_what_its_log_luminance_gradient_says() {
    // 150 and 200 decode to 0.30496 and 0.57755, so the gradient across the step is
    // ln(0.58147) - ln(0.30888) = 0.6324, the raw slope -0.7 * 0.6324 = -0.4427, and `tanh` brings
    // that to -0.4234. The wall away from the brighter side loses cosine — 0.5766/1.0860 = 0.5309
    // of it — and the wall on the far side of the same step gains, 1.4234/1.0860 = 1.3108.
    let slope = slope_across(150, 200, 1.0);
    assert!(
        (slope + 0.4234).abs() < 1.0e-3,
        "the fixture's own arithmetic moved: {slope}"
    );
    let ratios = brightening(&columns(150, 200), ACROSS, 1.0);
    let darkest = ratios.iter().copied().fold(f32::INFINITY, f32::min);
    let brightest = ratios.iter().copied().fold(0.0f32, f32::max);
    assert!(
        (darkest - across_the_tilt(slope)).abs() < NEAREST_RAY,
        "the steepest step darkened by {darkest}, and it says {}",
        across_the_tilt(slope)
    );
    assert!(
        (brightest - across_the_tilt(-slope)).abs() < NEAREST_RAY,
        "the step back brightened by {brightest}, and it says {}",
        across_the_tilt(-slope)
    );

    // **Stronger paint, steeper normal.** A wider step has to tilt further, or the gradient is not
    // what is being read.
    let harder = slope_across(120, 230, 1.0);
    assert!(
        across_the_tilt(harder) < across_the_tilt(slope) - 0.05,
        "the two fixtures do not separate: {} against {}",
        across_the_tilt(harder),
        across_the_tilt(slope)
    );
    let measured = brightening(&columns(120, 230), ACROSS, 1.0)
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    assert!(
        (measured - across_the_tilt(harder)).abs() < NEAREST_RAY,
        "the harder step darkened by {measured}, and it says {}",
        across_the_tilt(harder)
    );
}

#[test]
fn relief_tilts_only_across_the_direction_the_texture_varies_in() {
    // Every row of the texture is the same, so the vertical taps cancel and the normal cannot lean
    // along `v` at all. Under a sun in that plane the only thing left to see is the normal having
    // grown a component the light has none of, which costs it 1/sqrt(1 + s*s) — 0.9208 here.
    let slope = slope_across(150, 200, 1.0);
    let ratios = brightening(&columns(150, 200), ALONG, 1.0);
    let darkest = ratios.iter().copied().fold(f32::INFINITY, f32::min);
    let brightest = ratios.iter().copied().fold(0.0f32, f32::max);
    assert!(
        (darkest - along_the_tilt(slope)).abs() < NEAREST_RAY,
        "a column-only texture darkened by {darkest} along its own axis, and the normalisation \
         alone says {}",
        along_the_tilt(slope)
    );
    // And nothing brightened: a leak into `v` would lean half the steps toward this sun.
    assert!(
        brightest <= 1.0 + 1.0e-3,
        "a column-only texture brightened by {brightest} along its own axis"
    );
}

#[test]
fn the_relief_strength_scales_the_tilt_and_zero_turns_it_off() {
    // Zero is the mesh's own normal, exactly — the switch has to be an off switch and not a
    // quieter version of the same thing, or the A/B `docs/design.md` §5.1 asks for is not one.
    for ratio in brightening(&columns(150, 200), ACROSS, 0.0) {
        assert!(
            (ratio - 1.0).abs() < 1.0e-3,
            "relief 0 changed the shading by {ratio}"
        );
    }
    // And half strength is half the slope, which below the compression's knee is nearly half the
    // tilt: the raw slope is -0.4427 at full and -0.2213 at half, which `tanh` carries to -0.4234
    // and -0.2196.
    let half = slope_across(150, 200, 0.5);
    assert!(
        (half + 0.2196).abs() < 1.0e-3,
        "half the strength should be a little more than half the slope, not {half}"
    );
    let measured = brightening(&columns(150, 200), ACROSS, 0.5)
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    assert!(
        (measured - across_the_tilt(half)).abs() < NEAREST_RAY,
        "half strength darkened by {measured}, and half the slope says {}",
        across_the_tilt(half)
    );
}

#[test]
fn the_upscalers_guide_normal_is_the_mesh_normal_and_not_the_tilted_one() {
    // **Deliberate, and measured** — `docs/design.md` §8.90. A guide normal says which surface a
    // pixel is on, which is what history is reprojected against; painted relief is detail inside
    // one surface rather than a different surface, and handing the tilt to Ray Reconstruction costs
    // it most of its temporal accumulation.
    let scene = lit_wall(ACROSS);
    for normal in trace(&scene, &columns(150, 200), 1.0, Read::GuideNormal) {
        assert!(
            (normal - Vec3::NEG_X).length() < 1.0e-3,
            "the guide normal carried the tilt: {normal:?}"
        );
    }
}
