//! That the sky the shader draws is the sky `rtxmw_scene::Sky` describes.
//!
//! **One equation, two implementations.** The dome's shape is written in Rust so the host can
//! average it into the ambient a surface is lit by, and again in GLSL so a ray that hits nothing can
//! be given its own direction's colour. Nothing but this file stops the two drifting apart, and the
//! cost of them drifting is a frame lit by one sky and drawn with another.

use ash::vk;
use glam::{Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{CellId, Sky, TimeOfDay};

mod common;

const WIDTH: u32 = 160;
const HEIGHT: u32 = 160;

/// A cell with nothing in it, so every ray escapes and every pixel is sky.
///
/// **No ambient record**, which is what says "exterior" — an interior carries one and keeps its own
/// flat colour, and the dome is switched off for it.
fn empty() -> rtxmw_scene::StaticScene {
    let mut scene = common::scene_of(&[], &[], &[], &[], Vec3::ZERO);
    scene.ambient = None;
    scene
}

/// The frame a camera at `forward` sees, and the direction each of its pixels looked along.
fn looking(sky: Sky, forward: Vec3) -> (Vec<u8>, Vec<Vec3>) {
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
    renderer.set_bounce_samples(0);
    renderer.set_denoise_passes(0);
    renderer.set_fog(0.0);
    renderer.set_sky(sky);

    let mut uploader = gpu.uploader();
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            CellId::Exterior { x: 0, y: 0 },
            &empty(),
            &[],
        )
        .expect("scene should load");

    let eye = Vec3::ZERO;
    let view = glam::camera::rh::view::look_to_mat4(eye, forward, Vec3::Z);
    let fov = 75f32.to_radians();
    let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(fov, 1.0, 0.05);
    let constants = renderer.frame_constants(view, projection, eye);
    renderer
        .render_once(&mut uploader, &constants)
        .expect("trace should run");
    let pixels = readback::image_to_rgba8(
        &mut uploader,
        renderer.target(),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    )
    .expect("readback should succeed");
    drop(uploader);
    gpu.assert_no_validation_errors();

    // The same directions the shader built, from the same field of view: a pixel's clip coordinates
    // scaled by the half-angle's tangent, in the camera's own frame.
    let tangent = (fov * 0.5).tan();
    let right = forward.cross(Vec3::Z).normalize();
    let up = right.cross(forward);
    let mut directions = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let ndc = Vec2::new(
                (x as f32 + 0.5) / WIDTH as f32 * 2.0 - 1.0,
                1.0 - (y as f32 + 0.5) / HEIGHT as f32 * 2.0,
            ) * tangent;
            directions.push((forward + right * ndc.x + up * ndc.y).normalize());
        }
    }
    (pixels, directions)
}

/// The linear colour at one pixel. The target is written before any tone curve, so this is radiance
/// clipped to one rather than anything encoded.
fn at(pixels: &[u8], index: usize) -> Vec3 {
    let byte = index * 4;
    Vec3::new(
        pixels[byte] as f32,
        pixels[byte + 1] as f32,
        pixels[byte + 2] as f32,
    ) / 255.0
}

#[test]
fn every_pixel_of_the_sky_is_what_the_scene_crate_says_it_is() {
    // Three hours that exercise the whole dome: overhead, low enough to have a warm side, and under
    // the horizon where only the floor is left.
    for hour in [13.0, 19.5, 23.0] {
        let sky = Sky::at(TimeOfDay::hours(hour));
        let to_sun = -sky.sun.direction;
        // Facing across the sun rather than at it, so one edge of the frame is the sunward sky and
        // the other is the sky opposite — the axis the dome varies most along.
        let forward = to_sun.cross(Vec3::Z).normalize();
        let (pixels, directions) = looking(sky, forward);

        let (mut worst, mut worst_at) = (0.0f32, Vec3::ZERO);
        for (index, direction) in directions.iter().enumerate() {
            let wanted = sky.shape(*direction) + Sky::NIGHT_FLOOR;
            // Only where nothing is clipped: the target is 8-bit, so a channel at one says "at
            // least one" and cannot be compared against a number above it.
            if wanted.max_element() > 0.97 {
                continue;
            }
            let error = (at(&pixels, index) - wanted).abs().max_element();
            if error > worst {
                (worst, worst_at) = (error, *direction);
            }
        }
        // A little over one part in 255, which is what an 8-bit target can carry.
        assert!(
            worst < 0.006,
            "at {hour}:00 the shader and `Sky::shape` disagree by {worst} looking {worst_at:?}"
        );
    }
}

#[test]
fn the_sky_is_warm_toward_a_low_sun_and_cool_away_from_it() {
    // **What the whole change is for**, and it is worth asserting separately from the cross-check
    // above: that one would still pass if both implementations were a flat colour.
    let sky = Sky::at(TimeOfDay::hours(19.5));
    let to_sun = -sky.sun.direction;
    let horizon = |towards: Vec3| {
        let flat = Vec3::new(towards.x, towards.y, 0.0).normalize();
        sky.shape(flat)
    };
    let (sunward, away) = (horizon(to_sun), horizon(-to_sun));

    // Warm on the sun's side and cool on the other, which a sky with one colour cannot be.
    assert!(
        sunward.x > sunward.z,
        "the sunward horizon is not warm: {sunward:?}"
    );
    assert!(
        away.z > away.x,
        "the horizon opposite the sun is not cool: {away:?}"
    );
    // And brighter on the sun's side, because the other one is in the Earth's shadow.
    assert!(
        sunward.length() > 2.0 * away.length(),
        "{sunward:?} against {away:?}"
    );

    // Overhead keeps its blue at every hour — it is the thinnest air on the dome.
    for hour in [9.5, 13.0, 19.5] {
        let overhead = Sky::at(TimeOfDay::hours(hour));
        let zenith = overhead.shape(Vec3::Z);
        assert!(
            zenith.z > zenith.x,
            "the zenith is not blue at {hour}: {zenith:?}"
        );
    }
}
