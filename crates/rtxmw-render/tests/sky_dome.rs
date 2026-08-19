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
use rtxmw_scene::{CellId, Moon, Sky, WorldTime};

mod common;

const WIDTH: u32 = 160;
const HEIGHT: u32 = 160;

/// The view the dome is checked across, and the one a moon is.
const WIDE: f32 = 75.0;
const NARROW: f32 = 10.0;

/// A cell with nothing in it, so every ray escapes and every pixel is sky.
///
/// **No ambient record**, which is what says "exterior" — an interior carries one and keeps its own
/// flat colour, and the dome is switched off for it.
fn empty() -> rtxmw_scene::StaticScene {
    let mut scene = common::scene_of(&[], &[], &[], &[], Vec3::ZERO);
    scene.ambient = None;
    scene
}

/// The frame a camera at `forward` sees across `fov`, and the direction each pixel looked along.
///
/// **The field of view is a parameter because the moons need a narrow one.** Masser is 5.4 degrees
/// across, which in the 75 the dome tests use is ten pixels of a 160-square frame — too few to say
/// anything about a terminator. A ten-degree view puts the same disc across half the frame.
fn looking(sky: Sky, forward: Vec3, degrees: f32) -> (Vec<u8>, Vec<Vec3>) {
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
    let fov = degrees.to_radians();
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
    for hour in [12.0, 17.5, 23.0] {
        // **Stars off for the comparison**, because they are an addition on top of the dome rather
        // than part of its shape — `Sky::shape` has no notion of them and the shader draws them from
        // a hash. The test below is what says they are there at all; this one is about the dome.
        let sky = Sky {
            stars: 0.0,
            ..Sky::at(WorldTime::hours(hour))
        };
        let to_sun = -sky.sun.direction;
        // Facing across the sun rather than at it, so one edge of the frame is the sunward sky and
        // the other is the sky opposite — the axis the dome varies most along.
        let forward = to_sun.cross(Vec3::Z).normalize();
        let (pixels, directions) = looking(sky, forward, WIDE);

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
fn the_stars_come_out_at_night_and_only_at_night() {
    // The cross-check above turns the stars off to compare the dome, so something has to say they
    // are drawn at all — and that they are drawn *only* when the game's own schedule says.
    // How far the brightest pixel stands above the middle one — which is what a star *is*, and what
    // an empty sky of any brightness has none of.
    let contrast = |hour: f32| {
        let (pixels, _) = looking(
            Sky::at(WorldTime::hours(hour)),
            Vec3::new(0.3, 0.0, 0.95).normalize(),
            WIDE,
        );
        let mut values: Vec<f32> = (0..(WIDTH * HEIGHT) as usize)
            .map(|i| at(&pixels, i).max_element())
            .collect();
        values.sort_by(f32::total_cmp);
        values[values.len() - 1] / values[values.len() / 2].max(1e-4)
    };
    assert!(
        contrast(23.0) > 3.0,
        "no stars at 23:00: {}",
        contrast(23.0)
    );
    // Noon and mid-afternoon are both outside the schedule, and an empty sky is nearly flat.
    assert!(
        contrast(12.0) < 1.5,
        "something is out at noon: {}",
        contrast(12.0)
    );
    assert!(
        contrast(16.0) < 1.5,
        "something is out at 16:00: {}",
        contrast(16.0)
    );
}

#[test]
fn the_sky_is_warm_toward_a_low_sun_and_pale_and_dim_away_from_it() {
    // **What the whole change is for**, and it is worth asserting separately from the cross-check
    // above: that one would still pass if both implementations were a flat colour.
    let sky = Sky::at(WorldTime::hours(17.8));
    let to_sun = -sky.sun.direction;
    let flat = |towards: Vec3| Vec3::new(towards.x, towards.y, 0.0).normalize();
    let (sunward, away) = (sky.shape(flat(to_sun)), sky.shape(flat(-to_sun)));

    // Warm on the sun's side, which a sky with one colour cannot be.
    assert!(
        sunward.x > 4.0 * sunward.z,
        "the sunward horizon is not warm: {sunward:?}"
    );
    // And brighter there, because the other side is in the Earth's shadow.
    assert!(
        sunward.length() > 2.0 * away.length(),
        "{sunward:?} against {away:?}"
    );

    // **The horizon opposite it is pale rather than blue**, and that is the model rather than a
    // shortfall: thirty-eight atmospheres of air have forgotten what colour they scattered. Asserting
    // *cool* there passed on a difference of two parts in ten thousand, which is a strong-sounding
    // claim resting on rounding.
    let neutral = (away.max_element() - away.min_element()) / away.max_element();
    assert!(
        neutral < 0.01,
        "the horizon opposite the sun is not pale: {away:?}"
    );

    // The blue lives higher up, where the air is thin enough to keep it.
    let up = sky.shape((flat(-to_sun) + Vec3::Z * 0.577).normalize());
    assert!(
        up.z > 2.0 * up.x,
        "the sky above the anti-sun horizon is not blue: {up:?}"
    );

    // And overhead it is blue at every hour — the thinnest air on the dome.
    for hour in [9.0, 12.0, 17.5] {
        let overhead = Sky::at(WorldTime::hours(hour));
        let zenith = overhead.shape(Vec3::Z);
        assert!(
            zenith.z > zenith.x,
            "the zenith is not blue at {hour}: {zenith:?}"
        );
    }
}

#[test]
fn both_moons_are_the_discs_the_scene_crate_puts_in_the_sky() {
    // **The second equation written in two languages**, after the dome itself: `Moon::radiance` and
    // `moon_disc` in `lighting.glsl` are the same lit sphere, and only this stops them drifting.
    // The sign on the sphere's bulge was wrong once already — it inverts the phase, showing a
    // gibbous moon as a crescent, which no assertion about brightness or position would have caught.
    // **Stars off**, as the dome's own cross-check has them: a star landing on a moon's disc is
    // sixty times the sky's floor added to a pixel this is trying to attribute to the moon, and it
    // is drawn from a hash `Moon::radiance` knows nothing about.
    let sky = Sky {
        stars: 0.0,
        ..Sky::at(WorldTime::hours(23.0))
    };
    for (name, moon) in [("masser", sky.masser), ("secunda", sky.secunda)] {
        // Straight at it, so the disc is near the middle of the frame where a pixel's direction is
        // least distorted and the whole of it is in view.
        let forward = -moon.direction;
        let (pixels, directions) = looking(sky, forward, NARROW);

        let (mut worst, mut worst_at) = (0.0f32, Vec3::ZERO);
        let (mut covered, mut lit, mut dark) = (0, 0, 0);
        for (index, direction) in directions.iter().enumerate() {
            // Inside the disc is the cone test and nothing else, which is what says how *wide* the
            // moon is — the lit fraction says nothing about that, and a new moon has none of it.
            if direction.dot(-moon.direction) <= moon.angular_radius.cos() {
                continue;
            }
            covered += 1;
            let disc = moon.radiance(*direction, sky.sun.direction);
            match disc.max_element() > 0.5 * moon.colour.max_element() {
                true => lit += 1,
                false => dark += 1,
            }
            // Drawn *on top of* the sky rather than instead of it: a moon is behind the air, and
            // the airglow in front of it adds. So the pixel is both, which is what the shader sums.
            let wanted = sky.shape(*direction) + Sky::NIGHT_FLOOR + disc;
            if wanted.max_element() > 0.97 {
                continue;
            }
            let error = (at(&pixels, index) - wanted).abs().max_element();
            if error > worst {
                (worst, worst_at) = (error, *direction);
            }
        }

        // **The disc is the width the ini says**, measured off the frame rather than off the field
        // it was computed in: a moon of angular radius `r` seen through a view of half-angle `f`
        // covers `pi * (r * (WIDTH / 2) / tan(f))^2` pixels, near enough at these angles.
        let scale = (WIDTH as f32 * 0.5) / (NARROW.to_radians() * 0.5).tan();
        let expected = std::f32::consts::PI * (moon.angular_radius * scale).powi(2);
        let ratio = covered as f32 / expected;
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "{name} covered {covered} pixels where the ini's size wants {expected:.0}"
        );
        // And a terminator runs across it, or the comparison above would pass just as well against
        // a flat disc of one colour. A twelfth, because at this hour of the first day Secunda is all
        // but full and the shadowed part of it is a sliver down one limb.
        assert!(
            lit > covered / 12 && dark > covered / 12,
            "{name} has no terminator: {lit} lit, {dark} dark of {covered}"
        );
        // A little over one part in 255, which is what an 8-bit target can carry.
        assert!(
            worst < 0.006,
            "{name}: the shader and `Moon::radiance` disagree by {worst} looking {worst_at:?}"
        );
    }

    // And the bigger moon covers the more sky, which is the one thing a per-pixel comparison of two
    // agreeing implementations cannot tell you — they would agree about a moon of any size.
    assert!(sky.masser.angular_radius > 2.0 * sky.secunda.angular_radius);
}

#[test]
fn a_moon_that_has_set_is_not_drawn_and_the_day_has_none() {
    // Noon: both moons are the far side of the world, so the sky where they *would* be is empty.
    let noon = Sky::at(WorldTime::hours(12.0));
    assert_eq!(noon.masser.colour, Vec3::ZERO, "Masser is up at noon");
    assert_eq!(noon.secunda.colour, Vec3::ZERO, "Secunda is up at noon");

    // Which is a claim about the picture and not only about the numbers: pointing where a moon
    // would have stood draws nothing but sky.
    let forward = -Moon::masser(WorldTime::hours(12.0), noon.sun).direction;
    let (pixels, directions) = looking(noon, forward, NARROW);
    for (index, direction) in directions.iter().enumerate() {
        let wanted = noon.shape(*direction) + Sky::NIGHT_FLOOR;
        if wanted.max_element() > 0.97 {
            continue;
        }
        let error = (at(&pixels, index) - wanted).abs().max_element();
        assert!(
            error < 0.006,
            "something is drawn at noon looking {direction:?}"
        );
    }
}
