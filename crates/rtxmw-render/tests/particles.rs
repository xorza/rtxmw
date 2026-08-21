//! The shape an emitter puts on the screen, and what lights it.
//!
//! One plume, held still: the clock is zero in a test, so the noise field is a fixed one and every
//! number below is repeatable. What is asserted is the *shape* rather than any pixel's value —
//! everything after the trace is a monotone function of one number, and the shape is what no
//! exposure and no tone curve can manufacture.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3, Vec4};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{
    CellId, Instance, Material, Mesh, MeshId, ParticleEmitter, StaticScene, Submesh, Sun,
};

mod common;

const WIDTH: u32 = 192;
const HEIGHT: u32 = 192;

/// Where the plume hangs, how far it rises, and how wide it leaves.
///
/// The view is 75 degrees over 192 pixels, so at 200 units it spans `2 * 200 * tan(37.5)` = 307
/// units — 0.625 pixels a unit. A 120-unit plume is 75 pixels tall, centred on the frame, and a
/// 60-unit sprite gives it a 30-unit foot.
const DISTANCE: f32 = 200.0;
const HEIGHT_UNITS: f32 = 120.0;
const PIXELS_PER_UNIT: f32 = 0.625;

/// A backdrop far behind the plume, and one emitter in front of it.
fn one_plume(additive: bool, sunlit: bool) -> StaticScene {
    let reach = 4000.0;
    let backdrop = Mesh {
        positions: vec![
            Vec3::new(2000.0, -reach, -reach),
            Vec3::new(2000.0, reach, -reach),
            Vec3::new(2000.0, reach, reach),
            Vec3::new(2000.0, -reach, reach),
        ],
        normals: vec![Vec3::NEG_X; 4],
        uvs: vec![Vec2::ZERO; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material: 0,
            thin: false,
        }],
    };
    // Black, so nothing but the plume puts anything in the frame.
    let mut scene = common::scene_of(
        &[backdrop],
        &[Material {
            diffuse: Vec3::ZERO,
            ..Material::default()
        }],
        &[Instance {
            mesh: MeshId(0),
            transform: Affine3A::IDENTITY,
        }],
        &[],
        Vec3::splat(0.1),
    );
    if sunlit {
        scene.sun = Some(Sun {
            direction: Vec3::NEG_Z,
            colour: Vec3::splat(0.4),
            angular_radius: 0.0,
        });
    }
    scene.emitters.push(ParticleEmitter {
        // Half the rise below the eye, so the plume is centred on the frame.
        placement: Affine3A::from_translation(Vec3::new(DISTANCE, 0.0, -0.5 * HEIGHT_UNITS)),
        spread: Vec3::ZERO,
        speed: HEIGHT_UNITS,
        speed_variation: 0.0,
        declination: 0.0,
        declination_variation: 0.0,
        azimuth: 0.0,
        colour: Vec4::ONE,
        ramp: [Vec4::ONE; 3],
        ramp_mid: 0.5,
        size: 60.0,
        lifetime: 1.0,
        lifetime_variation: 0.0,
        gravity: Vec3::ZERO,
        additive,
    });
    scene
}

/// Renders the fixture and returns the traced radiance as bytes.
fn present(scene: &StaticScene) -> Vec<u8> {
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
    renderer.set_fog(0.0);
    // All three would carry the plume's light onto pixels it does not cover, and every assertion
    // here is about which pixels it does.
    renderer.set_glare(0.0);
    renderer.set_denoise_passes(0);

    let mut uploader = gpu.uploader();
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            CellId::Interior("fixture".to_owned()),
            scene,
            &[],
        )
        .expect("scene should load");

    let eye = Vec3::ZERO;
    let view = glam::camera::rh::view::look_to_mat4(eye, Vec3::X, Vec3::Z);
    let projection =
        glam::camera::rh::proj::vulkan::perspective_infinite_reverse(75f32.to_radians(), 1.0, 0.05);
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
    pixels
}

/// The green channel at a pixel.
fn value_at(pixels: &[u8], x: u32, y: u32) -> u8 {
    pixels[((y * WIDTH + x) * 4 + 1) as usize]
}

/// How wide the plume is drawn `risen` of the way up it, in pixels.
///
/// Averaged over three rows, because the noise erodes the outline into tongues and one row of that
/// is a sample of the noise rather than a measurement of the shape.
fn width_at(pixels: &[u8], risen: f32) -> f32 {
    let middle = (HEIGHT / 2) as i32;
    // The plume is centred on the frame and screen `y` grows downward, so a rise is a fall in `y`.
    let row = middle - ((risen - 0.5) * HEIGHT_UNITS * PIXELS_PER_UNIT) as i32;
    let mut total = 0.0;
    for row in [row - 1, row, row + 1] {
        let lit: Vec<u32> = (0..WIDTH)
            .filter(|&x| value_at(pixels, x, row as u32) > 0)
            .collect();
        total += match (lit.first(), lit.last()) {
            (Some(&first), Some(&last)) => (last - first + 1) as f32,
            _ => 0.0,
        };
    }
    total / 3.0
}

#[test]
fn a_flame_is_a_teardrop_and_not_a_pyramid() {
    let pixels = present(&one_plume(true, false));
    let (neck, belly, tip) = (
        width_at(&pixels, 0.06),
        width_at(&pixels, 0.3),
        width_at(&pixels, 0.85),
    );

    assert!(
        belly > 8.0,
        "the plume is not drawn at all: it is {belly} pixels across at its widest"
    );
    // **Narrow where it meets the fuel.** A flame is thinnest at the bottom — the gas has had no
    // room to expand and the reaction is still starting — and a plume widest at its foot presents a
    // flat base to anything looking at it, which is a pyramid and was reported as one.
    assert!(
        neck < belly,
        "the flame is {neck} pixels across at its foot against {belly} at its belly, so it is \
         widest at the bottom"
    );
    // **And tapered above it**, which is what makes a flame a flame rather than the opening cone
    // the gas is actually travelling in.
    assert!(
        tip < 0.6 * belly,
        "the flame is {tip} pixels across near its tip against {belly} at its belly"
    );
}

#[test]
fn a_plume_of_smoke_is_lit_by_the_sun_as_well_as_by_the_sky() {
    // **`sun_colour` is an irradiance**, so what a surface square-on to the sun leaves is
    // `albedo * sun_colour / pi`, and a puff — which the sun reaches all of rather than one
    // hemisphere of — comes out at very nearly that. Against an ambient of 0.1 and a sun of 0.4
    // that is `0.1 + 0.4/pi = 0.227` where it used to be 0.1, which is the difference between smoke
    // a daylit sky shows and a dark veil over whatever is behind it.
    let middle = (WIDTH / 2, HEIGHT / 2);
    let shaded = value_at(&present(&one_plume(false, false)), middle.0, middle.1);
    let sunlit = value_at(&present(&one_plume(false, true)), middle.0, middle.1);
    assert!(
        sunlit > shaded,
        "a puff in the sun reads {sunlit} against {shaded} in the shade"
    );

    // **And a flame is its own light**, so the sun does nothing to it either way — which is what
    // separates the two paths through the march.
    let flame = value_at(&present(&one_plume(true, false)), middle.0, middle.1);
    let flame_in_sun = value_at(&present(&one_plume(true, true)), middle.0, middle.1);
    assert_eq!(
        flame, flame_in_sun,
        "a flame's brightness is its own, not the sun's"
    );
}
