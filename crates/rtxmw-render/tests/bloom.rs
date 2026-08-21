//! The glow a bright thing leaves around itself, asserted on the bytes that reach a screen.
//!
//! **A profile rather than a level.** Every stage after the trace is a monotone function of one
//! number — auto-exposure is a scale and the tone curve a curve — so no single pixel's byte value
//! is worth asserting here without recomputing the whole chain. What no scale and no curve can
//! manufacture is a *gradient across a flat surface*: the backdrop is one material under one
//! ambient, so without a point spread every pixel of it away from the emitter is the same byte, and
//! a falling sequence is proof that light was carried there.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{CellId, Instance, Material, Mesh, MeshId, StaticScene, Submesh};

mod common;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;

/// How bright the emitter is against the wall behind it.
///
/// Two hundred against an ambient-lit twentieth: three and a half orders of magnitude, which is the
/// range a candle in a room actually spans and the range the glow has to survive being tone mapped
/// out of.
const EMISSIVE: f32 = 200.0;

/// A wall filling the view, with a small bright square stuck to the middle of it.
///
/// One mesh and two runs rather than two instances: the square sits a unit in front so it occludes
/// rather than z-fights, and both are flat and face the camera so nothing here depends on shading.
fn lamp_on_a_wall() -> StaticScene {
    let wall = |x: f32, half: f32| {
        [
            Vec3::new(x, -half, -half),
            Vec3::new(x, half, -half),
            Vec3::new(x, half, half),
            Vec3::new(x, -half, half),
        ]
    };
    let mut positions = wall(100.0, 400.0).to_vec();
    positions.extend(wall(99.0, 4.0));
    let mesh = Mesh {
        normals: vec![Vec3::NEG_X; positions.len()],
        uvs: vec![Vec2::ZERO; positions.len()],
        positions,
        indices: vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
        submeshes: vec![
            Submesh {
                first_index: 0,
                index_count: 6,
                material: 0,
                thin: false,
            },
            Submesh {
                first_index: 6,
                index_count: 6,
                material: 1,
                thin: false,
            },
        ],
    };
    common::scene_of(
        &[mesh],
        &[
            Material {
                diffuse: Vec3::splat(0.05),
                ..Material::default()
            },
            Material {
                emissive: Vec3::splat(EMISSIVE),
                ..Material::default()
            },
        ],
        &[Instance {
            mesh: MeshId(0),
            transform: Affine3A::IDENTITY,
        }],
        &[],
        Vec3::ONE,
    )
}

/// Renders the fixture and returns the display-ready bytes.
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
    // No bounce and no fog: both would carry light across the frame themselves, which is the one
    // thing this test must be the only source of.
    renderer.set_bounce_samples(0);
    renderer.set_fog(0.0);

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
        renderer.output(),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    )
    .expect("readback should succeed");
    drop(uploader);
    gpu.assert_no_validation_errors();
    pixels
}

/// The green channel at `(x, y)`, which for a grey fixture is its luminance.
fn value_at(pixels: &[u8], x: u32, y: u32) -> u8 {
    pixels[((y * WIDTH + x) * 4 + 1) as usize]
}

#[test]
fn a_bright_square_throws_light_across_the_flat_wall_it_is_stuck_to() {
    let pixels = present(&lamp_on_a_wall());
    let centre = WIDTH / 2;

    // The square is 8 units across at 99 units, and the view is 152 units wide over 256 pixels —
    // 1.68 pixels a unit — so it covers about 13 of them and everything sampled from 16 out is
    // wall.
    let along = |offset: u32| value_at(&pixels, centre + offset, centre);
    let profile: Vec<u8> = [16, 24, 40, 72, 120].map(along).to_vec();

    // **The emitter is still the emitter.** A blend that had scaled the frame rather than moved
    // light within it would show up here first.
    assert!(
        value_at(&pixels, centre, centre) > profile[0],
        "the square at {} is no brighter than the wall beside it at {}",
        value_at(&pixels, centre, centre),
        profile[0]
    );

    // **And the wall is not flat any more**, which is the whole assertion: one material, one
    // ambient, no bounce and no fog, so every one of these pixels was the same byte before there
    // was a point spread to carry the square's light onto them.
    for pair in profile.windows(2) {
        assert!(
            pair[0] > pair[1],
            "the glow does not fall away: profile {profile:?}"
        );
    }
    assert!(
        profile[4] > 0,
        "the far corner of the wall got none of it: profile {profile:?}"
    );

    // **Symmetric, because a point spread is.** A pyramid that read its neighbours with an offset
    // — the commonest way to get a tent filter wrong — leans the glow to one side, and a lean is
    // invisible in a falling sequence taken on one side alone.
    //
    // Mirrored about the *axis* rather than about a pixel: 256 pixels put the view's centre line
    // between 127 and 128, so `centre + offset` and `centre - 1 - offset` are the pair that stand
    // the same distance from it. Mirroring about pixel 128 instead compares a sample half a pixel
    // nearer the square with one half a pixel further, which on a gradient this steep is two bytes
    // of difference that has nothing to do with the filter.
    for offset in [16, 40, 120] {
        let (right, left) = (
            along(offset),
            value_at(&pixels, centre - 1 - offset, centre),
        );
        assert!(
            right.abs_diff(left) <= 1,
            "the glow leans: {right} to the right of the square against {left} to its left"
        );
    }
}
