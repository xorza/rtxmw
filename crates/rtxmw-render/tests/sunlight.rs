//! The sun's shadow, and the angular diameter that softens it.
//!
//! A real sun is half a degree across, which makes its penumbra narrow: seen from a camera the
//! same distance from a surface as the blocker is, it spans about one pixel. Widening it enough to
//! measure means putting the blocker far away and the camera close — here fifty times closer — and
//! that ratio is why the fixture looks the odd shape it does.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{Instance, Material, Mesh, MeshId, StaticScene, Submesh, Sun};

mod common;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;

/// Distance from the camera to the wall it looks at.
const WALL: f32 = 1000.0;
/// Where the camera stands, a hundred units short of it.
const EYE_X: f32 = WALL - 100.0;
/// How far behind the camera the blocker floats — fifty times the camera's own distance, which is
/// what widens the penumbra from one pixel to about seventy-five.
const BLOCKER_X: f32 = -4000.0;

/// A wall, and a half-plane far behind the camera that shadows its upper half.
///
/// The sun travels along `+X` exactly, so the shadow's edge lands on the wall at `z = 0` — the
/// image's centre row — and everything above it is in shadow.
fn shadowed_wall(angular_radius: f32) -> StaticScene {
    let quad = |x: f32, z: std::ops::Range<f32>| Mesh {
        positions: vec![
            Vec3::new(x, -6000.0, z.start),
            Vec3::new(x, 6000.0, z.start),
            Vec3::new(x, 6000.0, z.end),
            Vec3::new(x, -6000.0, z.end),
        ],
        normals: vec![Vec3::NEG_X; 4],
        uvs: vec![Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y],
        indices: vec![0, 1, 2, 0, 2, 3],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material: 0,
        }],
    };
    let mut scene = common::scene_of(
        &[quad(WALL, -6000.0..6000.0), quad(BLOCKER_X, 0.0..6000.0)],
        &[Material {
            diffuse: Vec3::splat(0.5),
            ..Material::default()
        }],
        &[0, 1].map(|mesh| Instance {
            mesh: MeshId(mesh),
            transform: Affine3A::IDENTITY,
        }),
        &[],
        // No ambient at all: whatever light reaches the wall came from the sun.
        Vec3::ZERO,
    );
    scene.sun = Some(Sun {
        direction: Vec3::X,
        colour: Vec3::splat(2.0),
        angular_radius,
    });
    scene
}

fn trace(scene: &StaticScene) -> Vec<u8> {
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
    // Neither filtering nor bouncing: both would blur the very edge being measured, and the
    // question here is what the shadow ray found.
    renderer.set_denoise_passes(0);
    renderer.set_bounce_samples(0);

    let mut uploader = gpu.uploader();
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            scene,
            &[],
        )
        .expect("scene should load");

    let eye = Vec3::new(EYE_X, 0.0, 0.0);
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

/// Mean red across a row, which on this wall is how much of the sun that row can see.
fn row(pixels: &[u8], y: u32) -> f32 {
    (WIDTH / 4..WIDTH * 3 / 4)
        .map(|x| pixels[((y * WIDTH + x) * 4) as usize] as f32)
        .sum::<f32>()
        / (WIDTH / 2) as f32
}

/// Rows lying between the plateaus rather than on either — the width of the penumbra.
fn penumbra_rows(pixels: &[u8]) -> usize {
    let (dark, lit) = (row(pixels, 8), row(pixels, HEIGHT - 8));
    let span = lit - dark;
    (8..HEIGHT - 8)
        .filter(|&y| {
            let value = row(pixels, y);
            value > dark + span * 0.2 && value < dark + span * 0.8
        })
        .count()
}

#[test]
fn the_suns_disc_is_what_makes_its_shadow_soft() {
    let real = trace(&shadowed_wall(Sun::default_daylight().angular_radius));
    // A sun of no size at all: the same light from the same direction, cast by a point.
    let point = trace(&shadowed_wall(0.0));

    let (dark, lit) = (row(&real, 8), row(&real, HEIGHT - 8));
    println!(
        "shadowed {dark:.1}, lit {lit:.1}, penumbra {} rows against {} for a point sun",
        penumbra_rows(&real),
        penumbra_rows(&point)
    );

    // There is a shadow at all: the top of the wall is dark and the bottom is not. The lit value
    // is the whole chain worked through — albedo 0.5, the Lambertian `1/pi`, a sun of 2.0 straight
    // on — which is 0.318, or 81 of 255.
    assert!(
        (lit - 81.0).abs() < 3.0,
        "the lit wall came out at {lit}, not the 81 it should be"
    );
    assert!(
        dark < 1.0,
        "the shadowed wall came out at {dark}, so nothing blocked the sun"
    );

    // A point sun gives an edge one or two pixels wide — the resolution's limit, not the light's.
    assert!(
        penumbra_rows(&point) <= 2,
        "a sun with no angular size still cast a {}-row penumbra",
        penumbra_rows(&point)
    );

    // The real disc spreads it over dozens. Half a degree across five thousand units is a 46-unit
    // penumbra, and at this framing 153 units of wall cover 256 rows — so about 77 rows of
    // gradient, of which the middle 20% to 80% measured here is 38.
    let rows = penumbra_rows(&real);
    assert!(
        (25..60).contains(&rows),
        "a half-degree sun gave a {rows}-row penumbra, where the geometry says about 38"
    );
}

#[test]
fn a_cell_with_no_sun_is_lit_by_nothing() {
    // How an interior says it has no sky: a black sun contributes nothing, needing no flag.
    let mut scene = shadowed_wall(Sun::default_daylight().angular_radius);
    scene.sun.as_mut().expect("the fixture has a sun").colour = Vec3::ZERO;
    let pixels = trace(&scene);
    assert_eq!(row(&pixels, HEIGHT - 8), 0.0);
}
