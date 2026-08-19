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
use rtxmw_scene::{CellId, Instance, Material, Mesh, MeshId, StaticScene, Submesh, Sun};

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
        // Wound so the triangles' own plane agrees with those normals — `cross(p1 - p0, p2 - p0)`
        // has to come out along -X too. It did not, and nothing noticed while shading consulted
        // only the authored normal; now that a hit is shaded as whichever face it met, a surface
        // whose winding disagrees with its normals is a surface lit from the wrong side.
        indices: vec![0, 2, 1, 0, 3, 2],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material: 0,
            thin: false,
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

/// Traces [`shadowed_wall`] from the spot that fixture is built around.
fn trace_wall(scene: &StaticScene) -> Vec<u8> {
    trace(scene, Vec3::new(EYE_X, 0.0, 0.0), Vec3::X)
}

fn trace(scene: &StaticScene, eye: Vec3, forward: Vec3) -> Vec<u8> {
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
    // And no fog, which scatters light into the shadow and would soften an edge these tests need
    // sharp — the whole measurement here is how wide the sun's own penumbra is.
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

    // Up is world up unless the camera is looking along it, which would leave the view undefined.
    let up = if forward.normalize().z.abs() > 0.9 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let view = glam::camera::rh::view::look_to_mat4(eye, forward, up);
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
    let real = trace_wall(&shadowed_wall(Sun::REAL_ANGULAR_RADIUS));
    // A sun of no size at all: the same light from the same direction, cast by a point.
    let point = trace_wall(&shadowed_wall(0.0));

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
    let mut scene = shadowed_wall(Sun::REAL_ANGULAR_RADIUS);
    scene.sun.as_mut().expect("the fixture has a sun").colour = Vec3::ZERO;
    let pixels = trace_wall(&scene);
    assert_eq!(row(&pixels, HEIGHT - 8), 0.0);
}

#[test]
fn the_back_of_a_sunlit_plane_is_dark_unless_it_is_a_sheet() {
    // **Morrowind hangs single-sided planes everywhere** — every tapestry, sail and leaf card — and
    // draws them from both sides. What the far side of one shows depends on what it is: the back of
    // something solid is in shade, while cloth passes a share of the light through.
    //
    // Both halves matter, and each guards a different thing.
    //
    // The dark half guards the *shading normal*, which faces whichever way the ray came from. Left
    // as the vertices authored it, the back of this plane reports the light landing on its front —
    // a surface lit through its own body. On a tree that reads as dark dust: a canopy is thousands
    // of cards packed below a pixel apiece, and neighbouring pixels landing on cards wound
    // oppositely came back at opposite brightnesses.
    //
    // The lit half guards the *ray offset*. Every ray leaving a surface is pushed off it along the
    // triangle's own plane, and that push has to go to the side the ray is travelling. Sent to the
    // viewer's side instead, the shadow ray from behind the cloth sets off toward the sun, meets
    // the cloth it started behind, and reports shadow — so the transmitted light never arrives.
    let plane = |thin: bool| Mesh {
        positions: vec![
            Vec3::new(-500.0, -500.0, 0.0),
            Vec3::new(500.0, -500.0, 0.0),
            Vec3::new(500.0, 500.0, 0.0),
            Vec3::new(-500.0, 500.0, 0.0),
        ],
        // Facing up, toward the sun, while the camera looks at it from underneath.
        normals: vec![Vec3::Z; 4],
        uvs: vec![Vec2::ZERO; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material: 0,
            thin,
        }],
    };

    let seen_from_below = |thin: bool| {
        let mut scene = common::scene_of(
            &[plane(thin)],
            &[Material {
                diffuse: Vec3::splat(0.5),
                ..Material::default()
            }],
            &[Instance {
                mesh: MeshId(0),
                transform: Affine3A::IDENTITY,
            }],
            &[],
            // No ambient: anything the camera sees came from the sun, through the plane.
            Vec3::ZERO,
        );
        scene.sun = Some(Sun {
            direction: Vec3::NEG_Z,
            colour: Vec3::splat(4.0),
            angular_radius: 0.0,
        });
        // Underneath, looking up at the back of it.
        let pixels = trace(&scene, Vec3::new(0.0, 0.0, -200.0), Vec3::Z);
        let centre = ((HEIGHT / 2) * WIDTH + WIDTH / 2) as usize * 4;
        pixels[centre + 1] as f32 / 255.0
    };

    // Albedo 0.5 against a sun of 4.0 straight on, through the Lambertian `1/pi`, is 0.64 — and
    // half of that comes through a sheet.
    let sheet = seen_from_below(true);
    assert!(
        (sheet - 0.32).abs() < 0.02,
        "a sheet backlit by the sun should transmit half of 0.64, got {sheet}"
    );
    let solid = seen_from_below(false);
    assert_eq!(
        solid, 0.0,
        "the back of a solid plane was lit through its own body"
    );
}

#[test]
fn a_surface_whose_normals_lean_shades_evenly_at_a_grazing_angle() {
    // A rug on a floor, seen from across the room. Its vertex normals lean, as Morrowind's do
    // wherever a flat thing was authored to catch the light — and at a grazing angle the leaning
    // carries some of them past the viewer, so that they point away from a face the camera is
    // looking straight at.
    //
    // **That is why the side a hit is shaded on is decided by the triangle's plane.** Decided by
    // the normal instead, the surface splits: the pixels whose interpolated normal happens to lean
    // away get shaded as the underside of the rug and go black, and the seam between the two halves
    // slides across the floor as the camera moves.
    let lean = 0.6f32;
    let rug = Mesh {
        positions: vec![
            Vec3::new(-700.0, -500.0, 0.0),
            Vec3::new(700.0, -500.0, 0.0),
            Vec3::new(700.0, 2500.0, 0.0),
            Vec3::new(-700.0, 2500.0, 0.0),
        ],
        // Leaning across the quad, from pointing left at one edge to pointing right at the other.
        normals: vec![
            Vec3::new(-lean, 0.0, 1.0 - lean).normalize(),
            Vec3::new(lean, 0.0, 1.0 - lean).normalize(),
            Vec3::new(lean, 0.0, 1.0 - lean).normalize(),
            Vec3::new(-lean, 0.0, 1.0 - lean).normalize(),
        ],
        uvs: vec![Vec2::ZERO; 4],
        // Wound so the plane points up, which is the side the camera is on.
        indices: vec![0, 1, 2, 0, 2, 3],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material: 0,
            thin: false,
        }],
    };
    let mut scene = common::scene_of(
        &[rug],
        &[Material {
            diffuse: Vec3::splat(0.5),
            ..Material::default()
        }],
        &[Instance {
            mesh: MeshId(0),
            transform: Affine3A::IDENTITY,
        }],
        &[],
        Vec3::ZERO,
    );
    // Straight down, so every normal that still points upward is lit by the same amount and the
    // only thing that can vary across the rug is which way the shading decided it faces.
    scene.sun = Some(Sun {
        direction: Vec3::NEG_Z,
        colour: Vec3::splat(4.0),
        angular_radius: 0.0,
    });

    // Low and off to one side: the ray reaches the far edge of the rug at a few degrees above it.
    let pixels = trace(
        &scene,
        Vec3::new(0.0, -450.0, 60.0),
        Vec3::new(0.15, 0.96, -0.12),
    );

    // A row just below the horizon, which is where the rug is most nearly edge-on and so where a
    // leaning normal is most likely to have been carried past the viewer. Further down the frame
    // the rays meet the floor too steeply for any of them to flip, and the seam is out of shot.
    let row = HEIGHT / 2 + 12;
    //
    // Every pixel of it, not the ones that happened to hit: the check below reads neighbours as
    // neighbours, and dropping a miss from the middle of the row would quietly make two pixels
    // either side of a hole adjacent.
    let lit: Vec<f32> = (0..WIDTH)
        .map(|x| {
            let at = ((row * WIDTH + x) * 4) as usize;
            assert!(pixels[at + 3] > 128, "pixel {x} of the row missed the rug");
            pixels[at + 1] as f32 / 255.0
        })
        .collect();

    // The leaning normals make the row legitimately brighter in the middle than at the edges, so
    // what is asserted is not flatness but *smoothness*: the lean turns gradually, and so must the
    // shading. Shading a hit as the underside of the rug does not turn gradually — it drops to
    // black at whichever pixel the interpolated normal crossed the viewer, which is a cliff.
    let mut steepest = 0.0f32;
    for pair in lit.windows(2) {
        steepest = steepest.max((pair[1] - pair[0]).abs());
    }
    let brightest = lit.iter().copied().fold(0.0f32, f32::max);
    let darkest = lit.iter().copied().fold(1.0f32, f32::min);
    assert!(brightest > 0.2, "the rug is not lit at all: {brightest}");
    // The fixture has to be leaning enough to be worth the test: a row of one value would pass the
    // smoothness check without ever exercising it.
    assert!(
        brightest - darkest > 0.03,
        "the row barely varies ({darkest} to {brightest}), so the normals are not leaning"
    );
    assert!(
        steepest < 0.05,
        "the shading steps by {steepest} between neighbouring pixels, from a row spanning \
         {darkest} to {brightest}"
    );
}
