//! Traces primary rays into an offscreen image and asserts on the pixels.
//!
//! This is where everything upstream is finally checked against reality. A wrong build-range
//! offset, a mirrored instance transform or a transposed projection all survive M3c and M3d
//! silently — they only become visible once a ray has to find the triangle. The scenes here are
//! small enough that where a hit lands can be worked out by hand.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_esm::EsmReader;
use rtxmw_gpu::{TestGpu, image_barrier, memory_barrier};
use rtxmw_render::{FrameConstants, GeometryBuffers, SceneAcceleration, VisibilityPass};
use rtxmw_scene::{Instance, Mesh, MeshId, ModelIndex, StaticScene};
use rtxmw_vfs::{DATA_DIR_VAR, morrowind_archives, morrowind_data_dir};

const CELL: &str = "Seyda Neen, Census and Excise Office";
const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const FOV_Y: f32 = 75.0;

/// Whether a pixel is a barycentric hit rather than the miss colour.
///
/// Barycentrics sum to one, so a hit's channels total about 255 where the background totals 57.
/// The gap is wide enough that half-float quantisation cannot blur the two together.
fn is_hit(pixel: &[u8]) -> bool {
    let total = pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32;
    total > 150
}

/// An axis-aligned quad in the world's YZ plane, facing back down -X toward a camera at the origin.
fn wall(x: f32, y: std::ops::Range<f32>, z: std::ops::Range<f32>) -> Mesh {
    Mesh {
        positions: vec![
            Vec3::new(x, y.start, z.start),
            Vec3::new(x, y.end, z.start),
            Vec3::new(x, y.end, z.end),
            Vec3::new(x, y.start, z.end),
        ],
        normals: vec![Vec3::NEG_X; 4],
        uvs: vec![Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

/// Traces `instances` from `eye` looking along `forward` and returns the image as 8-bit RGBA.
fn trace(meshes: &[Mesh], instances: &[Instance], eye: Vec3, forward: Vec3) -> Vec<u8> {
    let gpu = TestGpu::shared();
    let mut uploader = gpu.uploader();

    let geometry =
        GeometryBuffers::upload(gpu.memory(), &mut uploader, meshes).expect("upload failed");
    let acceleration = SceneAcceleration::build(
        gpu.device(),
        &mut uploader,
        gpu.physical().limits(),
        &geometry,
        instances,
    )
    .expect("build failed");

    let target = gpu
        .create_target(WIDTH, HEIGHT, vk::Format::R16G16B16A16_SFLOAT)
        .expect("could not create target");
    let mut pass = VisibilityPass::new(gpu.device()).expect("pipeline failed");
    pass.bind(acceleration.tlas(), target.image());

    let view = glam::camera::rh::view::look_to_mat4(eye, forward, Vec3::Z);
    let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
        FOV_Y.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
        0.05,
    );
    let constants = FrameConstants::new(view, projection, eye);

    let image = target.image().raw();
    let extent = target.extent();
    uploader
        .submit_and_wait(|device, cmd| {
            // SAFETY: the command buffer is recording, and the image and pipeline are alive.
            unsafe {
                image_barrier::transition(
                    device,
                    cmd,
                    image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                );
                pass.record(cmd, extent, &constants);
                memory_barrier::full(device, cmd);
            }
        })
        .expect("dispatch failed");

    let pixels = target
        .read_rgba8(&mut uploader, vk::ImageLayout::GENERAL)
        .expect("readback failed");
    drop(uploader);
    gpu.assert_no_validation_errors();
    pixels
}

/// The pixel at `(x, y)`, row-major from the top.
fn at(pixels: &[u8], x: u32, y: u32) -> &[u8] {
    let index = ((y * WIDTH + x) * 4) as usize;
    &pixels[index..index + 4]
}

/// How many pixels in the image are hits.
fn hit_count(pixels: &[u8]) -> usize {
    pixels.chunks_exact(4).filter(|p| is_hit(p)).count()
}

#[test]
fn a_wall_ahead_covers_the_centre_and_leaves_the_corners_empty() {
    // The camera sits at the origin looking east along +X. The wall is 100 units away and spans 50
    // units either side of the axis. At 75 degrees vertical field of view the half-height of the
    // view at that distance is 100 * tan(37.5 deg) = 76.7 units, so the wall covers the middle of
    // the image and falls short of the corners — which is what makes both assertions meaningful.
    let meshes = [wall(100.0, -50.0..50.0, -50.0..50.0)];
    let instances = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::IDENTITY,
    }];
    let pixels = trace(&meshes, &instances, Vec3::ZERO, Vec3::X);

    assert!(
        is_hit(at(&pixels, WIDTH / 2, HEIGHT / 2)),
        "the centre ray missed a wall directly ahead: {:?}",
        at(&pixels, WIDTH / 2, HEIGHT / 2)
    );
    for (x, y) in [
        (0, 0),
        (WIDTH - 1, 0),
        (0, HEIGHT - 1),
        (WIDTH - 1, HEIGHT - 1),
    ] {
        assert!(
            !is_hit(at(&pixels, x, y)),
            "corner ({x}, {y}) hit something outside the wall: {:?}",
            at(&pixels, x, y)
        );
    }

    // 100x100 units of wall inside a 153.4-unit-tall view is a little under half the height, and
    // the same in width since the image is square. Roughly a quarter of the pixels, then — loose
    // bounds, because the point is that it is a bounded patch rather than everything or nothing.
    let hits = hit_count(&pixels);
    let total = (WIDTH * HEIGHT) as usize;
    assert!(
        hits > total / 8 && hits < total / 2,
        "{hits} of {total} pixels hit, which is not a wall-sized patch"
    );
}

#[test]
fn geometry_to_the_north_appears_on_the_left_of_the_image() {
    // The handedness chain end to end. Looking east along +X with +Z up, the camera's right is
    // forward x up = X x Z = -Y, so world +Y (north) is to the camera's *left*. Vulkan's NDC does
    // not flip X, so left is low pixel x.
    //
    // Getting this wrong is the mirroring bug that survives every earlier test: a transposed
    // instance transform, a projection built for the wrong NDC convention, or an unprojection that
    // negated x would all still put a wall on screen — just on the wrong side.
    let meshes = [wall(100.0, 20.0..90.0, -40.0..40.0)];
    let instances = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::IDENTITY,
    }];
    let pixels = trace(&meshes, &instances, Vec3::ZERO, Vec3::X);

    let mut left = 0usize;
    let mut right = 0usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if is_hit(at(&pixels, x, y)) {
                if x < WIDTH / 2 {
                    left += 1;
                } else {
                    right += 1;
                }
            }
        }
    }

    assert!(left > 0, "nothing was hit at all");
    assert_eq!(right, 0, "{right} pixels hit on the right of the image");
}

#[test]
fn geometry_above_the_camera_appears_at_the_top_of_the_image() {
    // The other half of the handedness check. Vulkan's NDC is Y-down and the projection flips Y to
    // match, so world +Z (up) must land at low pixel y. A projection built for OpenGL's Y-up NDC
    // would put it at the bottom and nothing else would notice.
    let meshes = [wall(100.0, -40.0..40.0, 20.0..90.0)];
    let instances = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::IDENTITY,
    }];
    let pixels = trace(&meshes, &instances, Vec3::ZERO, Vec3::X);

    let mut top = 0usize;
    let mut bottom = 0usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if is_hit(at(&pixels, x, y)) {
                if y < HEIGHT / 2 {
                    top += 1;
                } else {
                    bottom += 1;
                }
            }
        }
    }

    assert!(top > 0, "nothing was hit at all");
    assert_eq!(bottom, 0, "{bottom} pixels hit in the bottom half");
}

#[test]
fn separate_meshes_both_render_at_their_own_offsets() {
    // The decisive test for the build-range triple. Two meshes in one buffer means the second has a
    // nonzero `first_vertex` and `primitive_offset`; if either is wrong, its structure is built from
    // the first mesh's vertices and both walls land in the same place. Placing them on opposite
    // sides means that failure shows up as one side going empty.
    let meshes = [
        wall(100.0, 20.0..90.0, -40.0..40.0),
        wall(100.0, -90.0..-20.0, -40.0..40.0),
    ];
    let instances = [
        Instance {
            mesh: MeshId(0),
            transform: Affine3A::IDENTITY,
        },
        Instance {
            mesh: MeshId(1),
            transform: Affine3A::IDENTITY,
        },
    ];
    let pixels = trace(&meshes, &instances, Vec3::ZERO, Vec3::X);

    let mut left = 0usize;
    let mut right = 0usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if is_hit(at(&pixels, x, y)) {
                if x < WIDTH / 2 {
                    left += 1;
                } else {
                    right += 1;
                }
            }
        }
    }

    assert!(left > 0, "the north wall (mesh 0) is missing");
    assert!(right > 0, "the south wall (mesh 1) is missing");
    // Mirror images of each other, so within a few pixels of the same area.
    let difference = left.abs_diff(right);
    assert!(
        difference * 20 < left + right,
        "{left} left against {right} right — the two meshes are not the same size on screen"
    );
}

#[test]
fn an_instance_transform_moves_the_geometry_it_places() {
    // The same mesh placed twice: once ahead and once behind. Only the one ahead may be visible, so
    // the instance transform has to be applied — a build that ignored it would draw both at the
    // origin and the hit count would not change between this and the single-instance case.
    let meshes = [wall(0.0, -50.0..50.0, -50.0..50.0)];
    let instances = [
        Instance {
            mesh: MeshId(0),
            transform: Affine3A::from_translation(Vec3::new(100.0, 0.0, 0.0)),
        },
        Instance {
            mesh: MeshId(0),
            transform: Affine3A::from_translation(Vec3::new(-100.0, 0.0, 0.0)),
        },
    ];
    let pixels = trace(&meshes, &instances, Vec3::ZERO, Vec3::X);

    assert!(is_hit(at(&pixels, WIDTH / 2, HEIGHT / 2)));
    let both = hit_count(&pixels);

    // With only the wall behind the camera, nothing should be visible at all.
    let behind = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::from_translation(Vec3::new(-100.0, 0.0, 0.0)),
    }];
    let pixels = trace(&meshes, &behind, Vec3::ZERO, Vec3::X);
    assert_eq!(
        hit_count(&pixels),
        0,
        "geometry behind the camera was drawn, so the transform was ignored"
    );
    assert!(both > 0);
}

#[test]
fn a_real_interior_traces_to_a_recognisable_image() {
    let Some(data) = morrowind_data_dir() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };
    let vfs = morrowind_archives().expect("the game is available");
    let bytes = std::fs::read(data.join("Morrowind.esm")).expect("Morrowind.esm should read");
    let esm = EsmReader::new(&bytes).expect("should parse");
    let models = ModelIndex::build(&esm).expect("model index should build");
    let scene = StaticScene::load_interior(&esm, &models, &vfs, CELL).expect("cell should load");

    // The centre of the cell's own geometry, which for this office lands inside the larger room.
    // There is no general "stand here" rule without tracing probe rays, so this is a fixture choice
    // that happens to work for this cell rather than something to reuse blindly.
    let eye = scene
        .bounds()
        .expect("a furnished cell has geometry")
        .centre();

    let pixels = trace(&scene.meshes, &scene.instances, eye, Vec3::X);
    let hits = hit_count(&pixels);
    let total = (WIDTH * HEIGHT) as usize;

    // "Recognisable" is really a claim about structure, so count how many distinct colours the hits
    // came back as. Barycentrics vary continuously across a triangle, so a view of real geometry
    // produces hundreds of shades; a single wrongly-placed triangle filling the view would produce
    // a smooth ramp of far fewer, and a broken traversal none at all.
    let mut shades: std::collections::HashSet<[u8; 3]> = std::collections::HashSet::new();
    for pixel in pixels.chunks_exact(4).filter(|p| is_hit(p)) {
        shades.insert([pixel[0], pixel[1], pixel[2]]);
    }

    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("seyda_neen.png");
    rtxmw_gpu::golden::write_png(&path, &pixels, WIDTH, HEIGHT);

    println!(
        "{CELL}: {hits} of {total} pixels hit ({:.0}%), {} distinct shades, from {eye:?}\n  wrote {}",
        hits as f64 / total as f64 * 100.0,
        shades.len(),
        path.display()
    );

    // Standing inside an enclosed room, most rays should land on a wall, the floor or furniture.
    // A near-empty image would mean the rays never found the geometry.
    assert!(
        hits > total * 2 / 5,
        "only {hits} of {total} pixels hit from inside the room"
    );
    assert!(
        shades.len() > 200,
        "only {} distinct shades — the view is not resolving separate triangles",
        shades.len()
    );
}
