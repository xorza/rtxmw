//! Where the shader says each pixel's surface was on the previous frame's screen.
//!
//! The arithmetic is asserted in `visibility_pass.rs`, against a double-precision projection. What
//! is left for a real trace is that the shader applies it to the surface the pixel actually hit —
//! at the depth it was found, and only where something was found at all.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{CellId, Instance, Material, Mesh, MeshId, StaticScene, Submesh};

mod common;

const SIZE: u32 = 64;

/// Two walls facing the camera, one at twice the distance of the other.
///
/// Depth is the whole point: a camera that steps sideways moves the near one across the screen
/// twice as far as the far one, and a motion vector that ignored the hit distance would move them
/// together.
fn steps() -> StaticScene {
    let wall = |x: f32, north: std::ops::Range<f32>, half: f32| Mesh {
        positions: vec![
            Vec3::new(x, north.start, -half),
            Vec3::new(x, north.end, -half),
            Vec3::new(x, north.end, half),
            Vec3::new(x, north.start, half),
        ],
        normals: vec![Vec3::NEG_X; 4],
        uvs: vec![Vec2::ZERO; 4],
        // Wound so the plane faces -X, back toward the camera.
        indices: vec![0, 2, 1, 0, 3, 2],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material: 0,
            thin: false,
        }],
    };
    // The camera looks east, so north is screen-left: a near wall covering only the northern half
    // fills the left of the view, and the far one stands behind it across the whole width. Both are
    // visible at once, so a single frame carries both depths.
    common::scene_of(
        &[
            wall(200.0, 0.0..400.0, 400.0),
            wall(400.0, -800.0..800.0, 800.0),
        ],
        &[Material::default()],
        &[0, 1].map(|mesh| Instance {
            mesh: MeshId(mesh),
            transform: Affine3A::IDENTITY,
        }),
        &[],
        Vec3::ONE,
    )
}

/// Renders `eyes` in order through one renderer and returns the last frame's motion vectors.
///
/// One renderer across both, because the previous frame is state it keeps: handing each frame its
/// own would make every frame the first one, which is exactly the case that reports no motion.
fn motion_after(eyes: &[Vec3]) -> Vec<f32> {
    let gpu = TestGpu::shared();
    let mut renderer = SceneRenderer::new(
        gpu.device(),
        gpu.physical(),
        gpu.memory(),
        vk::Extent2D {
            width: SIZE,
            height: SIZE,
        },
    )
    .expect("renderer should build");
    renderer.set_bounce_samples(0);
    renderer.set_denoise_passes(0);

    let mut uploader = gpu.uploader();
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            CellId::Interior("motion".to_owned()),
            &steps(),
            &[],
        )
        .expect("scene should load");

    let projection =
        glam::camera::rh::proj::vulkan::perspective_infinite_reverse(90f32.to_radians(), 1.0, 0.05);
    for eye in eyes {
        let view = glam::camera::rh::view::look_to_mat4(*eye, Vec3::X, Vec3::Z);
        let constants = renderer.frame_constants(view, projection, *eye);
        renderer
            .render_once(&mut uploader, &constants)
            .expect("frame should render");
    }

    let motion = readback::image_to_f32(&mut uploader, renderer.motion(), vk::ImageLayout::GENERAL)
        .expect("readback should succeed");
    drop(uploader);
    gpu.assert_no_validation_errors();
    motion
}

/// The motion vector at a pixel.
fn at(motion: &[f32], x: u32, y: u32) -> Vec2 {
    let index = ((y * SIZE + x) * 2) as usize;
    Vec2::new(motion[index], motion[index + 1])
}

#[test]
fn a_still_camera_reports_no_motion_and_a_stepping_one_reports_it_by_depth() {
    // The first frame has no history at all, and neither does a frame whose camera did not move;
    // both have to come back as zero rather than as whatever the allocator left in the image.
    let first = motion_after(&[Vec3::ZERO]);
    let still = motion_after(&[Vec3::ZERO, Vec3::ZERO]);
    for (name, frame) in [("the first frame", &first), ("a still camera", &still)] {
        let worst = (0..SIZE * SIZE)
            .map(|i| at(frame, i % SIZE, i / SIZE).length())
            .fold(0.0f32, f32::max);
        assert!(worst < 1.0e-3, "{name} reported {worst} pixels of motion");
    }

    // Now a step north. The camera looks east, so north is screen-left: the camera slides left and
    // a fixed surface therefore sits further *right* than it did. The vector points back to the
    // history, so it is negative — this is the sign a temporal filter adds to a pixel coordinate to
    // find last frame's sample, and getting it backwards smears the image the wrong way.
    let stepped = motion_after(&[Vec3::ZERO, Vec3::new(0.0, 4.0, 0.0)]);

    // The near wall fills the left of the image — north is screen-left — and the far one the right.
    let near = at(&stepped, SIZE / 4, SIZE / 2);
    let far = at(&stepped, 3 * SIZE / 4, SIZE / 2);
    println!("near wall {near:?}, far wall {far:?}");

    // A 90-degree view 64 pixels wide puts the screen edge at one unit per unit of depth, so a
    // pixel is 2/64 of the depth. Stepping 4 units north moves a surface 200 away by
    // 4 / (200 * 2 / 64) = 0.64 pixels, and one 400 away by half that.
    let expected_near = -4.0 / (200.0 * 2.0 / SIZE as f32);
    assert!(
        (near.x - expected_near).abs() < 0.05,
        "the near wall moved {near:?}, not {expected_near} pixels"
    );
    assert!(
        (far.x - expected_near / 2.0).abs() < 0.05,
        "the far wall moved {far:?}, not {} pixels",
        expected_near / 2.0
    );
    // Straight sideways: a step north tilts nothing.
    assert!(near.y.abs() < 0.01 && far.y.abs() < 0.01);
}
