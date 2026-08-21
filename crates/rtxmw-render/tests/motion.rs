//! Where the shader says each pixel's surface was on the previous frame's screen.
//!
//! The arithmetic is asserted in `visibility_pass.rs`, against a double-precision projection. What
//! is left for a real trace is that the shader applies it to the surface the pixel actually hit —
//! at the depth it was found, and only where something was found at all.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{CellId, Instance, Material, Mesh, MeshId, StaticScene, Submesh, TextureId};
use rtxmw_texture::{Texture, TextureFormat};

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
fn motion_after(eyes: &[Vec3], jitter: bool) -> Vec<f32> {
    motion_with(&steps(), &[], eyes, jitter, |_| {})
}

/// As [`motion_after`], calling `between` on the renderer after every frame but the last.
fn motion_with(
    scene: &StaticScene,
    textures: &[Option<Texture>],
    eyes: &[Vec3],
    jitter: bool,
    mut between: impl FnMut(&mut SceneRenderer),
) -> Vec<f32> {
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
    renderer.set_jitter(jitter);

    let mut uploader = gpu.uploader();
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            CellId::Interior("motion".to_owned()),
            scene,
            textures,
        )
        .expect("scene should load");

    let projection =
        glam::camera::rh::proj::vulkan::perspective_infinite_reverse(90f32.to_radians(), 1.0, 0.05);
    for (index, eye) in eyes.iter().enumerate() {
        let view = glam::camera::rh::view::look_to_mat4(*eye, Vec3::X, Vec3::Z);
        let constants = renderer.frame_constants(view, projection, *eye);
        renderer
            .render_once(&mut uploader, &constants)
            .expect("frame should render");
        if index + 1 < eyes.len() {
            between(&mut renderer);
        }
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
fn forgetting_the_history_makes_the_next_frame_a_first_frame() {
    // **The reset an upscaler needs**, and the case motion vectors cannot describe: the camera has
    // not moved *through* the world, it has arrived somewhere else — walking a door, or a cell
    // changing under it. Every vector into the previous frame is then a lie about a surface that
    // was not on screen, and a filter that believes them smears the old place over the new one.
    let step = [Vec3::ZERO, Vec3::new(0.0, 4.0, 0.0)];

    // The same two frames, once remembering and once not.
    let remembered = motion_after(&step, false);
    let forgotten = motion_with(&steps(), &[], &step, false, |renderer| {
        assert!(
            renderer.has_history(),
            "a frame that has been rendered leaves history behind"
        );
        renderer.forget_history();
        assert!(
            !renderer.has_history(),
            "the history survived being forgotten"
        );
    });

    let moved = at(&remembered, SIZE / 4, SIZE / 2);
    let reset = at(&forgotten, SIZE / 4, SIZE / 2);
    println!("stepping reports {moved:?} remembering and {reset:?} after a reset");
    assert!(
        moved.length() > 0.5,
        "the fixture reports no motion to forget: {moved:?}"
    );
    assert!(
        reset.length() < 1.0e-3,
        "a frame whose history was dropped still reported {reset:?}; it has to reproject onto \
         itself, which is what zero means"
    );
}

#[test]
fn jitter_moves_the_ray_and_leaves_the_motion_vector_alone() {
    // **The convention this pins.** The jitter is applied to the pixel's coordinate rather than to
    // the projection, so the hit it produces projects back through that same matrix to exactly that
    // coordinate — a motion vector measured against it therefore carries no jitter, and a camera
    // that did not move reports nothing however far inside its pixel the ray went.
    //
    // Measuring against the pixel *centre* instead is the plausible alternative, and it is wrong in
    // a way only this shows: it reports the frame's own jitter as motion, up to half a pixel of
    // history fetched from the wrong place every frame the camera is still.
    let still = motion_after(&[Vec3::ZERO, Vec3::ZERO], true);
    let worst = (0..SIZE * SIZE)
        .map(|i| at(&still, i % SIZE, i / SIZE).length())
        .fold(0.0f32, f32::max);
    println!("a still camera with jitter on reports {worst} pixels of motion");
    assert!(
        worst < 1.0e-3,
        "a still camera reported {worst} pixels of motion with jitter on; the jitter is leaking \
         into the vector"
    );

    // And the jitter is genuinely on, or the assertion above is about nothing: two frames of it
    // must trace different sub-pixel points and so land on different depths at a slope.
    let stepped = motion_after(&[Vec3::ZERO, Vec3::new(0.0, 4.0, 0.0)], true);
    let plain = motion_after(&[Vec3::ZERO, Vec3::new(0.0, 4.0, 0.0)], false);
    let moved = at(&stepped, SIZE / 4, SIZE / 2);
    let unmoved = at(&plain, SIZE / 4, SIZE / 2);
    println!("stepping reports {moved:?} jittered and {unmoved:?} not");
    assert!(
        (moved - unmoved).length() < 0.05,
        "jitter changed a real motion vector from {unmoved:?} to {moved:?}"
    );
}

#[test]
fn a_still_camera_reports_no_motion_and_a_stepping_one_reports_it_by_depth() {
    // The first frame has no history at all, and neither does a frame whose camera did not move;
    // both have to come back as zero rather than as whatever the allocator left in the image.
    let first = motion_after(&[Vec3::ZERO], false);
    let still = motion_after(&[Vec3::ZERO, Vec3::ZERO], false);
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
    let stepped = motion_after(&[Vec3::ZERO, Vec3::new(0.0, 4.0, 0.0)], false);

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

/// How far the wall below slides its texture each second, and how long a frame of it lasts.
///
/// One whole texture a second over a twentieth of a second, which is `MAX_ELAPSED` — the most a
/// frame's clock is allowed to carry for this.
const SCROLL: f32 = 1.0;
const FRAME: f32 = 0.05;

/// How wide the wall is, in world units, across one turn of its texture.
const WALL: f32 = 400.0;

/// One wall filling the view, with its texture sliding across it.
fn sliding_wall() -> StaticScene {
    let mesh = Mesh {
        positions: vec![
            Vec3::new(200.0, -0.5 * WALL, -0.5 * WALL),
            Vec3::new(200.0, 0.5 * WALL, -0.5 * WALL),
            Vec3::new(200.0, 0.5 * WALL, 0.5 * WALL),
            Vec3::new(200.0, -0.5 * WALL, 0.5 * WALL),
        ],
        normals: vec![Vec3::NEG_X; 4],
        // `u` runs the wall's width, so a slide along it is a slide across the screen.
        uvs: vec![Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y],
        indices: vec![0, 2, 1, 0, 3, 2],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material: 0,
            thin: false,
        }],
    };
    common::scene_of(
        &[mesh],
        // **Textured, because an untextured surface has nothing to slide.** The vector below does
        // not depend on what the picture is, only on there being one.
        &[Material {
            base_colour: Some(TextureId(0)),
            scroll: Vec2::new(SCROLL, 0.0),
            ..Material::default()
        }],
        &[Instance {
            mesh: MeshId(0),
            transform: Affine3A::IDENTITY,
        }],
        &[],
        Vec3::ONE,
    )
}

#[test]
fn a_texture_sliding_across_a_still_wall_still_moves() {
    // **The camera does not move between these two frames**, so every vector here would be exactly
    // zero if a motion vector described only the geometry. It cannot: what an upscaler reprojects
    // is what it can see move, and on a sheet with a texture running over it that is the texture.
    // Told nothing moved, Ray Reconstruction blends a second of frames that each hold the picture
    // somewhere else — which is Vivec's drains sharpening whenever the camera does and mushing over
    // the moment it stops.
    let still = [Vec3::ZERO, Vec3::ZERO];
    let plain = Some(Texture::from_pixels(
        TextureFormat::Rgba8,
        2,
        2,
        vec![255; 2 * 2 * 4],
    ));
    let motion = motion_with(&sliding_wall(), &[plain], &still, false, |renderer| {
        renderer.set_time(FRAME);
    });
    let middle = at(&motion, SIZE / 2, SIZE / 2);

    // **How far, worked through.** The wall spans 400 units across one turn of its texture, so a
    // scroll of one a second carries a texel `400 * SCROLL * FRAME` = 20 units along it in a frame.
    // The view is 90 degrees and the wall stands 200 off, so it spans 400 units across 64 pixels —
    // an eighth of a pixel per unit, and 20 units is 2.5 of them.
    let expected = WALL * SCROLL * FRAME * (SIZE as f32 / WALL);
    assert!(
        (middle.x.abs() - expected).abs() < 0.4,
        "a frame of sliding moved the history {} pixels across, not {expected}",
        middle.x.abs()
    );
    // Along the slide and not across it: `u` runs the wall's width, so the reprojection is sideways.
    assert!(
        middle.y.abs() < 0.4,
        "the history moved {} pixels up the screen, which nothing asked it to",
        middle.y.abs()
    );
}
