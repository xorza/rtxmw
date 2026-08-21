//! A texture walked across the surface it is painted on, asserted in pixels.
//!
//! **The whole of how Morrowind draws a moving fluid.** Vivec's water and Red Mountain's lava are
//! flat sheets with a `NiUVController` sliding the texture over them; nothing about the geometry
//! moves. What that has to come out as is a picture that shifts by a known amount when the clock
//! does, which is a thing a test can measure exactly.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{CellId, Instance, Material, Mesh, MeshId, StaticScene, Submesh, TextureId};
use rtxmw_texture::{Texture, TextureFormat};

mod common;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;

/// How far the texture slides each second, in texture coordinates.
///
/// A tenth of the texture a second, over a wall whose `u` runs nought to one across the *visible*
/// width — so after one second the picture has moved a tenth of the frame, which is 25.6 pixels of
/// 256 and is what the assertion counts. Small enough that the edge stays in the middle half of the
/// frame, where nothing else can be mistaken for it.
const SCROLL: f32 = 0.1;

/// Half the width the view covers at the wall, in world units.
///
/// The wall stands 100 units off and the view is 75 degrees, so it spans `100 * tan(37.5)` either
/// side of the axis. The wall itself is far bigger than that — it has to fill the corners too — and
/// its `u` is stretched to run nought to one across exactly this, so a texture coordinate and a
/// fraction of the frame are the same number.
const HALF_VIEW: f32 = 76.733;

/// Half black and half white, so the picture has exactly one edge to find.
///
/// Sixteen texels across a 256-pixel frame is a magnification of sixteen, so the sampler is reading
/// the top level and the edge stays an edge.
fn halves() -> Texture {
    let mut pixels = vec![0u8; 16 * 16 * 4];
    for row in 0..16 {
        for column in 8..16 {
            pixels[(row * 16 + column) * 4..][..4].copy_from_slice(&[255; 4]);
        }
    }
    Texture::from_pixels(TextureFormat::Rgba8, 16, 16, pixels)
}

/// A wall filling the view, textured across its whole width, sliding at [`SCROLL`].
fn sliding_wall() -> StaticScene {
    let reach = 400.0;
    // Nought to one across the view, which past the view's edge runs on past nought and one.
    let across = |y: f32| 0.5 + y / (2.0 * HALF_VIEW);
    let wall = Mesh {
        positions: vec![
            Vec3::new(100.0, -reach, -reach),
            Vec3::new(100.0, reach, -reach),
            Vec3::new(100.0, reach, reach),
            Vec3::new(100.0, -reach, reach),
        ],
        normals: vec![Vec3::NEG_X; 4],
        // `u` runs the width of the wall and `v` its height, so the edge in the texture is a
        // vertical line and the slide moves it sideways.
        uvs: vec![
            Vec2::new(across(-reach), 0.0),
            Vec2::new(across(reach), 0.0),
            Vec2::new(across(reach), 1.0),
            Vec2::new(across(-reach), 1.0),
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material: 0,
            thin: false,
        }],
    };
    common::scene_of(
        &[wall],
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

/// Renders the wall with the clock at `seconds` and returns the display-ready bytes.
fn present(scene: &StaticScene, seconds: f32) -> Vec<u8> {
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
    renderer.set_glare(0.0);
    // **The texture as painted, which is what the edge is.** De-lighting divides an estimate of the
    // painted lighting back out and relief tilts the normal by its gradient, and a black-to-white
    // step is nothing *but* gradient — both would soften the one thing being measured.
    renderer.set_delight(0.0);
    renderer.set_relief(0.0);
    renderer.set_time(seconds);

    let mut uploader = gpu.uploader();
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            CellId::Interior("fixture".to_owned()),
            scene,
            &[Some(halves())],
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

/// Where the black half meets the white half along the middle row, in pixels.
///
/// **Found as a crossing rather than as a step.** Sixteen texels magnified across 256 pixels puts
/// sixteen pixels of linear ramp between the two halves, so no pair of neighbours differs by much;
/// what is sharp is where the ramp passes the halfway value, and that is the same place however
/// wide the ramp is.
///
/// Searched over the middle half of the frame alone: the texture repeats, so its black meets its
/// white again at every whole coordinate — at the frame's own edges here — and only the crossing at
/// the middle of the texture is ever inside this window.
fn edge(pixels: &[u8]) -> u32 {
    let row = (HEIGHT / 2) as usize;
    let at = |x: usize| pixels[(row * WIDTH as usize + x) * 4];
    let window = (WIDTH / 4) as usize..(3 * WIDTH / 4) as usize;
    let levels: Vec<u8> = window.clone().map(at).collect();
    let halfway = u16::from(*levels.iter().min().expect("the window is not empty")).midpoint(
        u16::from(*levels.iter().max().expect("the window is not empty")),
    ) as u8;
    window
        .clone()
        .find(|&x| (at(x - 1) < halfway) != (at(x) < halfway))
        .unwrap_or_else(|| panic!("no edge in {window:?} of the wall")) as u32
}

#[test]
fn a_texture_slides_across_its_surface_by_the_rate_its_controller_names() {
    let wall = sliding_wall();
    let still = edge(&present(&wall, 0.0));
    let moved = edge(&present(&wall, 1.0));

    // **A quarter of the texture is a quarter of the frame**, because the wall's `u` runs nought to
    // one across the whole of it: 64 pixels of 256. Which way it goes depends on which way the
    // projection puts `+y`, and the shift is the assertion rather than the direction.
    let shift = still.abs_diff(moved);
    let wanted = (SCROLL * WIDTH as f32) as u32;
    assert!(
        shift.abs_diff(wanted) <= 2,
        "a second of sliding moved the edge from {still} to {moved} — {shift} pixels against the \
         {wanted} the rate calls for"
    );

    // **And the clock is what moves it.** Without this the assertion above would pass just as well
    // on a renderer that had ignored the rate and shifted the picture for some other reason.
    assert_eq!(
        still,
        edge(&present(&wall, 0.0)),
        "the same clock gave two different pictures"
    );
}
