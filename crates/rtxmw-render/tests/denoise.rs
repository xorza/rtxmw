//! The two halves of what demodulated denoising is supposed to do.
//!
//! Filtering the *lighting* rather than the finished image is the whole design, and it has two
//! consequences that pull against each other: noise in the light has to fall a lot, and detail in
//! the albedo must not move at all. A filter run on the composed frame would do well on the first
//! and fail the second, so both are asserted from the same render.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{Instance, Light, Material, Mesh, MeshId, StaticScene, Submesh};

mod common;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;

/// A wall filling the view, its left half twice as reflective as its right, with a lit floor
/// running up to its base.
///
/// The albedo step down the middle is the detail that must survive. The floor is what makes the
/// lighting noisy: a bounce ray from the wall either finds it or escapes into an ambient of
/// nothing, and four samples of that is a coarse dither — which is exactly the noise the filter is
/// for. Without it a lone wall's bounce rays all escape and every pixel agrees.
fn split_wall() -> StaticScene {
    let wall = |y: std::ops::Range<f32>, material: u32| Mesh {
        positions: vec![
            Vec3::new(100.0, y.start, -400.0),
            Vec3::new(100.0, y.end, -400.0),
            Vec3::new(100.0, y.end, 400.0),
            Vec3::new(100.0, y.start, 400.0),
        ],
        normals: vec![Vec3::NEG_X; 4],
        uvs: vec![Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y],
        indices: vec![0, 1, 2, 0, 2, 3],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material,
        }],
    };
    let floor = Mesh {
        positions: vec![
            Vec3::new(0.0, -400.0, -60.0),
            Vec3::new(100.0, -400.0, -60.0),
            Vec3::new(100.0, 400.0, -60.0),
            Vec3::new(0.0, 400.0, -60.0),
        ],
        normals: vec![Vec3::Z; 4],
        uvs: vec![Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y],
        indices: vec![0, 1, 2, 0, 2, 3],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material: 2,
        }],
    };
    // A panel standing in front of the wall's right-hand side, parallel to it and of the same
    // material. Nothing but *depth* distinguishes the two, which is the case the normal test cannot
    // see: at 60 units it is nearer the light than the wall at 100, so its lighting differs and
    // must not bleed across.
    let mut panel = wall(-400.0..-20.0, 1);
    for position in &mut panel.positions {
        position.x = 60.0;
    }

    let materials = [0.8, 0.4, 0.5].map(|albedo| Material {
        diffuse: Vec3::splat(albedo),
        ..Material::default()
    });
    let instances = [0, 1, 2, 3].map(|mesh| Instance {
        mesh: MeshId(mesh),
        transform: Affine3A::IDENTITY,
    });
    // North is to the camera's left, so the brighter half occupies the left of the image. No
    // ambient at all: a bounce ray that escapes then contributes nothing, which is what makes the
    // difference between finding the floor and missing it large enough to see.
    common::scene_of(
        &[wall(0.0..400.0, 0), wall(-400.0..0.0, 1), floor, panel],
        &materials,
        &instances,
        &[Light {
            position: Vec3::new(50.0, 0.0, 200.0),
            colour: Vec3::ONE,
            radius: 400.0,
        }],
        Vec3::ZERO,
    )
}

/// Renders with `passes` of filtering, reading the linear radiance rather than the display bytes.
///
/// The raw target, because tone mapping is a curve: it would compress the very differences these
/// assertions are about, and differently at each brightness.
fn trace(scene: &StaticScene, passes: u32) -> Vec<u8> {
    let gpu = TestGpu::shared();
    let mut renderer = SceneRenderer::new(
        gpu.device(),
        gpu.memory(),
        vk::Extent2D {
            width: WIDTH,
            height: HEIGHT,
        },
    )
    .expect("renderer should build");
    renderer.set_denoise_passes(passes);

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

fn red_at(pixels: &[u8], x: u32, y: u32) -> f32 {
    pixels[((y * WIDTH + x) * 4) as usize] as f32
}

/// Mean absolute difference between horizontally adjacent pixels across a band.
///
/// A measure of *noise* rather than of brightness: a smooth gradient changes by a fraction of a
/// level between neighbours, and single-sample lighting changes by many.
fn roughness(pixels: &[u8], x: std::ops::Range<u32>, rows: std::ops::Range<u32>) -> f32 {
    let mut total = 0.0;
    let mut count = 0.0;
    for y in rows {
        for column in x.clone() {
            total += (red_at(pixels, column + 1, y) - red_at(pixels, column, y)).abs();
            count += 1.0;
        }
    }
    total / count
}

#[test]
fn filtering_smooths_the_light_and_leaves_the_albedo_edge_alone() {
    let scene = split_wall();
    let raw = trace(&scene, 0);
    let filtered = trace(&scene, 4);

    // A band well clear of the seam, where the only variation is the indirect estimate's dither.
    let rough_raw = roughness(&raw, 30..110, 60..200);
    let rough_filtered = roughness(&filtered, 30..110, 60..200);
    println!("neighbour roughness: {rough_raw:.2} raw, {rough_filtered:.2} filtered");
    assert!(
        rough_filtered * 3.0 < rough_raw,
        "filtering barely smoothed anything: {rough_raw} to {rough_filtered}"
    );

    // Two edges have to survive, and they are different kinds. The first is a step in *albedo*
    // down the middle of the frame, which the filter never sees at all because it works on
    // lighting; the second is the junction with the floor, which it does see and must stop at.
    let column =
        |pixels: &Vec<u8>, x: u32| (60..200).map(|y| red_at(pixels, x, y)).sum::<f32>() / 140.0;
    let row = |pixels: &Vec<u8>, y: u32| (40..90).map(|x| red_at(pixels, x, y)).sum::<f32>() / 50.0;

    // Columns landing between the two plateaus rather than on one of them. A step has none; a ramp
    // spread over the filter's reach would have twenty.
    let transition_width = |pixels: &Vec<u8>| {
        let (left, right) = (column(pixels, 110), column(pixels, 146));
        let span = left - right;
        (112..145)
            .filter(|&x| {
                let value = column(pixels, x);
                value > right + span * 0.2 && value < right + span * 0.8
            })
            .count()
    };
    let (left, right) = (column(&filtered, 110), column(&filtered, 146));
    println!(
        "seam: {left:.1} to {right:.1} over {} columns ({} unfiltered)",
        transition_width(&filtered),
        transition_width(&raw)
    );
    assert!(
        left - right > 15.0,
        "the fixture has no albedo step to preserve: {left} against {right}"
    );
    assert!(
        transition_width(&filtered) <= 2,
        "the albedo step spread over {} columns; the filter is blurring surface detail, which \
         demodulating the lighting exists to prevent",
        transition_width(&filtered)
    );

    // The floor is half again as bright as the wall above it. Its lighting must not climb the
    // wall: the filter's taps reach sixteen pixels, and the normal and depth tests are the only
    // thing stopping them crossing a junction the two surfaces meet at.
    let plateau = row(&filtered, 208);
    let against_floor = row(&filtered, 224);
    let floor = row(&filtered, 236);
    println!("wall {plateau:.1} at rest, {against_floor:.1} beside the floor, floor {floor:.1}");
    assert!(
        floor - plateau > 15.0,
        "the fixture has no lighting step at the floor: {plateau} against {floor}"
    );
    assert!(
        (against_floor - plateau).abs() < 3.0,
        "the wall reads {against_floor} against the floor and {plateau} away from it; the floor's \
         light is bleeding up the wall, so the edge-stopping is not stopping"
    );

    // And the same again where only depth separates the surfaces. The panel at 60 units and the
    // wall at 100 are parallel and identically surfaced, so the normal test sees nothing between
    // them; only the depth term does.
    let wall_side = column(&filtered, 170);
    let against_panel = column(&filtered, 180);
    let panel_side = column(&filtered, 200);
    println!(
        "wall {wall_side:.1} at rest, {against_panel:.1} beside the panel, panel {panel_side:.1}"
    );
    assert!(
        (panel_side - wall_side).abs() > 8.0,
        "the fixture has no lighting step at the panel: {wall_side} against {panel_side}"
    );
    assert!(
        (against_panel - wall_side).abs() < 3.0,
        "the wall reads {against_panel} beside the panel and {wall_side} away from it; two \
         parallel surfaces are blurring into each other, so the depth term is doing nothing"
    );
}
