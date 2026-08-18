//! Exposure, tone curve and sRGB encoding, asserted on the bytes that reach a screen.
//!
//! These read `SceneRenderer::output` rather than `target`: the trace's own image is scene-referred
//! linear radiance, which is what every other test hand-computes against, while this is the
//! display-referred result of everything that happens to it afterwards.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{CellId, Instance, Material, Mesh, MeshId, StaticScene, Submesh};

mod common;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;

/// A quad far larger than the view, so every ray hits it and the frame is one flat surface.
///
/// Filling the frame matters: the histogram bins *every* pixel, and the sky colour a missed ray
/// writes would be a second population in it.
fn backdrop(albedo: Vec3) -> StaticScene {
    let mesh = Mesh {
        positions: vec![
            Vec3::new(100.0, -400.0, -400.0),
            Vec3::new(100.0, 400.0, -400.0),
            Vec3::new(100.0, 400.0, 400.0),
            Vec3::new(100.0, -400.0, 400.0),
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
    let material = Material {
        diffuse: albedo,
        ..Material::default()
    };
    common::scene_of(
        &[mesh],
        &[material],
        &[Instance {
            mesh: MeshId(0),
            transform: Affine3A::IDENTITY,
        }],
        &[],
        Vec3::ONE,
    )
}

/// Covers part of the backdrop with a strip of the scene's *second* material.
///
/// One unit in front of it so it occludes rather than z-fights, and in the same mesh so it needs no
/// second instance. At 99 units the view is 75.9 units tall either side of the axis, which is what
/// turns a `z` range into a fraction of the image.
fn add_strip(scene: &mut StaticScene, z: std::ops::Range<f32>) {
    let mesh = &mut scene.meshes[0];
    let base = mesh.positions.len() as u32;
    mesh.positions.extend([
        Vec3::new(99.0, -400.0, z.start),
        Vec3::new(99.0, 400.0, z.start),
        Vec3::new(99.0, 400.0, z.end),
        Vec3::new(99.0, -400.0, z.end),
    ]);
    mesh.normals.extend([Vec3::NEG_X; 4]);
    mesh.uvs.extend([Vec2::ZERO; 4]);
    let first_index = mesh.indices.len() as u32;
    mesh.indices
        .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    mesh.submeshes.push(Submesh {
        first_index,
        index_count: 6,
        material: 1,
    });
}

/// Renders `scene` and returns the display-ready bytes.
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
    // No bounce rays: the indirect estimator's dither would spread one flat surface across several
    // histogram bins, and every expectation below is computed for a single radiance.
    renderer.set_bounce_samples(0);

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

/// The red channel at `(x, y)`.
fn red_at(pixels: &[u8], x: u32, y: u32) -> u8 {
    pixels[((y * WIDTH + x) * 4) as usize]
}

/// What a flat frame of any brightness should come out at, worked through by hand.
///
/// Auto-exposure puts the mean luminance on the 0.18 key, so the tone curve sees 0.18. Its shadow
/// lift subtracts 0.04 from a value that far above the toe, leaving 0.14, which is below the 0.76
/// the curve starts compressing at and so passes through untouched. sRGB-encoding 0.14 gives
/// `1.055 * 0.14^(1/2.4) - 0.055 = 0.404`, or 103 of 255.
const FLAT_GREY: u8 = 103;

#[test]
fn a_flat_frame_lands_on_middle_grey_whatever_its_radiance() {
    // A hundredfold range, which is more than an interior and an exterior differ by. Auto-exposure
    // means the *same* bytes come out of both — that is the whole property, and it is also what
    // pins every stage: without the histogram the exposure would be wrong, without the tone curve
    // the value would be 0.18 rather than 0.14, and without the sRGB encode 0.14 would reach the
    // file as 36 rather than 103.
    let dim = present(&backdrop(Vec3::splat(0.02)));
    let bright = present(&backdrop(Vec3::splat(2.0)));

    for (name, pixels) in [("dim", &dim), ("bright", &bright)] {
        let value = red_at(pixels, WIDTH / 2, HEIGHT / 2);
        assert!(
            value.abs_diff(FLAT_GREY) <= 4,
            "{name} frame came out at {value}, not the {FLAT_GREY} a correctly exposed flat frame \
             should be — 36 would mean no sRGB encode, 168 a double one"
        );
    }

    let dim_value = red_at(&dim, WIDTH / 2, HEIGHT / 2);
    let bright_value = red_at(&bright, WIDTH / 2, HEIGHT / 2);
    assert!(
        dim_value.abs_diff(bright_value) <= 2,
        "a hundredfold change in radiance moved the exposed result from {dim_value} to \
         {bright_value}; auto-exposure is supposed to absorb it"
    );
}

#[test]
fn a_small_bright_patch_does_not_crush_the_rest_of_the_frame() {
    // Why the exposure is measured on a *log* histogram rather than a linear mean. An interior is
    // a large dark room with a few bright flames in it: here, five sixths of the frame at 0.02 and
    // the rest at 10, a five hundredfold difference.
    //
    // A linear mean of that is 1.72, which exposes by 0.105 and puts the dark part at 0.002 —
    // black, and 0 after encoding. The mean of the *logs* is 2^-4.12, which exposes by 3.13 and
    // lands the dark part at 44, leaving the bright strip to roll off instead of taking the room
    // down with it.
    let mut scene = backdrop(Vec3::splat(0.02));
    scene.materials.intern(Material {
        diffuse: Vec3::splat(10.0),
        ..Material::default()
    });
    add_strip(&mut scene, 50.0..400.0);

    let pixels = present(&scene);
    // Well below the strip, in the flat part.
    let dark = red_at(&pixels, WIDTH / 2, HEIGHT * 3 / 4);
    let lit = red_at(&pixels, WIDTH / 2, 8);
    println!("dark area {dark}, bright strip {lit}");

    assert!(
        dark > 25,
        "the dark part of the frame came out at {dark}; a linear mean would put it at 0, which is \
         what the log histogram exists to avoid"
    );
    assert!(
        lit > dark,
        "the bright strip ({lit}) is no brighter than the room around it ({dark})"
    );
}

#[test]
fn surfaces_with_no_light_on_them_are_left_out_of_the_average() {
    // The bottom half of the frame is a black surface, so its radiance is exactly zero — not dim,
    // *nothing*. Those pixels carry no information about how bright the scene is, and the histogram
    // keeps them in a bin of their own for exactly that reason.
    //
    // Counted, they halve the mean bin: 70 becomes 35, which reads as 2^-7.9 rather than 2^-5.7 and
    // opens the exposure from 9 to 42. The lit half then leaves the tone curve at 0.79 instead of
    // 0.14 and reaches the file at 232 rather than 103 — a blown-out frame, caused by averaging in
    // the parts of it that were never lit.
    let mut scene = backdrop(Vec3::splat(0.02));
    scene.materials.intern(Material {
        diffuse: Vec3::ZERO,
        ..Material::default()
    });
    add_strip(&mut scene, -400.0..0.0);

    let pixels = present(&scene);
    let unlit = red_at(&pixels, WIDTH / 2, HEIGHT * 3 / 4);
    let lit = red_at(&pixels, WIDTH / 2, HEIGHT / 4);
    println!("unlit half {unlit}, lit half {lit}");

    assert_eq!(unlit, 0, "a surface with no light on it should be black");
    assert!(
        lit.abs_diff(FLAT_GREY) <= 6,
        "the lit half came out at {lit}, not the {FLAT_GREY} it should be — 232 would mean the \
         unlit half was averaged into the exposure"
    );
}
