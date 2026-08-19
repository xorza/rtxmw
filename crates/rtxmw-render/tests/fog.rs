//! That fog thickens with distance and takes light from what stands in it.
//!
//! Every other file in this directory turns fog *off*, because it is atmosphere between the eye and
//! the surface rather than anything a surface does, and it would put a gradient across walls their
//! assertions need flat. This is the one that turns it on.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{Ambient, CellId, Instance, Light, Material, Mesh, MeshId, StaticScene, Submesh};

mod common;

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;

/// What the fog scatters, and how thickly. Bright and red so that neither can be confused with the
/// wall's own colour or with the lamp's.
const FOG_COLOUR: Vec3 = Vec3::new(0.6, 0.05, 0.05);
const FOG_DENSITY: f32 = 12.0;

/// A white wall filling the view `distance` away, facing the camera.
fn wall(distance: f32, lights: &[Light]) -> StaticScene {
    let half = distance;
    let mesh = Mesh {
        positions: vec![
            Vec3::new(distance, -half, -half),
            Vec3::new(distance, half, -half),
            Vec3::new(distance, half, half),
            Vec3::new(distance, -half, half),
        ],
        normals: vec![Vec3::NEG_X; 4],
        uvs: vec![Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y],
        indices: vec![0, 2, 1, 0, 3, 2],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material: 0,
            thin: false,
        }],
    };
    let mut scene = common::scene_of(
        &[mesh],
        &[Material {
            diffuse: Vec3::splat(0.6),
            ..Material::default()
        }],
        &[Instance {
            mesh: MeshId(0),
            transform: Affine3A::IDENTITY,
        }],
        lights,
        Vec3::splat(0.4),
    );
    // The fixture builder has no opinion about fog, so the cell's own record is set here — which is
    // where a real cell's comes from too.
    scene.ambient = Some(Ambient {
        colour: Vec3::splat(0.4),
        fog: FOG_COLOUR,
        fog_density: FOG_DENSITY,
        ..Ambient::default()
    });
    scene
}

/// The middle of the frame, tracing `scene` with `fog` of the cell's own applied.
fn centre(scene: &StaticScene, fog: f32) -> Vec3 {
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
    // No bounce and no filter: what is measured is the fog, and both would stir it into the rest.
    renderer.set_bounce_samples(0);
    renderer.set_denoise_passes(0);
    renderer.set_fog(fog);

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

    // A patch rather than a texel, since the march's start is jittered per pixel.
    let mut total = Vec3::ZERO;
    let mut count = 0.0;
    for y in HEIGHT / 2 - 8..HEIGHT / 2 + 8 {
        for x in WIDTH / 2 - 8..WIDTH / 2 + 8 {
            let at = ((y * WIDTH + x) * 4) as usize;
            total += Vec3::new(
                pixels[at] as f32,
                pixels[at + 1] as f32,
                pixels[at + 2] as f32,
            ) / 255.0;
            count += 1.0;
        }
    }
    total / count
}

#[test]
fn fog_thickens_with_the_distance_it_is_seen_through() {
    let near = wall(400.0, &[]);
    let far = wall(6000.0, &[]);

    // **Without fog the distance makes no difference at all**, which is what makes the comparison
    // below about the fog rather than about the wall: a flat surface lit only by ambient is the
    // same colour however far away it is.
    let (near_clear, far_clear) = (centre(&near, 0.0), centre(&far, 0.0));
    println!("unfogged: {near_clear:?} near, {far_clear:?} far");
    assert!(
        (near_clear - far_clear).abs().max_element() < 0.02,
        "the fixture's own colour depends on distance, so nothing below is about fog: \
         {near_clear:?} against {far_clear:?}"
    );

    let (near_fogged, far_fogged) = (centre(&near, 1.0), centre(&far, 1.0));
    println!("fogged:   {near_fogged:?} near, {far_fogged:?} far");

    // Transmittance is `exp(-sigma * distance)`, so the far wall keeps less of itself and takes
    // more of the fog. The fog is red and the wall is grey, so the red channel rises and the green
    // falls — asserting both is what distinguishes fog from the frame merely getting brighter.
    assert!(
        far_fogged.x > near_fogged.x + 0.05,
        "the far wall took no more of the red fog than the near one: {far_fogged:?} against \
         {near_fogged:?}"
    );
    assert!(
        far_fogged.y < near_fogged.y,
        "the far wall kept as much of its own green as the near one, so nothing was absorbed: \
         {far_fogged:?} against {near_fogged:?}"
    );
    // And the near wall is still mostly itself, or the test is comparing two fogs rather than a
    // fogged surface against a less fogged one.
    assert!(
        near_fogged.y > 0.1,
        "even the near wall is lost in fog, so this fixture cannot show a gradient: {near_fogged:?}"
    );
}

#[test]
fn a_lamp_in_the_fog_lights_it() {
    // The same wall and the same fog, with and without a lamp off to one side. The lamp faces away
    // from nothing and touches no surface the camera can see at the centre — what it changes is the
    // air, which is the whole claim.
    let dark = wall(6000.0, &[]);
    let lit = wall(
        6000.0,
        &[Light {
            position: Vec3::new(1200.0, 0.0, 0.0),
            colour: Vec3::new(0.0, 0.0, 1.0),
            radius: 3000.0,
        }],
    );

    let (without, with) = (centre(&dark, 1.0), centre(&lit, 1.0));
    println!("fog alone {without:?}, fog with a blue lamp in it {with:?}");

    // **Blue, which neither the fog nor the wall has any of.** A lamp that only lit surfaces would
    // leave the centre of this frame alone: the wall is six thousand units away and the light
    // reaches three thousand.
    assert!(
        with.z > without.z + 0.05,
        "the blue lamp put no blue into the air between the camera and the wall: {with:?} against \
         {without:?}"
    );
}
