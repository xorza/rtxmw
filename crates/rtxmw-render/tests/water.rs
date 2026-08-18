//! What a water surface does to the light that reaches it.
//!
//! Two properties, both of which a plausible-looking implementation can get wrong: how much of the
//! surface is mirror rather than window as the angle changes, and how what lies under it fades with
//! the distance the light travelled through it.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{
    CellId, Instance, Material, MaterialKind, Mesh, MeshId, StaticScene, Submesh, Sun,
};

mod common;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

/// A quad in the XY plane at `z`, large enough that every ray of the frame lands on it.
fn slab(z: f32, material: u32) -> Mesh {
    let reach = 4000.0;
    Mesh {
        positions: vec![
            Vec3::new(-reach, -reach, z),
            Vec3::new(reach, -reach, z),
            Vec3::new(reach, reach, z),
            Vec3::new(-reach, reach, z),
        ],
        normals: vec![Vec3::Z; 4],
        uvs: vec![Vec2::ZERO; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material,
        }],
    }
}

/// A floor `depth` units below water at z = 0, both filling the view.
fn flooded(depth: f32, floor: Material) -> StaticScene {
    common::scene_of(
        &[slab(0.0, 0), slab(-depth, 1)],
        &[
            Material {
                kind: MaterialKind::Water,
                ..Material::default()
            },
            floor,
        ],
        &[
            Instance {
                mesh: MeshId(0),
                transform: Affine3A::IDENTITY,
            },
            Instance {
                mesh: MeshId(1),
                transform: Affine3A::IDENTITY,
            },
        ],
        &[],
        // A dark sky, so a reflection of it is distinguishable from the floor beneath.
        Vec3::splat(0.02),
    )
}

/// Water over a floor that emits rather than reflects.
///
/// Emissive so that what reaches the camera is a known quantity attenuated by the water and nothing
/// else — no shadow rays, no bounce, no dependence on the sun.
fn pool(depth: f32) -> StaticScene {
    flooded(
        depth,
        Material {
            emissive: Vec3::ONE,
            diffuse: Vec3::ZERO,
            ..Material::default()
        },
    )
}

/// A sunlit sand floor `depth` below, with water over it when `covered`.
///
/// Diffuse rather than emissive, unlike [`pool`], because what is being measured here is whether the
/// *sun* reaches the bottom — which an emissive floor would answer for free.
fn seabed(depth: f32, covered: bool) -> StaticScene {
    let sand = Material {
        diffuse: Vec3::splat(0.5),
        ..Material::default()
    };
    let mut scene = if covered {
        flooded(depth, sand)
    } else {
        common::scene_of(
            &[slab(-depth, 0)],
            &[sand],
            &[Instance {
                mesh: MeshId(0),
                transform: Affine3A::IDENTITY,
            }],
            &[],
            Vec3::splat(0.02),
        )
    };
    // Straight down, so the floor faces it squarely and the geometry adds nothing to reason about.
    scene.sun = Some(Sun {
        direction: Vec3::NEG_Z,
        colour: Vec3::splat(2.0),
        angular_radius: 0.0,
    });
    scene
}

/// Traces `scene` from `eye` toward `forward` at time zero and returns the middle pixel's radiance.
fn centre_radiance(scene: &StaticScene, eye: Vec3, forward: Vec3) -> Vec3 {
    let pixels = frame_at(scene, eye, forward, 0.0);
    let index = ((HEIGHT / 2) * WIDTH + WIDTH / 2) as usize * 4;
    // The target is the trace's own image, before any tone curve, quantised on the way back.
    Vec3::new(
        pixels[index] as f32,
        pixels[index + 1] as f32,
        pixels[index + 2] as f32,
    ) / 255.0
}

/// Traces `scene` from `eye` toward `forward` with the clock at `time`, and returns the frame.
fn frame_at(scene: &StaticScene, eye: Vec3, forward: Vec3, time: f32) -> Vec<u8> {
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
    .expect("renderer");
    // No indirect bounce: it would add ambient to the floor and blur what is being measured.
    renderer.set_bounce_samples(0);
    // Unfiltered: every number here is one the trace computes, and smoothing is what would hide a
    // wrong one.
    renderer.set_denoise_passes(0);
    renderer.set_time(time);

    let mut uploader = gpu.uploader();
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            CellId::Interior("water".to_owned()),
            scene,
            &[],
        )
        .expect("scene should load");

    // Straight down is the interesting angle here and also the one where z-up degenerates, so the
    // up vector tips over rather than the camera being nudged off the axis being measured.
    let up = if forward.dot(Vec3::Z).abs() > 0.99 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let view = glam::camera::rh::view::look_to_mat4(eye, forward, up);
    let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
        60f32.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
        0.05,
    );
    let constants = renderer.frame_constants(view, projection, eye);
    renderer
        .render_once(&mut uploader, &constants)
        .expect("frame should render");

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

#[test]
fn water_is_a_window_head_on_and_a_mirror_at_a_glancing_angle() {
    // Straight down at shallow water: Schlick gives 2% reflectance at normal incidence, so almost
    // all of this is the floor seen through the water.
    let down = centre_radiance(&pool(50.0), Vec3::new(0.0, 0.0, 500.0), Vec3::NEG_Z);

    // Nearly along the surface, where reflectance approaches one and the floor all but disappears
    // behind a reflection of a sky darker than it.
    let grazing = centre_radiance(
        &pool(50.0),
        Vec3::new(0.0, -2000.0, 20.0),
        Vec3::new(0.0, 1.0, -0.01).normalize(),
    );

    assert!(
        down.y > 0.5,
        "looking down through 50 units of water should still show the floor, got {down:?}"
    );
    assert!(
        grazing.y < down.y * 0.5,
        "a glancing angle should reflect the dark sky rather than show the floor: \
         grazing {grazing:?} against {down:?}"
    );
}

#[test]
fn what_lies_under_water_fades_with_depth_and_reddens_first() {
    // The same white floor at two depths, viewed straight down. The floor emits, so the light's
    // path through the water is the depth itself — once, not there and back.
    let shallow = centre_radiance(&pool(100.0), Vec3::new(0.0, 0.0, 500.0), Vec3::NEG_Z);
    let deep = centre_radiance(&pool(400.0), Vec3::new(0.0, 0.0, 500.0), Vec3::NEG_Z);

    // Beer-Lambert against the extinction the shader declares — red 0.00643 per unit, green
    // 0.001429 — repeated here on purpose: this pins the *difference* between the channels, which
    // is the whole reason water has a colour. Tripling the depth from 100 to 400 units multiplies
    // the red-to-green ratio by `exp(-(0.00643 - 0.001429) * 300)`, and a single-channel extinction
    // would leave it at 1.
    let shallow_ratio = shallow.x / shallow.y;
    let deep_ratio = deep.x / deep.y;
    let expected = (-(0.00643f32 - 0.001429) * 300.0).exp();
    let measured = deep_ratio / shallow_ratio;
    assert!(
        (measured - expected).abs() < 0.03,
        "red should fall away by {expected} relative to green over that depth, measured {measured} \
         ({deep_ratio} against {shallow_ratio})"
    );
    assert!(
        deep.y < shallow.y,
        "the deeper floor must be dimmer: {deep:?} against {shallow:?}"
    );
    // Green survives the trip where red does not, which is why shallow coastal water reads green.
    assert!(
        deep.y > deep.x * 4.0,
        "at depth the surviving light should be green-dominant, got {deep:?}"
    );
}

#[test]
fn a_water_surface_writes_no_albedo_for_the_denoiser_to_multiply_by() {
    // Water is shaded whole and written past the filter: it has no albedo, and a mirror reflection
    // carries no noise worth filtering. If it wrote one, the composite would multiply its
    // illumination — zero — into the frame and the water would go black.
    let scene = pool(100.0);
    let lit = centre_radiance(&scene, Vec3::new(0.0, 0.0, 500.0), Vec3::NEG_Z);
    assert!(
        lit.length() > 0.1,
        "water composited to nothing, which is what a stray albedo write looks like: {lit:?}"
    );
}

#[test]
fn the_sun_reaches_the_bottom_through_water() {
    // **Water must not cast a shadow.** It is built into the acceleration structure like any other
    // surface, and a surface blocks shadow rays — so a seabed that was sunlit before there was a
    // sea has to still be sunlit with one over it, or the whole shallows go black in daylight.
    let depth = 100.0;
    let dry = centre_radiance(
        &seabed(depth, false),
        Vec3::new(0.0, 0.0, 500.0),
        Vec3::NEG_Z,
    );
    let submerged = centre_radiance(
        &seabed(depth, true),
        Vec3::new(0.0, 0.0, 500.0),
        Vec3::NEG_Z,
    );

    // What the water takes is what it absorbs along the camera's path back up: green loses
    // `exp(-0.001429 * 100)` = 0.867 of itself, and Fresnel reflects 2% of the rest away. So the
    // submerged floor should be a little dimmer than the dry one — not a fraction of it.
    let ratio = submerged.y / dry.y;
    assert!(
        ratio > 0.7 && ratio < 1.0,
        "the seabed should keep most of its sunlight under 100 units of water, \
         got {ratio} ({submerged:?} against {dry:?})"
    );
}

/// The green channel along the middle row, which is where the surface is looked at edge-on.
fn middle_row(pixels: &[u8]) -> Vec<f32> {
    let row = (HEIGHT / 2) as usize;
    (0..WIDTH as usize)
        .map(|x| pixels[(row * WIDTH as usize + x) * 4 + 1] as f32 / 255.0)
        .collect()
}

#[test]
fn waves_break_up_the_surface_and_travel_with_the_clock() {
    // Edge-on, where Fresnel is steepest and so most sensitive to the surface tilting: a flat
    // mirror gives one reflectance across the whole row, and waves give a different one per pixel.
    let eye = Vec3::new(0.0, -2000.0, 30.0);
    let forward = Vec3::new(0.0, 1.0, -0.012).normalize();
    let scene = pool(400.0);

    let still = middle_row(&frame_at(&scene, eye, forward, 0.0));
    let spread = still.iter().copied().fold(f32::MIN, f32::max)
        - still.iter().copied().fold(f32::MAX, f32::min);
    assert!(
        spread > 0.02,
        "a wavy surface should vary across the row; got a spread of {spread}"
    );

    // **The clock has to reach the shader**, which after moving the frame constants out of push
    // constants and into a buffer is not a given. A second later the same waves are somewhere else:
    // the shortest of them is 85 units long and travels at `sqrt(g/k)`, about 3 metres a second, so
    // a second moves it well past its own wavelength.
    let later = middle_row(&frame_at(&scene, eye, forward, 1.0));
    let moved: f32 = still
        .iter()
        .zip(&later)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / still.len() as f32;
    assert!(
        moved > 0.005,
        "the surface should have moved in a second; mean change was {moved}"
    );
}
