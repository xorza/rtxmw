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
    let mut scene = common::scene_of(
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
    );
    // What tells anything below the surface how deep it is, and so how the light reached it.
    scene.water_level = Some(0.0);
    scene
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
    let eye = Vec3::new(0.0, 0.0, 500.0);
    let dry = mean_green(&frame_at(&seabed(depth, false), eye, Vec3::NEG_Z, 0.0));
    let submerged = mean_green(&frame_at(&seabed(depth, true), eye, Vec3::NEG_Z, 0.0));

    // Green is absorbed over both legs — the sunlight's way down and the camera's way back up — at
    // `exp(-0.001429 * 100)` = 0.867 each, and Fresnel turns 2% of what is left away at the
    // surface. That puts the floor at about three quarters of its dry brightness. A *mean* rather
    // than a pixel because caustics move light around: any one point sits on a bright line or
    // between two, and only the average says how much arrived.
    let ratio = submerged / dry;
    assert!(
        (0.6..0.85).contains(&ratio),
        "the seabed should keep most of its sunlight under 100 units of water, \
         got {ratio} ({submerged} against {dry})"
    );
}

/// Mean green across the whole frame.
///
/// A mean rather than a pixel, because caustics move light *around*: any one point on a seabed sits
/// on a bright line or between two, and only the average says how much light arrived.
fn mean_green(pixels: &[u8]) -> f32 {
    let total: f32 = pixels.chunks_exact(4).map(|p| p[1] as f32).sum();
    total / (pixels.len() / 4) as f32 / 255.0
}

/// How much the frame varies about its own mean, as a fraction of it.
///
/// The measure of a caustic pattern: light gathered into lines rather than spread evenly.
fn relative_spread(pixels: &[u8]) -> f32 {
    let values: Vec<f32> = pixels
        .chunks_exact(4)
        .map(|p| p[1] as f32 / 255.0)
        .collect();
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance =
        values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / values.len() as f32;
    variance.sqrt() / mean.max(1e-6)
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

#[test]
fn caustics_gather_the_sunlight_without_creating_any() {
    // **The property that catches a wrong derivative.** A caustic is light moved, not made: the
    // pattern on a seabed is the sun's own light redistributed by the surface above it, so however
    // violently it shifts from moment to moment, the total arriving cannot change.
    let eye = Vec3::new(0.0, 0.0, 500.0);
    let scene = seabed(300.0, true);

    let first = frame_at(&scene, eye, Vec3::NEG_Z, 0.0);
    let second = frame_at(&scene, eye, Vec3::NEG_Z, 1.7);

    let moved: f32 = first
        .chunks_exact(4)
        .zip(second.chunks_exact(4))
        .map(|(a, b)| (a[1] as f32 - b[1] as f32).abs())
        .sum::<f32>()
        / (first.len() / 4) as f32
        / 255.0;
    let means = (mean_green(&first), mean_green(&second));
    assert!(
        moved > 0.01,
        "the pattern should have moved between the two frames; mean change was {moved}"
    );
    assert!(
        (means.0 - means.1).abs() < 0.1 * means.0,
        "the light arriving must not change with the pattern: {means:?}"
    );
}

#[test]
fn the_caustic_pattern_sharpens_with_depth() {
    // A surface is a lens, and a lens needs a throw. Just under the surface the light has had no
    // room to gather and the bottom is evenly lit; further down the same waves have pulled it into
    // lines. So contrast has to grow with depth — which is what pins the *depth* term in the
    // Jacobian rather than merely the curvature.
    let eye = Vec3::new(0.0, 0.0, 500.0);
    let shallow = relative_spread(&frame_at(&seabed(10.0, true), eye, Vec3::NEG_Z, 0.0));
    let deep = relative_spread(&frame_at(&seabed(400.0, true), eye, Vec3::NEG_Z, 0.0));

    assert!(
        deep > shallow * 3.0,
        "caustics should be far stronger at depth: {deep} against {shallow}"
    );
}

#[test]
fn the_waterline_leaves_no_seam_where_the_ground_meets_the_surface() {
    // **Edge-on, which is where the seam actually shows.** Looked at from above, three units of
    // water is nearly invisible whatever the shader does; at a glancing angle Fresnel turns the
    // same water into a mirror, so the pixel where the ground all but touches the surface reflects
    // the sky while the shore beside it shows sand. That step is the line the plane draws across
    // two cells in five, and closing it is what this checks.
    let eye = Vec3::new(0.0, -2000.0, 30.0);
    let forward = Vec3::new(0.0, 1.0, -0.014).normalize();
    let sand = Material {
        diffuse: Vec3::splat(0.5),
        ..Material::default()
    };
    // Three units of water, not zero: a floor exactly coplanar with the surface is a degenerate
    // scene, and the question here is what happens as the depth vanishes rather than at nothing.
    let mut touching = flooded(3.0, sand);
    touching.sun = Some(Sun {
        direction: Vec3::NEG_Z,
        colour: Vec3::splat(2.0),
        angular_radius: 0.0,
    });
    let dry = seabed(3.0, false);

    let with_water = mean_green(&frame_at(&touching, eye, forward, 0.0));
    let without = mean_green(&frame_at(&dry, eye, forward, 0.0));
    let difference = (with_water - without).abs() / without;
    assert!(
        difference < 0.05,
        "water over no water must look like no water: {with_water} against {without}"
    );
}

#[test]
fn looking_through_water_from_under_it_fades_with_distance() {
    // The camera below the surface, so the whole primary ray is under water rather than only the
    // part past a refraction. What it sees has to dim with how far away it is.
    let scene = seabed(600.0, true);
    let near = mean_green(&frame_at(
        &scene,
        Vec3::new(0.0, 0.0, -400.0),
        Vec3::NEG_Z,
        0.0,
    ));
    let far = mean_green(&frame_at(
        &scene,
        Vec3::new(0.0, 0.0, -100.0),
        Vec3::NEG_Z,
        0.0,
    ));

    // The floor is 200 units below the near camera and 500 below the far one. Green survives
    // `exp(-0.001429 * 200)` = 0.752 of the first and `exp(-0.001429 * 500)` = 0.489 of the
    // second, so the further view keeps about 65% of what the nearer one does.
    let ratio = far / near;
    let expected = (-0.001429f32 * 300.0).exp();
    assert!(
        (ratio - expected).abs() < 0.12,
        "the far view should keep about {expected} of the near one, got {ratio} \
         ({far} against {near})"
    );
}

#[test]
fn the_sky_through_snells_window_has_travelled_no_water() {
    // Looking up from under the surface, the refracted ray is in *air* — it is the sky, seen
    // through the window the critical angle leaves. Attenuating it as though it had crossed the
    // water is the easy mistake, and it turns the one bright thing down there green.
    //
    // The sky is made bright and the seabed dark so the two cannot be confused for one another.
    let mut scene = flooded(
        800.0,
        Material {
            diffuse: Vec3::ZERO,
            ..Material::default()
        },
    );
    scene.ambient = Some(rtxmw_scene::Ambient {
        colour: Vec3::splat(0.8),
        ..rtxmw_scene::Ambient::default()
    });

    // Just under the surface looking straight up, where the window is widest and the water between
    // the camera and it is a few units rather than a few hundred.
    let up = mean_green(&frame_at(&scene, Vec3::new(0.0, 0.0, -12.0), Vec3::Z, 0.0));
    assert!(
        up > 0.4,
        "the sky through the surface should be close to the sky itself, got {up}"
    );
}
