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

/// The extinction per world unit that `primary_visibility.comp` declares.
///
/// Repeated here because a shader constant is not visible from Rust, and **every expectation below
/// is derived from it** rather than written out — so tuning the water is one line here rather than
/// a hunt through five pieces of arithmetic that would otherwise quietly stop describing it.
const EXTINCTION: Vec3 = Vec3::new(0.004572, 0.000714, 0.001143);

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
            thin: false,
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

/// The traced radiance of a finished frame, as RGBA8.
fn target_pixels(uploader: &mut rtxmw_gpu::Uploader, renderer: &SceneRenderer) -> Vec<u8> {
    readback::image_to_rgba8(
        uploader,
        renderer.target(),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    )
    .expect("readback should succeed")
}

/// Traces `scene` from `eye` toward `forward` with the clock at `time`, and returns the frame.
fn frame_at(scene: &StaticScene, eye: Vec3, forward: Vec3, time: f32) -> Vec<u8> {
    rendered(scene, eye, forward, time, |_| {}, target_pixels)
}

/// Renders one frame of `scene` and hands the finished renderer to `read`.
///
/// Which image a test wants back is the only thing that varies — the trace's own radiance for the
/// numbers most of these assert, the material target for the guides — so the setup lives here once
/// rather than beside each. `dials` is for the one test that needs weather over the water; the rest
/// pass nothing.
fn rendered<T>(
    scene: &StaticScene,
    eye: Vec3,
    forward: Vec3,
    time: f32,
    dials: impl FnOnce(&mut SceneRenderer),
    read: impl FnOnce(&mut rtxmw_gpu::Uploader, &SceneRenderer) -> T,
) -> T {
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
    dials(&mut renderer);

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

    let read = read(&mut uploader, &renderer);
    drop(uploader);
    gpu.assert_no_validation_errors();
    read
}

/// The specular guide the trace wrote for the middle pixel: albedo in `xyz`, roughness in `w`.
///
/// **Read from the two places the upscaler reads them**, not from one convenient image: the albedo
/// is the material target's `rgb` and the roughness is the *normal* target's `w`, which is where
/// DLSS Ray Reconstruction's packed mode looks for it. A test reading both from one image would
/// keep passing if they were written to the wrong one.
fn guides_at(scene: &StaticScene, eye: Vec3, forward: Vec3) -> glam::Vec4 {
    rendered(
        scene,
        eye,
        forward,
        0.0,
        |_| {},
        |uploader, renderer| {
            let albedo = centre_texel(uploader, renderer.material());
            let roughness = centre_texel(uploader, renderer.normal_roughness()).w;
            albedo.truncate().extend(roughness)
        },
    )
}

/// The four channels `image` holds at the middle pixel.
fn centre_texel(uploader: &mut rtxmw_gpu::Uploader, image: &rtxmw_gpu::Image) -> glam::Vec4 {
    let middle = ((HEIGHT / 2) * WIDTH + WIDTH / 2) as usize * 4;
    let channels = readback::image_to_f32(uploader, image, vk::ImageLayout::GENERAL)
        .expect("readback should succeed");
    glam::Vec4::from_slice(&channels[middle..middle + 4])
}

#[test]
fn water_is_the_only_thing_that_reflects_and_it_says_so_in_the_guide() {
    // **What DLSS Ray Reconstruction reads to reproject a reflection.** A mirror does not move
    // across the screen the way the surface carrying it does — it moves with whatever is reflected —
    // so a filter given only depth smears every reflection. §5.2 records that this is easy to forget
    // and awkward to add late, which is why it is written before anything reads it.
    //
    // Vanilla Morrowind has no specular at all: `NiSpecularProperty` is force-disabled at this NIF
    // version, so every material in the world is matte and water is the single exception.
    // Under the surface looking *down*, so the floor is what the ray finds and no water is crossed
    // on the way — pointing up from here would hit the surface, which is the one thing that is not
    // matte.
    let seabed = guides_at(&pool(400.0), Vec3::new(0.0, 0.0, -100.0), Vec3::NEG_Z);
    println!("a matte floor reports {seabed:?}");
    assert_eq!(
        seabed.truncate(),
        Vec3::ZERO,
        "something other than water claimed a specular response"
    );
    assert_eq!(seabed.w, 1.0, "a matte surface is not fully rough");

    // Straight down onto water: the reflection is weakest head-on, so this is Fresnel at its floor
    // — a few per cent — and the surface is nearly a mirror.
    let above = guides_at(&pool(400.0), Vec3::new(0.0, 0.0, 200.0), Vec3::NEG_Z);
    println!("water from overhead reports {above:?}");
    assert!(
        above.x > 0.0 && above.x < 0.2,
        "water's specular albedo head-on is {}, not the few per cent Fresnel gives",
        above.x
    );
    assert!(
        above.w < 0.5,
        "water came out rougher than a matte floor at {}",
        above.w
    );
    // Grey: the surface reflects every wavelength alike, and a tint here would mean the Fresnel
    // term had picked up the water's colour.
    assert_eq!(above.x, above.y);
    assert_eq!(above.y, above.z);
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

    // Beer-Lambert, pinning the *difference* between the channels — which is the whole reason
    // water has a colour at all. Tripling the depth from 100 to 400 units multiplies the
    // red-to-green ratio by `exp(-(red - green) * 300)`, and a single-channel extinction would
    // leave it at 1.
    let shallow_ratio = shallow.x / shallow.y;
    let deep_ratio = deep.x / deep.y;
    let expected = (-(EXTINCTION.x - EXTINCTION.y) * 300.0).exp();
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

    // Green is absorbed over both legs — the sunlight's way down and the camera's way back up —
    // and Fresnel turns 2% of what is left away at the surface, so the floor keeps roughly
    // `exp(-green * depth)` squared of its dry brightness. A *mean* rather
    // than a pixel because caustics move light around: any one point sits on a bright line or
    // between two, and only the average says how much arrived.
    let ratio = submerged / dry;
    assert!(
        (0.6..0.98).contains(&ratio),
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
    // Two depths, either side of the offset a ray is pushed off a surface by — 1.5 units. Below
    // it, a refraction ray started on the far side of the plane would begin *under the ground*,
    // travel down through open air and report water of unbounded depth; on a gentle shore that
    // band is metres wide and drew a flat ribbon of scattering colour along the whole waterline.
    for depth in [0.6f32, 3.0] {
        let covered = seabed(depth, true);
        let dry = seabed(depth, false);

        let with_water = mean_green(&frame_at(&covered, eye, forward, 0.0));
        let without = mean_green(&frame_at(&dry, eye, forward, 0.0));
        let difference = (with_water - without).abs() / without;
        assert!(
            difference < 0.05,
            "{depth} units of water must look like no water: {with_water} against {without}"
        );
    }
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

    // The floor is 200 units below the near camera and 500 below the far one, so the further view
    // looks through 300 units more water than the nearer one.
    let ratio = far / near;
    let expected = (-EXTINCTION.y * 300.0).exp();
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

#[test]
fn the_same_water_looks_the_same_from_either_side_of_its_surface() {
    // Ten units above the surface and ten below, looking straight down at a floor two hundred
    // units under it. The only difference is those twenty units of water, so the two views have to
    // agree — anything counted on one side of the surface and not the other shows up here as a
    // step across it.
    let scene = seabed(200.0, true);
    let above = mean_green(&frame_at(
        &scene,
        Vec3::new(0.0, 0.0, 10.0),
        Vec3::NEG_Z,
        0.0,
    ));
    let below = mean_green(&frame_at(
        &scene,
        Vec3::new(0.0, 0.0, -10.0),
        Vec3::NEG_Z,
        0.0,
    ));
    let straight = below / above;
    assert!(
        (0.96..1.08).contains(&straight),
        "straight down, the two sides should agree: {below} against {above}"
    );

    // **At a slant they must not agree, and the difference is refraction.** Entering at 53 degrees
    // from the vertical, a ray is bent to 37 and reaches the floor in 200 / cos 37 = 250 units;
    // from below, nothing bends it and the same look costs 190 / cos 53 = 317. So the view from
    // above passes through 67 units less water and is legitimately the clearer of the two — which
    // is most of why water looks more transparent from a boat than from under it.
    let slant = Vec3::new(0.0, 0.8, -0.6).normalize();
    let over = mean_green(&frame_at(&scene, Vec3::new(0.0, -300.0, 10.0), slant, 0.0));
    let under = mean_green(&frame_at(&scene, Vec3::new(0.0, -300.0, -10.0), slant, 0.0));
    let expected = (-EXTINCTION.y * (316.7 - 250.0)).exp();
    let measured = under / over;
    assert!(
        measured < 1.0 && (measured - expected).abs() < 0.1,
        "the slanted view from below should be dimmer by about {expected}, got {measured} \
         ({under} against {over})"
    );
}

#[test]
fn the_sky_dims_with_depth_as_the_sun_does() {
    // **Skylight has to reach the bottom too.** The sun was being attenuated on its way down while
    // the sky was not, so a surface three metres under lit by a dimmed sun and a full-strength sky
    // came out brighter than either would allow — and every depth looked equally lit, which is the
    // opposite of what makes deep water read as deep.
    //
    // No sun at all here, so the ambient is the whole of the light. The camera sits twenty units
    // off the floor in both cases, so the path back to the eye is the same and only the light's own
    // journey down differs.
    let sand = Material {
        diffuse: Vec3::splat(0.5),
        ..Material::default()
    };
    // A bright sky, unlike the other fixtures here: with the dim one, both readings land on the
    // same two-of-255 and the ratio comes out at one by rounding rather than by physics.
    let lit = |depth: f32, eye: f32| {
        let mut scene = flooded(depth, sand);
        scene.ambient = Some(rtxmw_scene::Ambient {
            colour: Vec3::splat(1.6),
            ..rtxmw_scene::Ambient::default()
        });
        mean_green(&frame_at(
            &scene,
            Vec3::new(0.0, 0.0, eye),
            Vec3::NEG_Z,
            0.0,
        ))
    };
    let shallow = lit(100.0, -80.0);
    let deep = lit(600.0, -580.0);

    // Five hundred more units of water above it to get through.
    let expected = (-EXTINCTION.y * 500.0).exp();
    let measured = deep / shallow;
    assert!(
        (measured - expected).abs() < 0.12,
        "the deeper floor should keep about {expected} of the shallow one's skylight, \
         got {measured} ({deep} against {shallow})"
    );
}

/// One frame of the pool under `precipitation` at `time`, from an eye `height` above the water,
/// read however the caller needs it.
///
/// **Sixty units is close to the surface, because a ring is eleven centimetres across.** A cone
/// wider than that averages its crest and trough away exactly as it does for the swell — which is
/// correct, and which a camera four hundred units up at sixty-four pixels square reads as the
/// ripples not being there at all. That far view is its own case below, and reads the roughness
/// rather than the picture.
fn over_the_pool<T>(
    height: f32,
    precipitation: rtxmw_scene::Precipitation,
    time: f32,
    read: impl FnOnce(&mut rtxmw_gpu::Uploader, &SceneRenderer) -> T,
) -> T {
    let scene = pool(2_000.0);
    rendered(
        &scene,
        Vec3::new(0.0, 0.0, height),
        Vec3::new(1.0, 0.0, -0.9).normalize(),
        time,
        |renderer| {
            renderer.set_sky(rtxmw_scene::Sky {
                precipitation,
                ..rtxmw_scene::Sky::at(rtxmw_scene::WorldTime::hours(12.0))
            });
        },
        read,
    )
}

/// The traced picture of that frame.
fn surface_from(height: f32, precipitation: rtxmw_scene::Precipitation, time: f32) -> Vec<u8> {
    over_the_pool(height, precipitation, time, target_pixels)
}

/// Rain as `[Weather Rain]` describes it, but falling through a volume too small to draw.
///
/// **Which is what isolates the ripples.** A drop is drawn only within the weather's own diameter of
/// the eye, and a couple of units of that is nothing at all — so nothing reaches the transparency
/// layer, while the water still knows it is raining. Without that the comparison is hopeless: a
/// snowy frame differs from a dry one on the flakes in the air alone, and no amount of looking at
/// the surface separates the two.
///
/// Nor does reading the guides get round it, which was the other thing tried. Water writes
/// `-direction` into the normal target rather than its own normal, deliberately — see
/// `water_is_the_only_thing_that_reflects_and_it_says_so_in_the_guide` — so no ring reaches those
/// three channels. The *fourth* is a different quantity and does carry them, which is what the far
/// view at the end of the test reads; it is the ripples as roughness rather than as shape, and only
/// once they are too small to draw.
fn raining(snow: bool) -> rtxmw_scene::Precipitation {
    rtxmw_scene::Precipitation {
        count: 450.0,
        diameter: 2.0,
        height: 500.0,
        fall: 4025.0,
        snow,
    }
}

#[test]
fn rain_rings_the_water_and_snow_lands_on_it_without_a_sound() {
    let pixels = (WIDTH * HEIGHT) as usize;
    // Which pixels two frames of the same water disagree about, by more than the tonemap's own
    // rounding.
    let apart = |was: &[u8], now: &[u8]| -> Vec<bool> {
        was.chunks_exact(4)
            .zip(now.chunks_exact(4))
            .map(|(was, now)| (0..3).any(|c| was[c].abs_diff(now[c]) > 2))
            .collect()
    };
    let count = |mask: &[bool]| mask.iter().filter(|shown| **shown).count();

    // **Snow is the reference, and this is what earns it that.** It puts the same nothing in the
    // air — the diameter above is too small for a flake to be drawn at all — and it does not ring,
    // so a snowy frame and a dry one are the same frame down to the byte. Both halves of the
    // fixture's claim, and neither of them true by construction.
    let dry = surface_from(60.0, rtxmw_scene::Precipitation::NONE, 1.0);
    let snowed = surface_from(60.0, raining(true), 1.0);
    assert_eq!(
        count(&apart(&dry, &snowed)),
        0,
        "snow should land on water without a sound, and nothing should be drawn falling"
    );

    // **A fair share of the surface at any moment**, which is what rings living half a second on a
    // lattice this fine come to — every cell has an impact, and what varies is whether one is
    // passing under this pixel now.
    let ringing = apart(&snowed, &surface_from(60.0, raining(false), 1.0));
    let rings = count(&ringing);
    assert!(
        rings > pixels / 20,
        "rain should ring the water, and it moved {rings} of {pixels}"
    );

    // And the rings spread rather than standing still. **Measured against snow at each instant
    // rather than against rain a moment earlier**, which would be no test at all: the swell moves
    // too — `waves_break_up_the_surface_and_travel_with_the_clock` asserts exactly that — so two
    // rainy frames differ whether or not a single ring went anywhere. Differencing each against its
    // own snow cancels the swell and leaves the rings.
    //
    // **A twentieth of a second, which is short on purpose.** The cancellation is not perfect: a
    // ring's effect on a pixel rides on the slope it is added to, so a swell that moved changes
    // which pixels show a ring even where none of them travelled. That leak grows with the gap, and
    // it is what a third of a second was measuring. Over a twentieth the swell barely moves while
    // the ring front covers a fifth of its own wavelength — pinning the field to one instant and
    // leaving everything else alone drops this from 455 pixels to 97.
    let later = apart(
        &surface_from(60.0, raining(true), 1.05),
        &surface_from(60.0, raining(false), 1.05),
    );
    let moved = ringing
        .iter()
        .zip(&later)
        .filter(|(was, now)| was != now)
        .count();
    // More pixels change than were ringing to begin with, which is the whole pattern being
    // somewhere else rather than the same one nudged.
    assert!(
        moved > rings,
        "the rings should spread, not sit — {moved} pixels changed against {rings} ringing"
    );

    // **And far enough out to draw none of them, rain still dulls the water.** What a cone cannot
    // resolve is not gone, it is rough — `water_normal`'s own argument for the swell — or the far
    // half of a rainy bay is a mirror. From four hundred units up not one ring survives the fade,
    // so the roughness guide is the whole of what is left of them.
    //
    // Read from the guide rather than from the picture, because the picture cannot show it here: a
    // wider reflection cone over the smooth sky this fixture has returns the same radiance, and the
    // frames come back identical to the byte. What the lobe does is spread a *reflection*, and this
    // pool has nothing in its sky to spread.
    let roughness = |precipitation| {
        over_the_pool(400.0, precipitation, 1.0, |uploader, renderer| {
            centre_texel(uploader, renderer.normal_roughness()).w
        })
    };
    let (still, rough) = (roughness(raining(true)), roughness(raining(false)));
    assert!(
        rough > still * 1.5,
        "rain should roughen water too distant to ring — {rough} against {still}"
    );
}
