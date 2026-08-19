//! That fog thickens with distance and takes light from what stands in it.
//!
//! Every other file in this directory turns fog *off*, because it is atmosphere between the eye and
//! the surface rather than anything a surface does, and it would put a gradient across walls their
//! assertions need flat. This is the one that turns it on.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{
    Ambient, CellId, Instance, Light, Material, Mesh, MeshId, StaticScene, Submesh, Sun,
};

mod common;

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;

/// What the fog scatters. Bright and red so that it cannot be confused with the wall's own colour
/// or with the lamp's.
const FOG_COLOUR: Vec3 = Vec3::new(0.6, 0.05, 0.05);

/// How thickly, which is far above anything an interior ships.
///
/// An interior's recorded density is scaled down hard before it reaches the march — a room is meant
/// to hold a veil, and a veil across a fixture this small measures as noise. What is under test is
/// the integral over distance, and this buys enough of one to see.
const FOG_DENSITY: f32 = 140.0;

/// A flat quad from four corners, of the fixture's only material.
///
/// Its normals all face `-X` whatever the corners say, because the one quad here that is ever
/// *shaded* is the wall, and the wall faces the camera. The rest are only ever hit by a shadow ray,
/// which asks whether something is there and not which way it points.
fn quad(corners: [Vec3; 4]) -> Mesh {
    Mesh {
        positions: corners.to_vec(),
        normals: vec![Vec3::NEG_X; 4],
        uvs: vec![Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y],
        indices: vec![0, 2, 1, 0, 3, 2],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material: 0,
            thin: false,
        }],
    }
}

/// A wall of albedo `diffuse` filling the view `distance` away, facing the camera.
fn wall(distance: f32, lights: &[Light], diffuse: Vec3) -> StaticScene {
    let half = distance;
    let mesh = quad([
        Vec3::new(distance, -half, -half),
        Vec3::new(distance, half, -half),
        Vec3::new(distance, half, half),
        Vec3::new(distance, -half, half),
    ]);
    let mut scene = common::scene_of(
        &[mesh],
        &[Material {
            diffuse,
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
    let near = wall(400.0, &[], Vec3::splat(0.6));
    let far = wall(6000.0, &[], Vec3::splat(0.6));

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

/// The fog the two sunlit tests below stand in: thin, and blue where the fixture above is red.
///
/// **Dim on purpose.** Both measure what the *sun* puts into the air, and the fog's own colour is
/// the term they have to see past: against the red fixture's 0.6 the sun's backward scattering is
/// lost in the rounding, and the ratio those tests are about collapses to nothing.
const SUNLIT_FOG: Vec3 = Vec3::new(0.02, 0.02, 0.06);

/// Which way the sun's light travels in those tests — from ahead of the camera, and from behind it.
///
/// **Never level with the horizon.** Fog shadows itself over the column between a point and the sky
/// along the line to the sun, and for a sun exactly on the horizon that column never leaves the
/// layer, so it is infinite and no sunlight arrives anywhere. A fixture wanting any sun in its air
/// has to raise it. These two are 16.7 degrees up and 16.7 off the view ray, and they share that
/// climb — so the self-shadowing is identical between them and the phase function is the only thing
/// that differs.
const SUN_AHEAD: Vec3 = Vec3::new(-1.0, 0.0, -0.3);
const SUN_BEHIND: Vec3 = Vec3::new(1.0, 0.0, -0.3);

/// The sun for those tests, travelling `heading`.
fn sun(heading: Vec3) -> Sun {
    Sun {
        direction: heading.normalize(),
        colour: Vec3::splat(8.0),
        angular_radius: 0.004_654,
    }
}

/// Open air under `sun`, with `blockers` standing between it and the sky.
///
/// **Nothing in front of the camera at all**, which took a wrong fixture to arrive at: a wall put
/// there to bound the march also stands between the air and a sun ahead of the eye, so it shadowed
/// every ray and the sunlit case measured exactly the sunless one. What bounds the march here is
/// `FOG_REACH`, and the only geometry is an anchor behind the camera — the acceleration structure
/// wants something, and back there it is in the way of neither the eye nor the sun.
fn sunlit(sun: Sun, blockers: &[Mesh]) -> StaticScene {
    let mut meshes = vec![quad([
        Vec3::new(-3000.0, -100.0, -500.0),
        Vec3::new(-2000.0, -100.0, -500.0),
        Vec3::new(-2000.0, 100.0, 0.0),
        Vec3::new(-3000.0, 100.0, 0.0),
    ])];
    meshes.extend_from_slice(blockers);
    let instances: Vec<Instance> = (0..meshes.len())
        .map(|i| Instance {
            mesh: MeshId(i as u32),
            transform: Affine3A::IDENTITY,
        })
        .collect();
    let mut scene = common::scene_of(
        &meshes,
        &[Material {
            diffuse: Vec3::ZERO,
            ..Material::default()
        }],
        &instances,
        &[],
        Vec3::splat(0.02),
    );
    scene.sun = Some(sun);
    scene.ambient = Some(Ambient {
        colour: Vec3::splat(0.02),
        fog: SUNLIT_FOG,
        fog_density: FOG_DENSITY,
        ..Ambient::default()
    });
    scene
}

/// A lid over the whole march, high enough that the camera's own ray never reaches it.
///
/// It sits at `z = 1500` while the camera looks flat along `+X`, so nothing the eye sees is touched.
/// What it does cover is every line from the air to a sun 16.7 degrees up: those climb through
/// `z = 1500` about five thousand units downrange, and the march never leaves this quad.
fn lid() -> Mesh {
    quad([
        Vec3::new(-2000.0, -10000.0, 1500.0),
        Vec3::new(60000.0, -10000.0, 1500.0),
        Vec3::new(60000.0, 10000.0, 1500.0),
        Vec3::new(-2000.0, 10000.0, 1500.0),
    ])
}

/// The same air with no sun at all, which is what the fog's own colour scatters and nothing else.
fn sunless() -> Vec3 {
    let dark = Sun {
        colour: Vec3::ZERO,
        ..sun(SUN_AHEAD)
    };
    centre(&sunlit(dark, &[]), 1.0)
}

#[test]
fn the_fog_throws_the_sun_forward_and_scatters_almost_none_of_it_back() {
    // Nothing moves but the sun, from ahead of the eye to behind it. Same air, same distance, same
    // climb out of the layer — so the *only* term that differs between these two frames is the
    // angle the phase function is asked about.
    let (into, away) = (
        centre(&sunlit(sun(SUN_AHEAD), &[]), 1.0),
        centre(&sunlit(sun(SUN_BEHIND), &[]), 1.0),
    );
    let none = sunless();
    println!("facing the sun {into:?}, with it behind {away:?}, with no sun {none:?}");

    // Hand-computed, and it is the one number this fixture pins exactly: the fog scatters in
    // `1 - exp(-sigma * FOG_REACH)` of its own colour, where
    //   sigma = 140 * 0.006 (indoor scale) * 5e-4 (extinction) * 0.5 (even coverage) = 2.1e-4
    // so `0.02 * (1 - exp(-2.1e-4 * 30000))` = 0.01996 in red, against a sky the fog has all but
    // swallowed by then.
    assert!(
        (none.x - 0.01996).abs() < 0.004,
        "the fog's own inscatter should be 0.01996 with no sun in it, not {none:?}"
    );

    // **The ratio is the phase function and nothing else.** Everything that is not the phase
    // function cancels between these two frames, so what is left is `p(16.7 deg) / p(163.3 deg)`.
    // The readback averages sixteen pixels either side of the centre, which at this field of view
    // asks the phase anywhere from 9 to 24.4 degrees off the sun — so the ratio is a weighted mean
    // of pointwise ratios running from 21.3 to 59.5, and must land between them.
    //
    // **An isotropic fog would give exactly 1.0**, and a Henyey-Greenstein lobe at the g every
    // engine defaults to would give single figures. This is what a real droplet does.
    let ratio = (into.x - none.x) / (away.x - none.x);
    assert!(
        (21.3..=59.5).contains(&ratio),
        "the fog threw {ratio:.1} times as much of the sun forward as back, outside the 21.3 to \
         59.5 its own geometry allows: {into:?} against {away:?} over {none:?}"
    );
}

#[test]
fn what_stands_between_the_fog_and_the_sun_cuts_the_sun_out_of_it() {
    let open = centre(&sunlit(sun(SUN_AHEAD), &[]), 1.0);
    let shaded = centre(&sunlit(sun(SUN_AHEAD), &[lid()]), 1.0);
    let none = sunless();
    println!("open {open:?}, under the lid {shaded:?}, with no sun at all {none:?}");

    assert!(
        open.x > 10.0 * shaded.x,
        "the lid let the sun through into the air beneath it: {shaded:?} against an open {open:?}"
    );
    // **Equal to air with no sun, not merely darker than open air.** Anything above this and some
    // of the sun is leaking through the lid; anything below and the shadow has taken away some of
    // the fog's own light along with the sun's.
    assert!(
        (shaded - none).abs().max_element() < 0.004,
        "shadowed air should scatter exactly what sunless air does: {shaded:?} against {none:?}"
    );
}

#[test]
fn a_lamp_in_the_fog_lights_it() {
    // The same fog, with and without a lamp standing in it. What the lamp changes is the air, which
    // is the whole claim.
    //
    // **The wall is black**, and that is what makes the claim airtight rather than merely likely.
    // It used to be grey and far enough away that the lamp's recorded radius could not touch it —
    // until reach stopped being the recorded radius, the lamp grew past the wall, and the test went
    // on passing on light bouncing off it. An albedo of zero cannot return light however far the
    // lamp reaches, so the only place blue can come from is the air.
    let unlit = wall(6000.0, &[], Vec3::ZERO);
    let lit = wall(
        6000.0,
        &[Light {
            position: Vec3::new(1200.0, 0.0, 0.0),
            colour: Vec3::new(0.0, 0.0, 1.0),
            radius: 3000.0,
        }],
        Vec3::ZERO,
    );

    let (without, with) = (centre(&unlit, 1.0), centre(&lit, 1.0));
    println!("fog alone {without:?}, fog with a blue lamp in it {with:?}");

    // **Blue, which neither the fog nor the wall has any of**, and the wall returns nothing at all.
    assert!(
        with.z > without.z + 0.05,
        "the blue lamp put no blue into the air between the camera and the wall: {with:?} against \
         {without:?}"
    );
}
