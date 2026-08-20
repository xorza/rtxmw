//! That a flash reaches the frame: the sky it happens in, and the channel where one shows.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{
    CellId, Discharge, Instance, Lightning, Material, Mesh, MeshId, Sky, StaticScene, Submesh,
    WorldTime,
};

mod common;

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;

/// A thunderstorm's schedule, as `[Weather Thunderstorm]` writes it.
fn storm() -> Lightning {
    Lightning {
        frequency: 0.4,
        threshold: 0.6,
        decrement: 4.0,
    }
}

/// The sky the fixture stands under, which is where the deck's altitude comes from — and the flash
/// with it, so the camera below can be aimed at something the renderer agrees exists.
fn sky() -> Sky {
    Sky::at(WorldTime::hours(0.0))
}

/// Nothing but a floor far below, so what is measured is the sky and what is in it.
fn empty() -> StaticScene {
    let far = 200_000.0;
    let mesh = Mesh {
        positions: vec![
            Vec3::new(-far, -far, -50_000.0),
            Vec3::new(far, -far, -50_000.0),
            Vec3::new(far, far, -50_000.0),
            Vec3::new(-far, far, -50_000.0),
        ],
        normals: vec![Vec3::Z; 4],
        uvs: vec![Vec2::ZERO; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
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
            diffuse: Vec3::splat(0.02),
            ..Material::default()
        }],
        &[Instance {
            mesh: MeshId(0),
            transform: Affine3A::IDENTITY,
        }],
        &[],
        Vec3::splat(0.02),
    );
    scene.ambient = None;
    scene
}

/// The traced frame at `seconds`, looking from the origin toward `at`.
fn looking(lightning: Lightning, seconds: f32, at: Vec3) -> Vec<u8> {
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
    renderer.set_denoise_passes(0);
    renderer.set_fog(0.0);
    renderer.set_time(seconds);
    renderer.set_sky(Sky { lightning, ..sky() });

    let mut uploader = gpu.uploader();
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            CellId::Exterior { x: 0, y: 0 },
            &empty(),
            &[],
        )
        .expect("scene should load");
    let eye = Vec3::ZERO;
    let view = glam::camera::rh::view::look_to_mat4(eye, at.normalize(), Vec3::Z);
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

/// How many pixels the flash lit that the same sky without one did not.
///
/// **Against a dark frame rather than by absolute brightness**, because a midnight sky has a moon in
/// it and a moon is near white too — which is what an absolute threshold counted before this did.
fn arc(dark: &[u8], burning: &[u8]) -> usize {
    dark.as_chunks::<4>()
        .0
        .iter()
        .zip(burning.as_chunks::<4>().0)
        .filter(|(was, now)| (0..3).all(|c| now[c].saturating_sub(was[c]) > 60))
        .count()
}

#[test]
fn a_channel_is_drawn_where_the_discharge_has_one_and_nowhere_else() {
    // The schedule settles where the flash stands; the camera is aimed down the middle of it, so
    // whatever is drawn has to be on screen.
    let storm = storm();
    let flash = storm.flash(0.0, Vec3::ZERO, sky().clouds.altitude);
    let middle = (flash.source + flash.ground) * 0.5;
    assert_ne!(
        flash.kind,
        Discharge::Sheet,
        "the fixture wants a flash with a channel, and this one has none"
    );

    // **Two thirds of a second into the same storm is the reference**, because the flash lasts the
    // quarter second `Flash Decrement` buys and nothing is left of it by then.
    let dark = looking(storm, 0.7, middle);
    let burning = arc(&dark, &looking(storm, 0.0, middle));
    assert!(
        burning > 8,
        "a channel should be drawn where the discharge is — {burning} pixels of it"
    );

    // And a weather the ini gives no thunder draws nothing at all, which is nine of the ten.
    let clear = arc(&dark, &looking(Lightning::NONE, 0.0, middle));
    assert_eq!(
        clear, 0,
        "clear weather should draw no channel, not {clear}"
    );
}

/// How much of the channel's glow survives at `near` from a radius of `radius`, out of
/// `bolt_falloff` in `lightning.glsl`.
///
/// Repeated here because a shader function is not visible from Rust, and the property below is the
/// one thing standing between this feature and an artefact that came back twice.
fn bolt_falloff(near: f32, radius: f32) -> f32 {
    let x = (near * near) / (radius * radius);
    let window = (1.0 - x).max(0.0);
    window * window / (1.0 + GLARE * x)
}

/// How steeply the glare falls away from the channel, out of `BOLT_GLARE`.
const GLARE: f32 = 25.0;

/// How far past its radius `BOLT_BOUND` lets the bounding test reach.
///
/// **Checked where it is written rather than where it is used.** The bound must sit at or past the
/// radius the profile has already fallen to nothing at, or the cut lands where there is still
/// something to cut — which is the whole of the artefact below. It is a constant, so the check is a
/// constant too, and the compiler is the right thing to make it.
const BOUND: f32 = 1.25;
const _: () = assert!(BOUND >= 1.0, "the bound may not reach inside the glow");

#[test]
fn the_glow_reaches_zero_before_the_bound_cuts_it() {
    // **The artefact this exists to make impossible.** `bolt_along` skips the march for rays far
    // from the channel, and a skip is sound only where what is skipped is *nothing*. The profile was
    // `1 / (1 + x^2)` — the right shape, and one that never reaches zero — so the bound was always
    // discarding something, and the capsule it describes stood in the sky as a hard-edged pill
    // around the bolt. Pushing the bound out only made the discarded value smaller, and every time
    // the flash grew brighter the step came back: a fixed cut through a curve that never lands is a
    // step whose visibility is a matter of exposure, not of distance.
    //
    // Two facts make it impossible rather than merely faint, and both are pinned here.

    // One: the glow is exactly nothing at its own radius and beyond it.
    let radius = 40.0;
    assert_eq!(bolt_falloff(radius, radius), 0.0);
    assert_eq!(bolt_falloff(radius * 2.0, radius), 0.0);
    assert!(
        bolt_falloff(0.0, radius) > 0.99,
        "and all of it at the centre"
    );

    // Two: it arrives there flat. A profile that hit zero with slope still on it would leave a
    // crease rather than an edge — visible for the same reason and harder to see coming.
    let edge = bolt_falloff(radius * 0.99, radius);
    assert!(
        edge < 1e-3,
        "the glow should flatten into the sky, not run into it — {edge}"
    );

    // The bound sits at or past that radius — see `BOUND`, which the compiler checks.
}
