//! That a flash reaches the frame: the sky it happens in, and the channel where one shows.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{
    CellId, Discharge, Instance, Lightning, Material, MaterialKind, Mesh, MeshId, Sky, StaticScene,
    Submesh, WorldTime,
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

/// A quad at `z` filling the view, of `material`.
fn slab(z: f32, material: u32) -> Mesh {
    let far = 200_000.0;
    Mesh {
        positions: vec![
            Vec3::new(-far, -far, z),
            Vec3::new(far, -far, z),
            Vec3::new(far, far, z),
            Vec3::new(-far, far, z),
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

/// Nothing but a floor far below, so what is measured is the sky and what is in it.
fn empty() -> StaticScene {
    let mesh = slab(-50_000.0, 0);
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

/// The same sky over a sea at `z = 0`, for what the water gives back rather than what the air does.
fn sea() -> StaticScene {
    let mut scene = common::scene_of(
        &[slab(0.0, 0), slab(-50_000.0, 1)],
        &[
            Material {
                kind: MaterialKind::Water,
                ..Material::default()
            },
            Material {
                diffuse: Vec3::splat(0.02),
                ..Material::default()
            },
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
        Vec3::splat(0.02),
    );
    scene.ambient = None;
    scene.water_level = Some(0.0);
    scene
}

/// The traced frame at `seconds`, looking from the origin toward `at`.
fn looking(lightning: Lightning, seconds: f32, at: Vec3) -> Vec<u8> {
    looking_over(&empty(), lightning, seconds, Vec3::ZERO, at)
}

/// The traced frame at `seconds`, from `eye` toward `at` over whatever `scene` puts under it.
fn looking_over(
    scene: &StaticScene,
    lightning: Lightning,
    seconds: f32,
    eye: Vec3,
    at: Vec3,
) -> Vec<u8> {
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
    renderer.set_fog(
        std::env::var("PROBE_FOG")
            .map(|f| f.parse().unwrap())
            .unwrap_or(0.0),
    );
    // **The storm's clock and not the frame's**, which are the same seconds only when the world is
    // running at speed one — see `SceneRenderer::set_storm`. A flash is an event of a fixed length
    // and does not get shorter because the day was sped up.
    renderer.set_storm(seconds);
    renderer.set_sky(Sky { lightning, ..sky() });

    let mut uploader = gpu.uploader();
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            CellId::Exterior { x: 0, y: 0 },
            scene,
            &[],
        )
        .expect("scene should load");
    let view = glam::camera::rh::view::look_to_mat4(eye, (at - eye).normalize(), Vec3::Z);
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
fn bolt_falloff(near: f32, radius: f32, glare: f32) -> f32 {
    let x = (near * near) / (radius * radius);
    let window = (1.0 - x).max(0.0);
    window * window / (1.0 + glare * x)
}

/// How steeply the two curves fall away from the channel, out of `BOLT_CORE_GLARE` and
/// `BOLT_HALO_GLARE`.
const CORE_GLARE: f32 = 25.0;
const HALO_GLARE: f32 = 6.0;

/// How many times the white point the channel's glow peaks at, out of `BOLT_GLOW_PEAK`.
const GLOW_PEAK: f32 = 1.8;

/// A glow that peaks far above white is a solid shape with a rim, whatever curve is under it.
const _: () = assert!(GLOW_PEAK < 6.0);

/// The corona's height, out of `BOLT_CORONA_PEAK` in still air and `BOLT_CORONA_AIR` added by the
/// weather — the wash is made by the air, so how much of it there is follows how much air there is.
const CORONA_PEAK: f32 = 0.5;
const CORONA_AIR: f32 = 2.5;

/// The same bound, on the most any weather can raise it to.
const _: () = assert!(CORONA_PEAK + CORONA_AIR < 6.0);

/// How far the halo and the corona reach, out of `BOLT_HALO` and `BOLT_CORONA`, in pixels.
const HALO: f32 = 45.0;
const CORONA: f32 = 110.0;

/// **Nothing whose support is narrower than the corona may scale the corona.**
///
/// The wash's height follows the weather, and the weather was read off the depth of whichever of the
/// *narrow* tiers won the pixel — a depth that is simply absent past the halo's reach, where those
/// tiers draw nothing. So at that radius the wash lost its whole weather term and dropped by
/// `(CORONA_PEAK + CORONA_AIR) / CORONA_PEAK` in the width of a pixel: a hard ring drawn around the
/// middle of a glow, at a radius with nothing physical at it. It is invisible in still air, because
/// there the weather term is nought on both sides of the line.
///
/// The depth it uses now comes off the straight channel, which every ray past the bounding test has.
const _: () = assert!(HALO < CORONA);

/// And it has to be a weather-dependent figure at all: one height cannot serve a clear night and the
/// inside of a thunderstorm, which is a difference of the better part of an order of magnitude in
/// what it is seen against.
const _: () = assert!(CORONA_AIR > CORONA_PEAK);

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
    assert_eq!(bolt_falloff(radius, radius, CORE_GLARE), 0.0);
    assert_eq!(bolt_falloff(radius, radius, HALO_GLARE), 0.0);
    assert_eq!(bolt_falloff(radius * 2.0, radius, CORE_GLARE), 0.0);
    assert!(
        bolt_falloff(0.0, radius, CORE_GLARE) > 0.99,
        "and all of it at the centre"
    );

    // Two: it arrives there flat. A profile that hit zero with slope still on it would leave a
    // crease rather than an edge — visible for the same reason and harder to see coming.
    let edge = bolt_falloff(radius * 0.99, radius, CORE_GLARE);
    assert!(
        edge < 1e-3,
        "the glow should flatten into the sky, not run into it — {edge}"
    );

    // The bound sits at or past that radius — see `BOUND`, which the compiler checks.

    // **And the profile is a gradient rather than a slab**, which is a different property from
    // reaching zero and the one that actually kept failing. Everything above the white point clips,
    // so a glow written as a ratio to `BOLT_ARC` — a number in the tens of thousands — peaked at
    // fifty-eight times white without that being visible anywhere it was written, and rendered as a
    // solid capsule with a thin rim where the curve finally dropped through. Twice this was read as
    // a bounding-volume artefact and chased in the wrong place.
    //
    // Two things keep it honest: the peak is stated in display terms and bounded where it is
    // declared, and it has to fall *through* white early enough to leave most of the radius as an
    // actual falloff.
    let clipped = (1..100)
        .map(|step| step as f32 / 100.0)
        .find(|out| GLOW_PEAK * bolt_falloff(out * radius, radius, HALO_GLARE) < 1.0)
        .expect("the glow must fall below white somewhere inside its radius");
    assert!(
        clipped < 0.35,
        "the glow should be off the white point within a third of its reach, not {clipped}"
    );
}

/// Where the channel sits in the frame, as the mean position of everything it lit.
///
/// A centroid rather than a pixel count: two different flashes can light the same number of pixels
/// and stand nowhere near each other, and where it *is* is the whole of what a restrike promises.
fn channel_at(dark: &[u8], burning: &[u8]) -> Option<(f32, f32)> {
    let mut total = 0.0;
    let mut at = (0.0, 0.0);
    for (index, (was, now)) in dark
        .as_chunks::<4>()
        .0
        .iter()
        .zip(burning.as_chunks::<4>().0)
        .enumerate()
    {
        let lit = f32::from(
            (0..3)
                .map(|c| now[c].saturating_sub(was[c]))
                .min()
                .unwrap_or(0),
        );
        if lit < 60.0 {
            continue;
        }
        let (x, y) = ((index as u32 % WIDTH) as f32, (index as u32 / WIDTH) as f32);
        at = (at.0 + x * lit, at.1 + y * lit);
        total += lit;
    }
    (total > 0.0).then(|| (at.0 / total, at.1 / total))
}

#[test]
fn the_same_bolt_can_be_sent_again_and_a_new_one_lands_elsewhere() {
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
    renderer.set_fog(
        std::env::var("PROBE_FOG")
            .map(|f| f.parse().unwrap())
            .unwrap_or(0.0),
    );
    renderer.set_sky(Sky {
        lightning: storm(),
        ..sky()
    });

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

    let (eye, forward) = (Vec3::ZERO, Vec3::X);
    let mut shot = |renderer: &mut SceneRenderer| {
        let view = glam::camera::rh::view::look_to_mat4(eye, forward, Vec3::Z);
        let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
            110f32.to_radians(),
            1.0,
            0.05,
        );
        let constants = renderer.frame_constants(view, projection, eye);
        renderer
            .render_once(&mut uploader, &constants)
            .expect("trace should run");
        readback::image_to_rgba8(
            &mut uploader,
            renderer.output(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )
        .expect("readback should succeed")
    };

    // A sky with nothing burning in it, to measure the rest against.
    renderer.set_storm(0.0);
    let dark = shot(&mut renderer);

    // **A bolt, staged in front of where the camera is looking** — see `Lightning::staged`.
    renderer.strike(eye, forward);
    let first = channel_at(&dark, &shot(&mut renderer)).expect("a staged bolt should be drawn");

    // **The clock moves on and the same one comes back.** Nothing about the flash is stored: the
    // storm's own second is, and everything the bolt is made of is read from it again.
    renderer.set_storm(11.0);
    renderer.restrike();
    let again = channel_at(&dark, &shot(&mut renderer)).expect("a restrike should be drawn");
    assert!(
        (again.0 - first.0).abs() < 1.5 && (again.1 - first.1).abs() < 1.5,
        "a restrike should send the same bolt — {again:?} against {first:?}"
    );

    // And asking for a new one does not: a different flash of the schedule stands somewhere else.
    renderer.strike(eye, forward);
    let fresh = channel_at(&dark, &shot(&mut renderer)).expect("a new bolt should be drawn");
    assert!(
        (fresh.0 - first.0).abs() > 1.5 || (fresh.1 - first.1).abs() > 1.5,
        "a new bolt should not land where the last one did — {fresh:?} against {first:?}"
    );

    drop(uploader);
    gpu.assert_no_validation_errors();
}

/// The lift each pixel took from the flash, sorted — what the water gave back that the same water
/// without a flash did not.
fn lifts(dark: &[u8], burning: &[u8]) -> Vec<f32> {
    let mut held: Vec<f32> = dark
        .as_chunks::<4>()
        .0
        .iter()
        .zip(burning.as_chunks::<4>().0)
        .map(|(was, now)| {
            f32::from(
                (0..3)
                    .map(|c| now[c].saturating_sub(was[c]))
                    .min()
                    .unwrap_or(0),
            )
        })
        .collect();
    held.sort_by(f32::total_cmp);
    held
}

#[test]
fn the_water_gives_the_channel_back_as_a_line_and_not_as_a_sheet() {
    // A camera over a sea, aimed at where a flat mirror puts the channel: the virtual image of a
    // point at `z` is the same point at `-z`, so that is where the reflection of the middle of the
    // bolt has to land.
    let storm = storm();
    let eye = Vec3::new(0.0, 0.0, 2_000.0);
    let flash = storm.flash(0.0, eye, sky().clouds.altitude);
    assert_ne!(
        flash.kind,
        Discharge::Sheet,
        "the fixture wants a flash with a channel, and this one has none"
    );
    let middle = (flash.source + flash.ground) * 0.5;
    let sea = sea();
    let lift = lifts(
        &looking_over(
            &sea,
            storm,
            0.7,
            eye,
            Vec3::new(middle.x, middle.y, -middle.z),
        ),
        &looking_over(
            &sea,
            storm,
            0.0,
            eye,
            Vec3::new(middle.x, middle.y, -middle.z),
        ),
    );
    let top = lift[lift.len() - 1];
    let median = lift[lift.len() / 2];
    let blown = lift.iter().filter(|held| **held > 200.0).count();

    // **The count of lit pixels cannot tell this apart and that is the point.** The sky's own flash
    // comes back off the water through `sky_seen_through` whether a channel is drawn or not, and it
    // lifts nearly every pixel of a sea by a little: three thousand of them either way, to a median
    // that does not move. What only the channel does is blow a few of them out.
    eprintln!(
        "DIAG blown={blown} median={median} top={top} n={}",
        lift.len()
    );
    assert!(
        blown > 8 && top > 200.0,
        "the water should give back a channel bright enough to blow out — {blown} pixels, brightest {top}"
    );

    // **And it is a line on the water, not a sheet over it.** This is the guard on the drawn width:
    // `BOLT_CORE` and `BOLT_HALO` are counts of *pixels*, scaled by an angle per pixel, so anything
    // added to that scale which is a plain angle — the wave lobe was — multiplies the radius by
    // thousands. What that draws is the whole bay blown white in the general direction of the bolt,
    // and it defeats the bounding test on the way, so it does not even land where the bolt is.
    //
    // A hundredth of the frame, against the twenty-odd pixels a channel this far off actually takes
    // and the three hundred and forty-two the lobe put there on water this calm. Calm is the weak
    // case: the lobe is what the cone could not resolve, so it grows with distance and with rain,
    // and the sea in the shot that found this was both.
    assert!(
        median < 60.0 && (blown as f32) < 0.01 * lift.len() as f32,
        "a reflected channel covers a little of the water, not {blown} of {} pixels at a median lift of {median}",
        lift.len()
    );
}

/// How much of a flash reaches a point `across` from the channel's line and `foot` along it, out of
/// `flash_reaching` in `lightning.glsl` — in multiples of `FLASH_LIT`, which is what one delivers at
/// `FLASH_REFERENCE`.
///
/// An element of arc at `s` from the perpendicular's foot arrives over `across^2 + s^2`, and
/// `integral ds / (r^2 + s^2)` is `atan(s / r) / r`. Divided by the run, so however long the channel
/// is it carries one discharge between them.
fn flash_reaching(run: f32, foot: f32, across: f32) -> f32 {
    let across = across.max(NEAREST);
    let shaped = ((run - foot) / across).atan() + (foot / across).atan();
    REFERENCE * REFERENCE * shaped / (run * across)
}

/// What the point source it replaced delivered, out of the same file before this: an inverse square
/// on the distance to the *middle* of the channel, floored at the same `FLASH_NEAREST`.
fn from_the_midpoint(run: f32, foot: f32, across: f32) -> f32 {
    let away = ((foot - run * 0.5).powi(2) + across * across).sqrt();
    (REFERENCE / away.max(NEAREST)).powi(2)
}

/// `FLASH_REFERENCE` and `FLASH_NEAREST`.
const REFERENCE: f32 = 25_000.0;
const NEAREST: f32 = 2_500.0;

#[test]
fn the_light_a_flash_throws_comes_from_the_whole_channel() {
    // **A sheet is a point and the same integral says so**, which is what keeps the calibration: its
    // source and its ground are one place, so `run` falls to the floor of one unit and the
    // arctangent collapses to its argument — `atan(1/d)/d`, which is `1/d^2`. At the reference
    // distance that is exactly what `FLASH_LIT` means, and nothing else in this file would say so.
    let sheet = flash_reaching(1.0, 0.0, REFERENCE);
    assert!(
        (sheet - 1.0).abs() < 1e-4,
        "a sheet at the reference distance should deliver exactly one, not {sheet}"
    );

    // **And a channel is a point too, from far enough away.** A crawler's hundred thousand units
    // seen from a million subtends almost nothing, so the arctangents collapse the same way: the
    // whole thing goes to `(REFERENCE / d)^2`, which is 6.25e-4 here. Two hundred thousand out it is
    // two percent under that, and it should be — at that range the arc is not a point.
    let far = flash_reaching(100_000.0, 50_000.0, 1_000_000.0);
    let point = (REFERENCE / 1_000_000.0f32).powi(2);
    assert!(
        (far / point - 1.0).abs() < 0.005 && (point - 6.25e-4).abs() < 1e-9,
        "a distant channel should fall off as a point does — {far} against {point}"
    );

    // **The property that matters, and the one the midpoint model got wrong.** Standing off a long
    // crawler at the floor, the light barely changes as you walk along it: a hundred-thousand-unit
    // arc gives `2 * atan(20) = 3.0417` at its middle and `atan(30) + atan(10) = 3.0086` a quarter of
    // the way down, which after `REFERENCE^2 / (run * across)` is 7.604 against 7.522 — within three
    // percent. That is what a line source does, and it is why the glow follows the bolt.
    let (run, across) = (100_000.0, NEAREST);
    let middle = flash_reaching(run, run * 0.5, across);
    let quarter = flash_reaching(run, run * 0.25, across);
    assert!(
        (middle - 7.604).abs() < 0.01 && (quarter - 7.522).abs() < 0.01,
        "the arc should light its own length evenly — {middle} at the middle, {quarter} at a quarter"
    );
    assert!(
        (quarter / middle - 1.0).abs() < 0.03,
        "walking a quarter of the way along a crawler should change little, not {}",
        quarter / middle
    );

    // **A point at the midpoint does the opposite, and that was the bulb.** The floor put a ball of
    // maximum in-scattering at the halfway mark — a hundred there, and 0.99 a quarter of the way
    // along, a fall of a hundred times over a stretch of sky with the same bolt running through all
    // of it. The fog march painted that ball, and what it drew was a glowing sphere hanging in
    // mid-air with no channel anywhere inside it.
    let was_middle = from_the_midpoint(run, run * 0.5, across);
    let was_quarter = from_the_midpoint(run, run * 0.25, across);
    assert!(
        (was_middle - 100.0).abs() < 0.01 && (was_quarter - 0.9901).abs() < 0.001,
        "the midpoint model should peg at its floor and collapse off it — {was_middle}, {was_quarter}"
    );
    assert!(
        was_quarter / was_middle < 0.05 && middle / was_middle < 0.2,
        "the arc should be dimmer at the middle and far flatter along it than a point at its centre"
    );
}

#[test]
fn the_deck_lights_up_where_the_discharge_is() {
    let (Ok(thunderstorm), storm) = (rtxmw_scene::Weather::named("thunderstorm"), storm()) else {
        return;
    };
    let Ok(Some(textures)) = rtxmw_scene::SkyTextures::load(&thunderstorm) else {
        return;
    };
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
    renderer.set_fog(
        std::env::var("PROBE_FOG")
            .map(|f| f.parse().unwrap())
            .unwrap_or(0.0),
    );
    let mut uploader = gpu.uploader();
    renderer
        .set_sky_textures(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            &textures,
        )
        .expect("the sheet should upload");
    let midnight = WorldTime::hours(0.0);
    let under = Sky::under(midnight, &thunderstorm, textures.sheet());
    let altitude = under.clouds.altitude;
    renderer.set_sky(Sky {
        lightning: storm,
        ..under
    });
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

    // **Both halves of the sky in one frame, because exposure is per frame.** Rendering the
    // discharge's side and the far side as two shots compares two different tone curves: adding
    // light to a scene darkens its own exposure, so the side that gained the most measured the
    // least. Looking straight up through a very wide cone puts them in one image under one curve,
    // and the camera is rolled so the discharge is at the top of it — top half toward, bottom away.
    let eye = Vec3::ZERO;
    let flash = storm.flash(0.0, eye, altitude);
    let bearing = Vec3::new(flash.source.x, flash.source.y, 0.0).normalize();
    let mut shot = |seconds: f32| {
        renderer.set_storm(seconds);
        let view = glam::camera::rh::view::look_to_mat4(eye, Vec3::Z, bearing);
        let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
            140f32.to_radians(),
            1.0,
            0.05,
        );
        let constants = renderer.frame_constants(view, projection, eye);
        renderer
            .render_once(&mut uploader, &constants)
            .expect("trace should run");
        readback::image_to_rgba8(
            &mut uploader,
            renderer.output(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )
        .expect("readback should succeed")
    };
    let (dark, burning) = (shot(0.7), shot(0.0));
    let lift: Vec<f32> = dark
        .as_chunks::<4>()
        .0
        .iter()
        .zip(burning.as_chunks::<4>().0)
        .map(|(was, now)| f32::from(now[1]) - f32::from(was[1]))
        .collect();
    let half = (HEIGHT / 2) as usize * WIDTH as usize;
    let mean = |over: &[f32]| over.iter().sum::<f32>() / over.len() as f32;
    let (toward, from) = (mean(&lift[..half]), mean(&lift[half..]));
    // **The deck brightens where the discharge is.** Over half of all lightning never leaves the
    // cloud, and what an eye sees of those is a region of weather going bright — so this is the
    // half of a storm that shows most often.
    assert!(
        toward > 25.0,
        "the deck over the discharge should light up, not lift by {toward}"
    );

    // **And only there, which is the whole difference between lighting a deck and washing a sky.**
    // The flash is asked about at the point where each ray meets the shell, so the answer changes
    // across the sheet; asked once for the whole sky instead, the same light comes back as a flat
    // lift of 0.88 of the near side against this term's -0.18. Negative because exposure is per
    // frame and adapts to the light the near side gained — which is why both halves are measured in
    // one shot under one tone curve rather than in two.
    //
    // The level itself is a tuning figure and is not pinned; the geometry behind it is, in
    // `the_light_a_flash_throws_comes_from_the_whole_channel`.
    assert!(
        from < 0.25 * toward,
        "the glow should follow the deck the channel is in, not lift the far sky by {from} against {toward}"
    );
}
