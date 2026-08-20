//! That what falls out of the sky is there, that it moves, and that it stops where the game says.
//!
//! **The first version drew rain that stood still**, and nothing here would have caught it because
//! nothing here existed. The lattice was two-dimensional — hashed only across the fall — so every
//! drop in a column shared one sideways offset, and a streak a metre long where the drops are
//! centimetres apart fused the column into an unbroken rod. A rod of rain falling through itself
//! looks exactly like a rod of rain standing still.
//!
//! **The second version measured the exposure instead of the rain**, and every test here passed on
//! it. Counting pixels that differ at all sounds like the neutral thing to do and is not: metering
//! follows the frame's overall brightness, so putting rain in the air pulls every pixel down three
//! levels and putting snow in it pulls them down twenty. All 16,384 differed; 572 had a drop on
//! them. Every threshold in the file was written against the wrong one of those — one assertion
//! cleared its bar by seven pixels of rounding, and another compared two saturated counts and would
//! have *reversed* on the honest measure. `lit_by` is the answer: precipitation only ever adds
//! light, so the exposure it provokes can only subtract it, and a pixel that came out brighter than
//! the dry frame had something drawn on it.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{
    CellId, Instance, Material, Mesh, MeshId, Precipitation, Sky, StaticScene, Submesh, WorldTime,
};

mod common;

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;

/// A dark wall filling the view, so a streak has something to stand out against.
fn backdrop() -> StaticScene {
    let far = 3_000.0;
    let mesh = Mesh {
        positions: vec![
            Vec3::new(far, -far, -far),
            Vec3::new(far, far, -far),
            Vec3::new(far, far, far),
            Vec3::new(far, -far, far),
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
            diffuse: Vec3::splat(0.05),
            ..Material::default()
        }],
        &[Instance {
            mesh: MeshId(0),
            transform: Affine3A::IDENTITY,
        }],
        &[],
        Vec3::splat(0.2),
    );
    scene.ambient = None;
    scene
}

/// A frame of the plain backdrop under `weather`'s own sky at `seconds` on the world's clock.
fn falling(precipitation: Precipitation, seconds: f32) -> Vec<u8> {
    falling_in(&backdrop(), precipitation, seconds)
}

/// The same, over a scene of the caller's — which is how the water gets into one.
fn falling_in(scene: &StaticScene, precipitation: Precipitation, seconds: f32) -> Vec<u8> {
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
    // No fog: what is measured is what falls, and a march would put a gradient across it.
    renderer.set_fog(0.0);
    renderer.set_sky(Sky {
        precipitation,
        ..Sky::at(WorldTime::hours(12.0))
    });
    renderer.set_time(seconds);

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
    let eye = Vec3::ZERO;
    let view = glam::camera::rh::view::look_to_mat4(eye, Vec3::X, Vec3::Z);
    let projection =
        glam::camera::rh::proj::vulkan::perspective_infinite_reverse(75f32.to_radians(), 1.0, 0.05);
    let constants = renderer.frame_constants(view, projection, eye);
    renderer
        .render_once(&mut uploader, &constants)
        .expect("trace should run");
    // **The tone-mapped output, not the traced target.** What falls is written into a layer of its
    // own so Ray Reconstruction composites it rather than denoising it, which means it is not in
    // the frame until something puts it there — the upscaler when there is one, and the tone curve
    // when there is not. These tests attach no upscaler, so reading the finished image is what
    // covers both the drawing and the compositing.
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

/// How many levels brighter a pixel has to be before it counts as a streak.
///
/// **The exposure is the reason there is a threshold at all.** Metering follows the frame's overall
/// brightness, so putting rain in the air pulls the whole picture down a few levels and putting snow
/// in it pulls it down twenty — every pixel differs from the dry frame, whether a flake landed on it
/// or not. Four is above that drift and far below a streak: against a wall of 0.05 albedo the lit
/// ones clear twenty.
///
/// Only ever *downward*, which is what makes one constant enough. Precipitation adds light, so the
/// exposure it provokes can only darken; a pixel that came out brighter than the dry frame did so
/// because something was drawn on it.
const STREAK_LEVELS: u8 = 4;

/// Which pixels a weather lit, against the same frame with nothing falling.
fn lit_by(dry: &[u8], wet: &[u8]) -> Vec<bool> {
    dry.chunks_exact(4)
        .zip(wet.chunks_exact(4))
        .map(|(was, now)| (0..3).any(|c| now[c].saturating_sub(was[c]) > STREAK_LEVELS))
        .collect()
}

/// How many pixels a mask holds.
fn count(mask: &[bool]) -> usize {
    mask.iter().filter(|lit| **lit).count()
}

/// How many pixels are lit in one frame and not the other — what turned over between them.
///
/// A pattern that moved somewhere else entirely scores twice what it covers, since every pixel it
/// left went dark and every one it arrived at lit up. One that barely moved scores near nothing. So
/// the number worth reading is this over [`count`] of either frame, and it runs from 0 to 2.
fn turnover(before: &[bool], after: &[bool]) -> usize {
    before
        .iter()
        .zip(after)
        .filter(|(was, now)| was != now)
        .count()
}

/// How many of two frames' pixels differ at all, in any direction and by any amount.
///
/// **Only ever asked where the answer should be none.** Anything else and the exposure answers for
/// it: see `STREAK_LEVELS`.
fn differing(before: &[u8], after: &[u8]) -> usize {
    before
        .chunks_exact(4)
        .zip(after.chunks_exact(4))
        .filter(|(was, now)| was[..3] != now[..3])
        .count()
}

/// Rain as `[Weather Rain]` describes it, without needing the game installed.
fn rain() -> Precipitation {
    Precipitation {
        count: 450.0,
        diameter: 600.0,
        height: 500.0,
        fall: 4025.0,
        snow: false,
    }
}

/// Snow as `[Weather Snow]` describes it, for the comparisons rain is the other half of.
fn snow() -> Precipitation {
    Precipitation {
        count: 750.0,
        diameter: 800.0,
        height: 300.0,
        fall: 345.0,
        snow: true,
    }
}

#[test]
fn rain_falls_and_a_weather_with_none_drops_nothing() {
    let dry = falling(Precipitation::NONE, 0.0);
    let pixels = (WIDTH * HEIGHT) as usize;

    // **Present.** Against a wall this dark a streak is the only bright thing there is, so a pixel
    // the rain brightened is a pixel a drop was drawn on.
    let struck = count(&lit_by(&dry, &falling(rain(), 0.0)));
    assert!(
        struck > pixels / 100,
        "rain should reach more than a hundredth of the frame, not {struck} of {pixels}"
    );

    // **And it is rain rather than a wall of water.** Coverage is what a lattice gets wrong quietly:
    // the count above cannot tell more rain from drops drawn too wide or packed too close, and both
    // read as fog with edges rather than as weather. A tenth of the frame is the ceiling — three
    // times what a shower actually comes to here, and tripped by drawing a drop twice its width.
    assert!(
        struck < pixels / 10,
        "rain should not cover the frame — {struck} of {pixels}"
    );

    // **And a weather that drops nothing draws nothing**, which is six of the ten and every
    // interior — the shader leaves on the first line rather than marching an empty lattice. Byte for
    // byte, which is the one comparison the exposure cannot get into: it has nothing to shift.
    assert_eq!(differing(&dry, &falling(Precipitation::NONE, 4.0)), 0);
}

#[test]
fn the_drops_fall_rather_than_hanging_in_the_air() {
    // **A tenth of a second**, over which rain at 4,025 units a second travels 400 — thirty-eight
    // times the ten-unit streak it smears into, so no drop is within reach of where it was.
    let dry = falling(Precipitation::NONE, 0.0);
    let start = lit_by(&dry, &falling(rain(), 0.0));
    let later = lit_by(&dry, &falling(rain(), 0.1));
    let struck = count(&start);
    let carried = turnover(&start, &later);

    // More pixels changed hands than were ever lit, which is the whole pattern being somewhere else
    // rather than the same one nudged: every drop's pixels went dark and as many others lit up.
    assert!(
        carried > struck,
        "the drops should have moved on — {carried} pixels turned over against {struck} lit"
    );

    // And not more than a complete turnover, which is what says the two frames disagree about the
    // rain and nothing besides it. Twice the coverage is the ceiling; three times is slack.
    assert!(
        carried < struck * 3,
        "only the rain should have changed — {carried} against {struck} lit"
    );
}

/// How many lit pixels have a lit neighbour beside them, counted along each axis.
#[derive(Debug)]
struct Runs {
    down: usize,
    across: usize,
}

impl Runs {
    /// How far past round the shape sits: one where the two axes agree, higher the taller it is.
    fn elongation(&self) -> f32 {
        self.down as f32 / self.across.max(1) as f32
    }
}

/// Which way `mask`'s lit pixels run.
///
/// **What tells a streak from a blob.** The fall is straight down in these frames, so a drop smeared
/// over the shutter is tall and sub-pixel wide while a flake that barely moved is as wide as it is
/// high. Runs rather than extents, because at this size a streak is seven pixels and a flake is
/// one — there is nothing to fit a shape to, only neighbours to count.
fn runs(mask: &[bool]) -> Runs {
    let at = |x: u32, y: u32| mask[(y * WIDTH + x) as usize];
    let mut runs = Runs { down: 0, across: 0 };
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if !at(x, y) {
                continue;
            }
            runs.down += usize::from(y + 1 < HEIGHT && at(x, y + 1));
            runs.across += usize::from(x + 1 < WIDTH && at(x + 1, y));
        }
    }
    runs
}

#[test]
fn snow_is_drawn_round_and_slow_where_rain_is_drawn_as_a_streak() {
    // The two differ by the keys the ini gives them and by nothing written here: `Snow Gravity
    // Scale` makes a flake a tenth the weight of a drop for its area, so it drifts where rain falls,
    // and a flake is drawn wider because loose crystal is.
    let dry = falling(Precipitation::NONE, 0.0);
    let snowing = lit_by(&dry, &falling(snow(), 0.0));
    let raining = lit_by(&dry, &falling(rain(), 0.0));

    // **Wider**, and by a lot: more flakes, a wider cylinder, and each drawn three times the radius
    // of a drop.
    assert!(
        count(&snowing) > count(&raining) * 4,
        "snow should cover far more of the frame than rain — {} against {}",
        count(&snowing),
        count(&raining)
    );

    // **And round rather than streaked, which is the whole of what a fall speed does to a shape.** A
    // streak is how far the drop travelled while the shutter was open, so rain at nine metres a
    // second smears fifteen centimetres and is drawn a hundredth of that across — tall and
    // sub-pixel wide — while a flake at three quarters of a metre a second moves less than its own
    // width and comes out as the disc it is. Measured down the screen against across it, because
    // the fall is straight down here.
    // **Bounded on the excess over round, not on the ratio itself.** Snow covers a third of the
    // frame, so flakes touch and the neighbours that fusing adds land in both directions equally —
    // which pulls any ratio measured here toward one whatever the flakes are shaped like. What
    // survives that is how far *past* one it sits: a sixth of the way with a flake drawn round,
    // and a hundred times that with rain's own streak length handed to it.
    let streaked = runs(&raining);
    let round = runs(&snowing);
    assert!(
        streaked.elongation() > 4.0,
        "a drop should be drawn as a streak — {streaked:?} is {} past round",
        streaked.elongation()
    );
    assert!(
        round.elongation() < 1.15,
        "a flake should be drawn round — {round:?} is {} past round",
        round.elongation()
    );

    // **Slower, over a window short enough to tell the two apart.** Two thousandths of a second
    // carries a drop 8 units, three quarters of the streak it is drawn as, so nearly all of it has
    // gone; a flake goes 0.7 against the 2.1 it is drawn as, so most of it is still there.
    //
    // Read as a fraction of what each covers, because the absolute counts cannot be compared: snow
    // lights ten times as many pixels, so it turns more of them over while moving a tenth as far.
    // That is the trap the first version of this test fell into, before the exposure hid it.
    let turned = |precipitation, lit: &[bool]| {
        turnover(lit, &lit_by(&dry, &falling(precipitation, 0.002))) as f32 / count(lit) as f32
    };
    let drifted = turned(snow(), &snowing);
    let fell = turned(rain(), &raining);
    assert!(
        fell > 1.5,
        "rain should have turned over completely — {fell} of what it covers"
    );
    assert!(
        drifted < 1.0,
        "snow should still be most of where it was — {drifted} of what it covers"
    );
}

#[test]
fn nothing_falls_under_the_water_and_a_surface_below_the_eye_cuts_it_off() {
    // **Every ray submerged**, with the surface far overhead. A drop that reached the water stopped
    // being a drop, and the lattice knows nothing about that on its own — so this hung a downpour in
    // the bay beside the eye, lit by a sun the water had already taken most of.
    let mut under = backdrop();
    under.water_level = Some(20_000.0);
    let poured = differing(
        &falling_in(&under, Precipitation::NONE, 0.0),
        &falling_in(&under, rain(), 0.0),
    );
    assert_eq!(
        poured, 0,
        "rain reached under the water, on {poured} pixels"
    );

    // **And a surface below the eye cuts the downward rays at it**, which is the half a hit does not
    // already cover: a ray that finds nothing within the weather's reach would otherwise carry the
    // rain straight down through open water. Fifty units under the eye, so the steeper of those rays
    // cross before the hundred it takes for a streak to be worth drawing at all.
    let mut bay = backdrop();
    bay.water_level = Some(-50.0);

    // **Each against its own dry frame**, because a water level is not free of the rest of the
    // picture: the backdrop runs on below it, a wall the sun has to reach through water is darker,
    // and the exposure follows that across every pixel — which is what `lit_by` is proof against and
    // a plain difference is not. What is left of the bias runs the wrong way for this test, since a
    // streak stands out further against the darker wall.
    let lower = |dry: &[u8], wet: &[u8]| {
        let half = (HEIGHT / 2 * WIDTH) as usize * 4;
        count(&lit_by(&dry[half..], &wet[half..]))
    };
    let over_water = lower(
        &falling_in(&bay, Precipitation::NONE, 0.0),
        &falling_in(&bay, rain(), 0.0),
    );
    let over_land = lower(&falling(Precipitation::NONE, 0.0), &falling(rain(), 0.0));
    assert!(
        over_water < over_land * 3 / 4,
        "open water should take the rain out of the rays that reach it — {over_water} against \
         {over_land}"
    );
}
