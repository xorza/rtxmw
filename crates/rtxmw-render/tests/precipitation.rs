//! That what falls out of the sky is there, that it moves, and that it stops where the game says.
//!
//! **The first version drew rain that stood still**, and nothing here would have caught it because
//! nothing here existed. The lattice was two-dimensional — hashed only across the fall — so every
//! drop in a column shared one sideways offset, and a streak a metre long where the drops are
//! centimetres apart fused the column into an unbroken rod. A rod of rain falling through itself
//! looks exactly like a rod of rain standing still.

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

/// A frame under `weather`'s own sky at `seconds` on the world's clock.
fn falling(precipitation: Precipitation, seconds: f32) -> Vec<u8> {
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
            &backdrop(),
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

/// How many of two frames' pixels differ at all.
fn moved(before: &[u8], after: &[u8]) -> usize {
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

#[test]
fn rain_falls_and_a_weather_with_none_drops_nothing() {
    let dry = falling(Precipitation::NONE, 0.0);
    let wet = falling(rain(), 0.0);
    let pixels = (WIDTH * HEIGHT) as usize;

    // **Present.** Against a wall this dark a streak is the only bright thing there is, so what
    // moves between the two frames is the rain and nothing else.
    let struck = moved(&dry, &wet);
    assert!(
        struck > pixels / 100,
        "rain should reach more than a hundredth of the frame, not {struck} of {pixels}"
    );

    // **And a weather that drops nothing draws nothing**, which is six of the ten and every
    // interior — the shader leaves on the first line rather than marching an empty lattice.
    assert_eq!(moved(&dry, &falling(Precipitation::NONE, 4.0)), 0);
}

#[test]
fn the_drops_fall_rather_than_hanging_in_the_air() {
    // **A tenth of a second**, over which rain at 4,025 units a second travels 400 — six times the
    // metre-long streak it smears into, so every drop has left where it was.
    let start = falling(rain(), 0.0);
    let later = falling(rain(), 0.1);
    let pixels = (WIDTH * HEIGHT) as usize;

    let struck = moved(&falling(Precipitation::NONE, 0.0), &start);
    let carried = moved(&start, &later);
    assert!(
        carried > struck / 2,
        "the drops should have moved on — {carried} pixels changed against {struck} the rain \
         covers at all"
    );
    assert!(carried < pixels, "the whole frame should not be rain");
}

#[test]
fn snow_falls_slower_and_wider_than_rain() {
    // The two differ by the keys the ini gives them and by nothing written here: `Snow Gravity
    // Scale` makes a flake a tenth the weight of a drop for its area, so it drifts where rain
    // falls, and a flake is drawn wider because loose crystal is.
    let flakes = Precipitation {
        count: 750.0,
        diameter: 800.0,
        height: 300.0,
        fall: 345.0,
        snow: true,
    };
    let still = falling(Precipitation::NONE, 0.0);
    let snowing = moved(&still, &falling(flakes, 0.0));
    assert!(snowing > 0, "snow should reach the frame at all");

    // **Slower, measured over a window rain crosses six times.** In a tenth of a second a flake
    // travels 34 units against a drop's 400, so far less of the frame turns over.
    let snow_carried = moved(&falling(flakes, 0.0), &falling(flakes, 0.1));
    let rain_carried = moved(&falling(rain(), 0.0), &falling(rain(), 0.1));
    assert!(
        snow_carried < rain_carried,
        "snow should drift where rain falls — {snow_carried} against {rain_carried}"
    );
}
