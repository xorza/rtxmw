//! Steady-state rendering must not allocate on the heap.
//!
//! A renderer that allocates per frame gets unpredictable frame times: the allocator occasionally
//! takes a slow path, and at 60 fps a single 2 ms stall is a dropped frame. Averages hide that; a
//! hard zero does not. The number here is a budget, not a guideline — raising it needs a reason
//! recorded in the commit.
//!
//! This runs as a `#[test]` rather than a benchmark deliberately. It is a correctness property with
//! a pass/fail answer, it wants a debug build, and criterion would triple the compile time to
//! measure something that is not about wall-clock speed.
//!
//! **Scope:** a whole frame of the real renderer with a stationary camera — building the frame
//! constants, recording the trace, submitting it, and waiting for it. That is everything the engine
//! does per frame except swapchain acquire and present, which need a surface.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::TestGpu;
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{Ambient, Instance, Light, Material, MeshId, Submesh};

mod common;

// The profiler needs to see every allocation in this binary, so it must be the global allocator.
// This is why the test lives in its own integration-test binary rather than beside the unit tests.
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Frames run before measuring, to let one-time setup settle: device bring-up, lazily built
/// function tables, and any first-call caching inside the driver.
const WARMUP_FRAMES: usize = 16;

/// Frames measured. Large enough that a single stray allocation is unambiguous, small enough to
/// stay fast.
const MEASURED_FRAMES: usize = 256;

/// Allocations permitted across the whole measured window, not per frame.
///
/// Zero is the intended value and the one currently observed. It is expressed as a budget rather
/// than `== 0` so a driver that allocates a fixed amount on some path can be accommodated
/// deliberately, with the number and the reason visible here.
const ALLOCATION_BUDGET: u64 = 0;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const CELL: &str = "Seyda Neen, Census and Excise Office";

/// A lit box of two quads, for when the game is not installed.
///
/// Deliberately not empty: the frame path branches on having a scene, and measuring the branch that
/// does nothing would prove nothing. This exercises the trace, the cutout and a shadow ray.
fn synthetic_scene() -> rtxmw_scene::StaticScene {
    let quad = |x: f32| rtxmw_scene::Mesh {
        positions: vec![
            Vec3::new(x, -80.0, -80.0),
            Vec3::new(x, 80.0, -80.0),
            Vec3::new(x, 80.0, 80.0),
            Vec3::new(x, -80.0, 80.0),
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
    let mut scene = common::scene_of(
        &[quad(200.0), quad(120.0)],
        &[Material::default()],
        &[
            Instance {
                mesh: MeshId(0),
                transform: Affine3A::IDENTITY,
            },
            Instance {
                mesh: MeshId(1),
                transform: Affine3A::from_translation(Vec3::new(0.0, 100.0, 0.0)),
            },
        ],
    );
    scene.lights = vec![Light {
        position: Vec3::new(100.0, 0.0, 60.0),
        colour: Vec3::ONE,
        radius: 400.0,
    }];
    scene.ambient = Some(Ambient {
        colour: Vec3::splat(0.05),
        ..Ambient::default()
    });
    scene
}

#[test]
fn steady_state_rendering_does_not_allocate() {
    let _profiler = dhat::Profiler::builder().testing().build();

    // Outside the measured window on purpose: device bring-up, scene decoding and every upload are
    // load-time work, and all of them allocate freely.
    let gpu = TestGpu::shared();
    let (scene, textures, what) = match common::load_cell(CELL) {
        Some(cell) => (cell.scene, cell.textures, CELL),
        None => (synthetic_scene(), Vec::new(), "synthetic scene"),
    };

    let mut renderer = SceneRenderer::new(
        gpu.device(),
        gpu.memory(),
        vk::Extent2D {
            width: WIDTH,
            height: HEIGHT,
        },
    )
    .expect("renderer should build");

    let mut uploader = gpu.uploader();
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            &scene,
            &textures,
        )
        .expect("scene should load");

    // A camera that never moves, which is the case the budget is about: nothing changes between
    // frames, so anything the heap does is the renderer's own doing.
    let eye = scene.bounds().map_or(Vec3::ZERO, |b| b.centre());
    let view = glam::camera::rh::view::look_to_mat4(eye, Vec3::X, Vec3::Z);
    let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
        75f32.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
        0.05,
    );

    // The frame constants are rebuilt inside the loop rather than hoisted, because the engine
    // rebuilds them every frame — hoisting would exclude the one part of the frame that touches
    // the camera at all.
    for _ in 0..WARMUP_FRAMES {
        let constants = renderer.frame_constants(view, projection, eye);
        renderer
            .render_once(&mut uploader, &constants)
            .expect("warmup frame failed");
    }

    let before = dhat::HeapStats::get();
    for _ in 0..MEASURED_FRAMES {
        let constants = renderer.frame_constants(view, projection, eye);
        renderer
            .render_once(&mut uploader, &constants)
            .expect("measured frame failed");
    }
    let after = dhat::HeapStats::get();

    let blocks = after.total_blocks - before.total_blocks;
    let bytes = after.total_bytes - before.total_bytes;

    println!(
        "{what}, {MEASURED_FRAMES} frames at {WIDTH}x{HEIGHT}: {blocks} allocations, {bytes} bytes \
         ({:.3} allocations/frame)",
        blocks as f64 / MEASURED_FRAMES as f64,
    );

    drop(uploader);
    gpu.assert_no_validation_errors();

    // `<=` reads as `== 0` while the budget is zero, which clippy flags. Keeping the comparison
    // means raising the budget stays a one-line change to the constant.
    #[allow(clippy::absurd_extreme_comparisons)]
    {
        assert!(
            blocks <= ALLOCATION_BUDGET,
            "{blocks} allocations across {MEASURED_FRAMES} frames ({bytes} bytes) exceeds the \
             budget of {ALLOCATION_BUDGET}; open dhat-heap.json in \
             https://nnethercote.github.io/dh_view/dh_view.html and sort by \"Total (blocks)\""
        );
    }
}
