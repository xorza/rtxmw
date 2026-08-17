//! Steady-state frame submission must not allocate on the heap.
//!
//! A renderer that allocates per frame gets unpredictable frame times: the allocator occasionally
//! takes a slow path, and at 60 fps a single 2 ms stall is a dropped frame. The number here is a
//! budget, not a guideline — raising it needs a reason recorded in the commit.
//!
//! This runs as a `#[test]` rather than a benchmark deliberately. It is a correctness property with
//! a pass/fail answer, it wants a debug build, and criterion would triple the compile time to
//! measure something that is not about wall-clock speed.
//!
//! Scope: everything a frame does between "the previous submission finished" and "this one has
//! completed" — fence reset, command buffer reset and record, barrier, submit, wait. Swapchain
//! acquire and present are not covered, because they need a window; fold them in when there is a
//! headless path through the real frame loop.

use ash::vk;
use rtxmw_gpu::TestGpu;

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

/// Arbitrary; the colour does not matter, only that the same work happens every frame.
const CLEAR: [f32; 4] = [0.1, 0.2, 0.3, 1.0];

#[test]
fn steady_state_frame_submission_does_not_allocate() {
    let _profiler = dhat::Profiler::builder().testing().build();

    // Outside the measured window on purpose: this builds the whole device.
    let gpu = TestGpu::shared();
    let target = gpu
        .create_target(64, 64, vk::Format::R8G8B8A8_UNORM)
        .expect("could not create render target");

    for _ in 0..WARMUP_FRAMES {
        target.clear(CLEAR).expect("warmup frame failed");
    }

    let before = dhat::HeapStats::get();
    for _ in 0..MEASURED_FRAMES {
        target.clear(CLEAR).expect("measured frame failed");
    }
    let after = dhat::HeapStats::get();

    let blocks = after.total_blocks - before.total_blocks;
    let bytes = after.total_bytes - before.total_bytes;

    println!(
        "{MEASURED_FRAMES} frames: {blocks} allocations, {bytes} bytes \
         ({:.3} allocations/frame)",
        blocks as f64 / MEASURED_FRAMES as f64,
    );

    gpu.assert_no_validation_errors();

    // `<=` reads as `== 0` while the budget is zero, which clippy flags. Keeping the comparison
    // means raising the budget stays a one-line change to the constant.
    #[allow(clippy::absurd_extreme_comparisons)]
    let within_budget = blocks <= ALLOCATION_BUDGET;
    assert!(
        within_budget,
        "steady-state frames allocated {blocks} times ({bytes} bytes) over {MEASURED_FRAMES} \
         frames, budget is {ALLOCATION_BUDGET}.\n\
         a heap profile was written to dhat-heap.json — open it at \
         https://nnethercote.github.io/dh_view/dh_view.html and sort by \"Total (blocks)\" to see \
         which call sites are responsible."
    );
}
