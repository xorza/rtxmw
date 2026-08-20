//! That what the driver compiled is kept for the next run.
//!
//! **The whole of what starting up costs is one module.** `primary_visibility.comp` includes every
//! `.glsl` in the directory and comes to 172,043 words of SPIR-V; turning that into machine code
//! takes about three seconds on an Ada card, where the other five modules together take one
//! millisecond. Nothing reused it — `VkPipelineCache` was passed as null at every creation site —
//! so a second renderer in the same process paid the three seconds again in full, and the suite,
//! which builds about thirty across fourteen processes, spent 101 seconds mostly on that.
//!
//! **What is asserted here is the file, not the clock.** Timing it looks like the obvious test and
//! is not one: this driver keeps a shader cache of its own, so a build with `VkPipelineCache::null`
//! still comes back fast once that has seen the module — the stopwatch cannot tell whose cache
//! answered. Whether *this* renderer wrote *its* cache out is unambiguous.

use ash::vk;
use rtxmw_gpu::TestGpu;
use rtxmw_render::SceneRenderer;

#[test]
fn the_compiled_pipelines_are_kept_for_the_next_run() {
    // **Before anything touches the GPU**, because the device reads this path when it is created
    // and `TestGpu::shared` creates it on first use. This test binary holds nothing else, so
    // nothing has run yet.
    //
    // SAFETY: no other thread exists yet — the harness has not handed this binary a second test —
    // and every reader of the variable runs after this line.
    let at = std::env::temp_dir().join(format!("rtxmw-pipelines-test-{}.bin", std::process::id()));
    unsafe { std::env::set_var("RTXMW_PIPELINE_CACHE", &at) };
    let _ = std::fs::remove_file(&at);

    let gpu = TestGpu::shared();
    let renderer = SceneRenderer::new(
        gpu.device(),
        gpu.physical(),
        gpu.memory(),
        vk::Extent2D {
            width: 64,
            height: 64,
        },
    )
    .expect("renderer should build");
    drop(renderer);

    let kept = std::fs::metadata(&at).map(|of| of.len()).unwrap_or(0);
    let _ = std::fs::remove_file(&at);
    // Half a megabyte on this driver; what matters is that a blob went out at all, since an empty
    // one is what a cache nothing was ever put into hands back.
    assert!(
        kept > 1024,
        "building a renderer should leave its compiled pipelines behind — {kept} bytes at {at:?}"
    );
}
