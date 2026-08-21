//! M7's own done-when: what one sample a pixel is worth once Ray Reconstruction has it.
//!
//! **The half of the milestone that had never been measured.** `upscaler_stability.rs` shows the
//! frame holds still and the frame-timing table shows it fits the budget; neither says the picture
//! is *right*. Stability is not accuracy — a filter that returned last frame unchanged would pass
//! both — so what is missing is a comparison against a converged reference, which is what this is.
//!
//! Three candidates against one reference, because a single error figure means nothing on its own:
//! the trace as it comes, the à-trous filter Ray Reconstruction replaced, and Ray Reconstruction.
//! The filter is in here as the thing that has to be beaten, since replacing it is a decision this
//! number is the evidence for.
//!
//! Its own test binary, and so its own process — NGX is global per device and the SDK does not
//! promise to survive being initialised twice at once, which is why `upscaler_stability.rs` is a
//! file of its own too.
#![cfg(feature = "dlss")]

use ash::vk;
use rtxmw_gpu::{Uploader, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_render::dlss::{Paths, Preset, Upscaler};
use rtxmw_scene::LoadedCell;

mod common;

use common::ngx_gpu::{CELL, NgxGpu, Viewing};

/// The size the comparison is made at, and the size the reference is rendered at.
///
/// **Not §5.3's 3840×2160**, which the reference would be sixteen times as expensive at. What the
/// ratio between a preset's render size and its output decides is how much Ray Reconstruction has
/// to invent, and that is fixed per preset rather than per resolution.
///
/// It is not free of the frame's size altogether, which is why the reference is rendered at exactly
/// this and not at some convenient other: `cone_spread_from` derives a ray's cone from the target's
/// height, so a coarser render samples coarser mips. DLAA traces at this size and is compared like
/// for like; Performance traces at half of it, and the mip it lands on is part of what the figure
/// below is measuring.
const SIZE: vk::Extent2D = vk::Extent2D {
    width: 960,
    height: 540,
};

/// Bounce rays a reference frame casts, and how many such frames are averaged.
///
/// **Four thousand samples where the milestone asked for a thousand, because the extra ones are
/// nearly free — and bought as samples rather than as frames.** Every estimator here draws from a
/// stream that rotates with the frame index, so frames converge all of them and samples converge
/// the bounce estimator alone; but a frame is read back over the bus and a sample is not. Over
/// sixty-four frames, sixteen, thirty-two and sixty-four samples give a residual of 0.0062, 0.0045
/// and 0.0034 for 0.52, 0.57 and 0.63 seconds — the square root the theory promises, for a tenth of
/// what buying it in frames costs, which is what says the residual is the bounce estimator's rather
/// than the shadows'. So the same four thousand at half the frames: 0.0040 for 0.48 s.
const REFERENCE_SAMPLES: u32 = 128;
const REFERENCE_FRAMES: u32 = 32;

/// How many à-trous passes the filter candidate runs, which is `denoiser::DEFAULT_PASSES`.
///
/// Written out because that constant is `pub(crate)` and a test binary is outside the crate. What
/// is under test is the filter a renderer with no upscaler actually runs, so if the default moves
/// this has to move with it — nothing here will notice on its own.
const FILTER_PASSES: u32 = 4;

/// How many frames Ray Reconstruction is given to settle before the frame that is measured.
///
/// **A still camera is the case the milestone names**, and a temporal resolve given no history is
/// being asked for something it does not claim to do. Sixteen is past where the error stops falling
/// — the run below prints the curve, so this is read off a measurement rather than guessed.
const SETTLE: u32 = 16;

/// The floor under the relative error's denominator, in units of the frame's own mean.
///
/// **Rousselle's ε, carried into the frame's own units, which the published figure is not.** It is
/// what makes this a *relative* error rather than one decided by whichever handful of pixels is
/// brightest — the frame is scene-referred and unbounded, so a plain RMSE over linear radiance
/// measures the ceiling lamp and nothing else. But 1e-2 assumes an image sitting around one, and an
/// interior here traces at three hundredths of mid grey: taken literally the floor would swamp
/// every denominator in the frame and quietly turn this back into the absolute error it exists to
/// avoid. Dividing both frames by the reference's mean first is what makes the number mean the same
/// thing in a cave and on a beach.
const EPSILON: f64 = 1e-2;

/// One preset's answer: where the error settled, and how it got there.
#[derive(Debug)]
struct Reconstruction {
    error: f32,
    settling: Vec<f32>,
}

/// The mean radiance of a frame's colour channels, which every error below is measured in units of.
fn mean_radiance(frame: &[f32]) -> f64 {
    let pixels = frame.as_chunks::<4>().0;
    let total: f64 = pixels
        .iter()
        .flat_map(|pixel| &pixel[..3])
        .map(|channel| f64::from(*channel))
        .sum();
    total / (pixels.len() * 3) as f64
}

/// Relative RMSE of `image` against `reference`, both in units of `level`.
///
/// Colour only: alpha carries a specular distance rather than a channel of the picture.
fn rel_rmse(image: &[f32], reference: &[f32], level: f64) -> f32 {
    assert_eq!(image.len(), reference.len(), "two frames of the same size");
    let pixels = image.as_chunks::<4>().0;
    let mut total = 0.0f64;
    for (a, b) in pixels.iter().zip(reference.as_chunks::<4>().0) {
        for channel in 0..3 {
            let (x, y) = (f64::from(a[channel]) / level, f64::from(b[channel]) / level);
            let difference = x - y;
            total += difference * difference / (y * y + EPSILON);
        }
    }
    (total / (pixels.len() * 3) as f64).sqrt() as f32
}

/// Adds `frame` into `sum`, which is how a reference is built out of frames that each hold noise.
///
/// **In linear radiance, which is the only space an average means anything in.** The tone curve is
/// not linear, so the mean of tone-mapped frames is not the tone-mapped mean — it is darker,
/// because the curve is concave where a firefly lands.
fn accumulate(sum: &mut [f64], frame: &[f32]) {
    assert_eq!(sum.len(), frame.len(), "the same frame every time");
    for (into, value) in sum.iter_mut().zip(frame) {
        *into += f64::from(*value);
    }
}

/// `measured` with the reference's own noise taken back out of it.
///
/// **Two independent errors add in quadrature**, so a candidate compared against a reference that
/// still carries noise of its own reads high by exactly that: `measured² = true² + residual²`.
/// Subtracting it back is what lets the reference stay affordable rather than asking for the four
/// times the samples it would take to make the term negligible. It assumes only that the two are
/// independent, which holds because they draw from different sample streams — see `sample_stream`.
fn without_reference_noise(measured: f32, residual: f32) -> f32 {
    (measured * measured - residual * residual).max(0.0).sqrt()
}

#[test]
fn one_sample_a_pixel_reconstructs_to_something_near_a_converged_frame() {
    let Some(NgxGpu {
        instance,
        physical,
        device,
        memory,
        mut uploader,
    }) = NgxGpu::new(c"rtxmw-reconstruction")
    else {
        return;
    };

    let mut renderer =
        SceneRenderer::new(&device, &physical, &memory, SIZE).expect("renderer should build");
    // **Real content, or nothing.** What a reconstruction has to invent is texture detail at pixel
    // scale and geometry finer than a ray, and a synthetic fixture has neither — a flat wall is
    // reconstructed perfectly by anything, including doing nothing at all.
    let Some(cell) = LoadedCell::load_interior(CELL).expect("cell should load") else {
        eprintln!("skipping: {CELL} needs the game, and a fixture cannot stand in for it here");
        return;
    };
    renderer
        .load_scene(
            &device,
            &mut uploader,
            physical.limits(),
            cell.id.clone(),
            &cell.scene,
            &cell.textures,
        )
        .expect("scene should load");
    let camera = Viewing::of(&cell.scene, SIZE);

    // The linear frame the renderer just produced, at whatever size it produced it: the upscaler's
    // where there is one, the trace's own where there is not.
    let frame = |renderer: &mut SceneRenderer, uploader: &mut Uploader| -> Vec<f32> {
        let constants = renderer.frame_constants(camera.view, camera.projection, camera.eye);
        renderer
            .render_once(uploader, &constants)
            .expect("trace should run");
        match renderer.upscaled() {
            Some(image) => readback::image_to_f32(uploader, image, vk::ImageLayout::GENERAL),
            None => readback::image_to_f32(
                uploader,
                renderer.target(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ),
        }
        .expect("readback should succeed")
    };

    // **Jittered, because the reference has to be antialiased.** Ray Reconstruction resolves
    // sub-pixel detail out of the jitter it is given, so a reference sampled at pixel centres would
    // charge it for every edge it got *right*. Averaging the jittered frames is what an
    // antialiased ground truth is.
    renderer.set_jitter(true);
    // No filter under the reference: an à-trous pass is a bias, and a biased ground truth flatters
    // whichever candidate is biased the same way.
    renderer.set_denoise_passes(0);
    renderer.set_bounce_samples(REFERENCE_SAMPLES);
    let started = std::time::Instant::now();
    let mut later = vec![0.0f64; (SIZE.width * SIZE.height * 4) as usize];
    let mut earlier = vec![0.0f64; later.len()];
    for index in 0..REFERENCE_FRAMES {
        let taken = frame(&mut renderer, &mut uploader);
        // **Split in two runs rather than by parity, because the jitter is Halton in base two.**
        // Bit reversal puts every odd index in the right half of the pixel and every even one in
        // the left, so alternating frames between the halves separates them by half a pixel — and
        // what came back was that shift measured at every edge in the frame, four times the noise
        // it was supposed to be measuring. Consecutive runs each cover the pixel evenly.
        accumulate(
            match index < REFERENCE_FRAMES / 2 {
                true => &mut earlier,
                false => &mut later,
            },
            &taken,
        );
    }
    let half: Vec<f32> = earlier
        .iter()
        .map(|total| (total / f64::from(REFERENCE_FRAMES / 2)) as f32)
        .collect();
    let reference: Vec<f32> = earlier
        .iter()
        .zip(&later)
        .map(|(a, b)| ((a + b) / f64::from(REFERENCE_FRAMES)) as f32)
        .collect();
    println!(
        "reference: {REFERENCE_FRAMES} frames x {REFERENCE_SAMPLES} spp in {:.2} s",
        started.elapsed().as_secs_f32()
    );

    // **What the reference's own noise is, before anything is measured against it**, and it is a
    // measurement rather than an estimate. The halves are independent at half the samples each, so
    // `half - reference` is exactly `(one half's noise - the other's) / 2`, whose variance is the
    // *full* reference's — and taken against the full reference it carries the same denominator
    // every figure below does. Any candidate landing near this is being measured against noise.
    let level = mean_radiance(&reference);
    let residual = rel_rmse(&half, &reference, level);
    println!(
        "the frame's mean radiance is {level:.4}, and the reference's own noise {residual:.4}"
    );

    renderer.set_bounce_samples(1);
    renderer.set_jitter(false);
    let measured = rel_rmse(&frame(&mut renderer, &mut uploader), &reference, level);
    let raw = without_reference_noise(measured, residual);
    renderer.set_denoise_passes(FILTER_PASSES);
    let measured = rel_rmse(&frame(&mut renderer, &mut uploader), &reference, level);
    let filtered = without_reference_noise(measured, residual);
    println!("1 spp as traced {raw:.4}, à-trous {filtered:.4}");

    let data = NgxGpu::scratch();
    let mut reconstructed = Vec::new();
    for preset in [Preset::Dlaa, Preset::Performance] {
        let upscaler = Upscaler::new(
            &instance,
            &physical,
            &device,
            &mut uploader,
            SIZE,
            preset,
            Paths {
                data: &data,
                feature_libraries: std::path::Path::new(rtxmw_render::dlss::FEATURE_LIBRARIES),
            },
        )
        .unwrap_or_else(|e| panic!("Ray Reconstruction should build at {preset:?}: {e}"));
        let render_size = upscaler.render_size();
        renderer
            .resize(&memory, render_size)
            .expect("the trace should follow the upscaler's render size");
        renderer
            .set_upscaler(&memory, Some(upscaler))
            .expect("the upscaler should attach");
        // Nothing before this frame is history it can use: the trace has just changed size.
        renderer.forget_history();

        // The curve as the history fills, which is what `SETTLE` is read off.
        let mut settling = Vec::new();
        for _ in 0..SETTLE {
            let taken = frame(&mut renderer, &mut uploader);
            settling.push(without_reference_noise(
                rel_rmse(&taken, &reference, level),
                residual,
            ));
        }
        let error = *settling.last().expect("SETTLE is not zero");
        println!(
            "{preset:?} traces {}x{}: {error:.4} after {SETTLE} frames, from {:.4} on the first",
            render_size.width, render_size.height, settling[0]
        );
        println!("  curve: {settling:.4?}");
        reconstructed.push(Reconstruction { error, settling });
        renderer
            .set_upscaler(&memory, None)
            .expect("the upscaler should detach");
    }
    drop(uploader);
    if let Some(log) = instance.validation_log() {
        let errors = log.errors_on_this_thread();
        assert!(
            errors.is_empty(),
            "{} validation error(s) from the frames above",
            errors.len()
        );
    }

    // Named rather than indexed, so nothing here depends on the order the loop above ran in.
    let [dlaa, performance]: [Reconstruction; 2] = reconstructed
        .try_into()
        .expect("one answer per preset asked for");

    // **The correction has to stay a correction.** Every figure above has the reference's own noise
    // taken out of it in quadrature, which is exact and is also where a reference too short would
    // hide: at `residual` equal to the measurement the answer would be zero whatever the candidate
    // did. Under the best candidate's own error, the term moves the reading by under a fifth.
    assert!(
        residual < dlaa.error,
        "the reference carries {residual} of its own noise against the best candidate's {}, so the \
         quadrature correction is doing the measuring — raise REFERENCE_SAMPLES, which buys the \
         same convergence at a tenth of what frames cost",
        dlaa.error
    );

    // **What the milestone asked for.** A hundredth of the frame's own level per pixel against a
    // frame traced at four thousand samples, where the trace it was handed stands at a fifth. The
    // bound is not quite twice the measurement, which on a project pinned to one class of hardware
    // is the room a driver revision needs and no more.
    assert!(
        dlaa.error < 0.015,
        "one sample a pixel reconstructs to {} of the frame's own level, which is not comparable \
         to a converged frame",
        dlaa.error
    );
    // And it has to be doing the work: the trace it is handed is the thing it improves on.
    assert!(
        raw > dlaa.error * 8.0,
        "one sample a pixel comes out at {raw} and Ray Reconstruction leaves it at {}, which is \
         not a reconstruction",
        dlaa.error
    );
    // **It has to beat the filter it replaced**, which is the decision this number is evidence for:
    // Ray Reconstruction was put in place *of* the à-trous passes rather than beside them.
    assert!(
        filtered > dlaa.error * 2.0,
        "Ray Reconstruction reconstructs no better than the {FILTER_PASSES} à-trous passes it \
         replaced: {} against {filtered}",
        dlaa.error
    );

    // **The history is what does it**, which the curve says and a single figure does not: a resolve
    // given no previous frame has one sample a pixel and nothing else. Its first frame measures
    // level with the à-trous filter, and everything past that is accumulation.
    assert!(
        dlaa.settling[0] > dlaa.error * 2.0,
        "the error does not fall as the history fills — {:.4} on the first frame against {} on the \
         last, so nothing is accumulating",
        dlaa.settling[0],
        dlaa.error
    );

    // **Upscaling costs accuracy, and the price is what §5.3 is buying with.** Performance traces a
    // quarter of the pixels, so three of every four are reconstructed from history and neighbours —
    // and it still lands nearer the reference than a full-resolution trace under the old filter.
    // Bounded against the raw trace rather than against `filtered`, which it beats by too little to
    // hold a test on.
    assert!(
        raw > performance.error * 4.0,
        "at Performance the reconstruction is no better than a quarter of the raw trace's error: \
         {} against {raw}",
        performance.error
    );
}
