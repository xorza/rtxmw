//! That Ray Reconstruction holds still when the camera does.
//!
//! **The property no other check can see.** A feature that builds, evaluates, returns success and
//! passes the validation layer can still shake by a pixel a frame, and a single rendered frame —
//! which is what every other measurement here is — cannot tell. `docs/design.md` §8.30 is the bug
//! this exists for: the jitter offset handed to NGX had the wrong sign on both axes, and the only
//! thing that noticed was a person watching the window.
//!
//! Its own test binary, and so its own process. NGX is global per device and the SDK does not
//! promise to survive being initialised twice at once, which the unit test in `dlss/mod.rs` already
//! relies on — two NGX users in one binary would be racing.
#![cfg(feature = "dlss")]

use ash::vk;
use glam::Vec3;
use rtxmw_gpu::{
    Device, Instance, Memory, PhysicalDevice, Presentation, Uploader, Validation, readback,
};
use rtxmw_render::SceneRenderer;
use rtxmw_render::dlss::{Paths, Preset, Requirements, Upscaler};
use rtxmw_scene::LoadedCell;

/// Output size, and — at DLAA — the size traced too.
///
/// **DLAA rather than an upscaling preset**, so that what this measures is the temporal resolve
/// alone. Any mode that upscales would fold reconstruction error into the same number and leave a
/// failure ambiguous.
/// The cell this renders, and the one the shake was seen in.
const CELL: &str = "Seyda Neen, Census and Excise Office";

const SIZE: vk::Extent2D = vk::Extent2D {
    width: 1280,
    height: 720,
};

/// Root mean square difference between two renders, over colour on `0..1`.
///
/// Colour only: the tone curve writes an opaque frame, so alpha is 255 in both and including it
/// would only divide the result by a constant.
fn rms(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "two renders of the same size");
    let mut total = 0.0f64;
    let mut count = 0usize;
    for (x, y) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0.iter()) {
        for channel in 0..3 {
            let difference = (f64::from(x[channel]) - f64::from(y[channel])) / 255.0;
            total += difference * difference;
            count += 1;
        }
    }
    (total / count as f64).sqrt() as f32
}

#[test]
fn a_still_camera_gives_ray_reconstruction_a_still_image() {
    let required = Requirements::query().expect("the SDK should answer");
    let instance = Instance::new(c"rtxmw-stability", &[], Validation::Record)
        .expect("an instance should build");
    let Ok(physical) = PhysicalDevice::select(&instance, Presentation::NotNeeded) else {
        eprintln!("skipping: no device this renderer can use");
        return;
    };
    for name in &required.device {
        assert!(
            physical.supports(name),
            "{} does not offer {name:?}, which NGX says it needs",
            physical.name()
        );
    }
    let device = Device::new(&instance, &physical, &required.device)
        .expect("a device should build with what NGX asked for");
    let memory = Memory::new(&instance, &physical, &device).expect("memory should come up");
    let mut uploader = Uploader::new(&device, &memory, physical.graphics_queue_family())
        .expect("an uploader should build");

    let data = std::env::temp_dir().join("rtxmw-ngx");
    std::fs::create_dir_all(&data).expect("a scratch directory should be creatable");
    let upscaler = Upscaler::new(
        &instance,
        &physical,
        &device,
        &mut uploader,
        SIZE,
        Preset::Dlaa,
        Paths {
            data: &data,
            feature_libraries: std::path::Path::new(rtxmw_render::dlss::FEATURE_LIBRARIES),
        },
    )
    .unwrap_or_else(|e| panic!("Ray Reconstruction should build: {e}"));
    assert_eq!(
        upscaler.render_size(),
        SIZE,
        "DLAA should trace at the size it produces, or this is measuring an upscale too"
    );

    let mut renderer = SceneRenderer::new(&device, &physical, &memory, upscaler.render_size())
        .expect("renderer should build");
    renderer
        .set_upscaler(&memory, Some(upscaler))
        .expect("the upscaler should attach");
    // **Real content, or nothing.** A synthetic fixture was tried twice and neither version could
    // fail on the fault: nine bars each way gave a frame thirty-six edges wide and passed with every
    // sign combination, and two-pixel bars separated the populations by 1.5×. What makes a
    // misaligned history visible is texture detail at pixel scale, which only real content has — so
    // a machine without the game skips, rather than running a check that cannot fail.
    let Some(cell) = LoadedCell::load_interior(CELL).expect("cell should load") else {
        eprintln!("skipping: {CELL} needs the game, and a synthetic scene cannot show this");
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

    // One camera, never moved, for every frame below — and placed by the scene rather than by hand
    // so that it faces geometry whichever of the two loaded.
    let eye = cell
        .scene
        .bounds()
        .map_or(Vec3::ZERO, |bounds| bounds.centre());
    let view = glam::camera::rh::view::look_to_mat4(eye, Vec3::X, Vec3::Z);
    let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
        75f32.to_radians(),
        SIZE.width as f32 / SIZE.height as f32,
        0.05,
    );
    let frame = |renderer: &mut SceneRenderer, uploader: &mut Uploader| -> Vec<u8> {
        let constants = renderer.frame_constants(view, projection, eye);
        renderer
            .render_once(uploader, &constants)
            .expect("trace should run");
        readback::image_to_rgba8(
            uploader,
            renderer.output(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )
        .expect("readback should succeed")
    };

    // The first frame is told to reset and has no history to be stable against, so the pair being
    // compared starts at the second.
    for _ in 0..8 {
        frame(&mut renderer, &mut uploader);
    }
    let before = frame(&mut renderer, &mut uploader);
    let after = frame(&mut renderer, &mut uploader);

    let moved = rms(&before, &after);
    println!("consecutive frames differ by {moved:.5} RMS with the camera held still");
    drop(uploader);
    // DLSS records its own commands, and a success from NGX says nothing about whether they were
    // valid — the layer is what has an opinion about the resources they touched.
    if let Some(log) = instance.validation_log() {
        let errors = log.errors_on_this_thread();
        assert!(
            errors.is_empty(),
            "{} validation error(s) from the frames above",
            errors.len()
        );
    }

    // **The bound sits between two measured populations, not around one.** On this cell and this
    // GPU the correct signs measure 0.00090; inverting Y alone gives 0.00371, X alone 0.00485, and
    // both 0.00731. The geometric mean of the two nearest is 0.0018, which leaves each side a
    // factor of two — retune it from those figures rather than by nudging, and if a different GPU
    // lands between them then the scene, not the number, is what needs work.
    assert!(
        moved < 0.0018,
        "the frame moves by {moved} RMS with the camera still. Ray Reconstruction resolves across \
         frames, so this is the jitter it is told about disagreeing with the jitter the trace \
         applied — see docs/design.md §8.30, where the sign was wrong on both axes"
    );
    // And it has to be a real picture: a black frame is perfectly stable.
    let lit = after.as_chunks::<4>().0.iter().filter(|p| p[0] > 8).count();
    assert!(
        lit * 4 > after.len() / 4,
        "only {lit} pixels of the frame are lit, so stability here means nothing"
    );
}
