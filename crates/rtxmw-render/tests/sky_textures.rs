//! Changing the weather while a device is running, which is a bindless slot being written twice.
//!
//! **The interesting half is the second write, not the first.** A weather's cloud sheet goes into a
//! slot the texture array reserves, and filling that slot drops the image that was in it — so a
//! second weather has to replace the descriptor as well as the memory, or the sky keeps drawing the
//! sheet it was built with while every number the host derives comes from the new one. That failure
//! is invisible from the host side, which is why it is asserted against pixels here.

use ash::vk;
use glam::Vec3;
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{CellId, Sky, SkyTextures, Weather, WorldTime};

mod common;

const EXTENT: vk::Extent2D = vk::Extent2D {
    width: 96,
    height: 96,
};

/// A cell with nothing in it, so every ray escapes and every pixel is sky.
fn empty() -> rtxmw_scene::StaticScene {
    let mut scene = common::scene_of(&[], &[], &[], &[], Vec3::ZERO);
    scene.ambient = None;
    scene
}

/// How many of two frames' bytes differ.
fn differing(before: &[u8], after: &[u8]) -> usize {
    before
        .chunks_exact(4)
        .zip(after.chunks_exact(4))
        .filter(|(was, now)| was[..3] != now[..3])
        .count()
}

#[test]
fn a_second_weather_replaces_the_sheet_the_first_one_left_on_the_device() {
    // Two weathers whose sheets could not look more different: clear's cirrus averages a quarter of
    // an alpha, overcast's covers the dome. Without the game there is nothing to read and nothing
    // to say.
    let (Ok(clear), Ok(overcast)) = (Weather::named("clear"), Weather::named("overcast")) else {
        return;
    };
    let (Ok(Some(clear_sheet)), Ok(Some(overcast_sheet))) =
        (SkyTextures::load(&clear), SkyTextures::load(&overcast))
    else {
        return;
    };

    let gpu = TestGpu::shared();
    let mut renderer = SceneRenderer::new(gpu.device(), gpu.physical(), gpu.memory(), EXTENT)
        .expect("renderer should build");
    renderer.set_bounce_samples(0);
    renderer.set_denoise_passes(0);
    renderer.set_fog(0.0);

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

    // Looking up rather than out, so most of the frame is the layer rather than the horizon the
    // sheet fades out toward.
    let eye = Vec3::ZERO;
    let forward = Vec3::new(0.4, 0.0, 0.9).normalize();
    let view = glam::camera::rh::view::look_to_mat4(eye, forward, Vec3::Z);
    let projection =
        glam::camera::rh::proj::vulkan::perspective_infinite_reverse(75f32.to_radians(), 1.0, 0.05);

    let noon = WorldTime::hours(12.0);
    let mut frame = |weather: &Weather, textures: &SkyTextures| {
        renderer
            .set_sky_textures(
                gpu.device(),
                &mut uploader,
                gpu.physical().limits(),
                textures,
            )
            .expect("the sheet should upload");
        renderer.set_sky(Sky::under(noon, weather, textures.sheet()));
        let constants = renderer.frame_constants(view, projection, eye);
        renderer
            .render_once(&mut uploader, &constants)
            .expect("trace should run");
        readback::image_to_rgba8(
            &mut uploader,
            renderer.target(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )
        .expect("readback should succeed")
    };

    let under_clear = frame(&clear, &clear_sheet);
    let under_overcast = frame(&overcast, &overcast_sheet);
    // And back, which is the assertion that the slot is a slot rather than a one-way door: a third
    // write has to land on the same picture the first one did.
    let clear_again = frame(&clear, &clear_sheet);
    drop(uploader);
    gpu.assert_no_validation_errors();

    let pixels = (EXTENT.width * EXTENT.height) as usize;
    // The two sheets differ over most of the sky, so a slot that did not take the second write
    // would show as almost nothing moving. A fifth of the frame is far above what noise could do
    // and far below what a coincidence could reach.
    let moved = differing(&under_clear, &under_overcast);
    assert!(
        moved > pixels / 5,
        "only {moved} of {pixels} pixels changed between clear and overcast"
    );
    assert_eq!(
        differing(&under_clear, &clear_again),
        0,
        "coming back to clear should come back to the same sky"
    );
}
