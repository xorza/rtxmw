//! How the ground picks its colour from the four terrain textures nearest it.
//!
//! A cell names one texture per 512-unit tile and nothing in between, so the whole question is what
//! happens between two tiles that named different ones. The answer has to be a bilinear ramp
//! between the four surrounding tile *centres*, and it has to run in the same direction the tiles
//! do — a transposed or rotated weight set still produces a smooth image, which is why the two
//! assertions below are about where each texture ends up rather than about smoothness.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{
    CellId, Instance, Material, MaterialKind, Mesh, MeshId, Submesh, TerrainLayers, TextureId,
};
use rtxmw_texture::{Texture, TextureFormat};

mod common;

/// Side of one terrain tile in world units, as `TERRAIN_TILE` in `surface.glsl` declares it.
const TILE: f32 = 512.0;

const SIZE: u32 = 64;

/// The camera looks straight down from this height with a 90° field of view, so the frame covers
/// exactly `HEIGHT` units either side of the point below it — one whole tile across.
const HEIGHT: f32 = TILE / 2.0;

/// A one-texel texture of a single colour.
///
/// One texel on purpose: it takes the sampler, the mip selection and the UV interpolation out of
/// the measurement entirely, so a pixel's colour is the blend weights and nothing else.
fn flat(colour: [u8; 3]) -> Texture {
    Texture::from_pixels(
        TextureFormat::Rgba8,
        1,
        1,
        vec![colour[0], colour[1], colour[2], 255],
    )
}

/// A horizontal plane at z = 0 covering the square of tile centres the camera looks at.
fn ground() -> Mesh {
    let (low, high) = (TILE / 2.0, TILE * 1.5);
    Mesh {
        positions: vec![
            Vec3::new(low, low, 0.0),
            Vec3::new(high, low, 0.0),
            Vec3::new(high, high, 0.0),
            Vec3::new(low, high, 0.0),
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
    }
}

/// Traces the plane with `layers` as its four terrain textures and returns the frame.
fn trace(layers: [u32; 4]) -> Vec<u8> {
    // Primaries and white, so every channel of every texture is either 0 or 255 and its sRGB decode
    // is exactly 0.0 or 1.0. That is what lets the expectations below be plain arithmetic on the
    // weights rather than arithmetic on the transfer function as well.
    let palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]];
    // Only as many as `layers` actually names: the renderer decodes one texture per path in the
    // scene's list and checks the two agree, and a fixture naming one texture four times has a list
    // of one.
    let named = layers.iter().copied().max().unwrap() as usize + 1;
    let textures: Vec<_> = palette[..named].iter().map(|&c| Some(flat(c))).collect();
    let material = Material {
        kind: MaterialKind::Terrain(TerrainLayers(layers.map(TextureId))),
        ..Material::default()
    };
    let scene = common::scene_of(
        &[ground()],
        &[material],
        &[Instance {
            mesh: MeshId(0),
            transform: Affine3A::IDENTITY,
        }],
        &[],
        // Full ambient and no lights, so what comes back is the albedo unchanged.
        Vec3::ONE,
    );

    let gpu = TestGpu::shared();
    let mut renderer = SceneRenderer::new(
        gpu.device(),
        gpu.physical(),
        gpu.memory(),
        vk::Extent2D {
            width: SIZE,
            height: SIZE,
        },
    )
    .expect("renderer should build");
    // Both off: an indirect bounce would add light the plane did not have, and filtering would
    // smooth the very ramp being measured.
    renderer.set_bounce_samples(0);
    // And no fog: it is atmosphere between the eye and the ground rather than the ground's own
    // colour, and every expectation here is a blend weight worked out by hand.
    renderer.set_fog(0.0);
    // And no glare, which is fog's counterpart in the display chain: it moves light between
    // pixels by construction, and every expectation here is about one pixel at a time.
    renderer.set_glare(0.0);
    renderer.set_denoise_passes(0);

    let mut uploader = gpu.uploader();
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            CellId::Interior("terrain".to_owned()),
            &scene,
            &textures,
        )
        .expect("scene should load");

    let eye = Vec3::new(TILE, TILE, HEIGHT);
    // Straight down is where a z-up camera degenerates, so the up vector tips onto +Y — which also
    // fixes the frame's orientation: world +X runs right and world +Y runs up the image.
    let view = glam::camera::rh::view::look_to_mat4(eye, Vec3::NEG_Z, Vec3::Y);
    let projection =
        glam::camera::rh::proj::vulkan::perspective_infinite_reverse(90f32.to_radians(), 1.0, 0.05);
    let constants = renderer.frame_constants(view, projection, eye);
    renderer
        .render_once(&mut uploader, &constants)
        .expect("frame should render");

    let pixels = readback::image_to_rgba8(
        &mut uploader,
        renderer.target(),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    )
    .expect("readback should succeed");
    drop(uploader);
    gpu.assert_no_validation_errors();
    pixels
}

/// The blend weights at the pixel `(column, row)`, worked out from the camera rather than the
/// shader: at 90° from `HEIGHT` up, the frame spans `TILE` units, so a pixel centre is a known
/// fraction of the way between the two tile centres on each axis.
///
/// **Flat, then across, then flat**, rather than a ramp over the whole tile — see `ground_colour`.
/// The fraction is where the point sits between two tile centres; the weight is zero over the first
/// quarter of that, one over the last, and interpolates over the middle half, which puts the whole
/// transition in the 256 units straddling the tile boundary.
fn weights_at(column: u32, row: u32) -> Vec2 {
    let across = (column as f32 + 0.5) / SIZE as f32;
    // Row zero is the top of the image and world +Y is up, so the row index runs against +Y.
    let up = 1.0 - (row as f32 + 0.5) / SIZE as f32;
    (Vec2::new(across, up) * 2.0 - 0.5).clamp(Vec2::ZERO, Vec2::ONE)
}

fn pixel(pixels: &[u8], column: u32, row: u32) -> Vec3 {
    let index = ((row * SIZE + column) * 4) as usize;
    Vec3::new(
        pixels[index] as f32,
        pixels[index + 1] as f32,
        pixels[index + 2] as f32,
    ) / 255.0
}

#[test]
fn the_ground_blends_its_four_tiles_across_their_boundaries_and_nowhere_else() {
    let pixels = trace([0, 1, 2, 3]);

    // Red is layer 0, green layer 1, blue layer 2 and layer 3 is all three at once, so each channel
    // reads back the sum of the weights of the layers carrying it:
    //
    //   red   = (1-x)(1-y) + xy
    //   green = x(1-y) + xy = x
    //   blue  = (1-x)y + xy = y
    //
    // Green and blue are therefore the two weights themselves, which is what makes a transposed or
    // mirrored weight set visible rather than merely different.
    let mut worst = 0.0f32;
    for row in 0..SIZE {
        for column in 0..SIZE {
            let w = weights_at(column, row);
            let expected = Vec3::new((1.0 - w.x) * (1.0 - w.y) + w.x * w.y, w.x, w.y);
            worst = worst.max((pixel(&pixels, column, row) - expected).abs().max_element());
        }
    }
    // Two levels of eight-bit quantisation, and nothing else: this is an exact prediction of every
    // pixel in the frame.
    println!("worst channel error across the frame: {:.4}", worst);
    assert!(
        worst < 2.5 / 255.0,
        "the ground is not blending its four tiles as the profile says; worst channel is off by \
         {worst}"
    );

    // **The property the first version of this got wrong.** A ramp from one tile centre to the next
    // leaves nowhere on the map drawing a single texture: every point is a mix of four, and a tile
    // reads as a translucent square laid over its neighbours rather than as ground. The middle half
    // of a tile has to be that tile alone.
    let inside = pixel(&pixels, SIZE / 8, SIZE - 1 - SIZE / 8);
    println!("well inside the first tile: {inside:?}");
    assert!(
        (inside - Vec3::X).length() < 2.5 / 255.0,
        "a point well inside one tile drew {inside:?} rather than that tile — layer 0 — alone"
    );

    // The corners, spelled out, so a failure says which texture went where rather than only that
    // some pixel was wrong.
    let corner = |column, row| pixel(&pixels, column, row);
    let (near, far) = (0, SIZE - 1);
    println!(
        "corners: bottom-left {:?} bottom-right {:?} top-left {:?} top-right {:?}",
        corner(near, far),
        corner(far, far),
        corner(near, near),
        corner(far, near),
    );
    // Layer 0 is the tile below-left of the point, so it owns the corner at low x and low y — the
    // *bottom*-left of an image whose +Y runs upward.
    assert!(
        corner(near, far).x > 0.9 && corner(near, far).y < 0.1,
        "layer 0 is not at low x, low y: {:?}",
        corner(near, far)
    );
    assert!(
        corner(far, far).y > 0.9 && corner(far, far).x < 0.1,
        "layer 1 is not at high x, low y: {:?}",
        corner(far, far)
    );
    assert!(
        corner(near, near).z > 0.9 && corner(near, near).x < 0.1,
        "layer 2 is not at low x, high y: {:?}",
        corner(near, near)
    );
    assert!(
        corner(far, near).min_element() > 0.9,
        "layer 3 is not at high x, high y: {:?}",
        corner(far, near)
    );
}

#[test]
fn four_tiles_naming_one_texture_leave_the_ground_flat() {
    // The case that is most of a cell: nothing to interpolate, and the blend has to be an identity
    // rather than a wash of whatever the weights happen to be. A shader that dropped a weight, or
    // normalised the four to something other than one, fails here while still looking smooth.
    let pixels = trace([1; 4]);
    let mut worst = 0.0f32;
    for row in 0..SIZE {
        for column in 0..SIZE {
            worst = worst.max((pixel(&pixels, column, row) - Vec3::Y).abs().max_element());
        }
    }
    println!("worst channel error against flat green: {:.4}", worst);
    assert!(
        worst < 2.5 / 255.0,
        "four tiles naming the same texture did not come out as that texture; off by {worst}"
    );
}
