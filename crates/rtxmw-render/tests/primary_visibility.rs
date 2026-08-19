//! Traces primary rays into an offscreen image and asserts on the pixels.
//!
//! This is where everything upstream is finally checked against reality. A wrong build-range
//! offset, a mirrored instance transform or a transposed projection all survive M3c and M3d
//! silently — they only become visible once a ray has to find the triangle. The scenes here are
//! small enough that where a hit lands can be worked out by hand.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{
    AlphaMode, CellId, Instance, Light, LoadedCell, Material, Mesh, MeshId, StaticScene, Submesh,
    TextureId,
};
use rtxmw_texture::{Texture, TextureFormat};
use rtxmw_vfs::DATA_DIR_VAR;

mod common;

const CELL: &str = "Seyda Neen, Census and Excise Office";
const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const FOV_Y: f32 = 75.0;

/// Whether a ray hit anything, from the flag the shader writes into alpha.
///
/// Read from alpha rather than inferred from the colour: material colours can be arbitrarily dark,
/// so brightness stopped being evidence of a hit the moment the output stopped being barycentric.
fn is_hit(pixel: &[u8]) -> bool {
    pixel[3] > 128
}

/// The colour of a pixel, ignoring the hit flag.
fn shade(pixel: &[u8]) -> [u8; 3] {
    [pixel[0], pixel[1], pixel[2]]
}

/// A quad spanned by `across` and `up` from `corner`, with the normal and material given.
///
/// Wound so its triangles' own plane comes out along `normal`. That is not decoration: the renderer
/// shades a hit as whichever face of the triangle the ray met, so a quad whose winding disagrees
/// with the normal it was handed is a quad lit from the wrong side — and the fixture, not the
/// renderer, would be what the test then measured.
fn quad(corner: Vec3, across: Vec3, up: Vec3, normal: Vec3, material: u32) -> Mesh {
    let wound = if across.cross(up).dot(normal) < 0.0 {
        vec![0, 2, 1, 0, 3, 2]
    } else {
        vec![0, 1, 2, 0, 2, 3]
    };
    Mesh {
        positions: vec![corner, corner + across, corner + across + up, corner + up],
        normals: vec![normal; 4],
        uvs: vec![Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y],
        indices: wound,
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material,
            thin: false,
        }],
    }
}

/// An axis-aligned quad in the world's YZ plane, facing back down -X toward a camera at the origin.
fn wall(x: f32, y: std::ops::Range<f32>, z: std::ops::Range<f32>) -> Mesh {
    quad(
        Vec3::new(x, y.start, z.start),
        Vec3::new(0.0, y.end - y.start, 0.0),
        Vec3::new(0.0, 0.0, z.end - z.start),
        Vec3::NEG_X,
        0,
    )
}

/// Traces `instances` from `eye` looking along `forward` and returns the image as 8-bit RGBA.
fn trace(meshes: &[Mesh], instances: &[Instance], eye: Vec3, forward: Vec3) -> Vec<u8> {
    trace_textured(meshes, &[Material::default()], &[], instances, eye, forward)
}

/// As [`trace`], with textures for the scene's path list to sample.
///
/// Full ambient and no lights, so an unlit trace shows albedo unchanged — which is what every test
/// written before lighting existed is asserting about.
fn trace_textured(
    meshes: &[Mesh],
    materials: &[Material],
    textures: &[Option<Texture>],
    instances: &[Instance],
    eye: Vec3,
    forward: Vec3,
) -> Vec<u8> {
    let scene = common::scene_of(meshes, materials, instances, &[], Vec3::ONE);
    trace_scene(&scene, textures, eye, forward, 0, Read::Radiance)
}

/// Renders one frame of `scene` through the production renderer and reads it back.
///
/// Deliberately the real `SceneRenderer` rather than a test-local assembly of the same pieces: a
/// replica would drift from what the engine actually runs, and every one of these assertions is
/// meant to be about the engine.
///
/// `bounces` is how many diffuse bounce rays each pixel casts. Zero everywhere above, because at
/// zero the indirect estimator collapses to the flat `albedo * ambient` fill those tests were
/// written against — everything they assert is about geometry, materials or direct light, and
/// occluded ambient would only add noise to it.
fn trace_scene(
    scene: &StaticScene,
    textures: &[Option<Texture>],
    eye: Vec3,
    forward: Vec3,
    bounces: u32,
    read: Read,
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
    renderer.set_bounce_samples(bounces);
    // Unfiltered, because every assertion in this file is about what the *trace* computes: radiance
    // worked out by hand, the Monte Carlo error of the indirect estimator, the fraction of rays
    // that hit. Smoothing the lighting is exactly what would hide the last of those.
    renderer.set_denoise_passes(0);
    // And unfogged, for the same reason: fog is atmosphere between the eye and the surface rather
    // than anything the surface does, and it varies with world position — so it would put a
    // gradient across a wall these tests need flat, and change what a scene renders as when it is
    // moved away from the origin.
    renderer.set_fog(0.0);

    // One uploader for the whole trace, held across load and render. It wraps a single queue, so
    // the harness serialises every test through it — which is exactly why the renderer borrows one
    // rather than making its own.
    let mut uploader = gpu.uploader();
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            CellId::Interior("fixture".to_owned()),
            scene,
            textures,
        )
        .expect("scene should load");

    let view = glam::camera::rh::view::look_to_mat4(eye, forward, Vec3::Z);
    let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
        FOV_Y.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
        0.05,
    );
    let constants = renderer.frame_constants(view, projection, eye);
    renderer
        .render_once(&mut uploader, &constants)
        .expect("trace should run");

    let image = match read {
        Read::Radiance => renderer.target(),
        Read::Tonemapped => renderer.output(),
    };
    let pixels =
        readback::image_to_rgba8(&mut uploader, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .expect("readback should succeed");
    drop(uploader);
    gpu.assert_no_validation_errors();
    pixels
}

/// Which of the renderer's images a trace reads back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Read {
    /// The pass's own output, in the radiance units a test can compute by hand.
    Radiance,
    /// After exposure and tonemapping — the bytes the window would show.
    ///
    /// What a test wants when it is judging the *image*: a Morrowind interior lit by candles sits
    /// in the darkest few of 256 levels as raw radiance, so counting distinct colours there counts
    /// how dim the room is rather than how much of it is textured.
    Tonemapped,
}

/// The pixel at `(x, y)`, row-major from the top.
fn at(pixels: &[u8], x: u32, y: u32) -> &[u8] {
    let index = ((y * WIDTH + x) * 4) as usize;
    &pixels[index..index + 4]
}

/// How many pixels in the image are hits.
fn hit_count(pixels: &[u8]) -> usize {
    pixels.chunks_exact(4).filter(|p| is_hit(p)).count()
}

#[test]
fn a_wall_ahead_covers_the_centre_and_leaves_the_corners_empty() {
    // The camera sits at the origin looking east along +X. The wall is 100 units away and spans 50
    // units either side of the axis. At 75 degrees vertical field of view the half-height of the
    // view at that distance is 100 * tan(37.5 deg) = 76.7 units, so the wall covers the middle of
    // the image and falls short of the corners — which is what makes both assertions meaningful.
    let meshes = [wall(100.0, -50.0..50.0, -50.0..50.0)];
    let instances = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::IDENTITY,
    }];
    let pixels = trace(&meshes, &instances, Vec3::ZERO, Vec3::X);

    assert!(
        is_hit(at(&pixels, WIDTH / 2, HEIGHT / 2)),
        "the centre ray missed a wall directly ahead: {:?}",
        at(&pixels, WIDTH / 2, HEIGHT / 2)
    );
    for (x, y) in [
        (0, 0),
        (WIDTH - 1, 0),
        (0, HEIGHT - 1),
        (WIDTH - 1, HEIGHT - 1),
    ] {
        assert!(
            !is_hit(at(&pixels, x, y)),
            "corner ({x}, {y}) hit something outside the wall: {:?}",
            at(&pixels, x, y)
        );
    }

    // 100x100 units of wall inside a 153.4-unit-tall view is a little under half the height, and
    // the same in width since the image is square. Roughly a quarter of the pixels, then — loose
    // bounds, because the point is that it is a bounded patch rather than everything or nothing.
    let hits = hit_count(&pixels);
    let total = (WIDTH * HEIGHT) as usize;
    assert!(
        hits > total / 8 && hits < total / 2,
        "{hits} of {total} pixels hit, which is not a wall-sized patch"
    );
}

#[test]
fn geometry_to_the_north_appears_on_the_left_of_the_image() {
    // The handedness chain end to end. Looking east along +X with +Z up, the camera's right is
    // forward x up = X x Z = -Y, so world +Y (north) is to the camera's *left*. Vulkan's NDC does
    // not flip X, so left is low pixel x.
    //
    // Getting this wrong is the mirroring bug that survives every earlier test: a transposed
    // instance transform, a projection built for the wrong NDC convention, or an unprojection that
    // negated x would all still put a wall on screen — just on the wrong side.
    let meshes = [wall(100.0, 20.0..90.0, -40.0..40.0)];
    let instances = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::IDENTITY,
    }];
    let pixels = trace(&meshes, &instances, Vec3::ZERO, Vec3::X);

    let mut left = 0usize;
    let mut right = 0usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if is_hit(at(&pixels, x, y)) {
                if x < WIDTH / 2 {
                    left += 1;
                } else {
                    right += 1;
                }
            }
        }
    }

    assert!(left > 0, "nothing was hit at all");
    assert_eq!(right, 0, "{right} pixels hit on the right of the image");
}

#[test]
fn two_materials_in_one_mesh_shade_differently() {
    // The direct test of the hit-to-material chain. Both halves are in the *same* mesh, so they
    // share an instance and a custom index — only `geometry_index` distinguishes them. If it were
    // ignored, or the submesh table were not flat, both halves would resolve to the same entry and
    // come back one colour while still filling exactly the same pixels.
    let mut mesh = wall(100.0, 20.0..90.0, -40.0..40.0);
    let south = wall(100.0, -90.0..-20.0, -40.0..40.0);
    let base = mesh.positions.len() as u32;
    mesh.positions.extend(south.positions);
    mesh.normals.extend(south.normals);
    mesh.uvs.extend(south.uvs);
    mesh.indices.extend(south.indices.iter().map(|i| i + base));
    mesh.submeshes.push(Submesh {
        first_index: 6,
        index_count: 6,
        material: 1,
        thin: false,
    });

    // Two materials distinguishable by their own colour. Identity alone is no longer enough now
    // that shading reads the material rather than hashing its index — which is the point: a colour
    // difference here means the right *entry* was fetched, not merely a different one.
    let materials = [
        Material {
            diffuse: Vec3::new(1.0, 0.0, 0.0),
            ..Material::default()
        },
        Material {
            diffuse: Vec3::new(0.0, 1.0, 0.0),
            ..Material::default()
        },
    ];
    let instances = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::IDENTITY,
    }];
    let pixels = trace_textured(&[mesh], &materials, &[], &instances, Vec3::ZERO, Vec3::X);

    let mut left = std::collections::HashSet::new();
    let mut right = std::collections::HashSet::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let pixel = at(&pixels, x, y);
            if is_hit(pixel) {
                if x < WIDTH / 2 {
                    left.insert(shade(pixel));
                } else {
                    right.insert(shade(pixel));
                }
            }
        }
    }

    assert_eq!(
        left.len(),
        1,
        "the north wall is not one material: {left:?}"
    );
    assert_eq!(
        right.len(),
        1,
        "the south wall is not one material: {right:?}"
    );
    assert_ne!(
        left, right,
        "both submeshes resolved to the same material, so the geometry index was ignored"
    );
}

#[test]
fn geometry_above_the_camera_appears_at_the_top_of_the_image() {
    // The other half of the handedness check. Vulkan's NDC is Y-down and the projection flips Y to
    // match, so world +Z (up) must land at low pixel y. A projection built for OpenGL's Y-up NDC
    // would put it at the bottom and nothing else would notice.
    let meshes = [wall(100.0, -40.0..40.0, 20.0..90.0)];
    let instances = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::IDENTITY,
    }];
    let pixels = trace(&meshes, &instances, Vec3::ZERO, Vec3::X);

    let mut top = 0usize;
    let mut bottom = 0usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if is_hit(at(&pixels, x, y)) {
                if y < HEIGHT / 2 {
                    top += 1;
                } else {
                    bottom += 1;
                }
            }
        }
    }

    assert!(top > 0, "nothing was hit at all");
    assert_eq!(bottom, 0, "{bottom} pixels hit in the bottom half");
}

#[test]
fn separate_meshes_both_render_at_their_own_offsets() {
    // The decisive test for the build-range triple. Two meshes in one buffer means the second has a
    // nonzero `first_vertex` and `primitive_offset`; if either is wrong, its structure is built from
    // the first mesh's vertices and both walls land in the same place. Placing them on opposite
    // sides means that failure shows up as one side going empty.
    let meshes = [
        wall(100.0, 20.0..90.0, -40.0..40.0),
        wall(100.0, -90.0..-20.0, -40.0..40.0),
    ];
    let instances = [
        Instance {
            mesh: MeshId(0),
            transform: Affine3A::IDENTITY,
        },
        Instance {
            mesh: MeshId(1),
            transform: Affine3A::IDENTITY,
        },
    ];
    let pixels = trace(&meshes, &instances, Vec3::ZERO, Vec3::X);

    let mut left = 0usize;
    let mut right = 0usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if is_hit(at(&pixels, x, y)) {
                if x < WIDTH / 2 {
                    left += 1;
                } else {
                    right += 1;
                }
            }
        }
    }

    assert!(left > 0, "the north wall (mesh 0) is missing");
    assert!(right > 0, "the south wall (mesh 1) is missing");
    // Mirror images of each other, so within a few pixels of the same area.
    let difference = left.abs_diff(right);
    assert!(
        difference * 20 < left + right,
        "{left} left against {right} right — the two meshes are not the same size on screen"
    );
}

#[test]
fn an_instance_transform_moves_the_geometry_it_places() {
    // The same mesh placed twice: once ahead and once behind. Only the one ahead may be visible, so
    // the instance transform has to be applied — a build that ignored it would draw both at the
    // origin and the hit count would not change between this and the single-instance case.
    let meshes = [wall(0.0, -50.0..50.0, -50.0..50.0)];
    let instances = [
        Instance {
            mesh: MeshId(0),
            transform: Affine3A::from_translation(Vec3::new(100.0, 0.0, 0.0)),
        },
        Instance {
            mesh: MeshId(0),
            transform: Affine3A::from_translation(Vec3::new(-100.0, 0.0, 0.0)),
        },
    ];
    let pixels = trace(&meshes, &instances, Vec3::ZERO, Vec3::X);

    assert!(is_hit(at(&pixels, WIDTH / 2, HEIGHT / 2)));
    let both = hit_count(&pixels);

    // With only the wall behind the camera, nothing should be visible at all.
    let behind = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::from_translation(Vec3::new(-100.0, 0.0, 0.0)),
    }];
    let pixels = trace(&meshes, &behind, Vec3::ZERO, Vec3::X);
    assert_eq!(
        hit_count(&pixels),
        0,
        "geometry behind the camera was drawn, so the transform was ignored"
    );
    assert!(both > 0);
}

/// A quad with UVs spanning the whole texture, so a texel's alpha maps to a screen region.
fn uv_wall(x: f32, y: std::ops::Range<f32>, z: std::ops::Range<f32>) -> Mesh {
    let mut mesh = wall(x, y, z);
    // Corner order matches `wall`: (y0,z0), (y1,z0), (y1,z1), (y0,z1).
    mesh.uvs = vec![
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 0.0),
    ];
    mesh
}

/// A 2x1 texture: opaque white on the left texel, fully transparent on the right.
fn half_transparent() -> Texture {
    Texture::from_pixels(
        TextureFormat::Rgba8,
        2,
        1,
        vec![255, 255, 255, 255, 255, 255, 255, 0],
    )
}

#[test]
fn a_transparent_texel_is_traced_through_rather_than_hit() {
    // The alpha test in the candidate loop. Half the texture is transparent, so half the quad must
    // not register a hit at all — the ray passes through and reaches the background. Marking the
    // geometry opaque, or dropping the candidate loop, fills the whole quad instead.
    let meshes = [uv_wall(100.0, -60.0..60.0, -60.0..60.0)];
    let materials = [Material {
        base_colour: Some(TextureId(0)),
        alpha: AlphaMode::Mask(0.5),
        ..Material::default()
    }];
    let instances = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::IDENTITY,
    }];
    let pixels = trace_textured(
        &meshes,
        &materials,
        &[Some(half_transparent())],
        &instances,
        Vec3::ZERO,
        Vec3::X,
    );

    // The quad spans y in [-60, 60] with u running 0..1 across it, and world +Y is screen-left — so
    // the opaque half (u < 0.5) lands on the *right* of the image.
    let mut left = 0usize;
    let mut right = 0usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if is_hit(at(&pixels, x, y)) {
                if x < WIDTH / 2 {
                    left += 1;
                } else {
                    right += 1;
                }
            }
        }
    }

    assert!(right > 0, "the opaque half of the texture was not drawn");
    assert_eq!(
        left, 0,
        "{left} pixels hit where the texture is transparent — the cutout did not happen"
    );
}

#[test]
fn an_opaque_material_ignores_its_texture_alpha() {
    // The same transparent texture on an opaque material must draw solid: the build marks that run
    // `OPAQUE`, traversal commits without asking, and the cutout never runs. This is what keeps the
    // alpha test off the 3,982 opaque materials that do not need it.
    let meshes = [uv_wall(100.0, -60.0..60.0, -60.0..60.0)];
    let materials = [Material {
        base_colour: Some(TextureId(0)),
        ..Material::default()
    }];
    let instances = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::IDENTITY,
    }];
    let pixels = trace_textured(
        &meshes,
        &materials,
        &[Some(half_transparent())],
        &instances,
        Vec3::ZERO,
        Vec3::X,
    );

    let mut left = 0usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH / 2 {
            if is_hit(at(&pixels, x, y)) {
                left += 1;
            }
        }
    }
    assert!(
        left > 0,
        "an opaque material was cut out by its texture alpha"
    );
}

/// Mean brightness of the hit pixels.
fn mean_brightness(pixels: &[u8]) -> f32 {
    let mut total = 0.0;
    let mut count = 0.0;
    for pixel in pixels.chunks_exact(4).filter(|p| is_hit(p)) {
        total += (pixel[0] as f32 + pixel[1] as f32 + pixel[2] as f32) / 3.0;
        count += 1.0;
    }
    if count == 0.0 { 0.0 } else { total / count }
}

#[test]
fn a_light_brightens_what_it_reaches() {
    let meshes = [wall(200.0, -150.0..150.0, -150.0..150.0)];
    let materials = [Material::default()];
    let instances = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::IDENTITY,
    }];
    let ambient = Vec3::splat(0.05);
    let light = Light {
        position: Vec3::new(100.0, 0.0, 0.0),
        colour: Vec3::ONE,
        radius: 300.0,
    };

    let dark = common::scene_of(&meshes, &materials, &instances, &[], ambient);
    let bright = common::scene_of(
        &meshes,
        &materials,
        &instances,
        std::slice::from_ref(&light),
        ambient,
    );
    let unlit = trace_scene(&dark, &[], Vec3::ZERO, Vec3::X, 0, Read::Radiance);
    let lit = trace_scene(&bright, &[], Vec3::ZERO, Vec3::X, 0, Read::Radiance);

    assert!(
        mean_brightness(&unlit) > 0.0,
        "the wall was not drawn at all"
    );
    assert!(
        mean_brightness(&lit) > mean_brightness(&unlit) * 2.0,
        "adding a light moved mean brightness only from {} to {}",
        mean_brightness(&unlit),
        mean_brightness(&lit)
    );
}

#[test]
fn an_occluder_between_light_and_surface_casts_a_shadow() {
    // Worked out by hand. The light sits at (100, 0, 100) and the lit wall at x = 200, so a ray
    // from the light to the wall's centre (200, 0, 0) crosses x = 150 at z = 50. An occluder there
    // spanning z in [20, 80] therefore blocks it. The centre pixel's own view ray travels along
    // z = 0 and passes *under* that occluder, so it still sees wall in both traces — which is what
    // makes the two comparable.
    let lit_wall = wall(200.0, -150.0..150.0, -150.0..150.0);
    let occluder = wall(150.0, -30.0..30.0, 20.0..80.0);
    let materials = [Material::default()];
    let light = Light {
        position: Vec3::new(100.0, 0.0, 100.0),
        colour: Vec3::ONE,
        radius: 400.0,
    };
    let ambient = Vec3::splat(0.05);
    let place = |mesh: u32| Instance {
        mesh: MeshId(mesh),
        transform: Affine3A::IDENTITY,
    };

    let open = common::scene_of(
        std::slice::from_ref(&lit_wall),
        &materials,
        &[place(0)],
        std::slice::from_ref(&light),
        ambient,
    );
    let shadowed = common::scene_of(
        &[lit_wall, occluder],
        &materials,
        &[place(0), place(1)],
        std::slice::from_ref(&light),
        ambient,
    );
    let clear = trace_scene(&open, &[], Vec3::ZERO, Vec3::X, 0, Read::Radiance);
    let blocked = trace_scene(&shadowed, &[], Vec3::ZERO, Vec3::X, 0, Read::Radiance);

    let centre = WIDTH / 2;
    let sum = |p: &[u8]| p[0] as u32 + p[1] as u32 + p[2] as u32;
    let open = at(&clear, centre, centre);
    let shadow = at(&blocked, centre, centre);

    assert!(
        is_hit(open) && is_hit(shadow),
        "the wall centre was not hit"
    );
    assert!(
        sum(open) > sum(shadow) * 2,
        "the occluder cast no shadow: {open:?} lit against {shadow:?} blocked"
    );
    // Ambient still reaches it — a shadow is unlit, not black.
    assert!(sum(shadow) > 0, "the shadowed point lost its ambient too");
}

#[test]
fn a_shadow_edge_is_soft_rather_than_binary() {
    // Counting brightness levels across the boundary does not work: with no ambient, the lit wall
    // already varies row to row through attenuation and the cosine term, so a *hard* shadow also
    // produces many levels. What isolates the shadow is the ratio against the same scene without
    // the occluder — that cancels the shading entirely and leaves visibility alone.
    let lit_wall = wall(200.0, -150.0..150.0, -150.0..150.0);
    let occluder = wall(150.0, -30.0..30.0, 20.0..80.0);
    let materials = [Material::default()];
    let light = Light {
        position: Vec3::new(100.0, 0.0, 100.0),
        colour: Vec3::ONE,
        radius: 400.0,
    };
    let place = |mesh: u32| Instance {
        mesh: MeshId(mesh),
        transform: Affine3A::IDENTITY,
    };
    let trace = |meshes: &[Mesh], instances: &[Instance]| {
        let scene = common::scene_of(
            meshes,
            &materials,
            instances,
            std::slice::from_ref(&light),
            Vec3::ZERO,
        );
        trace_scene(&scene, &[], Vec3::ZERO, Vec3::X, 0, Read::Radiance)
    };

    let clear = trace(std::slice::from_ref(&lit_wall), &[place(0)]);
    let blocked = trace(&[lit_wall, occluder], &[place(0), place(1)]);

    // Walk the middle column and record how much of the light each row still receives.
    let mut penumbra = 0usize;
    let mut fully_lit = 0usize;
    let mut fully_dark = 0usize;
    for y in 0..HEIGHT {
        let open = at(&clear, WIDTH / 2, y);
        let shut = at(&blocked, WIDTH / 2, y);
        if !is_hit(open) || !is_hit(shut) {
            continue;
        }
        let sum = |p: &[u8]| (p[0] as f32 + p[1] as f32 + p[2] as f32).max(1.0);
        let visibility = sum(shut) / sum(open);
        if visibility > 0.9 {
            fully_lit += 1;
        } else if visibility < 0.1 {
            fully_dark += 1;
        } else {
            penumbra += 1;
        }
    }

    assert!(
        fully_lit > 0 && fully_dark > 0,
        "the column never crossed the shadow: {fully_lit} lit, {fully_dark} dark"
    );
    // A point light gives at most the one row that straddles the edge. An emitter with area gives a
    // band of rows that see part of it, and that band is the penumbra.
    assert!(
        penumbra >= 3,
        "only {penumbra} partly-lit rows — the shadow edge is hard"
    );
}

#[test]
fn an_unoccluded_lit_wall_is_smooth() {
    // Nothing can block this wall, so visibility is 1.0 everywhere and the only variation should be
    // the falloff. Speckle here would mean shadow rays hitting the surface they started on.
    let meshes = [wall(200.0, -150.0..150.0, -150.0..150.0)];
    let materials = [Material::default()];
    let instances = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::IDENTITY,
    }];
    let light = Light {
        position: Vec3::new(100.0, 0.0, 0.0),
        colour: Vec3::ONE,
        radius: 300.0,
    };
    let scene = common::scene_of(
        &meshes,
        &materials,
        &instances,
        std::slice::from_ref(&light),
        Vec3::ZERO,
    );
    let pixels = trace_scene(&scene, &[], Vec3::ZERO, Vec3::X, 0, Read::Radiance);

    // Neighbour-to-neighbour brightness jumps along the centre row. A smooth falloff changes by a
    // few units per pixel; self-occlusion makes it jump by the full lit value at random pixels.
    let sum = |p: &[u8]| p[0] as i32 + p[1] as i32 + p[2] as i32;
    let mut worst = 0;
    let mut jumps = 0;
    let mut prev: Option<i32> = None;
    for x in 0..WIDTH {
        let p = at(&pixels, x, HEIGHT / 2);
        if !is_hit(p) {
            continue;
        }
        let v = sum(p);
        if let Some(q) = prev {
            let d = (v - q).abs();
            worst = worst.max(d);
            if d > 20 {
                jumps += 1;
            }
        }
        prev = Some(v);
    }
    println!("centre row: worst neighbour jump {worst}, {jumps} jumps over 20");
    assert!(
        jumps == 0,
        "{jumps} discontinuities on an unoccludable surface"
    );
}

#[test]
fn a_real_interior_traces_to_a_recognisable_image() {
    let Some(cell) = LoadedCell::load_interior(CELL).expect("cell should load") else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };
    let missing = cell.missing_textures();
    let entrances = cell.entrances;
    let scene = cell.scene;

    // Where the game itself puts a traveller arriving through the door, dropped onto whatever is
    // under it — the same rule the engine's own viewpoint uses.
    //
    // It replaces the centre of the cell's bounds, which this test used to stand at on the grounds
    // that it "happens to work for this cell". It does not: that point is inside a ceiling pillar,
    // and the test was reading the dim inside of a beam while claiming to look at a room.
    let door = entrances.first().expect("a cell with a way in");
    let feet = scene
        .ground_below(door.arrival + Vec3::Z * 20.0)
        .expect("a floor under the doorway");
    let eye = door.arrival.truncate().extend(feet + 128.0);

    let textures = cell.textures;

    let lit = common::scene_of(
        &scene.meshes,
        scene.materials.materials(),
        &scene.instances,
        &scene.lights,
        scene.ambient.map_or(Vec3::ZERO, |a| a.colour),
    );
    let pixels = trace_scene(&lit, &textures, eye, door.facing, 0, Read::Tonemapped);
    let hits = hit_count(&pixels);
    let total = (WIDTH * HEIGHT) as usize;

    // Each distinct colour is a distinct material, so this counts how many surfaces are in view.
    // One would mean every hit resolved to the same table entry, which is exactly what a constant
    // geometry index or a misread custom index produces — and it would still fill the screen.
    let mut shades: std::collections::HashSet<[u8; 3]> = std::collections::HashSet::new();
    for pixel in pixels.chunks_exact(4).filter(|p| is_hit(p)) {
        shades.insert(shade(pixel));
    }

    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("seyda_neen.png");
    readback::write_png_opaque(&path, &pixels, WIDTH, HEIGHT);

    let cutouts = scene
        .materials
        .materials()
        .iter()
        .filter(|m| !matches!(m.alpha, AlphaMode::Opaque))
        .count();
    println!(
        "  {cutouts} of {} materials run the alpha cutout, {} lights, ambient {:?}",
        scene.materials.materials().len(),
        scene.lights.len(),
        scene.ambient.map(|a| a.colour)
    );
    println!(
        "{CELL}: {hits} of {total} pixels hit ({:.0}%), {} distinct shades from {} textures \
         ({missing} missing), from {eye:?}\n  wrote {}",
        hits as f64 / total as f64 * 100.0,
        shades.len(),
        textures.len(),
        path.display()
    );

    // Standing inside an enclosed room, most rays should land on a wall, the floor or furniture.
    // A near-empty image would mean the rays never found the geometry.
    assert!(
        hits > total * 2 / 5,
        "only {hits} of {total} pixels hit from inside the room"
    );
    // Textured surfaces vary per texel, so this is now counting sampled colour rather than material
    // identity: hundreds of shades means UVs are being interpolated and the array is being read. A
    // handful would mean flat material colour with the texture lookup doing nothing.
    assert!(
        shades.len() > 500,
        "only {} distinct shades — the textures are not being sampled",
        shades.len()
    );

    // The same cell with indirect light on, which is the only place the bounce path meets real
    // geometry: 104 models, cutout materials and thirteen lights rather than two quads.
    //
    // A sealed room is the strong case for the prediction. Almost every bounce ray hits a wall
    // instead of escaping, so the ambient fill is occluded nearly everywhere and the mean has to
    // fall — the flat fill is the ceiling this can approach and never exceed.
    let bounced = trace_scene(&lit, &textures, eye, Vec3::X, 4, Read::Radiance);
    let flat_mean = mean_brightness(&pixels);
    let bounced_mean = mean_brightness(&bounced);
    println!("  mean brightness: {flat_mean:.2} flat, {bounced_mean:.2} with 4 bounces");
    assert_eq!(
        hit_count(&bounced),
        hits,
        "indirect light changed which pixels hit, which it cannot"
    );
    assert!(
        bounced_mean < flat_mean,
        "an enclosed room got no darker under occluded ambient: {flat_mean} to {bounced_mean}"
    );
}

/// Where the camera sits for every bounce test: half way up the wall it looks at, 100 units back.
const BOUNCE_EYE: Vec3 = Vec3::new(0.0, 0.0, 40.0);

/// Rows where the centre column meets the wall low down and high up.
///
/// The camera is at z = 40 looking east with a 75-degree vertical field of view, so at the wall's
/// 100-unit distance the view is `2 * 100 * tan(37.5) = 153.5` units tall and one row is 0.6 units.
/// Row 174 of 256 is therefore `40 - 76.73 * (2 * 174.5 / 256 - 1) = 12.1` units up the wall, and
/// row 77 is 70.3 — both far enough from the wall's 0..80 edges that a patch around either stays
/// on it.
const LOW_ROW: u32 = 174;
const HIGH_ROW: u32 = 77;

/// A wall to look at with a floor running up to its base, and one light high above the floor.
///
/// The two surfaces are the whole point: the wall is white and the light is white, so *every* bit
/// of colour on the wall arrived by reflection off the floor, and the red and blue channels of one
/// pixel are a direct measurement of it that needs no second render to compare against.
///
fn bounce_scene(floor_albedo: Vec3, ambient: Vec3, lights: &[Light]) -> StaticScene {
    let meshes = [
        quad(
            Vec3::new(100.0, -40.0, 0.0),
            Vec3::new(0.0, 80.0, 0.0),
            Vec3::new(0.0, 0.0, 80.0),
            Vec3::NEG_X,
            0,
        ),
        quad(
            Vec3::new(40.0, -40.0, 0.0),
            Vec3::new(60.0, 0.0, 0.0),
            Vec3::new(0.0, 80.0, 0.0),
            Vec3::Z,
            1,
        ),
    ];
    let materials = [
        Material::default(),
        Material {
            diffuse: floor_albedo,
            ..Material::default()
        },
    ];
    let instances = [
        Instance {
            mesh: MeshId(0),
            transform: Affine3A::IDENTITY,
        },
        Instance {
            mesh: MeshId(1),
            transform: Affine3A::IDENTITY,
        },
    ];
    common::scene_of(&meshes, &materials, &instances, lights, ambient)
}

/// A wall filling the view, glowing by `emissive` and reflecting `albedo`.
///
/// Nothing else lights it: no lamp, no ambient, and the caller asks for no bounce — so what reaches
/// the camera is the emissive term and nothing that could stand in for it.
fn glowing_wall(albedo: Vec3, emissive: Vec3) -> StaticScene {
    let meshes = [quad(
        Vec3::new(100.0, -80.0, -40.0),
        Vec3::new(0.0, 160.0, 0.0),
        Vec3::new(0.0, 0.0, 160.0),
        Vec3::NEG_X,
        0,
    )];
    let materials = [Material {
        diffuse: albedo,
        emissive,
        ..Material::default()
    }];
    let instances = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::IDENTITY,
    }];
    common::scene_of(&meshes, &materials, &instances, &[], Vec3::ZERO)
}

#[test]
fn an_emissive_surface_is_lit_by_its_glow_rather_than_painted_with_it() {
    // **The bug this is for.** Emissive used to be written beside the albedo rather than through it,
    // so a surface that glowed showed its glow and nothing of itself — a Bitter Coast mushroom cap,
    // whose material carries 0.5 where its stalk carries zero, came out a flat white disc with no
    // texture in it at all. The original engine sums emission with the other light and multiplies
    // the whole by the texture; `docs/design.md` §8.36.
    let glow = Vec3::splat(0.25);
    let bright = trace_scene(
        &glowing_wall(Vec3::ONE, glow),
        &[],
        BOUNCE_EYE,
        Vec3::X,
        0,
        Read::Radiance,
    );
    let dim = trace_scene(
        &glowing_wall(Vec3::splat(0.25), glow),
        &[],
        BOUNCE_EYE,
        Vec3::X,
        0,
        Read::Radiance,
    );

    let (lit, unlit) = (
        patch_mean(&bright, WIDTH / 2, HEIGHT / 2).x,
        patch_mean(&dim, WIDTH / 2, HEIGHT / 2).x,
    );
    println!("the same glow on albedo 1.0 reads {lit:.3}, on 0.25 reads {unlit:.3}");

    // Both have to be lit at all, or the ratio below is two zeroes agreeing.
    assert!(
        lit > 0.05,
        "an emissive surface with nothing else on it should be visible, got {lit}"
    );
    // **Four to one, because the albedos are.** Emissive is light the surface receives, so what
    // leaves it is that light times what it reflects — quartering the albedo quarters the result.
    // Written past the albedo instead, both would read the same and this ratio would be one.
    let ratio = lit / unlit;
    assert!(
        (ratio - 4.0).abs() < 0.4,
        "quartering the albedo changed the glow by {ratio:.2}x rather than 4x, so the emissive \
         term is not going through the albedo: {lit} against {unlit}"
    );
}

/// The colour-bleed fixture: a red floor under the white wall, lit from nearly overhead.
///
/// The light lands square on the floor and glances off the wall at about 0.2 of full incidence,
/// which keeps the direct term low enough that even a single bounce sample landing on the brightest
/// part of the floor cannot push a pixel past 1.0. Clipping there would flatten exactly the
/// differences these tests measure, and the convergence test needs the estimator to stay linear.
fn bleed_scene() -> StaticScene {
    bounce_scene(
        Vec3::new(0.5, 0.05, 0.05),
        Vec3::ZERO,
        &[Light {
            position: Vec3::new(40.0, 0.0, 300.0),
            colour: Vec3::ONE,
            radius: 600.0,
        }],
    )
}

/// The mean colour over a square of pixels, in the target's own 0..1 units.
///
/// A patch rather than a single pixel, because a handful of bounce rays is far too few for one
/// pixel to carry a value: each sample returns the floor's radiance or nothing, so a pixel holds
/// one of five levels. Averaging is what turns that dither back into the quantity it dithers.
fn patch_mean(pixels: &[u8], centre_x: u32, centre_y: u32) -> Vec3 {
    /// Half-width of the patch, so 17 by 17 pixels.
    const HALF: u32 = 8;

    let mut total = Vec3::ZERO;
    let mut count = 0.0;
    for y in centre_y - HALF..=centre_y + HALF {
        for x in centre_x - HALF..=centre_x + HALF {
            let p = at(pixels, x, y);
            assert!(is_hit(p), "patch pixel ({x}, {y}) missed the wall");
            total += Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) / 255.0;
            count += 1.0;
        }
    }
    total / count
}

#[test]
fn a_bounce_carries_the_floors_colour_onto_the_wall_above_it() {
    // A red floor under a white wall, lit by a white light. Nothing in the scene can put red on
    // the wall except light that reflected off the floor to get there, so the red-minus-blue gap
    // at a pixel *is* the indirect term, measured without a reference image.
    let scene = bleed_scene();

    let flat = trace_scene(&scene, &[], BOUNCE_EYE, Vec3::X, 0, Read::Radiance);
    let bounced = trace_scene(&scene, &[], BOUNCE_EYE, Vec3::X, 16, Read::Radiance);

    let flat_low = patch_mean(&flat, WIDTH / 2, LOW_ROW);
    let bounced_low = patch_mean(&bounced, WIDTH / 2, LOW_ROW);
    let bounced_high = patch_mean(&bounced, WIDTH / 2, HIGH_ROW);
    println!(
        "wall at z=12: flat {flat_low:?} bounced {bounced_low:?}\n\
         wall at z=70: bounced {bounced_high:?}"
    );

    // With no bounce rays the wall is exactly as achromatic as the light and the albedo make it —
    // white on white, so the channels agree to the last quantisation step.
    assert!(
        (flat_low.x - flat_low.z).abs() < 0.01,
        "an unbounced white wall under a white light is not grey: {flat_low:?}"
    );

    // At z = 12 the floor fills the lower half of the hemisphere and then some: the far edge is
    // 58.5 units away against 12 of height, so every direction steeper than 11.5 degrees below
    // horizontal lands on it, which is 37% of the cosine-weighted hemisphere. Times the floor's
    // 0.42 of outgoing red that is 0.16 of red against 0.016 of blue, a gap of about 0.14.
    let low_gap = bounced_low.x - bounced_low.z;
    assert!(
        low_gap > 0.08,
        "the floor put only {low_gap} of red on the wall above it: {bounced_low:?}"
    );

    // At z = 70 the floor is below 50 degrees of elevation, which is 6.5% of the same hemisphere —
    // a fifth of the exposure and so a fifth of the tint. Bleed that did not fall off with distance
    // from the bleeder would be a constant fudge rather than a gathered quantity.
    let high_gap = bounced_high.x - bounced_high.z;
    assert!(
        low_gap > 3.0 * high_gap,
        "bleed barely fell off with distance: {low_gap} low against {high_gap} high"
    );
}

#[test]
fn ambient_is_occluded_by_what_a_surface_faces() {
    // The same geometry with the floor black and the light gone, lit only by ambient. A bounce ray
    // that escapes sees the ambient; one that lands on the black floor sees nothing. So the wall's
    // brightness reads off how much sky each part of it can see — which is ambient occlusion, and
    // it is the half of this milestone that changes how an interior looks rather than what colour
    // it is.
    let scene = bounce_scene(Vec3::ZERO, Vec3::splat(0.5), &[]);

    let flat = trace_scene(&scene, &[], BOUNCE_EYE, Vec3::X, 0, Read::Radiance);
    let occluded = trace_scene(&scene, &[], BOUNCE_EYE, Vec3::X, 16, Read::Radiance);

    let flat_low = patch_mean(&flat, WIDTH / 2, LOW_ROW).x;
    let flat_high = patch_mean(&flat, WIDTH / 2, HIGH_ROW).x;
    let low = patch_mean(&occluded, WIDTH / 2, LOW_ROW).x;
    let high = patch_mean(&occluded, WIDTH / 2, HIGH_ROW).x;
    println!("ambient only: flat {flat_low:.3}/{flat_high:.3}, occluded {low:.3}/{high:.3}");

    // Unoccluded, ambient is a flat fill: a white wall under 0.5 of it is 0.5 everywhere, with no
    // way to tell the bottom from the top.
    assert!(
        (flat_low - 0.5).abs() < 0.01 && (flat_high - 0.5).abs() < 0.01,
        "unoccluded ambient is not flat: {flat_low} at the bottom, {flat_high} at the top"
    );

    // 37% of the hemisphere blocked at z = 12 leaves 0.31; 6.5% at z = 70 leaves 0.47. The bounds
    // are loose either side of those, because what is being asserted is that the occluded fraction
    // is *gathered from the geometry* — a fixed darkening would fail the second half.
    assert!(
        low < 0.40 && low > 0.22,
        "the wall's foot should keep roughly two thirds of the ambient, got {low}"
    );
    assert!(
        high > 0.42 && high < flat_high,
        "the wall's top sees nearly all of the ambient but not all of it, got {high}"
    );
}

/// Root-mean-square difference between two renders, over the pixels both of them hit.
///
/// Against a converged reference this is the Monte Carlo error of the indirect estimate, in the
/// 0..1 units the target stores — the baseline M7's denoiser has to beat.
/// The same handful of surfaces, placed `offset` from the world origin.
///
/// Instance transforms rather than moved vertices, which is how the engine places everything but
/// terrain: a mesh is authored about its own origin and the transform puts it somewhere.
fn scattered_at(offset: Vec3) -> StaticScene {
    let meshes = [
        // A wall filling the view, and two panels in front of it at different depths — enough
        // structure that a ray aimed a pixel or two wrong lands on something else.
        wall(300.0, -200.0..200.0, -200.0..200.0),
        wall(150.0, -40.0..40.0, -40.0..40.0),
        wall(80.0, -15.0..15.0, -60.0..-20.0),
    ];
    let materials = [0.9, 0.35, 0.6].map(|albedo| Material {
        diffuse: Vec3::splat(albedo),
        ..Material::default()
    });
    let instances = [0, 1, 2].map(|mesh| Instance {
        mesh: MeshId(mesh),
        transform: Affine3A::from_translation(offset),
    });
    common::scene_of(
        &meshes,
        &materials,
        &instances,
        &[Light {
            position: offset + Vec3::new(60.0, 40.0, 90.0),
            colour: Vec3::splat(2.0),
            radius: 600.0,
        }],
        Vec3::splat(0.15),
    )
}

#[test]
fn a_scene_far_from_the_origin_renders_as_it_does_at_the_origin() {
    // **A world is not a neighbourhood of the origin, and the arithmetic has to know it.** The
    // unprojection used to hand back a world-space point on the near plane which the shader then
    // differenced against the camera — two numbers the size of the world, whose difference is the
    // near distance. At the origin that is exact, so nothing showed; at Seyda Neen it aimed every
    // ray more than a hundred pixels wide, and moving the camera moved the error, which is what
    // made it look like the picture was shaking.
    //
    // Placing the *same* scene at the far corner of Vvardenfell and asking for the same picture is
    // the end-to-end statement of that: whatever the renderer does, where the world sits must not
    // be part of it.
    let eye = Vec3::ZERO;
    let here = trace_scene(
        &scattered_at(Vec3::ZERO),
        &[],
        eye,
        Vec3::X,
        0,
        Read::Radiance,
    );

    let far = Vec3::new(200_000.0, 150_000.0, 900.0);
    let there = trace_scene(
        &scattered_at(far),
        &[],
        eye + far,
        Vec3::X,
        0,
        Read::Radiance,
    );

    // Both have to be pictures of something, or agreeing is easy.
    let covered = |image: &[u8]| image.chunks_exact(4).filter(|p| is_hit(p)).count();
    assert!(
        covered(&here) > (WIDTH * HEIGHT / 4) as usize,
        "the fixture fills only {} pixels at the origin",
        covered(&here)
    );
    assert_eq!(
        covered(&here),
        covered(&there),
        "the scene covers a different number of pixels once it is moved, so rays are landing \
         somewhere else entirely"
    );

    let error = rmse(&here, &there);
    println!("rmse between the origin and {far:?}: {error:.5}");
    // Not bit-identical and cannot be: the vertices themselves are `f32`, and at 200,000 units the
    // spacing between representable positions is 0.024 — so the geometry really is a shade
    // different out there. A hundredth is far inside that and far outside what the old
    // unprojection did, which moved whole surfaces off the frame.
    assert!(
        error < 0.01,
        "the same scene renders differently at {far:?}: rmse {error}"
    );
}

fn rmse(image: &[u8], reference: &[u8]) -> f32 {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for (a, b) in image.chunks_exact(4).zip(reference.chunks_exact(4)) {
        if !is_hit(a) || !is_hit(b) {
            continue;
        }
        for channel in 0..3 {
            let d = (a[channel] as f64 - b[channel] as f64) / 255.0;
            sum += d * d;
            count += 1;
        }
    }
    assert!(count > 0, "the two renders share no hit pixels");
    (sum / count as f64).sqrt() as f32
}

#[test]
fn the_indirect_estimate_converges_as_its_sample_count_rises() {
    // The number M7 needs a baseline for. Sampling is stratified only by the hash, so the error
    // should fall as one over the square root of the sample count — halving for every quadrupling.
    // A systematic difference between the sample counts would not shrink at all, and a bias would
    // leave the error flattening out well above zero.
    let scene = bleed_scene();

    let reference = trace_scene(&scene, &[], BOUNCE_EYE, Vec3::X, 256, Read::Radiance);
    let errors: Vec<f32> = [1u32, 4, 16]
        .into_iter()
        .map(|samples| {
            rmse(
                &trace_scene(&scene, &[], BOUNCE_EYE, Vec3::X, samples, Read::Radiance),
                &reference,
            )
        })
        .collect();
    println!(
        "indirect RMSE against 256 samples: 1 spp {:.4}, 4 spp {:.4}, 16 spp {:.4}",
        errors[0], errors[1], errors[2]
    );

    // Four times the samples is half the error. Asserting 1.6 rather than 2.0 leaves room for the
    // reference's own residual error, which adds in quadrature and matters most at the low end.
    assert!(
        errors[0] > errors[1] * 1.6,
        "quadrupling the samples barely helped: {} to {}",
        errors[0],
        errors[1]
    );
    assert!(
        errors[1] > errors[2] * 1.6,
        "quadrupling the samples barely helped: {} to {}",
        errors[1],
        errors[2]
    );
    assert!(
        errors[2] < 0.05,
        "16 samples is still {} from converged, which is a bias rather than noise",
        errors[2]
    );
}

/// A cutout mask that alternates texel by texel, over a chain whose coarser levels are solid.
///
/// The shape of every alpha-masked thing in the game: a mask whose detail is finer than a distant
/// pixel, and whose average — which is what the coarser levels hold — is a perfectly good answer
/// for one.
fn checkered_cutout() -> Texture {
    const SIZE: usize = 16;
    let mut finest = Vec::with_capacity(SIZE * SIZE * 4);
    for y in 0..SIZE {
        for x in 0..SIZE {
            finest.extend_from_slice(&[255, 255, 255, if (x + y) % 2 == 0 { 255 } else { 0 }]);
        }
    }
    // Every coarser level solid, which is what averaging a half-and-half checker gives once the
    // detail is gone — rounded up, so a level that can no longer see the holes keeps the surface.
    let coarse = |n: usize| vec![255u8; n * n * 4];
    let levels = [coarse(8), coarse(4), coarse(2), coarse(1)];
    let refs: Vec<&[u8]> = std::iter::once(finest.as_slice())
        .chain(levels.iter().map(Vec::as_slice))
        .collect();
    Texture::from_levels(TextureFormat::Rgba8, SIZE as u32, SIZE as u32, &refs)
}

#[test]
fn a_cutout_seen_from_far_enough_stops_being_a_stipple() {
    // **A binary test on a point sample is a coin toss once the mask is finer than the pixel.**
    // Read at its finest level regardless of distance, this checker keeps or drops each pixel at
    // random — which is the crawling sparkle along the fringe of a rug and the speckle over a tree.
    // Read at the level the ray cone can actually resolve, the holes have already averaged away and
    // the surface is simply there.
    // The mask tiled twenty times across a wall far enough away that one pixel covers several
    // texels of it — the whole point being minification, which is where a point sample goes wrong.
    // At 4,000 units a pixel spans about 24 units and a texel about 7.5.
    let mut wall = uv_wall(4000.0, -1200.0..1200.0, -1200.0..1200.0);
    for uv in &mut wall.uvs {
        *uv *= 20.0;
    }
    let meshes = [wall];
    let materials = [Material {
        base_colour: Some(TextureId(0)),
        alpha: AlphaMode::Mask(0.5),
        ..Material::default()
    }];
    let instances = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::IDENTITY,
    }];
    let pixels = trace_textured(
        &meshes,
        &materials,
        &[Some(checkered_cutout())],
        &instances,
        Vec3::ZERO,
        Vec3::X,
    );

    // The wall is far enough to fill the middle of the frame, so every pixel there lands on it.
    // Any miss is a hole the cutout punched at random.
    let mut holes = 0;
    let mut sampled = 0;
    // Well inside the wall's own outline: at this distance the frame is far wider than the wall,
    // so a looser window would count the sky beyond its edges as holes.
    for y in 3 * HEIGHT / 8..5 * HEIGHT / 8 {
        for x in 3 * WIDTH / 8..5 * WIDTH / 8 {
            sampled += 1;
            if !is_hit(at(&pixels, x, y)) {
                holes += 1;
            }
        }
    }
    assert!(
        holes * 20 < sampled,
        "{holes} of {sampled} pixels fell through a mask that averages to solid at this distance"
    );
}

#[test]
fn a_sheet_lit_from_behind_transmits_and_a_solid_one_does_not() {
    // Morrowind hangs single layers of triangles everywhere — sails, banners, rugs — and a layer
    // has no back for its own shadow to fall on. Lit from the far side it glows; the same quad
    // marked solid is black there, because a solid's body is in the way.
    //
    // The two light positions mirror each other about the wall, so distance, attenuation and the
    // cosine are identical between them and the only difference left is the side. That is what
    // makes the expected ratio exactly `TRANSMISSION`, with nothing else to cancel out.
    let mut sheet = wall(200.0, -150.0..150.0, -150.0..150.0);
    sheet.submeshes[0].thin = true;
    let solid = wall(200.0, -150.0..150.0, -150.0..150.0);
    let materials = [Material::default()];
    let instances = [Instance {
        mesh: MeshId(0),
        transform: Affine3A::IDENTITY,
    }];
    // Dim enough that the front value lands mid-range: at full brightness both sides clip at one
    // and a half is indistinguishable from a whole.
    let lamp = |x: f32| Light {
        position: Vec3::new(x, 0.0, 0.0),
        colour: Vec3::splat(0.15),
        radius: 400.0,
    };

    // No ambient at all: the direct term is the whole image, so a transmitted value cannot be a
    // fill light in disguise.
    let trace_with = |mesh: &Mesh, light: Light| {
        let scene = common::scene_of(
            std::slice::from_ref(mesh),
            &materials,
            &instances,
            &[light],
            Vec3::ZERO,
        );
        patch_mean(
            &trace_scene(&scene, &[], Vec3::ZERO, Vec3::X, 0, Read::Radiance),
            WIDTH / 2,
            HEIGHT / 2,
        )
        .x
    };

    let front = trace_with(&solid, lamp(100.0));
    assert!(
        front > 0.2,
        "the fixture is too dim to tell a half of it from quantisation: {front}"
    );
    assert_eq!(
        trace_with(&solid, lamp(300.0)),
        0.0,
        "a solid wall was lit through from behind"
    );

    // The near side is untouched — transmission adds a term for the far side rather than splitting
    // the one the front already had.
    let sheet_front = trace_with(&sheet, lamp(100.0));
    assert!(
        (sheet_front - front).abs() < 0.01,
        "marking a run thin changed how it is lit from the front: {front} became {sheet_front}"
    );

    // And the far side arrives at half, which is the transmission the shader declares.
    let sheet_back = trace_with(&sheet, lamp(300.0));
    assert!(
        (sheet_back - front * 0.5).abs() < 0.01,
        "a sheet lit from behind returned {sheet_back}, not half of {front}"
    );
}
