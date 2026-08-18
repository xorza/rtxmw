//! Traces primary rays into an offscreen image and asserts on the pixels.
//!
//! This is where everything upstream is finally checked against reality. A wrong build-range
//! offset, a mirrored instance transform or a transposed projection all survive M3c and M3d
//! silently — they only become visible once a ray has to find the triangle. The scenes here are
//! small enough that where a hit lands can be worked out by hand.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_esm::EsmReader;
use rtxmw_gpu::{TestGpu, image_barrier, memory_barrier};
use rtxmw_render::{
    FrameConstants, GeometryBuffers, MaterialBuffers, SceneAcceleration, TextureArray,
    VisibilityPass,
};
use rtxmw_scene::{
    AlphaMode, Instance, Material, Mesh, MeshId, ModelIndex, StaticScene, Submesh, TextureId,
};
use rtxmw_texture::{Texture, TextureFormat};
use rtxmw_vfs::{DATA_DIR_VAR, morrowind_archives, morrowind_data_dir};

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

/// An axis-aligned quad in the world's YZ plane, facing back down -X toward a camera at the origin.
fn wall(x: f32, y: std::ops::Range<f32>, z: std::ops::Range<f32>) -> Mesh {
    Mesh {
        positions: vec![
            Vec3::new(x, y.start, z.start),
            Vec3::new(x, y.end, z.start),
            Vec3::new(x, y.end, z.end),
            Vec3::new(x, y.start, z.end),
        ],
        normals: vec![Vec3::NEG_X; 4],
        uvs: vec![Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y],
        indices: vec![0, 1, 2, 0, 2, 3],
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: 6,
            material: 0,
        }],
    }
}

/// Traces `instances` from `eye` looking along `forward` and returns the image as 8-bit RGBA.
fn trace(meshes: &[Mesh], instances: &[Instance], eye: Vec3, forward: Vec3) -> Vec<u8> {
    trace_with(meshes, &[Material::default()], instances, eye, forward)
}

/// As [`trace`], with an explicit material table.
fn trace_with(
    meshes: &[Mesh],
    materials: &[Material],
    instances: &[Instance],
    eye: Vec3,
    forward: Vec3,
) -> Vec<u8> {
    trace_textured(meshes, materials, &[], instances, eye, forward)
}

/// As [`trace_with`], with textures for the scene's path list to sample.
fn trace_textured(
    meshes: &[Mesh],
    materials: &[Material],
    textures: &[Option<Texture>],
    instances: &[Instance],
    eye: Vec3,
    forward: Vec3,
) -> Vec<u8> {
    let gpu = TestGpu::shared();
    let mut uploader = gpu.uploader();

    let geometry = GeometryBuffers::upload(&mut uploader, meshes).expect("upload failed");
    let acceleration = SceneAcceleration::build(
        gpu.device(),
        &mut uploader,
        gpu.physical().limits(),
        &geometry,
        materials,
        instances,
    )
    .expect("build failed");

    let tables = MaterialBuffers::upload(&mut uploader, &geometry, materials)
        .expect("material upload failed");

    let array =
        TextureArray::upload(gpu.device(), &mut uploader, textures).expect("texture upload failed");

    let target = gpu
        .create_target(WIDTH, HEIGHT, vk::Format::R16G16B16A16_SFLOAT)
        .expect("could not create target");
    let mut pass = VisibilityPass::new(gpu.device(), array.len()).expect("pipeline failed");
    pass.bind(
        acceleration.tlas(),
        target.image(),
        &geometry,
        &tables,
        &array,
    );

    let view = glam::camera::rh::view::look_to_mat4(eye, forward, Vec3::Z);
    let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
        FOV_Y.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
        0.05,
    );
    let constants = FrameConstants::new(view, projection, eye);

    let image = target.image().raw();
    let extent = target.extent();
    uploader
        .submit_and_wait(|device, cmd| {
            // SAFETY: the command buffer is recording, and the image and pipeline are alive.
            unsafe {
                image_barrier::transition(
                    device,
                    cmd,
                    image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                );
                pass.record(cmd, extent, &constants);
                memory_barrier::full(device, cmd);
            }
        })
        .expect("dispatch failed");

    let pixels = target
        .read_rgba8(&mut uploader, vk::ImageLayout::GENERAL)
        .expect("readback failed");
    drop(uploader);
    gpu.assert_no_validation_errors();
    pixels
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
    let pixels = trace_with(&[mesh], &materials, &instances, Vec3::ZERO, Vec3::X);

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

#[test]
fn a_real_interior_traces_to_a_recognisable_image() {
    let Some(data) = morrowind_data_dir() else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };
    let vfs = morrowind_archives().expect("the game is available");
    let bytes = std::fs::read(data.join("Morrowind.esm")).expect("Morrowind.esm should read");
    let esm = EsmReader::new(&bytes).expect("should parse");
    let models = ModelIndex::build(&esm).expect("model index should build");
    let scene = StaticScene::load_interior(&esm, &models, &vfs, CELL).expect("cell should load");

    // The centre of the cell's own geometry, which for this office lands inside the larger room.
    // There is no general "stand here" rule without tracing probe rays, so this is a fixture choice
    // that happens to work for this cell rather than something to reuse blindly.
    let eye = scene
        .bounds()
        .expect("a furnished cell has geometry")
        .centre();

    // Decode every texture the cell names. The 45 dangling references across the whole library are
    // why this is `Option`: a miss becomes the fallback slot rather than a failure.
    let mut textures = Vec::with_capacity(scene.materials.textures().len());
    let mut missing = 0usize;
    for path in scene.materials.textures() {
        let decoded = vfs
            .read(path)
            .ok()
            .and_then(|bytes| Texture::decode(&bytes).ok());
        if decoded.is_none() {
            missing += 1;
        }
        textures.push(decoded);
    }

    let pixels = trace_textured(
        &scene.meshes,
        scene.materials.materials(),
        &textures,
        &scene.instances,
        eye,
        Vec3::X,
    );
    let hits = hit_count(&pixels);
    let total = (WIDTH * HEIGHT) as usize;

    // Each distinct colour is a distinct material, so this counts how many surfaces are in view.
    // One would mean every hit resolved to the same table entry, which is exactly what a constant
    // geometry index or a misread custom index produces — and it would still fill the screen.
    let mut shades: std::collections::HashSet<[u8; 3]> = std::collections::HashSet::new();
    for pixel in pixels.chunks_exact(4).filter(|p| is_hit(p)) {
        shades.insert(shade(pixel));
    }

    // Alpha carries the hit flag, so a miss is alpha zero — which a PNG viewer composites as
    // transparent and shows as whatever is behind it. For a picture to look at, the flag has to be
    // dropped, or half the image reads as blown-out white when it is really background.
    let mut opaque = pixels.clone();
    for pixel in opaque.chunks_exact_mut(4) {
        pixel[3] = 0xFF;
    }
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("seyda_neen.png");
    rtxmw_gpu::golden::write_png(&path, &opaque, WIDTH, HEIGHT);

    let cutouts = scene
        .materials
        .materials()
        .iter()
        .filter(|m| !matches!(m.alpha, AlphaMode::Opaque))
        .count();
    println!(
        "  {cutouts} of {} materials run the alpha cutout",
        scene.materials.materials().len()
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
}
