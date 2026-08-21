//! The per-frame acceleration structure path, and what it costs.
//!
//! **`docs/design.md` M12 step one.** Skeletal animation needs the vertices of a placement rewritten
//! every frame and the structures over them rebuilt inside the frame's own command buffer — none of
//! which the renderer did for anything before this. What that costs decides the shape of the rest of
//! the milestone, and the number that decides it is measured here rather than assumed: at the
//! content's own scale a rebuild may well beat a refit, because a structure that has to stay
//! refittable is traversed by every ray in the frame and built once.
//!
//! Sized from the shipped content: the busiest cell in the game places twenty-two animated
//! references, and the mean skinned mesh is 1,401 vertices and 1,678 triangles.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{
    Bone, CellId, Channel, DeformingInstance, INFLUENCES, Influence, Instance, Material, Mesh,
    MeshId, NO_PARENT, Rig, RigId, StaticScene, Submesh, Sun,
};

mod common;

const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;

/// What the busiest cell in the game places, and what a skinned mesh averages.
const PLACEMENTS: u32 = 22;
const ROWS: u32 = 29;
const COLUMNS: u32 = 29;

/// Frames measured, after the same number thrown away.
///
/// A first frame builds rather than refits — an update has to have something to update — and the
/// first few of anything on a GPU are the driver warming up. Neither is what a steady frame costs.
const WARMUP: u32 = 24;
const MEASURED: u32 = 96;

/// A grid of `ROWS` by `COLUMNS` quads, which is the size of mesh this is about.
///
/// A lattice rather than one big quad because what is being measured is a build over triangles: the
/// vertex count and the triangle count are the inputs, and a mesh with four vertices would measure
/// the call overhead instead.
fn lattice() -> Mesh {
    let (mut positions, mut normals, mut uvs, mut indices) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for row in 0..=ROWS {
        for column in 0..=COLUMNS {
            let u = column as f32 / COLUMNS as f32;
            let v = row as f32 / ROWS as f32;
            positions.push(Vec3::new(u * 120.0 - 60.0, 0.0, v * 120.0 - 60.0));
            normals.push(Vec3::NEG_Y);
            uvs.push(Vec2::new(u, v));
        }
    }
    let stride = COLUMNS + 1;
    for row in 0..ROWS {
        for column in 0..COLUMNS {
            let corner = row * stride + column;
            // Wound so the triangles' own plane comes out along the normals they carry. A quad
            // whose winding disagrees with its normals is shaded as its own back, which is a fully
            // lit fixture rendering black.
            indices.extend([corner, corner + 1, corner + stride]);
            indices.extend([corner + 1, corner + stride + 1, corner + stride]);
        }
    }
    let index_count = indices.len() as u32;
    Mesh {
        positions,
        normals,
        uvs,
        indices,
        submeshes: vec![Submesh {
            first_index: 0,
            index_count,
            material: 0,
            thin: false,
        }],
    }
}

/// A rig that bends the lattice: one joint held still, one that swings, blended across its width.
///
/// **A skinned bend rather than a rigid turn**, so the measurement is of the thing the milestone is
/// about: every vertex reads four influences, blends four matrices and lands somewhere its
/// neighbour did not.
fn bend() -> Rig {
    let turn = |time: f32, degrees: f32| {
        let half = degrees.to_radians() / 2.0;
        // Stored `w` first, and about `+Z` — across the lattice's own plane, so the normals turn
        // with it and the shading has something to see.
        rtxmw_scene::QuaternionKey {
            time,
            value: [half.cos(), 0.0, 0.0, half.sin()],
        }
    };
    let mut influences = Vec::with_capacity((ROWS as usize + 1) * (COLUMNS as usize + 1));
    for _ in 0..=ROWS {
        for column in 0..=COLUMNS {
            let along = column as f32 / COLUMNS as f32;
            let mut influence = Influence::default();
            influence.bones[0] = 0;
            influence.bones[1] = 1;
            influence.weights[0] = 1.0 - along;
            influence.weights[1] = along;
            influences.push(influence);
        }
    }
    assert_eq!(
        INFLUENCES, 4,
        "the fixture leaves two slots empty on purpose"
    );
    Rig {
        mesh: MeshId(0),
        parents: vec![NO_PARENT, NO_PARENT],
        rest: vec![Affine3A::IDENTITY, Affine3A::IDENTITY],
        channels: vec![
            Channel::default(),
            Channel {
                rotations: vec![turn(0.0, 0.0), turn(1.0, 40.0), turn(2.0, 0.0)],
                ..Channel::default()
            },
        ],
        bones: vec![
            Bone {
                joint: 0,
                inverse_bind: Affine3A::IDENTITY,
            },
            Bone {
                joint: 1,
                inverse_bind: Affine3A::IDENTITY,
            },
        ],
        influences,
        // No named spans: the fixture is one animation and the whole of it, which is what a banner
        // is too.
        groups: Vec::new(),
        playing: 0.0..2.0,
    }
}

/// `PLACEMENTS` copies of it in a grid, every one deforming out of step with its neighbours.
///
/// Lit by a sun rather than by ambient alone, because ambient is what a surface receives regardless
/// of which way it faces: under it a deformation moves the geometry and changes nothing about the
/// picture, and the assertion that the vertices reached the trace would pass over a path that did
/// nothing at all.
fn scene(deforming: bool) -> StaticScene {
    let mut scene = common::scene_of(
        &[lattice()],
        &[Material::default()],
        &[],
        &[],
        Vec3::splat(0.05),
    );
    scene.sun = Some(Sun {
        // Across the surfaces rather than along the view, so a normal leaning by the deformation's
        // own slope is the largest change in the frame.
        direction: Vec3::new(-1.0, 1.0, -0.4).normalize(),
        colour: Vec3::splat(3.0),
        angular_radius: 0.00465,
    });
    let across = 6u32;
    let placed: Vec<Instance> = (0..PLACEMENTS)
        .map(|index| Instance {
            mesh: MeshId(0),
            transform: Affine3A::from_translation(Vec3::new(
                (index % across) as f32 * 130.0 - 325.0,
                400.0,
                (index / across) as f32 * 130.0 - 195.0,
            )),
        })
        .collect();
    match deforming {
        // A phase apiece, so a row of them does not move as one body — which is what a single
        // structure rebuilt for all of them would look like if the placements were not separate.
        true => {
            scene.deforming = placed
                .into_iter()
                .enumerate()
                .map(|(index, instance)| DeformingInstance {
                    instance,
                    rig: RigId(0),
                    phase: index as f32 * 0.37,
                })
                .collect();
            scene.rigs = vec![bend()];
        }
        false => scene.instances = placed,
    }
    scene
}

/// What one configuration measured.
#[derive(Debug, Clone, Copy)]
struct Measured {
    /// Milliseconds of device time in the animation stage, averaged over the measured frames.
    animate: f32,
    /// And in the trace, which is where a structure built for updatability is paid for.
    trace: f32,
}

/// Renders `scene` for enough frames to have a steady figure, and returns the mean of the last ones.
///
/// The real `SceneRenderer` down to the command buffer, because a replica of the frame path is
/// exactly the thing this cannot afford to be measuring.
fn measure(
    scene: &StaticScene,
    refit: bool,
    frames: u32,
    pixels: Option<&mut Vec<u8>>,
) -> Measured {
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
    renderer.set_fog(0.0);
    // Before the scene: the choice is baked into the structures when they are created.
    renderer.set_refit_deforming(refit);

    let mut uploader = gpu.uploader();
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            CellId::Interior("deforming".to_owned()),
            scene,
            &[],
        )
        .expect("scene should load");

    let eye = Vec3::new(0.0, -400.0, 0.0);
    let view = glam::camera::rh::view::look_to_mat4(eye, Vec3::Y, Vec3::Z);
    let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
        75f32.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
        0.05,
    );

    let mut animate = 0.0;
    let mut trace = 0.0;
    let mut measured = 0u32;
    for frame in 0..frames {
        // A moving clock, or the deformation would be the same every frame and the structures would
        // be rebuilt over vertices that had not moved.
        renderer.set_time(frame as f32 * (1.0 / 60.0));
        let constants = renderer.frame_constants(view, projection, eye);
        renderer
            .render_once(&mut uploader, &constants)
            .expect("frame should render");
        if frame >= WARMUP {
            let timings = renderer.timings().expect("timings should read back");
            animate += timings.animate;
            trace += timings.trace;
            measured += 1;
        }
    }
    if let Some(out) = pixels {
        *out = readback::image_to_rgba8(
            &mut uploader,
            renderer.target(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )
        .expect("readback should succeed");
    }
    drop(uploader);
    gpu.assert_no_validation_errors();
    Measured {
        animate: animate / measured.max(1) as f32,
        trace: trace / measured.max(1) as f32,
    }
}

/// How far two renders differ, as a mean absolute channel difference over what both of them hit.
///
/// Over the hits rather than the frame, because most of this frame is the empty room behind the
/// lattices and averaging over it would divide any answer by five for no reason.
fn differing(a: &[u8], b: &[u8]) -> f32 {
    let mut total = 0.0;
    let mut counted = 0u32;
    for (a, b) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
        if a[3] < 128 || b[3] < 128 {
            continue;
        }
        total += a[..3]
            .iter()
            .zip(&b[..3])
            .map(|(a, b)| f32::from(*a) - f32::from(*b))
            .map(f32::abs)
            .sum::<f32>()
            / 3.0;
        counted += 1;
    }
    total / counted.max(1) as f32
}

#[test]
fn the_vertices_a_frame_writes_are_the_ones_it_traces() {
    // **The whole chain in one assertion.** The deform pass writes a region of the shared streams,
    // a bottom level is built over that region inside the frame, the top level is rebuilt over the
    // bottom levels, and a ray finds the result. Any link missing and the two frames below are
    // identical — a build that read the bind pose, a barrier that let the trace run first, a top
    // level left holding the bounds it had.
    let moving = scene(true);
    let (mut early, mut late) = (Vec::new(), Vec::new());
    measure(&moving, false, WARMUP, Some(&mut early));
    measure(&moving, false, WARMUP + 20, Some(&mut late));
    let apart = differing(&early, &late);
    assert!(
        apart > 2.0,
        "two poses a third of a second apart differ by {apart:.3}/255 across what they both hit, \
         which is not a deformation reaching the picture"
    );

    // And a scene with nothing deforming is untouched by any of it: it must render exactly as it
    // did before this path existed, twice running.
    let still = scene(false);
    let (mut once, mut again) = (Vec::new(), Vec::new());
    measure(&still, false, WARMUP, Some(&mut once));
    measure(&still, false, WARMUP + 20, Some(&mut again));
    // Not bit-identical: the sampler's hash streams move with the frame counter, so a shadow ray
    // lands somewhere else and a handful of pixels either side of an edge flip. What must not
    // happen is the geometry moving.
    let noise = differing(&once, &again);
    assert!(
        noise < 0.5,
        "a scene with nothing deforming moved by {noise:.3}/255 between two frames"
    );
}

#[test]
fn refitting_and_rebuilding_are_both_measured_before_either_is_chosen() {
    let moving = scene(true);
    let frames = WARMUP + MEASURED;
    let rebuilt = measure(&moving, false, frames, None);
    let refitted = measure(&moving, true, frames, None);
    let still = measure(&scene(false), false, frames, None);

    let mesh = lattice();
    println!(
        "{PLACEMENTS} placements of {} vertices and {} triangles, {WIDTH}x{HEIGHT}",
        mesh.positions.len(),
        mesh.indices.len() / 3
    );
    println!(
        "  rebuild  animate {:.3} ms, trace {:.3} ms",
        rebuilt.animate, rebuilt.trace
    );
    println!(
        "  refit    animate {:.3} ms, trace {:.3} ms",
        refitted.animate, refitted.trace
    );
    println!(
        "  static   animate {:.3} ms, trace {:.3} ms",
        still.animate, still.trace
    );

    // **A frame over nothing that moves pays nothing for this existing.** The stage is skipped
    // entirely, so what is left between its two timestamps is nothing at all.
    assert!(
        still.animate < 0.02,
        "a static frame spent {:.3} ms animating nothing",
        still.animate
    );
    // And the path is real: rebuilding twenty-two structures and a top level over them cannot be
    // free, so a zero here is a stage that recorded nothing.
    assert!(
        rebuilt.animate > 0.0 && refitted.animate > 0.0,
        "the animation stage measured no time at all: {rebuilt:?}, {refitted:?}"
    );

    // **Refitting is the cheaper build**, by rather more than the margin here allows for: measured
    // at 0.108 ms against 0.242. This is what `DEFAULT_REFIT` rests on, and if it ever stops being
    // true the default is wrong rather than this assertion.
    assert!(
        refitted.animate < rebuilt.animate * 0.8,
        "refitting took {:.3} ms against rebuilding's {:.3}, so it is no longer the cheaper build",
        refitted.animate,
        rebuilt.animate
    );
    // **And it costs the traversal nothing**, which is the half of the trade that was expected to
    // bite: a structure that has to stay refittable cannot be organised as freely, and every ray in
    // the frame goes through it. Measured equal to three decimal places at a deviation of a third
    // of the mesh's own size.
    assert!(
        refitted.trace < rebuilt.trace * 1.1,
        "refittable structures traced at {:.3} ms against rebuilt ones' {:.3}, so updatability has \
         started costing the frame what it did not",
        refitted.trace,
        rebuilt.trace
    );
}
