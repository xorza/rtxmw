//! What a shower leaves on what it lands on.
//!
//! **The drops in the air are the half that is easy to believe and the wrong half to stop at.** A
//! surface under rain goes darker — a film keeps handing the light back to the substrate to absorb
//! another share of, which is Lekner and Dorf's answer to why anything is darker wet — and picks the
//! sky up off the top of it. Neither reads without the other, and both stop dead at whatever the
//! rain cannot reach.

use ash::vk;
use glam::{Affine3A, Vec2, Vec3};
use rtxmw_gpu::{TestGpu, readback};
use rtxmw_render::SceneRenderer;
use rtxmw_scene::{
    CellId, Falling, Instance, Material, Mesh, MeshId, Precipitation, Sky, StaticScene, Submesh,
    WorldTime,
};

mod common;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

/// How dark a film leaves an albedo, out of the constants `wetness.glsl` derives.
///
/// Repeated here because a shader constant is not visible from Rust, and the expectation below is
/// derived from it rather than written out.
fn soaked(albedo: f32) -> f32 {
    const ENTERING: f32 = 0.0918;
    const RETURNING: f32 = 0.475;
    (1.0 - ENTERING) * albedo * (1.0 - RETURNING) / (1.0 - albedo * RETURNING)
}

/// How much of the environment a film returns at `cosine` off its normal, out of `wetness.glsl`.
///
/// Lazarov's fit to the split-sum environment BRDF, repeated here for the same reason `soaked` is: a
/// shader constant is not visible from Rust, and every expectation below is derived from this rather
/// than written out.
/// The most of the film a porous world ever holds standing on its surface, out of `FILM_HELD`.
///
/// Only the deepest of the grain reaches it — `FILM_STANDS` gates the rest away — and a ring
/// passing over can carry it the rest of the way to a mirror, which is what `FILM_RING_HELD` is
/// for. So this is a ceiling on what any pixel returns rather than a factor every pixel gets.
const HELD: f32 = 0.55;

/// How rough water that has soaked in is, and how rough water still standing is — `FILM_ROUGHNESS`
/// and `FILM_PUDDLE`. Every pixel of a filmed surface lies between them.
const SOAKED_IN: f32 = 0.55;
const STANDING: f32 = 0.12;

fn film_response(cosine: f32, roughness: f32) -> f32 {
    const C0: [f32; 4] = [-1.0, -0.0275, -0.572, 0.022];
    const C1: [f32; 4] = [1.0, 0.0425, 1.04, -0.04];
    let r: Vec<f32> = (0..4).map(|at| roughness * C0[at] + C1[at]).collect();
    let a004 = (r[0] * r[0]).min((-9.28 * cosine).exp2()) * r[0] + r[1];
    let (scale, bias) = (-1.04 * a004 + r[2], 1.04 * a004 + r[3]);
    0.02 * scale + bias
}

/// A level floor of known albedo, with `roof` sheets of it stacked overhead.
fn ground(albedo: f32, roof: &[f32]) -> StaticScene {
    let quad = |z: f32| {
        let reach = 4_000.0;
        Mesh {
            positions: vec![
                Vec3::new(-reach, -reach, z),
                Vec3::new(reach, -reach, z),
                Vec3::new(reach, reach, z),
                Vec3::new(-reach, reach, z),
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
    };
    let mut meshes = vec![quad(0.0)];
    meshes.extend(roof.iter().map(|z| quad(*z)));
    let instances: Vec<_> = (0..meshes.len())
        .map(|at| Instance {
            mesh: MeshId(at as u32),
            transform: Affine3A::IDENTITY,
        })
        .collect();
    common::scene_of(
        &meshes,
        &[Material {
            diffuse: Vec3::splat(albedo),
            ..Material::default()
        }],
        &instances,
        &[],
        Vec3::splat(0.3),
    )
}

/// Rain as `[Weather Rain]` describes it.
fn rain() -> Precipitation {
    Precipitation {
        count: 450.0,
        diameter: 600.0,
        height: 500.0,
        fall: 4_025.0,
        kind: Falling::Rain,
    }
}

/// Where the camera stands and what it looks at. Both fixtures below aim at the floor's origin, so
/// the only thing that separates them is the angle they meet it at.
#[derive(Debug, Clone, Copy)]
struct View {
    eye: Vec3,
    forward: Vec3,
}

/// Straight down from a hundred units, so the floor fills the frame and the view is head on — where
/// the film's response is at its smallest and the sheen is the least of what is being measured.
const OVERHEAD: View = View {
    eye: Vec3::new(0.0, 0.0, 100.0),
    forward: Vec3::NEG_Z,
};

/// Down the floor's own length from just above it, which is where the response runs away with the
/// frame if nothing holds it back.
const ALONG: View = View {
    eye: Vec3::new(0.0, -600.0, 12.0),
    forward: Vec3::new(0.0, 1.0, -0.02),
};

/// The specular guide at the wettest pixel of a frame: how much of it is film, and how sharp.
#[derive(Debug)]
struct Pooled {
    returned: f32,
    roughness: f32,
}

/// What `read` makes of the middle of a frame of `scene` under `precipitation`, seen from `view`.
fn middle<T>(
    scene: &StaticScene,
    precipitation: Precipitation,
    view: View,
    read: impl FnOnce(&mut rtxmw_gpu::Uploader, &SceneRenderer) -> T,
) -> T {
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
    // No fog: what is measured is what the film did, and a march would lay a gradient over it.
    renderer.set_fog(0.0);
    renderer.set_sky(Sky {
        precipitation,
        ..Sky::at(WorldTime::hours(12.0))
    });

    let mut uploader = gpu.uploader();
    renderer
        .load_scene(
            gpu.device(),
            &mut uploader,
            gpu.physical().limits(),
            CellId::Exterior { x: 0, y: 0 },
            scene,
            &[],
        )
        .expect("scene should load");
    // Any up will do but the one the view is looking along, which `look_to_mat4` cannot make a basis
    // from — straight down needs a different one from the near-horizontal shot.
    let up = if view.forward.z.abs() > 0.9 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let looking = glam::camera::rh::view::look_to_mat4(view.eye, view.forward.normalize(), up);
    let projection =
        glam::camera::rh::proj::vulkan::perspective_infinite_reverse(60f32.to_radians(), 1.0, 0.05);
    let constants = renderer.frame_constants(looking, projection, view.eye);
    renderer
        .render_once(&mut uploader, &constants)
        .expect("trace should run");
    let held = read(&mut uploader, &renderer);
    drop(uploader);
    gpu.assert_no_validation_errors();
    held
}

/// The four channels `image` holds at the middle pixel.
///
/// The layout is the caller's because the two ends of the frame are left in different ones: the
/// guides stay `GENERAL` for whatever reads them next, and the traced target is handed on ready to
/// copy.
fn texel(
    uploader: &mut rtxmw_gpu::Uploader,
    image: &rtxmw_gpu::Image,
    layout: vk::ImageLayout,
) -> glam::Vec4 {
    let at = ((HEIGHT / 2) * WIDTH + WIDTH / 2) as usize * 4;
    let channels =
        readback::image_to_f32(uploader, image, layout).expect("readback should succeed");
    glam::Vec4::from_slice(&channels[at..at + 4])
}

/// The specular guide the trace wrote for the floor: how much of it is film, and how sharp.
fn film(scene: &StaticScene, precipitation: Precipitation, view: View) -> Pooled {
    middle(scene, precipitation, view, |uploader, renderer| {
        let mut read = |image| {
            readback::image_to_f32(uploader, image, vk::ImageLayout::GENERAL)
                .expect("readback should succeed")
        };
        let specular = read(renderer.material());
        let roughness = read(renderer.normal_roughness());
        let at = specular
            .as_chunks::<4>()
            .0
            .iter()
            .enumerate()
            .max_by(|(_, one), (_, two)| one[0].total_cmp(&two[0]))
            .expect("the frame should have pixels");
        Pooled {
            returned: at.1[0],
            roughness: roughness[at.0 * 4 + 3],
        }
    })
}

/// The green the trace put in the middle of the frame.
fn lit(scene: &StaticScene, precipitation: Precipitation, view: View) -> f32 {
    middle(scene, precipitation, view, |uploader, renderer| {
        texel(
            uploader,
            renderer.target(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )
        .y
    })
}

#[test]
fn rain_darkens_and_glosses_what_the_sky_can_see() {
    let scene = ground(0.5, &[]);

    // **Dry is the matte the rest of the world is.** Nothing in vanilla Morrowind carries a
    // specular — `NiSpecularProperty` is force-disabled at this NIF version — so a floor's guide
    // says mirror-nothing, roughness one, until the weather changes it.
    let dry = film(&scene, Precipitation::NONE, OVERHEAD);
    assert_eq!(
        (dry.returned, dry.roughness),
        (0.0, 1.0),
        "a dry floor should carry no specular at all"
    );

    // Head on, so Fresnel is at its normal-incidence floor of water's 2% and the film is nearly all
    // window — which is exactly what makes the *darkening* below measurable on its own.
    //
    // **Water stands somewhere, and it is a puddle where it does.** Head on the response is at its
    // normal-incidence floor, so even standing water returns a fraction of a percent — which is the
    // whole of what a puddle seen from straight above gives back. Pinning both ends catches a
    // response that drifted and a grain that stopped pooling anywhere alike.
    let pool = film(&scene, rain(), OVERHEAD);
    let head_on = film_response(1.0, pool.roughness);
    assert!(
        (head_on * 0.05..=head_on).contains(&pool.returned),
        "a puddle seen head on should return a share of {head_on}, not {}",
        pool.returned
    );

    // **Between soaked in and standing, and this reads the standing end.** `film` finds the wettest
    // pixel in the frame, which is where the water pooled — and a puddle is smooth, where the damp
    // ground around it takes the roughness of the stone it soaked into. Both ends are pinned because
    // a film that stopped varying would sit at one of them.
    assert!(
        (STANDING..SOAKED_IN).contains(&pool.roughness),
        "the wettest pixel should be a pool, smoother than damp stone — {}",
        pool.roughness
    );

    // **And darker by the closed form, not by a tint.** Light that leaves a 0.5 substrate is turned
    // back into it by total internal reflection at the top of the film — 0.475 of it, Egan and
    // Hilgeman's average at water's index — and what survives the whole series is 0.313, five
    // eighths of dry, once the film is thick enough to count.
    //
    // **Between drowned and barely damp**, which is the range the grain leaves it in: the middle of
    // the frame carries whatever depth the noise put there, and `FILM_THINNEST` is the shallowest
    // that goes. A surface under any film keeps less of itself than a dry one and none of them keeps
    // less than one under a full film.
    let drowned = soaked(0.5) / 0.5;
    let expected = 1.0 + (drowned - 1.0) * 0.35;
    let (dry, wet) = (
        lit(&scene, Precipitation::NONE, OVERHEAD),
        lit(&scene, rain(), OVERHEAD),
    );
    let measured = wet / dry;
    assert!(
        (drowned..=expected).contains(&measured),
        "a filmed floor should keep between {drowned} and {expected} of its dry self, not \
         {measured} ({wet} against {dry})"
    );
}

#[test]
fn what_the_rain_cannot_reach_stays_dry() {
    // A second sheet overhead, above the eye rather than between it and the floor, so the camera
    // still sees the floor and the rain does not.
    let sheltered = ground(0.5, &[200.0]);
    let sheltered = film(&sheltered, rain(), OVERHEAD);
    assert_eq!(
        (sheltered.returned, sheltered.roughness),
        (0.0, 1.0),
        "a floor under a roof should stay as dry as the weather it never met"
    );

    // And it is the roof doing it rather than the fixture: the same floor with the sheet taken away
    // is wet in the same weather.
    let open = film(&ground(0.5, &[]), rain(), OVERHEAD).returned;
    assert!(open > 0.0, "the open floor should be wet — {open}");

    // **Snow lands rather than soaking**, which is the same line `waves.glsl` draws for the ripples
    // and for the same reason: the ini writes `Rain Ripples=1` beside `Snow Ripples=0`.
    let flakes = Precipitation {
        count: 750.0,
        diameter: 800.0,
        height: 300.0,
        fall: 345.0,
        kind: Falling::Snow,
    };
    let snowed = film(&ground(0.5, &[]), flakes, OVERHEAD).returned;
    assert_eq!(
        snowed, 0.0,
        "snow should leave the ground dry, not {snowed}"
    );
}

#[test]
fn a_dark_floor_loses_more_of_itself_than_a_bright_one() {
    // **Which is the whole point of the series rather than a multiplier.** Each pass the film hands
    // the light back, a bright substrate returns more of it and a dark one keeps less — so the two
    // do not darken by the same fraction, and a flat scale could not tell them apart.
    let dark = soaked(0.1) / 0.1;
    let bright = soaked(0.8) / 0.8;
    assert!(
        dark < bright,
        "a dark floor should keep less of itself — {dark} against {bright}"
    );

    let ratio = |albedo: f32| {
        let scene = ground(albedo, &[]);
        lit(&scene, rain(), OVERHEAD) / lit(&scene, Precipitation::NONE, OVERHEAD)
    };
    let (measured_dark, measured_bright) = (ratio(0.1), ratio(0.8));
    assert!(
        measured_dark < measured_bright,
        "and the trace should agree — {measured_dark} against {measured_bright}"
    );
}

#[test]
fn a_film_takes_from_the_diffuse_what_it_reflects() {
    // **The bug this exists to keep out.** The reflection was added on top of the surface's own
    // response and the response was left alone — so at a glancing angle, where Schlick's term runs
    // to one, a deck came back lit as wood *plus* a whole sheet of sky over it. The wood was still
    // there and could not be seen for the sky summed on it, and the whole thing read as plastic.
    // More light leaving than arrived is the short way to say it.
    //
    // **Measured as how much of the floor's own colour survives, at two angles.** A brighter floor
    // returns more than a darker one, and the *gap* between them is what the diffuse term is worth
    // in the frame — so how far that gap closes as the view goes glancing is exactly how much of the
    // diffuse the film took. Reading it this way needs nothing known about what is being reflected,
    // which is what makes it a test rather than a second copy of the shader.
    let gap = |view: View| {
        let bright = lit(&ground(0.8, &[]), rain(), view);
        let dark = lit(&ground(0.2, &[]), rain(), view);
        bright - dark
    };
    let (level, glancing) = (gap(OVERHEAD), gap(ALONG));
    // **A small shift, and small is the honest answer.** The film returns a fiftieth of the
    // environment head on and about a tenth at this angle — the split-sum figure, once the geometric
    // shadowing a rough surface has is accounted for — so what it takes from the floor's own colour
    // moves by that much and no more. Left alone, as it was, the gap would not move at all, which is
    // what this catches: the bar sits between the two.
    assert!(
        glancing < level * 0.93,
        "a glancing film should leave the floor less of itself — {glancing} against {level} level"
    );
    assert!(
        glancing > 0.0,
        "and not all of it: some of the floor should still show — {glancing}"
    );

    // And this is the angle where the bug could bite: the film returns several times more here than
    // it does seen from above.
    let glancing = film(&ground(0.5, &[]), rain(), ALONG);
    let above = film(&ground(0.5, &[]), rain(), OVERHEAD).returned;
    assert!(
        glancing.returned > above * 3.0,
        "a glancing view should be far more film — {} against {above} from above",
        glancing.returned
    );

    // **And bounded by what the surface can physically return.** Schlick alone reaches one at
    // grazing whatever the roughness, and there is no microfacet term under any of this to say
    // otherwise — the facets that would have to face the eye at this angle are shadowed by the ones
    // in front of them. The split-sum fit carries that shadowing, and porosity takes its share on
    // top: a world of mud and thatch cannot hand back more of the sky than it holds on its surface.
    let ceiling = film_response(0.0, glancing.roughness) * HELD;
    assert!(
        glancing.returned <= ceiling,
        "a film cannot return more than the {ceiling} its roughness and porosity allow, not \
         {}",
        glancing.returned
    );
}
