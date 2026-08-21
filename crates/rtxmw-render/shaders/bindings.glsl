// The descriptor set, and the shapes the buffers in it hold.
//
// Every other module reads from here and none of them declare a binding of their own, so this file
// is the whole of what the host has to agree with. The structs are laid out for `scalar` block
// layout and match their `repr(C)` counterparts field for field; the Rust side pins the sizes.

// Pulled in here so every pass that reads this file has `LUMA` without asking for it — see the
// note in `colour.glsl` for why one constant needs a file of its own.
#include "colour.glsl"

layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;

// Emissive surfaces and the sky, which the composite adds the lit result on top of. Keeping them
// here rather than in an image of their own costs nothing: they are the one part of the frame that
// is neither noisy nor demodulated, so they want neither the denoiser nor an albedo to divide by.
layout(set = 0, binding = 1, rgba16f) uniform writeonly image2D target;

// One entry per acceleration structure geometry, addressed by the two halves a hit reports.
struct Geometry {
    uint first_index;
    uint first_vertex;
    uint material;
    // Flags about the run rather than its material. Declared in `material_buffers.rs` too.
    uint flags;
};

// This run's triangles are a sheet — a sail, a rug, a leaf — with nothing behind them.
const uint GEOMETRY_THIN = 1u;

struct Material {
    vec3 diffuse;
    float opacity;
    vec3 emissive;
    float alpha_cutoff;
    uint base_colour;
    // Which shading model this surface runs. Declared in `material_buffers.rs` too, and pinned
    // there by a test, because a shader cannot see a Rust constant.
    uint kind;
    // The four ground textures a `KIND_TERRAIN` surface blends, packed sixteen bits apiece.
    uint terrain_layers0;
    uint terrain_layers1;
};

const uint KIND_DIFFUSE = 0u;
const uint KIND_WATER = 1u;
const uint KIND_TERRAIN = 2u;

// The side of one terrain texture tile, in world units. A cell is sixteen of them across, which is
// why a cell's origin drops out of the blend below: it is always a whole number of tiles.
const float TERRAIN_TILE = 512.0;

// Instance mask bits. A ray asking for `MASK_SOLID` alone cannot see water; `MASK_ANY` sees all.
const uint MASK_SOLID = 0x01u;
const uint MASK_ANY = 0xFFu;

// Scalar layout so these match the host structs field for field. Under the default std430 rules a
// `vec3` would be padded to sixteen bytes and every entry after the first would be misread.
layout(set = 0, binding = 2, scalar) readonly buffer Geometries {
    Geometry geometries[];
};

layout(set = 0, binding = 3, scalar) readonly buffer Materials {
    Material materials[];
};

layout(set = 0, binding = 4, scalar) readonly buffer Indices {
    uint indices[];
};

struct Attributes {
    vec3 normal;
    vec2 uv;
};

layout(set = 0, binding = 5, scalar) readonly buffer VertexAttributes {
    Attributes attributes[];
};

struct Light {
    vec3 position;
    float radius;
    vec3 colour;
    /// How large the emitter is, which is what gives its shadows a penumbra.
    float source_radius;
};

layout(set = 0, binding = 6, scalar) readonly buffer Lights {
    Light lights[];
};

layout(set = 0, binding = 7, scalar) readonly buffer Positions {
    vec3 positions[];
};

// The G-buffer. Splitting the frame into what a surface *is* and what light reaches it is what
// makes denoising possible at all: the noise lives entirely in the lighting, and filtering that on
// its own leaves every texel of albedo detail untouched. Recombining is one multiply.
//
// Depth rides in the normal's fourth channel because the two are always wanted together — they are
// the pair an edge-stopping filter tests to decide whether two pixels are the same surface.
layout(set = 0, binding = 8, rgba16f) uniform writeonly image2D albedo_target;
layout(set = 0, binding = 9, rgba16f) uniform writeonly image2D normal_roughness_target;
layout(set = 0, binding = 10, rgba16f) uniform writeonly image2D illumination_target;
// Where each pixel's surface was on the previous frame's screen, as a displacement in pixels.
layout(set = 0, binding = 11, rg32f) uniform writeonly image2D motion_target;
// The mirror-like part of a surface's response and how sharp it is — see `Guides` in `surface.glsl`.
// The specular *distance* rides in the albedo target's alpha, which is otherwise a constant.
layout(set = 0, binding = 12, rgba16f) uniform writeonly image2D material_target;
// Clip depth in `r` for the upscaler, distance from the eye in `g` for the filter.
layout(set = 0, binding = 13, rg32f) uniform writeonly image2D depth_target;

// Last binding, and the only runtime-sized one: Vulkan allows a variable descriptor count only on
// the final element of a set. Slot zero is the fallback, so a material's texture id addresses
// `id + 1`.
// The light grid: cell `i` owns `light_grid_indices[light_grid_offsets[i] .. [i + 1]]`, which is
// why the offsets carry a trailing sentinel. Built in `light_grid.rs`.
layout(set = 0, binding = 15, scalar) readonly buffer LightGridOffsets {
    uint light_grid_offsets[];
};

layout(set = 0, binding = 16, scalar) readonly buffer LightGridIndices {
    uint light_grid_indices[];
};

// What falls out of the sky, written here instead of into the frame so the upscaler composites it
// rather than denoising it — see `GBuffer::transparency` for why rain and Ray Reconstruction do not
// get on. Premultiplied colour, and the coverage beside it because NGX takes the two separately.
layout(set = 0, binding = 17, rgba16f) uniform writeonly image2D transparency_target;
layout(set = 0, binding = 18, r16f) uniform writeonly image2D transparency_opacity_target;

// One plume, as the host builds it — see `GpuEmitter`. Every number here comes out of the game's
// own `NiParticleSystemController`; what it draws with does not. See `particles.glsl`.
struct Emitter {
    vec3 origin;
    // How far anything can get from that origin, which is the sphere a ray rejects it by.
    float reach;
    // The direction parcels actually leave in, unit length — the emission cone's axis and not the
    // node's, since six of the game's vents point straight down.
    vec3 axis;
    // How wide the plume is where it leaves, in world units.
    float foot;
    // Where gravity has carried a parcel by the end of its life.
    vec3 drop;
    // How far a parcel gets before its life runs out, and how long that takes — which together
    // are the speed the noise is advected upward at.
    float height;
    float lifetime;
    vec4 colour;
    // The tint keyed over the rise: at the foot, at `ramp_mid`, and at the top. An albedo for
    // smoke; a blackbody's own colours for fire.
    vec4 ramp[3];
    float ramp_mid;
    // How far the plume opens over its own travel, as the sine of the cone's half-angle.
    float flare;
    // One where it burns rather than being lit.
    float additive;
    // What separates one emitter's noise from the next.
    uint seed;
};

layout(set = 0, binding = 19, scalar) readonly buffer Emitters {
    Emitter emitters[];
};

// Last, because a variable descriptor count is only allowed on a set's final binding.
layout(set = 0, binding = 20) uniform sampler2D textures[];

// Where a texture id's colour lands in the array above.
//
// Slot zero is the fallback a material with no texture takes, the next three are the sky's own
// pictures — two moon portraits and the weather's cloud sheet — and every texture after them is
// followed by the shading map estimated from it. One array rather than two because Vulkan allows a
// variable descriptor count only on a set's final binding.
//
// The sky's are reserved rather than appended because a material's id addresses a *fixed* pair of
// slots: anything inserted after the first cell had loaded would move every id behind it.
const uint SKY_SLOTS = 3u;

uint colour_slot(uint id) {
    return 1u + SKY_SLOTS + 2u * id;
}

uint shading_slot(uint id) {
    return 2u + SKY_SLOTS + 2u * id;
}

// Stands in for a material with no base colour texture. Declared as `NO_TEXTURE` in
// `material_buffers.rs` too, and pinned there by a test, because a shader cannot see a Rust
// constant.
const uint NO_TEXTURE = 0xFFFFFFFFu;

// How many sinusoids the surface is summed from. Declared again in `wave_spectrum.rs`, whose test
// pins the two together.
const int WAVE_COUNT = 32;

// One of them, as the host builds it: twenty tightly packed bytes, matching `GpuWave` field for
// field.
struct Wave {
    vec2 direction;
    float wavenumber;
    float amplitude;
    // Radians of phase per second, from the dispersion relation at the shelf's depth. Carried
    // rather than derived, because `sqrt(g k)` is only its deep-water limit and a shore is where
    // that limit stops holding.
    float speed;
};

// A buffer rather than push constants: the block reached the 128 bytes Vulkan guarantees, and then
// waves needed a clock. Scalar layout, so it matches the `repr(C)` struct in `visibility_pass.rs`
// field for field. The matrices arrive already multiplied — their product is all unprojection
// needs, and it is one thing to send rather than two to keep consistent.
// One of Morrowind's two moons: where it is, how wide, what its lit face radiates, and what it
// delivers to a surface facing it.
//
// **A sphere rather than a sprite.** The disc is drawn by reconstructing the sphere's own normal at
// the pixel and asking whether the sun reaches it, so the terminator is the real one — a crescent
// points at the sun because nothing else could have drawn it. `face` is the slot in `textures` its
// vanilla portrait sits at, or zero for a flat disc of `colour` where none was uploaded.
struct Moon {
    vec3 direction;
    float cos_radius;
    vec3 colour;
    uint face;
    vec3 light;
    // Luminance of the portrait's own mean texel, which is what the portrait is divided by so its
    // mottling multiplies `colour` as a ratio rather than replacing it with a level.
    float face_mean;
    // The axis the moon turns about, which its face is held upright against rather than against the
    // horizon — a locked moon's orientation follows its orbit, so the face turns as it crosses.
    vec3 pole;
    // How far the shading leans on Lommel-Seeliger rather than Lambert. See `Moon::lunar_lambert`.
    float lunar_lambert;
};

// Where `precip_kind` is cut, as thresholds rather than values — see `Falling` in `rtxmw-scene`.
//
// Here rather than in `precipitation.glsl` because `waves.glsl` and `wetness.glsl` both ask whether
// it is raining, and both are included before that file is.
const float PRECIP_RAIN = 0.5;
const float PRECIP_SNOW = 1.5;

layout(set = 0, binding = 14, scalar) readonly buffer Frame {
    // Clip coordinates to an offset from the eye, in world axes — *not* the inverse
    // view-projection, which would land a world-space point the shader then has to subtract the
    // camera position from. See `ndc_to_world_offset` in `visibility_pass.rs` for what that cost.
    mat4 ndc_to_world_offset;
    // The previous frame's `projection * rotation`: an offset from *that* frame's eye, to its clip
    // coordinates. Built in `visibility_pass.rs`.
    mat4 previous_clip_from_offset;
    // This frame's, for the clip depth an upscaler reprojects with. The inverse above turns a pixel
    // into a ray; this turns the hit back into the depth the pixel would have had.
    mat4 clip_from_offset;
    vec3 camera_position;
    // Sub-pixel offset added to the pixel centre, in pixels. Zero unless an upscaler asked for it,
    // and never part of a motion vector — see `visibility_pass.rs`.
    vec2 jitter;
    // How far the eye moved since the previous frame, `now - before`. Small, and differenced on the
    // host where both positions are known — never here, where they are world-scale.
    vec3 camera_motion;
    // Reciprocal of the light grid's cell size, then the corner it is addressed from and how many
    // cells it spans. Zero dimensions is a scene with no lights.
    float light_grid_scale;
    vec3 light_grid_origin;
    uvec3 light_grid_dimensions;
    vec3 ambient;
    float cone_spread;
    // The direction the sun's light *travels*, so the direction to it is the negation.
    vec3 sun_direction;
    float sun_cos_radius;
    // Zero when the cell has no sky, which costs no branch: every term it feeds is a multiply.
    vec3 sun_colour;
    uint bounce_samples;
    // Where the water surface sits, or negative infinity where there is none — so `water_level - z`
    // is simply never positive for a dry cell.
    float water_level;
    // Seconds since the engine started. Zero in a screenshot and in every test, so the water they
    // see is a definite shape rather than whenever they happened to run.
    float time;
    // Which frame this is. It moves the hash streams every frame, so a still camera does not redraw
    // the same noise — see `sample_stream` in `lighting.glsl`.
    uint sequence;
    // How much of the lighting painted into a texture to divide back out, from zero for the texture
    // as shipped to one for the whole estimate. See `baked_shading` in `surface.glsl`.
    float delight;
    // How far the relief painted into a texture tilts the normal it is shaded by, from zero for the
    // vertex normal alone to one for the whole of it. See `relief.glsl`.
    float relief;
    // The radiance the cell's fog scatters, how thickly it sits, and how much of the whole effect
    // to apply — see `fog.glsl`.
    vec3 fog;
    float fog_density;
    float fog_strength;
    // The sky dome's shape: the warm end of every tint on it, how far the sunward side has gone
    // toward that, and what the whole shape is multiplied by. Zero scale is an interior, which has
    // no dome — see `Sky::shape` in `rtxmw-scene`, which this draws per pixel.
    vec3 sky_warm;
    float sky_warmth;
    float sky_scale;
    vec3 sky_floor;
    float sky_stars;
    // The weather's own medium, in front of everything the sky has in it: its colour, whose scale
    // means nothing because it is given the brightness of whatever it stands over, and how much of
    // the sky it has taken over. Zero under clear and indoors — see `Veil` in `rtxmw-scene`.
    vec3 sky_veil;
    float sky_veiled;
    // One where the fog should be an even haze rather than banks — indoors, where the air is still
    // and a room is smaller than a single bank would be, and out of doors however far the weather's
    // own wind has stirred them out.
    float fog_uniform;
    // Which way the air moves and how hard, out of the weather's `Wind Speed`. Zero is dead still,
    // which foggy and snow are; nine tenths is a blight storm. See `Sky::wind` in `rtxmw-scene`.
    vec2 fog_wind;
    // How deep the layer stands, against clear weather's in still air: the weather's own
    // `Land Fog Depth` and its wind together — see `Sky::fog_lift` in `rtxmw-scene`.
    float fog_lift;
    // What is in the air: how far apart they stand, how far the volume reaches, which of the three
    // they are, and what the air does with them. Spacing is zero for the four weathers carrying
    // nothing — see `Precipitation` in `rtxmw-scene`.
    //
    // **The kind is ordered so one comparison answers the commonest question.** Rain is nought and
    // anything above it is not rain, which is what wets a surface and rings water; snow is one and
    // ash is two, and those two differ in how wide they are drawn and how dark they come out.
    float precip_spacing;
    float precip_reach;
    float precip_kind;
    vec3 precip_fall;
    // Masser and Secunda, which the sky draws as spheres and the surfaces are lit by — see `Moon`
    // above and `moon_disc` in `lighting.glsl`. Black and zero-width for a cell with no sky.
    Moon masser;
    Moon secunda;
    // The cloud layer: how high it sits, how far the world curves under it, how much of it one tile
    // of the painted sheet spans, and how far the wind has carried that sheet across it.
    float cloud_altitude;
    float cloud_world_radius;
    float cloud_tile;
    vec2 cloud_drift;
    // What a lit cloud and a shadowed one radiate — sun plus sky, and sky alone.
    vec3 cloud_lit;
    vec3 cloud_shadowed;
    // The sheet's own mean luminance, which its texels are read as a ratio to, then how much sky
    // the layer covers and where the sheet sits in `textures`. Either at zero is no layer at all.
    float cloud_mean;
    float cloud_cover;
    uint cloud_sheet;
    // The sinusoids the sea is summed from, built on the host from an empirical spectrum rather
    // than a series chosen by eye — see `wave_spectrum.rs`.
    // The flash, if one is burning: what it puts into the sky, which of the three shapes it took —
    // 0 sheet, 1 in-cloud, 2 to ground — where the channel stands and where it ends. A zero radiance
    // is no flash, which is every frame of every weather but one.
    vec3 flash_radiance;
    float flash_kind;
    uint flash_seed;
    vec3 flash_source;
    vec3 flash_ground;
    // How many of `emitters` the resident cells placed — see `particles.glsl`.
    uint emitter_count;
    Wave waves[WAVE_COUNT];
} frame;
