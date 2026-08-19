// The descriptor set, and the shapes the buffers in it hold.
//
// Every other module reads from here and none of them declare a binding of their own, so this file
// is the whole of what the host has to agree with. The structs are laid out for `scalar` block
// layout and match their `repr(C)` counterparts field for field; the Rust side pins the sizes.

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
layout(set = 0, binding = 9, rgba16f) uniform writeonly image2D normal_depth_target;
layout(set = 0, binding = 10, rgba16f) uniform writeonly image2D illumination_target;

// Last binding, and the only runtime-sized one: Vulkan allows a variable descriptor count only on
// the final element of a set. Slot zero is the fallback, so a material's texture id addresses
// `id + 1`.
// The light grid: cell `i` owns `light_grid_indices[light_grid_offsets[i] .. [i + 1]]`, which is
// why the offsets carry a trailing sentinel. Built in `light_grid.rs`.
layout(set = 0, binding = 12, scalar) readonly buffer LightGridOffsets {
    uint light_grid_offsets[];
};

layout(set = 0, binding = 13, scalar) readonly buffer LightGridIndices {
    uint light_grid_indices[];
};

layout(set = 0, binding = 14) uniform sampler2D textures[];

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
layout(set = 0, binding = 11, scalar) readonly buffer Frame {
    // Clip coordinates to an offset from the eye, in world axes — *not* the inverse
    // view-projection, which would land a world-space point the shader then has to subtract the
    // camera position from. See `ndc_to_world_offset` in `visibility_pass.rs` for what that cost.
    mat4 ndc_to_world_offset;
    vec3 camera_position;
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
    // The sinusoids the sea is summed from, built on the host from an empirical spectrum rather
    // than a series chosen by eye — see `wave_spectrum.rs`.
    Wave waves[WAVE_COUNT];
} frame;
