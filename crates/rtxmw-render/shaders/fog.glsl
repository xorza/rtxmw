// Fog that pools on the ground, drifts, and is lit by whatever is standing in it.
//
// **Marched rather than integrated.** An exponential falloff with height has a closed form — the
// whole ray's fog in a handful of instructions — but only while the density is uniform across the
// horizontal plane, and a fog that never moves is the one thing this was asked not to be. Adding
// noise to the density is what costs the march.
//
// The result is folded into the two channels the trace already writes rather than computed later:
// the composite forms `emitted + albedo * lighting`, and
//
//     (emitted + albedo * lighting) * T + inscatter
//         == (emitted * T + inscatter) + albedo * (lighting * T)
//
// so attenuating each half and adding the scattered light to the first gives the fogged frame with
// no extra image, no extra binding, and the lights already to hand here.

// Steps along the view ray.
//
// The integrand is smooth — an exponential times a low-frequency noise — so the count buys evenness
// rather than detail, and the jittered start below turns what is left of the banding into noise the
// temporal filter removes.
const uint FOG_STEPS = 24u;

// Height above the fog's base at which its density falls to `1/e`, in Morrowind units.
//
// Seventy units to the metre, so about thirty-seven of them: a layer deep enough to fill a valley
// and still thin out over the hill beside it. It began at four metres, which swallowed a doorstep
// and nothing else.
const float FOG_HEIGHT = 2600.0;

// Where the fog's density peaks when the cell has no water to gather over: sea level outdoors, and
// close enough to a floor to serve indoors.
const float FOG_BASE = 0.0;

// The height the fog pools at.
//
// A dry cell has no water level — the shader is handed negative infinity for one — so it falls back
// to the origin rather than putting the fog infinitely far below the world.
float fog_base() {
    return isinf(frame.water_level) ? FOG_BASE : frame.water_level;
}

// How far a ray that hits nothing carries fog. Beyond this the sky is the sky.
const float FOG_REACH = 30000.0;

// How large one cell of the coarsest drift noise is.
//
// **Sized to what the march can resolve where it is looked at.** The steps bunch near the camera —
// see `fog_depth` — so the first few hundred units are sampled finely enough to show cells this
// size, which is what puts shape into fog at eye level rather than only from a ridge.
const float FOG_GRAIN = 900.0;

// Octaves of noise, and the heading and speed each one drifts on.
//
// **The differing speeds are what stops it reading as a texture.** One field scrolling rigidly past
// is a pattern in motion; three shearing against each other at their own rates make the shapes
// themselves form and pull apart, which is what fog actually does. The headings differ for the same
// reason, and the third has a little vertical drift so banks rise and settle rather than only
// sliding.
const int FOG_OCTAVES = 3;
const vec3 FOG_WIND[FOG_OCTAVES] = vec3[FOG_OCTAVES](
    vec3(11.0, 7.0, 0.0),
    vec3(-6.0, 14.0, 2.5),
    vec3(19.0, -4.0, -1.5)
);

// Frequency step between octaves. Not two, so the lattices never line up and repeat.
const float FOG_LACUNARITY = 2.27;

// How far the field drags itself sideways before it is sampled, in units.
//
// **Domain warping**: rather than adding more octaves, the *coordinate* is displaced by a noise of
// its own, so the shapes stretch and curl instead of staying the roughly round blobs a sum of
// octaves gives. Quilez's `fbm(p + w·fbm(p))`, at one level and with a single-octave warp — the full
// construction is an fbm per component per level, which at twenty-four samples a pixel is a
// different budget from this.
//
// Horizontal only. The vertical shape of this fog is the height falloff, and warping across it would
// only blur the layer it is meant to have.
const float FOG_WARP = 450.0;

// Below this the air is clear, and the fog reaches full thickness at one.
//
// **This is what makes it patchy rather than merely uneven.** Scaling the density by a noise gives
// fog that is everywhere and varies; cutting it off gives banks with gaps between them, which is
// what a valley at dawn looks like.
// The band has to sit inside the noise's own range. Averaging octaves narrows it — three of them
// land mostly within a quarter either side of a half — so a threshold picked for one octave's spread
// clears almost everything: at `0.42..1.0` the fog all but vanished.
const float FOG_CLEARING = 0.44;
const float FOG_SOLID = 0.66;

// What the coverage is when the fog is even rather than banked. The band's own mean, so indoors and
// out differ in character rather than in how much air there is.
const float FOG_EVEN = 0.5;

// What the recorded density means as an extinction coefficient, per unit.
//
// The record's number is the original engine's dial rather than a physical quantity — its `0.75`
// indoors is a mood, not a measurement — so this is what turns one into the other.
const float FOG_EXTINCTION = 5.0e-4;

// Where along the ray the step ending at `fraction` of the way through reaches.
//
// **Uniform steps spend their samples where nobody is looking.** A ray that runs thirty thousand
// units gives the first hundred — where the fog is close enough to have any shape to it — a
// twentieth of one sample, and lays the other twenty-three across ground too far away to resolve.
// So the structure was only visible from high up, where everything is far and the coarse octave is
// all there is.
//
// Squaring bunches them at the near end: with twenty-four steps the first spans about fifty units
// and the last a couple of thousand. It is the same reasoning that makes a froxel grid slice its
// frustum exponentially rather than evenly.
float fog_depth(float fraction) {
    return fraction * fraction;
}

float fog_value(ivec3 cell) {
    return float(hash(uvec4(uvec3(cell + 4096), 0u)) & 0xFFFFu) / 65535.0;
}

// Trilinear value noise, which is as much structure as a drifting haze needs.
float fog_noise(vec3 p) {
    ivec3 base = ivec3(floor(p));
    vec3 f = fract(p);
    // Smoothstep, so the lattice does not show as a grid of creases.
    f = f * f * (3.0 - 2.0 * f);
    float total = 0.0;
    for (int corner = 0; corner < 8; ++corner) {
        ivec3 offset = ivec3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        vec3 weight = mix(1.0 - f, f, vec3(offset));
        total += fog_value(base + offset) * weight.x * weight.y * weight.z;
    }
    return total;
}

// Fractal noise: three octaves, each drifting on its own heading at its own speed, over a domain
// dragged sideways by a noise of its own.
float fog_fbm(vec3 position) {
    // Two samples of the same field at far-apart offsets, which is a cheap way to get a vector out
    // of a scalar noise — they are uncorrelated enough for a displacement.
    vec3 coarse = position / (FOG_GRAIN * 2.0);
    vec2 warp = vec2(
        fog_noise(coarse),
        fog_noise(coarse + vec3(5.2, 1.3, 7.1))
    ) - 0.5;
    position.xy += warp * FOG_WARP;

    float total = 0.0;
    float weight = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    for (int octave = 0; octave < FOG_OCTAVES; ++octave) {
        vec3 at = position * frequency + FOG_WIND[octave] * frame.time;
        total += amplitude * fog_noise(at / FOG_GRAIN);
        weight += amplitude;
        amplitude *= 0.5;
        frequency *= FOG_LACUNARITY;
    }
    return total / weight;
}

// Extinction at a point, in world space.
//
// **Measured from the water, not from the origin.** Fog gathers over water and drains off high
// ground, so the surface a cell records is the level it should pool at — and above the layer there
// is none of it, which is what standing on a hill is supposed to look like.
float fog_density_at(vec3 position) {
    float above = max(position.z - fog_base(), 0.0);
    float height = exp(-above / FOG_HEIGHT);
    // **Even, indoors.** Banks are a thing weather does to a landscape; a room is smaller than one
    // bank and its air is still, so what belongs there is a faint uniform haze. `FOG_EVEN` is the
    // band's own mean, so switching between them changes the character and not the amount.
    //
    // `patch` is what `coverage` wants to be called, and GLSL reserves that for tessellation.
    float banks = smoothstep(FOG_CLEARING, FOG_SOLID, fog_fbm(position));
    float coverage = mix(banks, FOG_EVEN, frame.fog_uniform);
    return frame.fog_density * FOG_EXTINCTION * height * coverage;
}

// The radiance scattering toward the eye from a point in the fog.
//
// **Every lamp that reaches it**, through the same grid a surface uses, so a lantern shows as a
// halo in the murk rather than lighting only what it stands on. Unshadowed: a shaft needs a ray per
// light per step, which is a different order of cost from this.
//
// No phase function. Isotropic scattering is the honest default for a fog with no measured one, and
// the alternative is a lobe chosen to look right, which is a thing to tune once there is something
// to tune it against.
vec3 fog_light(vec3 position) {
    vec3 total = frame.fog;
    uvec2 near = lights_reaching(position);
    for (uint k = near.x; k < near.y; ++k) {
        Light light = lights[light_grid_indices[k]];
        vec3 offset = light.position - position;
        float reach = length(offset);
        if (reach < light.radius) {
            total += light.colour * attenuation(reach, light.radius);
        }
    }
    return total;
}

// `colour` and `lighting` as they reach the eye through `distance` units of fog.
//
// Returns the transmittance in `w`, which the caller multiplies its lighting by, and the light
// scattered in along the way in `xyz`.
vec4 fog_along(vec3 origin, vec3 direction, float distance, uvec2 pixel) {
    if (frame.fog_density <= 0.0 || frame.fog_strength <= 0.0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    float span = min(distance, FOG_REACH);

    // A different offset into every step for every pixel and every frame, so what would be
    // twenty-four visible shells becomes noise the filter and the upscaler both remove.
    float offset = unit_pair(hash(uvec4(sample_stream(pixel), STREAM_FOG, 0u))).x;

    float transmittance = 1.0;
    vec3 scattered = vec3(0.0);
    float behind = 0.0;
    for (uint i = 0u; i < FOG_STEPS; ++i) {
        float ahead = fog_depth(float(i + 1u) / float(FOG_STEPS)) * span;
        float stride = ahead - behind;
        vec3 position = origin + direction * (behind + stride * offset);
        behind = ahead;

        float extinction = fog_density_at(position);
        // Absorbed over this step, and what scatters in is lit where it sits.
        float absorbed = 1.0 - exp(-extinction * stride);
        scattered += transmittance * absorbed * fog_light(position);
        transmittance *= 1.0 - absorbed;
    }

    // The strength dial fades the whole effect rather than the density, so zero is the frame
    // untouched however thick the cell says its fog is.
    return vec4(
        scattered * frame.fog_strength,
        mix(1.0, transmittance, frame.fog_strength)
    );
}
