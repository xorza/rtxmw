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

/// Height above the fog's base at which its density falls to `1/e`, in Morrowind units.
///
/// Seventy units to the metre, so this is about four metres: deep enough to swallow a doorstep and
/// shallow enough to stand clear of.
const float FOG_HEIGHT = 280.0;

// Where the fog's density peaks. Sea level, which is what the outdoors is measured from and close
// enough to an interior's floor to serve there too.
const float FOG_BASE = 0.0;

// How far a ray that hits nothing carries fog. Beyond this the sky is the sky.
const float FOG_REACH = 30000.0;

// How large one cell of the drift noise is, and how fast the whole field moves.
const float FOG_GRAIN = 900.0;
const vec3 FOG_WIND = vec3(14.0, 9.0, 0.0);

// How far the noise swings the density either side of the height falloff.
const float FOG_VARIATION = 0.55;

// How much of the fog does *not* fall off with height.
//
// **Height fog alone only fills hollows.** Stand above the layer and the air ahead is clear however
// far it goes, which is not what distance looks like — a hillside a mile off is paler than one at
// hand whatever the altitude. This fraction is the haze that gives that, and the rest is the part
// that pools on the ground and drifts.
const float FOG_FLOOR = 0.22;

// What the recorded density means as an extinction coefficient, per unit.
//
// The record's number is the original engine's dial rather than a physical quantity — its `0.75`
// indoors is a mood, not a measurement — so this is what turns one into the other.
const float FOG_EXTINCTION = 1.2e-4;

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

// Extinction at a point, in world space.
float fog_density_at(vec3 position) {
    float above = max(position.z - FOG_BASE, 0.0);
    float height = mix(exp(-above / FOG_HEIGHT), 1.0, FOG_FLOOR);
    float drift = fog_noise(position / FOG_GRAIN + FOG_WIND * frame.time / FOG_GRAIN);
    float varied = mix(1.0 - FOG_VARIATION, 1.0 + FOG_VARIATION, drift);
    return frame.fog_density * FOG_EXTINCTION * height * varied;
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
    float step = span / float(FOG_STEPS);

    // A different offset into the first step for every pixel and every frame, so what would be
    // twenty-four visible shells becomes noise the filter and the upscaler both remove.
    float offset = unit_pair(hash(uvec4(sample_stream(pixel), STREAM_FOG, 0u))).x;

    float transmittance = 1.0;
    vec3 scattered = vec3(0.0);
    for (uint i = 0u; i < FOG_STEPS; ++i) {
        vec3 position = origin + direction * ((float(i) + offset) * step);
        float extinction = fog_density_at(position);
        // Absorbed over this step, and what scatters in is lit where it sits.
        float absorbed = 1.0 - exp(-extinction * step);
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
