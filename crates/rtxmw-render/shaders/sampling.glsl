// Turning a pixel and a sample index into directions, without a random number in sight.
//
// **The hash moves every frame**, which it did not until a temporal filter arrived. A fixed seed
// dithers and holds still, and with only an à-trous pass to smooth it that is the better trade —
// reseeding turns the residual into crawling static with nothing to average it away. Ray
// Reconstruction inverts the trade: it accumulates across frames, so a pattern that never changes
// is not noise it can remove but detail it preserves.
//
// The cost to the path that does not use it is small and real. Consecutive frames of a still
// camera, filtered, went from bit-identical to 0.7% RMSE apart. The stream constants keep one
// estimator's samples from repeating another's, and `sample_stream` keeps this frame's from
// repeating the last.

// The hash stream a pixel draws from this frame.
//
// **Two pixels never collide within a frame**, because exclusive-or with a fixed word is a bijection
// — the rotation changes which stream each pixel gets, not how many there are.
uvec2 sample_stream(uvec2 pixel) {
    return pixel ^ uvec2(frame.sequence * 0x9E3779B9u, frame.sequence * 0x85EBCA6Bu);
}

// The Lambertian BRDF's normalisation.
//
// Absorbed into the host's `INTENSITY` until now, where a single scale on the only lighting term
// was unobservable. It is observable from here: the indirect estimator in `lighting.glsl`
// integrates over the hemisphere and picks up a compensating factor of pi, so the ratio between
// direct and indirect light is only right when this sits where it belongs.
const float INV_PI = 0.318309886;

// Distinct hash streams, so the direction a bounce takes and the shadow rays it then casts are not
// drawn from the same numbers. Bounce `b`'s shadow rays use `STREAM_INDIRECT + b`, which is why
// this is the last one.
const uint STREAM_DIRECT = 0u;
const uint STREAM_BOUNCE = 1u;
const uint STREAM_INDIRECT = 2u;

// Added to whichever stream is asking, so the sun's samples never repeat a lamp's — and the moons'
// never repeat the sun's, which matters at dusk when all three are up at once and one pattern
// across all of them would correlate their penumbrae into a single band.
const uint STREAM_SUN = 4096u;
const uint STREAM_WATER_REFLECT = 8192u;
const uint STREAM_FOG = 12288u;
const uint STREAM_WATER_REFRACT = 8193u;
const uint STREAM_MASSER = 16384u;
const uint STREAM_SECUNDA = 20480u;
const uint STREAM_FILM = 24576u;

// A cheap, stable hash of four integers, for decorrelating one pixel's samples from its
// neighbours'.
//
// Stable rather than per-frame: without temporal accumulation, reseeding every frame turns the
// noise into crawling static. A fixed pattern dithers instead, which holds still.
uint hash(uvec4 v) {
    uint h = v.x * 0x8DA6B343u ^ v.y * 0xD8163841u ^ v.z * 0xCB1AB31Fu ^ v.w * 0x165667B1u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    return h;
}

// Two values in `0..1` from one hash, taking a half each.
vec2 unit_pair(uint seed) {
    return vec2(float(seed & 0xFFFFu), float((seed >> 16) & 0xFFFFu)) / 65535.0;
}

// A direction within `cos_radius` of `axis`, uniform over the cone's solid angle.
//
// How an area light is sampled when it is infinitely far away: the sun has no position to aim at,
// only a direction and a width, so a shadow ray picks a point on the disc it subtends. Uniform over
// solid angle rather than over the disc, which keeps the penumbra even instead of bunching samples
// at its centre.
vec3 cone_direction(vec3 axis, float cos_radius, vec2 u) {
    float cos_theta = mix(cos_radius, 1.0, u.x);
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    float phi = 6.2831853 * u.y;

    // Branchless orthonormal basis around the axis (Duff et al. 2017). The sign trick is what
    // avoids the degeneracy a naive cross product has when the axis nears the reference vector.
    float sign_z = axis.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign_z + axis.z);
    float b = axis.x * axis.y * a;
    vec3 tangent = vec3(1.0 + sign_z * axis.x * axis.x * a, sign_z * b, -sign_z * axis.x);
    vec3 bitangent = vec3(b, sign_z + axis.y * axis.y * a, -axis.y);

    return tangent * (cos(phi) * sin_theta) + bitangent * (sin(phi) * sin_theta) + axis * cos_theta;
}

// A point on the unit sphere, from two values in `0..1`.
vec3 sphere_point(vec2 u) {
    float z = 1.0 - 2.0 * u.x;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 6.2831853 * u.y;
    return vec3(r * cos(phi), r * sin(phi), z);
}

// A direction on the hemisphere around `normal`, distributed by the cosine of its angle to it.
//
// Malley's method: displacing the normal by a uniform point on the unit sphere lands exactly
// cosine-distributed, which is the one distribution that cancels the cosine term in the rendering
// equation and leaves the estimator as albedo times the mean radiance.
vec3 cosine_direction(vec3 normal, vec2 u) {
    vec3 d = normal + sphere_point(u);
    float len = length(d);
    // The sphere point can land antipodal to the normal and collapse the sum. Vanishingly rare and
    // not impossible, and normalizing zero would put a NaN in the frame.
    return len > 1.0e-4 ? d / len : normal;
}
