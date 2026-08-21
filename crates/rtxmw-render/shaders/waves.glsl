// The shape of the water surface, as a sum of the sinusoids the host chose for it.
//
// **One height field, differentiated twice.** The normal is its gradient and the caustics are its
// curvature, so the two cannot disagree about where a crest is — which they would the moment either
// sampled a field of its own. `wave_spectrum.rs` decides which waves are in the sum; this decides
// what they look like at a point.

// How far a wave draws the water toward its own crest, as a fraction of its height.
//
// Gerstner's waves move each particle in a circle rather than up and down, so the surface gathers
// toward the crests and stretches across the troughs: peaks sharpen, hollows flatten. That is the
// shape a real swell has, and every ocean renderer applies it.
//
// **At this sea state it is worth almost nothing, and it is kept because it is right rather than
// because it shows.** Its contribution to the Jacobian is the steepness `A k`, which sums to 0.58
// across the spectrum, against a refraction term that reaches several times that in deep water — so
// turning it from nought to one moves a few hundred pixels of a shore by at most a twentieth of
// their value. It would matter on a surface steep enough to fold, which these waves are not: one is
// the trochoid limit, where the crest comes to a point, and even there they cannot reach it.
//
// What it does buy outright is the zero-depth case in `water.glsl`'s `caustic`, which is wrong
// without it.
const float WAVE_CHOPPINESS = 1.0;

// How far the swell carries the ripples riding on it, and over what distance that carrying turns.
//
// **A sum of plane waves is a lattice, and curvature is where that shows.** The second derivative
// weights a component by `A k^2`, which climbs with wavenumber however the spectrum falls, so the
// shortest few dominate it whatever else is in the sum. A handful of plane waves crossing is a
// grid, and that grid was the caustics: a tiling of near-identical cells, regular enough to read as
// a texture rather than as water.
//
// So the short waves are carried. On real water they ride on the long swell and are swept along
// with its orbital motion, which bends their crests and slides them out of step from one trough to
// the next — the same effect that makes a real pool's caustics wander rather than tile. A drift of
// a few units is most of a wavelength to the shortest waves and a rounding error to the longest, so
// one displacement field does the whole job without touching the swell it is derived from.
//
// It still earns its place under the empirical spectrum, whose thirty-two directions might have
// been thought to make it redundant: turning it off moves three quarters of a shore's pixels.
const float WAVE_DRIFT = 13.0;
const float WAVE_DRIFT_LENGTH = 640.0;

// Morrowind's gravity, in world units per second squared: 8.96 m/s^2 across 69.99 units to the
// metre.
//
// Only the drift field uses it now. The spectrum's own components carry the speeds the host worked
// out for them, from a dispersion relation that accounts for the depth of the shelf — `sqrt(g k)`
// is only its deep-water limit, and a shore is where that limit stops holding.
const float WATER_GRAVITY = 627.1;

// One wave component at a point: how much of it the ray cone can still resolve, its wavenumber, and
// its phase there.
//
// **Shared by the normal and the curvature**, which is the whole point: those are the first and
// second derivatives of one height field, and the dispersion that makes long waves outrun short
// ones has to be the same physics in both or the light lands where the surface is not.
struct WaveSample {
    float detail;
    float wavenumber;
    float phase;
    float amplitude;
    vec2 direction;
};

// Where the ripples have been carried to by the time they are sampled — see [`WAVE_DRIFT`].
//
// Two long crossing swells at angles the octave sequence never takes, so the drift cannot fall into
// step with the waves it displaces. Both callers of `sample_wave` pass their positions through this
// first, and they have to agree: the caustics differentiate the same field the normals come from.
vec2 drifted(vec2 p, float time) {
    float wavenumber = TAU / WAVE_DRIFT_LENGTH;
    float speed = sqrt(WATER_GRAVITY * wavenumber);
    float along = wavenumber * dot(vec2(0.8347, 0.5507), p) - speed * time;
    float across = wavenumber * dot(vec2(-0.4132, 0.9106), p) - speed * time * 0.83;
    return p + WAVE_DRIFT * vec2(sin(along) + 0.6 * cos(across),
                                 cos(along) - 0.6 * sin(across));
}

// How much of a wave that long a cone this wide can still tell apart, from none of it to all.
//
// A wave narrower than the pixel looking at it is averaged away rather than drawn: a ray cone a
// wavelength wide covers a crest and a trough whose slopes cancel, and picking one of them instead
// is what makes distant water a field of crawling white sparks. A property of the cone rather than
// of what raised the wave, so the swell and the rain's rings fade by the same rule.
float resolved(float wavelength, float footprint) {
    return 1.0 - smoothstep(0.25 * wavelength, 0.75 * wavelength, footprint);
}

WaveSample sample_wave(int index, vec2 p, float time, float footprint) {
    Wave component = frame.waves[index];
    float wavelength = TAU / component.wavenumber;
    WaveSample wave;
    wave.amplitude = component.amplitude;
    wave.direction = component.direction;
    wave.detail = resolved(wavelength, footprint);
    wave.wavenumber = component.wavenumber;
    wave.phase = wave.wavenumber * dot(wave.direction, p) - component.speed * time;
    return wave;
}

// How far apart the raindrops that leave a ring on the water land, in world units.
//
// **Not how many fall, which is thousands to the square metre a second.** A real rain deposits far
// more drops than a surface could show as separate rings — what an eye picks out is a few tens at a
// time, each living about half a second. Twenty units is twenty-nine centimetres, which puts a dozen
// impacts on a square metre and a handful of them ringing at any moment.
const float RIPPLE_CELL = 20.0;

// How long one ring lasts before it has spread into nothing, in seconds.
const float RIPPLE_LIFE = 0.6;

// How fast a ring spreads, in world units a second.
//
// Capillary-gravity waves on water cannot travel slower than 0.23 m/s, which is where the surface
// tension and the gravity branches of the dispersion relation meet, and a splash ring runs out at
// twice that: thirty-five units is half a metre a second, so a ring reaches about thirty centimetres
// before `RIPPLE_LIFE` is up.
const float RIPPLE_SPEED = 35.0;

// The wavelength of the ring itself, in world units — about eleven centimetres, which is the scale
// capillary ripples actually take.
const float RIPPLE_LENGTH = 8.0;

// How steep a fresh ring is, as slope at its crest.
//
// Per ring, and rings overlap — nine cells are summed — so what it comes to as a *field* is the
// number to compare against the sea: the variance below works out to an rms slope of 0.20 against
// the spectrum's summed 0.58. About a third of the swell's, which is enough to break a reflection
// where it lands and gone again within `RIPPLE_LIFE`.
const float RIPPLE_STEEPNESS = 0.30;

// What the rain does to the water it lands on, as a slope added to the wave field's.
//
// **Rings from a lattice of impacts, the same trick the drops themselves use.** Each cell of the
// water plane holds one impact at a hashed place, repeating on its own phase, and what it leaves is
// a ring expanding at `RIPPLE_SPEED` and dying at `RIPPLE_LIFE`. The slope points away from the
// impact, because that is what a ring is.
//
// **Rain only, which the file says rather than this.** `[Weather]` carries `Rain Ripples=1` and
// `Snow Ripples=0`: a flake lands on water without a splash, and Bethesda wrote both halves down.
//
// `unresolved` comes back as the mean square slope the cone averaged away, on the same terms the
// spectrum reports its own — see `water_normal`. Rings are eleven centimetres across and stop being
// resolvable within a hundred metres of the eye, and rain on water at that range is a *duller*
// sheet rather than a smooth one, so throwing their slope away instead of roughening with it is
// what would put a mirror on the far half of a bay in a downpour.
vec2 ripple_slope(vec2 p, float time, float footprint, out float unresolved) {
    unresolved = 0.0;
    if (frame.precip_spacing <= 0.0 || frame.precip_kind > PRECIP_RAIN) {
        return vec2(0.0);
    }
    float detail = resolved(RIPPLE_LENGTH, footprint);
    float lost = 1.0 - detail * detail;

    vec2 slope = vec2(0.0);
    float wavenumber = TAU / RIPPLE_LENGTH;
    vec2 base = floor(p / RIPPLE_CELL);
    // The eight neighbours as well as the cell itself: a ring outlives its own cell, so an impact
    // next door is still ringing over this one.
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 cell = base + vec2(x, y);
            uint noise = hash(uvec4(uvec2(ivec2(cell) + 4096), 0u, 0u));
            vec2 jitter = vec2(float(noise & 0xFFFu), float((noise >> 12) & 0xFFFu)) / 4095.0;
            float offset = float(noise >> 24) / 255.0;
            vec2 fell = (cell + jitter) * RIPPLE_CELL;
            // Where this impact is in its own life, which each cell keeps its own phase of so the
            // whole surface does not ring at once.
            float age = fract(time / RIPPLE_LIFE + offset) * RIPPLE_LIFE;

            vec2 away = p - fell;
            float distance = length(away);
            float front = RIPPLE_SPEED * age;
            // Behind the front there is nothing yet, and the ring fades as it spreads — its energy
            // over a circumference that grows, and its own life running out.
            if (distance > front || front <= 0.0) {
                continue;
            }
            float faded = (1.0 - age / RIPPLE_LIFE) * front / max(distance, front * 0.15);
            float steepness = RIPPLE_STEEPNESS * min(faded, 1.0);
            // Whatever share of this ring the cone could not resolve, kept as variance rather than
            // dropped — a sinusoid of steepness `s` has mean square slope `s^2 / 2`, which is the
            // argument `water_normal` makes for the swell and holds here unchanged.
            unresolved += lost * 0.5 * steepness * steepness;
            if (detail <= 0.0) {
                continue;
            }
            slope += away / max(distance, 1e-4)
                   * (detail * steepness * cos(wavenumber * (distance - front)));
        }
    }
    return slope;
}

// The surface normal at a point, from the gradient of the wave height field.
//
// The height is `sum(A * sin(k * dot(d, p) - w * t))` and this is its slope, so the two cannot
// disagree — which matters more than it sounds: the caustics stage differentiates this same field
// again, and a normal that did not come from the height would put the light in the wrong place.
//
// The quad itself stays flat. Displacing two triangles would buy nothing a normal does not, and the
// silhouette of water against a shore is set by the terrain behind it rather than by the surface.
vec3 water_normal(vec2 p, float time, float footprint, out float unresolved) {
    vec2 here = drifted(p, time);
    vec2 slope = vec2(0.0);
    unresolved = 0.0;
    for (int i = 0; i < WAVE_COUNT; ++i) {
        WaveSample wave = sample_wave(i, here, time, footprint);
        float steepness = wave.amplitude * wave.wavenumber;
        // **What the cone could not resolve is not gone, it is rough.** Averaging a crest and a
        // trough into a flat facet throws away real slope, and a surface that lost its slope
        // reflects like polished plastic. Keeping the variance of what was dropped — a sinusoid of
        // steepness `s` has mean square slope `s^2 / 2` — is what lets it come back as a widened
        // specular lobe instead. This is LEAN mapping's argument, in the one dimension it needs.
        unresolved += (1.0 - wave.detail * wave.detail) * 0.5 * steepness * steepness;
        if (wave.detail <= 0.0) {
            continue;
        }
        slope += wave.direction * (wave.detail * steepness * cos(wave.phase));
    }
    // What the rain adds on top, which is not part of the spectrum and must not be: the caustics
    // differentiate the swell and a ring is far too small to carry one. Its roughness joins the
    // spectrum's, though, because a cone that cannot resolve a ring lost real slope either way.
    float rain;
    slope += ripple_slope(p, time, footprint, rain);
    unresolved += rain;
    return normalize(vec3(-slope, 1.0));
}
