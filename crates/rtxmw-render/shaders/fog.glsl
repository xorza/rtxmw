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

// Height above the fog's base at which its density falls to `1/e`, in Morrowind units, for clear
// weather in dead still air. Every weather scales it by `frame.fog_lift`.
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

// Octaves of noise, and the heading and speed each one churns on — the air's own turbulence, which
// is there in a dead calm. What a *wind* adds on top is `FOG_GALE`, below.
//
// **The differing speeds are what stops it reading as a texture.** One field scrolling rigidly past
// is a pattern in motion; three shearing against each other at their own rates make the shapes
// themselves form and pull apart, which is what fog actually does. The headings differ for the same
// reason, and the third has a little vertical drift so banks rise and settle rather than only
// sliding.
const int FOG_OCTAVES = 3;
const vec3 FOG_CHURN[FOG_OCTAVES] = vec3[FOG_OCTAVES](
    vec3(11.0, 7.0, 0.0),
    vec3(-6.0, 14.0, 2.5),
    vec3(19.0, -4.0, -1.5)
);

// What a `Wind Speed` of one comes to in world units a second.
//
// **Advection, which is not what `FOG_CHURN` is.** Those three drag the octaves past each other on
// headings that disagree, which is what makes the shapes form and pull apart — air doing that in a
// dead calm is the whole reason a still fog is not a frozen texture. This is the separate thing a
// wind adds: the entire field carried downwind together, on the heading the cloud layer drifts
// along, because there is one wind over a landscape.
//
// **Read as a wind rather than picked, which is what it took to make an ash storm look like one.**
// The first figure here was 120, chosen against `FOG_GRAIN` so that the strongest weather crossed
// one cell of the coarsest noise in about nine seconds — and nine seconds to cross thirteen metres
// is 1.4 metres a second, which is a still afternoon rather than a storm. Twenty metres a second is
// a Beaufort 8 gale, and seventy units to the metre makes that 1,400. The ten then land where their
// names say: clear's 0.1 is a two-metre breeze, rain's 0.3 is six, thunderstorm's 0.5 is ten,
// ashstorm's 0.8 is sixteen, and blight and blizzard blow eighteen.
//
// `frame.time` runs at the clock's own rate rather than the game's thirty-times one, so this is a
// wind rather than the time-lapse §8.56 caught the cloud layer in.
const float FOG_GALE = 1400.0;

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

// How many shadow rays the sun gets in the fog, and so how many pieces the march is cut into.
//
// **Not one per step.** A ray costs about four march steps here, so shadowing every one of the
// twenty-four would cost more than four times what the whole fog costs now. What the industry
// spends on this is the calibration worth having: a froxel volume traces one ray per cell, and at
// the grid NVIDIA ships in Remix that works out at about three quarters of a ray per output pixel.
// Eight is over ten times that sample density, and at full screen resolution rather than an eighth
// of it — which is where a shaft through a tree keeps its edges.
//
// **Eight rather than four because eight measured at 0.18 ms more**, which buys the step from a
// smear off a hillside to a beam: against a ray-per-step reference the error runs 0.0155, 0.0134,
// 0.0087, 0.0048 for one, two, four and eight. They are perfectly coherent — every one of them
// points at the same sun — which is why the eighth costs so little.
const uint FOG_SHADOW_RAYS = 8u;

// How many march steps one of those rays answers for. `FOG_SHADOW_RAYS` must divide `FOG_STEPS`.
const uint FOG_STEPS_PER_RAY = FOG_STEPS / FOG_SHADOW_RAYS;

// Below this fraction of what the sky scatters into the fog, the sun does not get a shadow ray.
//
// **What makes the cost fall only where the shafts are.** Ninety degrees off the sun the phase
// function is two thousandths of its forward value, so the sun puts less light into the air there
// than the rounding on the sky's term — and a shaft cut out of light that faint is a shaft nobody
// can see. Looking away from the sun, this is the whole of what fog costs.
const float FOG_SHAFT_FLOOR = 0.02;

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
    // The whole field carried downwind, before the octaves are dragged past each other — one
    // displacement rather than three, because a wind moves the air it is in rather than shearing it.
    //
    // **Minus, for the reason `cloud_uv` subtracts its own drift.** A bank sits at a fixed
    // coordinate in the noise, so sampling from further upwind as the clock runs is what carries it
    // past; adding would walk the whole field into the wind.
    position.xy -= frame.fog_wind * (frame.time * FOG_GALE);
    for (int octave = 0; octave < FOG_OCTAVES; ++octave) {
        vec3 at = position * frequency + FOG_CHURN[octave] * frame.time;
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
    // **Air only, and under a bay there is none.** The layer pools *at* the water rather than in
    // it: `above` clamps at the surface, so every point below one was coming out at the layer's
    // full thickness — a whole second medium laid over the water's own, which `water.glsl` already
    // attenuates and colours. Looking down into Seyda Neen's bay that put the fog's grey between
    // the eye and the seabed twice over.
    //
    // `water_level - z` is never positive for a dry cell, which is what the negative infinity in
    // the frame constants is for — so this costs a dry interior nothing and needs no flag.
    if (frame.water_level - position.z > 0.0) {
        return 0.0;
    }
    float above = max(position.z - fog_base(), 0.0);
    // **How deep the layer stands**, which the host settles out of the weather's own fog depth and
    // its wind together — see `Sky::fog_lift`. This is what makes the fog agree with §8.61's veil
    // rather than contradict it: the veil says eight of the ten fill the sky, and a medium that
    // filled the sky while pooling in a 37-metre bank was two answers to one question.
    float height = exp(-above / (FOG_HEIGHT * frame.fog_lift));
    // **Even, indoors — and evener the harder it blows.** Banks are a thing weather does to a
    // landscape; a room is smaller than one bank and its air is still, so what belongs there is a
    // faint uniform haze. Out of doors the same turbulence that lifts the layer mixes it, so a
    // blight storm is nine tenths of the way to a flat wall of dust while a fog bank keeps every
    // gap it has. `FOG_EVEN` is the band's own mean, so moving between them changes the character
    // and not the amount.
    //
    // The host settles which of the two this is, because it is one number for the whole frame and
    // this runs twenty-four times a ray.
    //
    // `patch` is what `coverage` wants to be called, and GLSL reserves that for tessellation.
    float banks = smoothstep(FOG_CLEARING, FOG_SOLID, fog_fbm(position));
    float coverage = mix(banks, FOG_EVEN, frame.fog_uniform);
    return frame.fog_density * FOG_EXTINCTION * height * coverage;
}

// The mean diameter of the fog's water droplets, in micrometres.
//
// **The one dial on the shape of the sun's halo.** Radiation fog runs from a few micrometres to
// about twenty, and the forward peak sharpens brutally with size: at five the fog scatters 1,300
// times an isotropic one straight down the sun's line, at eight 4,300, at thirty 81,000. Eight is
// a thick coastal fog, and it is the size the halo was chosen at rather than measured at.
const float FOG_DROPLET = 8.0;

// The solid angle of the whole sphere, inverted: what an isotropic phase function is worth.
const float INV_FOUR_PI = INV_PI * 0.25;

// The largest of a colour's three channels.
float brightest(vec3 colour) {
    return max(colour.x, max(colour.y, colour.z));
}

// Henyey-Greenstein, normalised so that it integrates to one over the sphere.
float henyey_greenstein(float g, float cos_theta) {
    float g2 = g * g;
    float denominator = 1.0 + g2 - 2.0 * g * cos_theta;
    return INV_FOUR_PI * (1.0 - g2) / (denominator * sqrt(denominator));
}

// What the fog sends toward the eye per steradian, `cos_theta` off the sun's line.
//
// **Mie, not Henyey-Greenstein.** A single HG lobe is the usual choice and it cannot do this shape:
// real droplets throw a diffraction peak within a degree of the light that is orders of magnitude
// above anything a lobe with a single `g` reaches, and they still send a sixth of isotropic
// *backwards*. Both are what a fog looks like — the blaze around a low sun and the fact that fog is
// not black when you turn away from it. Jendersie and d'Eon fit a HG peak blended with Draine's
// function to tabulated Mie over droplet diameters of five to fifty micrometres, which is a pair of
// lobes and four `exp`s rather than a table:
// <https://research.nvidia.com/labs/rtr/approximate-mie/>.
//
// **Per steradian, and that is not a detail.** The sky's term needs no phase function at all — it
// arrives from every direction, and a phase function integrates to one over the sphere, so the whole
// of it scatters in whatever shape the fog has. The sun arrives from one direction, as irradiance,
// and what comes back toward the eye is that irradiance times the phase function *per steradian*.
// Normalising this to "isotropic is 1" instead — which is the convention the lamps below are
// written in — makes the sun `4*pi` times too bright, which is a white-out with an exposure system
// underneath it doing its best.
//
// One evaluation for a whole ray. The sun is directional, so the angle between the view ray and it
// is the same at every point along the march — which is the only reason a phase function of this
// shape is affordable.
float fog_phase(float cos_theta) {
    float peak_g = exp(-0.0990567 / (FOG_DROPLET - 1.67154));
    float bulk_g = exp(-2.20679 / (FOG_DROPLET + 3.91029) - 0.428934);
    float alpha = exp(3.62489 - 8.29288 / (FOG_DROPLET + 5.52825));
    float bulk_share = exp(-0.599085 / (FOG_DROPLET - 0.641583) - 0.665888);

    // Draine's function is Henyey-Greenstein with a `1 + alpha*cos^2` term over the normalisation
    // that term costs.
    float bulk = henyey_greenstein(bulk_g, cos_theta)
               * (1.0 + alpha * cos_theta * cos_theta)
               / (1.0 + alpha * (1.0 + 2.0 * bulk_g * bulk_g) / 3.0);
    return mix(henyey_greenstein(peak_g, cos_theta), bulk, bulk_share);
}

// The sun's light as it scatters into a ray heading `direction`, before anything shadows it.
//
// Black for a cell with no sky, which is how an interior says it has no sun.
vec3 fog_sunlight(vec3 direction) {
    if (frame.sun_colour == vec3(0.0)) {
        return vec3(0.0);
    }
    return frame.sun_colour * fog_phase(dot(direction, -frame.sun_direction));
}

// The optical depth between a point of the given `extinction` and the sky above it, along the line
// to the sun.
//
// **Fog shadows itself, and leaving that out is what makes single scattering white out.** Light
// arriving at a point deep in a bank has crossed the whole bank to get there; without the term, every
// point in the fog is lit as though it were the first one the sun touched, and a phase function that
// aims the sun's light at the eye then multiplies a quantity that was already several times too
// large.
//
// Closed form rather than a second march. The density falls off exponentially with height, so the
// column along a straight line out of it integrates to `sigma * H / cos(zenith)` — the same integral
// an atmosphere's optical depth uses, and the one Golubev writes out for exponential media. Its
// assumption is that the coverage a point sits in continues along that line, which is what a bank
// looks like from inside it and is wrong only near a bank's edge, where the fog is thin and the term
// is close to one anyway.
float fog_sun_depth(float extinction) {
    // A sun on the horizon lights an infinite column of fog; the floor is what keeps that finite.
    float climb = max(-frame.sun_direction.z, 1e-3);
    return extinction * FOG_HEIGHT / climb;
}

// The radiance scattering toward the eye from a point in the fog.
//
// **Every lamp that reaches it**, through the same grid a surface uses, so a lantern shows as a
// halo in the murk rather than lighting only what it stands on. Unshadowed: a shaft needs a ray per
// light per step, which is a different order of cost from this.
//
// **Isotropic, and `1/4*pi` is what isotropic actually is.** A lamp reaches this point as
// irradiance, the same as the sun does, and what comes back toward the eye is that irradiance times
// a phase function *per steradian* — so a lamp with no phase function still owes the factor. It was
// missing, and lamps lit the air twelve and a half times more strongly than a sky of the same
// radiance did.
//
// Not the real phase function, unlike the sun's: a lamp's angle to the view ray changes at every
// step and for every lamp, where the sun's is fixed for a whole march. It would also be a firefly
// waiting to happen — the forward peak is 4,300 times isotropic, and a lamp sits at a finite
// distance where a march step can land almost exactly on the line from the eye through it.
//
// `sun` arrives with everything already taken out of it — the phase function, the shadow ray, the
// fog's own column, the water overhead — so this is a sum and nothing more.
vec3 fog_light(vec3 position, vec3 sun) {
    vec3 lamps = vec3(0.0);
    uvec2 near = lights_reaching(position);
    for (uint k = near.x; k < near.y; ++k) {
        Light light = lights[light_grid_indices[k]];
        vec3 offset = light.position - position;
        float reach = length(offset);
        if (reach < light.radius) {
            lamps += light.colour * attenuation(reach, light.radius);
        }
    }
    return frame.fog + sun + INV_FOUR_PI * lamps;
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

    vec3 sun = fog_sunlight(direction);
    bool shafts = brightest(sun) > FOG_SHAFT_FLOOR * brightest(frame.fog);

    float transmittance = 1.0;
    vec3 scattered = vec3(0.0);
    float behind = 0.0;
    for (uint s = 0u; s < FOG_SHADOW_RAYS; ++s) {
        float reach = fog_depth(float((s + 1u) * FOG_STEPS_PER_RAY) / float(FOG_STEPS)) * span;
        // **One ray decides the whole stretch, from a point drawn anywhere along it.** Holding an
        // answer across several steps is what a froxel does too, and the jitter is what keeps it
        // from being a decision always taken at the same place: over frames the probe walks the
        // stretch, so a shaft's edge lands between two neighbours as noise rather than as a step.
        //
        // Aimed at a point on the sun's disc rather than at its centre, which costs nothing and
        // softens the edge. That matters more here than it does on a surface: the march samples
        // visibility far more coarsely than the fog varies, and a gradient wider than the sampling
        // is the only thing that keeps a hard edge from aliasing along the ray.
        float visible = 1.0;
        if (shafts) {
            vec3 probe = origin + direction * mix(behind, reach, offset);
            vec2 u = unit_pair(hash(uvec4(sample_stream(pixel), STREAM_FOG, 1u + s)));
            vec3 towards = cone_direction(-frame.sun_direction, frame.sun_cos_radius, u);
            visible = occluded(probe, towards, RAY_MAX) ? 0.0 : 1.0;
        }

        for (uint k = 0u; k < FOG_STEPS_PER_RAY; ++k) {
            uint i = s * FOG_STEPS_PER_RAY + k + 1u;
            float ahead = fog_depth(float(i) / float(FOG_STEPS)) * span;
            float stride = ahead - behind;
            vec3 position = origin + direction * (behind + stride * offset);
            behind = ahead;

            float extinction = fog_density_at(position);
            // Absorbed over this step, and what scatters in is lit where it sits.
            float absorbed = 1.0 - exp(-extinction * stride);
            // Everything between the sun and this point: what the geometry stopped, what the fog
            // itself absorbed on the way down, and what any water overhead took out of it.
            vec3 reaching = sun * visible
                          * exp(-fog_sun_depth(extinction))
                          * daylight_reaching(position);
            scattered += transmittance * absorbed * fog_light(position, reaching);
            transmittance *= 1.0 - absorbed;
        }
    }

    // The strength dial fades the whole effect rather than the density, so zero is the frame
    // untouched however thick the cell says its fog is.
    return vec4(
        scattered * frame.fog_strength,
        mix(1.0, transmittance, frame.fog_strength)
    );
}
