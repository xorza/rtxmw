// Flames, steam and smoke, as volumes the ray walks rather than as the sprites the game drew.
//
// **The numbers are Morrowind's and the drawing is not.** An emitter's file says where it is, which
// way it points, how fast it lets go, how long a parcel lives, how wide it opens and how much of it
// there is — and every one of those describes a *plume* as well as it describes a spray of quads.
// What it also carries is a photograph: `tx_firealpha10` is a pinkish tan puff shot at some
// unrecorded exposure, and multiplying a flame by it is how fire came out pink. So the parameters
// are kept and the pictures are dropped. `docs/design.md` §8.103.
//
// **What replaces them.** A density field shaped by those parameters, roiled by noise advected
// along the emitter's own axis at the emitter's own speed, marched front to back:
//
//   - **Fire emits and does not absorb.** Its colour is the one thing about a flame that is not a
//     matter of taste — a blackbody at the temperature of the gas — and it cools as it rises, so
//     the hue reddens and the brightness falls as the fourth power without anything being told to
//     fade. `blackbody::colour` and `ParticleEmitter::burn` do that on the host.
//   - **Smoke scatters and absorbs.** Beer-Lambert along the ray, Henyey-Greenstein about the sun,
//     a short march toward it for self-shadowing, and the powder term a backlit puff needs. That is
//     the standard cloud recipe, and it is what stops smoke being a flat disc that punches a hole
//     in whatever is behind it.
//
// Nothing is simulated and nothing is stored: the field is a closed form in position and the clock,
// so a frame allocates nothing and any starting time gives the same picture.

// Most emitters a ray walks, whatever the host uploaded.
const uint EMITTER_LIMIT = 1024u;

// Steps taken through one emitter's bounding sphere, and toward the sun from each of them.
//
// **The sphere is what makes this affordable.** A candle's plume is five units across and the test
// against it throws the emitter away for almost every pixel of the frame, so the march is paid for
// only where there is something to march through. The light steps are for smoke alone — a flame
// emits and casts no shadow of its own.
const uint PLUME_STEPS = 32u;
const uint PLUME_LIGHT_STEPS = 4u;

// How many eddies span the plume's own height.
//
// **Tied to the plume rather than to the world**, so a candle's flame and a lava vent's are equally
// broken up rather than the small one being a smooth blob and the large one static.
const float PLUME_EDDIES = 6.0;

// Where the noise's own range is taken from, before it is stretched to `0..1`.
//
// Averaging octaves of anything piles the result up around the middle — three of them land almost
// entirely between a third and two thirds — so the ends have to be brought in before the erosion
// below has anything to work with.
const float PLUME_FLOOR = 0.35;
const float PLUME_CEILING = 0.72;

// How deeply the noise eats into the plume's shape.
//
// At zero the outline is the shape's own — a cone, which reads as a triangle. At one the noise can
// remove the plume entirely wherever it is thin, which leaves gaps in the middle of a flame. Six
// tenths breaks the edge into tongues and leaves the core alone.
const float PLUME_EROSION = 0.6;

// How thin the noise makes the *inside* of a plume where it is thinnest.
//
// **Erosion alone only works on the edge**, because it is a subtraction and the middle of a plume
// is far enough above it to survive whatever the noise says. That is right for a flame, whose core
// really is solid — and wrong for a spray, whose shape has no structure of its own to show: a
// hemispherical burst passes its radial test everywhere inside itself, so what is left is a smooth
// ball with a nibbled rim, and Vivec's drains came out as cotton wool. Multiplying as well as
// subtracting puts billows through the middle of it.
const float PLUME_GRAIN = 0.3;

// Octaves of noise, and what each is worth against the one before it.
const uint PLUME_OCTAVES = 3u;
const float PLUME_LACUNARITY = 2.3;
const float PLUME_GAIN = 0.55;

// Where the plume starts and stops, as fractions of the height a parcel reaches.
//
// **Neither end is a hard edge.** The bottom is where the fuel is and the gas has had no room to
// spread; the top is where it has cooled past glowing and thinned into the air. Cutting either
// square is the fault the sprites had, one scale up.
const float PLUME_FOOT = 0.10;
const float PLUME_HEAD = 0.55;
// A flame's tongues break up where the gas stops burning, which is well short of where a parcel
// stops travelling — so the head fade takes the top half of the rise and what is left is a rounded
// crown rather than the long thin spike a later fade leaves.
const float PLUME_HEAD_FLAME = 0.55;

// Where the radial falloff begins, as a fraction of the plume's radius at that height.
const float PLUME_CORE = 0.45;

// How far the plume leans off its own axis, as a fraction of its radius, and over how many bends.
//
// **A flame licks; it does not stand.** The axis of a real plume wanders — buoyancy is unstable, so
// the column snakes and the tongue at the top is somewhere else from one moment to the next. Two
// noise channels read along the axis give that for nothing, and reading them at a position that
// *rises* with the gas is what sends each bend travelling upward instead of the whole column
// waving like a flag nailed at the bottom.
const float PLUME_WANDER = 0.55;
const float PLUME_BENDS = 2.5;

// The sine of the half-angle past which a plume is a burst rather than a column.
//
// Forty-five degrees. Below it the file is describing something that goes *somewhere* — a flame, a
// chimney, a vent — and above it something that goes everywhere: `ex_waterfall_mist_01` writes a
// half-angle of `pi/2` and sprays into a whole hemisphere.
const float PLUME_BURST = 0.7;

// The shape of a flame, as a fraction of its widest: at the fuel, where its belly sits, at the tip.
//
// **Fire tapers where smoke spreads, and that is not a stylistic choice.** The luminous zone is
// where there is still fuel burning; above it the gas has been consumed and what is left rises and
// cools out of sight. So the *shape* of a flame is a taper even though the gas cone it sits in is
// opening the whole way — which is why the first attempt, built on the cone alone, came out as an
// upside-down flame filling a fireplace.
//
// **And it necks at the bottom, which the second attempt did not.** A flame is thinnest where it
// meets the fuel: the gas has not had room to expand and the reaction is still starting. Widest at
// the foot instead, the plume presents a flat disc to anything looking up at it and a flat base to
// anything looking level — which is a pyramid, and was reported as one. A quarter of the way up is
// where the belly sits.
const float FLAME_NECK = 0.42;
const float FLAME_BELLY = 0.25;
const float FLAME_TIP = 0.12;

// How bright fire is where it is thick enough to hide itself.
//
// **A flame is a grey body, and that is what makes one number enough.** It emits *and* absorbs, so
// what leaves a deep flame is not the sum of everything inside it — it is what the outermost layer
// radiates, because the layers behind that one are hidden by it. Radiance saturates at the
// blackbody value however deep the fire is, which is why a bonfire is not a thousand times brighter
// than a candle: it is the same brightness over more of the screen.
//
// Emission alone, with no absorption, is what a sprite-based renderer does by piling additive
// quads, and it is why the first volumetric attempt filled a fireplace with white — forty units of
// plume summed forty units' worth. With the absorption in, this is a *level* rather than a gain,
// and it means one thing: how bright the surface of a flame is. The colour is not here at all — it
// comes from the temperature, and `blackbody::colour` hands it over at unit luminance so this can
// say the level on its own.
const float FLAME_RADIANCE = 0.5;

// Optical depth straight up through a plume, from its foot to where it thins away.
//
// **Dimensionless, and against the plume rather than the world**, which is what lets a candle's
// three units and a lava vent's four hundred both come out looking like what they are. A flame is
// thick enough to hide its own far side; smoke is a veil that mostly does not.
const float FLAME_DEPTH = 4.0;
const float SMOKE_DEPTH = 1.5;

// How strongly smoke throws light forward, for Henyey-Greenstein.
//
// **Droplets and grains scatter forward**, which is why a plume lit from behind has a bright rim
// and one lit from in front is flat. Six tenths is the middle of what the cloud literature uses.
const float SMOKE_ANISOTROPY = 0.6;

// How dark the outside of a backlit puff goes — the "powder" term.
//
// Single scattering alone makes a thin edge *bright*, because there is little between it and the
// sun; what actually happens is that a thin edge has too little medium to have scattered much
// toward the eye at all, so it darkens. This is the standard correction for that.
const float SMOKE_POWDER = 1.6;

// A value in `0..1` at one lattice corner.
float plume_corner(ivec3 at) {
    return float(hash(uvec4(uvec3(at + 4096), 0x9E37u)) & 0xFFFFu) / 65535.0;
}

// Smooth value noise in three dimensions, in `0..1`.
//
// Value rather than gradient noise: eight hashes against twelve dot products, and what is wanted
// here is a lumpy field rather than one with a particular spectrum. Fire hides the difference and
// smoke is filtered by its own scattering.
float plume_noise(vec3 p) {
    ivec3 cell = ivec3(floor(p));
    vec3 f = fract(p);
    // The cubic weights, which are what make the lattice invisible.
    f = f * f * (3.0 - 2.0 * f);
    float total = 0.0;
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                vec3 weight = mix(1.0 - f, f, vec3(i, j, k));
                total += plume_corner(cell + ivec3(i, j, k)) * weight.x * weight.y * weight.z;
            }
        }
    }
    return total;
}

// Several octaves of it, normalised so the sum still runs `0..1`.
float plume_fbm(vec3 p) {
    float total = 0.0;
    float scale = 1.0;
    float weight = 1.0;
    float carried = 0.0;
    for (uint octave = 0u; octave < PLUME_OCTAVES; ++octave) {
        total += weight * plume_noise(p * scale);
        carried += weight;
        scale *= PLUME_LACUNARITY;
        weight *= PLUME_GAIN;
    }
    return total / carried;
}

// Where a point sits inside a plume: how far up it is, and how much is there.
struct Inside {
    // Fraction of the way from the emitter to where a parcel stops, `0..1`.
    float risen;
    // What is there, `0..1`, with the noise already in it.
    float density;
};

// The emitter's colour ramp at `fraction` of the way up the plume.
//
// Two linear segments. For smoke that is the file's own ramp, which is an albedo; for fire it is
// the blackbody one the host built out of the gas temperature.
vec4 particle_tint(Emitter emitter, float fraction) {
    if (fraction < emitter.ramp_mid) {
        return mix(emitter.ramp[0], emitter.ramp[1], fraction / max(emitter.ramp_mid, 1e-4));
    }
    return mix(emitter.ramp[1], emitter.ramp[2],
               (fraction - emitter.ramp_mid) / max(1.0 - emitter.ramp_mid, 1e-4));
}

// Reads the plume at a world position. `burning` shapes it as a flame rather than as smoke.
Inside plume_at(Emitter emitter, vec3 world, float time, bool burning) {
    float height = emitter.height;
    vec3 offset = world - emitter.origin;

    // **Gravity bends the plume rather than tilting it**, which is what a parabola looks like: a
    // parcel twice as far along has been falling twice as long, so the carry goes as the square.
    // Taken out of the position before the shape is read, so everything below is a straight cone.
    offset -= emitter.drop * (length(offset) / height);

    // **How far a parcel has got, and the two cases are not a blend.** A narrow plume is directed:
    // what it has done is *rise*, and it widens with that. A cone past `PLUME_BURST` has no axis
    // worth speaking of — `ex_waterfall_mist_01` writes a half-angle of `pi/2` and sprays into a
    // whole hemisphere, so most of what it throws goes sideways or down — and there the distance
    // from the emitter is what a parcel's life has bought. Measuring a *column* by distance instead
    // is what widened every base in the game into a flat disc, because a point out to the side of a
    // flame's foot has travelled as far as one halfway up it.
    float up = dot(offset, emitter.axis);
    bool burst = emitter.flare > PLUME_BURST;
    float risen = (burst ? length(offset) : up) / height;
    if (risen <= 0.0 || risen >= 1.0) {
        return Inside(0.0, 0.0);
    }

    // A teardrop for fire and an opening cone for smoke — see `FLAME_NECK` and `Emitter::flare`.
    float opening = emitter.foot + risen * height * emitter.flare;
    float radius = burning
                 ? opening * mix(FLAME_NECK, 1.0, smoothstep(0.0, FLAME_BELLY, risen))
                           * mix(1.0, FLAME_TIP, smoothstep(FLAME_BELLY, 1.0, risen))
                 : opening;

    // A stable pair of axes across the plume. Built from the emitter's own rather than passed,
    // because only the direction it leaves in means anything and the turn about that is arbitrary.
    vec3 sideways = normalize(cross(emitter.axis,
                                    abs(emitter.axis.z) < 0.99 ? vec3(0.0, 0.0, 1.0)
                                                               : vec3(1.0, 0.0, 0.0)));
    vec3 other = cross(emitter.axis, sideways);

    // Where this slice of the plume has wandered to, in that cross-section.
    float bend = (up - height * time / emitter.lifetime) * (PLUME_BENDS / height);
    vec2 lean = (vec2(plume_noise(vec3(bend, 11.3, 4.1)), plume_noise(vec3(bend, 27.7, 8.9))) - 0.5)
              * (PLUME_WANDER * radius * risen);
    vec2 across = vec2(dot(offset, sideways), dot(offset, other)) - lean;

    float shape = (1.0 - smoothstep(PLUME_CORE, 1.0, length(across) / radius))
                * smoothstep(0.0, PLUME_FOOT, risen)
                * (1.0 - smoothstep(burning ? PLUME_HEAD_FLAME : PLUME_HEAD, 1.0, risen));
    if (shape <= 0.0) {
        return Inside(risen, 0.0);
    }

    // **The noise rises with the gas.** Height over lifetime is the speed the file gives it, and
    // subtracting the distance travelled from the sampled position leaves a field that is static in
    // the plume's own frame — so every eddy climbs at exactly that speed for one subtraction, and
    // nothing is kept between frames.
    vec3 drift = emitter.axis * (height * time / emitter.lifetime);
    float roil = plume_fbm((world - drift) * (PLUME_EDDIES / height));
    // **Stretched, because value noise does not use its own range.** Three octaves of it pile up
    // around the middle and almost never reach either end, so the erosion below would have almost
    // nothing to bite on. This is what puts the contrast back.
    roil = clamp((roil - PLUME_FLOOR) / (PLUME_CEILING - PLUME_FLOOR), 0.0, 1.0);

    // **Eroded rather than dimmed, which is the difference between a flame and a triangle.**
    // Multiplying the shape by the noise leaves the shape's own outline standing and only varies
    // what is inside it, so a tapered plume reads as a smooth cone however much detail is painted
    // within. Subtracting instead makes the noise decide where the plume *ends*: a place the noise
    // is thin needs the shape to be thick to survive at all, so the edge comes apart into tongues.
    //
    // Subtracted and not renormalised, which the first attempt did: dividing by the noise again
    // pushes everything the erosion did not remove to full density, and the flame came back as a
    // ragged *column* with no grade left inside it.
    return Inside(risen, max(shape * mix(PLUME_GRAIN, 1.0, roil)
                                 - PLUME_EROSION * (1.0 - roil),
                             0.0));
}

// What is left of the sun by the time it reaches `world` through the plume itself.
float plume_shadow(Emitter emitter, vec3 world, float time, float height) {
    vec3 toward = -frame.sun_direction;
    float step = height / float(PLUME_LIGHT_STEPS);
    float depth = 0.0;
    for (uint i = 0u; i < PLUME_LIGHT_STEPS; ++i) {
        depth += plume_at(emitter, world + toward * (step * (float(i) + 0.5)), time, false).density
               * step;
    }
    return exp(-SMOKE_DEPTH * depth / height);
}

// Henyey-Greenstein, with the `1/4pi` left to the caller's own normalisation.
float plume_phase(float cosine) {
    float g = SMOKE_ANISOTROPY;
    float denominator = 1.0 + g * g - 2.0 * g * cosine;
    return (1.0 - g * g) / max(denominator * sqrt(denominator), 1e-4);
}

// Everything one emitter puts along the ray, added into `colour` and taken out of `through`.
void emitter_along(Emitter emitter, vec3 origin, vec3 direction, float span, uvec2 pixel,
                   inout vec3 colour, inout float through) {
    // The segment of the ray inside the emitter's bounding sphere, which is the only part of it
    // that can hold anything.
    vec3 to = emitter.origin - origin;
    float along = dot(to, direction);
    float perpendicular = dot(to, to) - along * along;
    float radius = emitter.reach * emitter.reach;
    if (perpendicular > radius) {
        return;
    }
    float half_chord = sqrt(max(radius - perpendicular, 0.0));
    float enter = max(along - half_chord, 0.0);
    float leave = min(along + half_chord, span);
    if (leave <= enter) {
        return;
    }

    float height = emitter.height;
    float step = (leave - enter) / float(PLUME_STEPS);
    // **Dithered per pixel and not per frame.** Marching from the same offset in every pixel draws
    // the bounding sphere's own shells across the plume; offsetting each pixel breaks them up. What
    // it must not do is *move*: this layer is handed to the upscaler to composite rather than to
    // denoise, so anything reseeded every frame stays in the picture as crawling static instead of
    // being averaged away. A hash of the pixel alone is a fixed pattern, which the upscaler's own
    // sub-pixel jitter then walks across and resolves — the same argument `hash` itself makes.
    float jitter = unit_pair(hash(uvec4(pixel, emitter.seed, 0x5EEDu))).x;
    bool burning = emitter.additive > 0.0;
    float phase = burning ? 0.0 : plume_phase(dot(direction, -frame.sun_direction));
    // **Whether the sun reaches this plume at all, asked once for the whole of it.** A puff is lit
    // by the sun as much as by the sky (§8.102), and taking that unshadowed left Vivec's drains
    // blazing white in an alcove under a hundred feet of stone. One ray from the middle of the
    // plume rather than one per step: these are a few metres across and the sun either finds them
    // or it does not, so a per-step answer would cost thirty times as much to say the same thing.
    float lit = 0.0;
    if (!burning) {
        vec3 middle = emitter.origin + emitter.axis * (0.5 * height);
        lit = occluded(middle, -frame.sun_direction, RAY_MAX) ? 0.0 : 1.0;
    }

    for (uint i = 0u; i < PLUME_STEPS; ++i) {
        if (through < 0.01) {
            break;
        }
        vec3 at = origin + direction * (enter + step * (float(i) + jitter));
        Inside inside = plume_at(emitter, at, frame.time, burning);
        if (inside.density <= 0.0) {
            continue;
        }
        vec4 over = emitter.colour * particle_tint(emitter, inside.risen);
        // **The same integrator for both, and only the source term differs.** What a step adds is
        // whatever it holds times the fraction of the ray that step stops, and what it takes away
        // is the rest — which is the emission-absorption equation written once. Sampling the source
        // at a point and multiplying by the step length instead is what makes a march brighten as
        // it gets finer; this is the same answer at any step count.
        float depth = burning ? FLAME_DEPTH : SMOKE_DEPTH;
        float extinction = inside.density * over.a * depth / height;
        float leaving = exp(-extinction * step);

        vec3 source;
        if (burning) {
            // A flame's ramp is a blackbody's: the colour is the temperature of the gas, and the
            // fall toward the top is the fourth power of it.
            source = over.rgb * FLAME_RADIANCE;
        } else {
            // **And smoke is lit rather than glowing.** What it sends to the eye is the sun through
            // whatever of itself stands in the way, thrown forward by the phase function, plus the
            // sky from every direction at once.
            float sunlit = plume_shadow(emitter, at, frame.time, height);
            float powder = 1.0 - exp(-SMOKE_POWDER * inside.density);
            source = over.rgb
                   * (frame.sun_colour * (INV_PI * phase * sunlit * powder * lit) + frame.ambient);
        }
        colour += source * (through * (1.0 - leaving));
        through *= leaving;
    }
}

// Everything every emitter puts along the ray: premultiplied colour, and what is left of the ray.
//
// Matches `precipitation_along`, which the caller sums with — see the note there.
vec4 particles_along(vec3 origin, vec3 direction, float span, uvec2 pixel) {
    vec3 colour = vec3(0.0);
    float through = 1.0;
    uint count = min(frame.emitter_count, EMITTER_LIMIT);
    for (uint index = 0u; index < count; ++index) {
        emitter_along(emitters[index], origin, direction, span, pixel, colour, through);
    }
    return vec4(colour, through);
}
