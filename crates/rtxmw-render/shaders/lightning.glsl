// What a flash does to the sky it happens in.
//
// **The light is not here.** A flash reaches the frame constants already folded into the ambient —
// see `FLASH_AMBIENT` — so the rain in the air, the film on the ground, the fog and the indirect
// bounce are all lit by it without one of them knowing lightning exists. What is left for this file
// is the flash being *seen*: the deck it happens in going bright, and the channel where one shows.

// How much of the flash the cloud deck scatters back out toward the eye.
//
// **A discharge inside a cloud is a lamp inside a paper lantern.** Almost none of the channel's light
// leaves in a straight line; it is scattered by the water it is buried in and comes back out over an
// enormous area, which is why a sheet flash lights a whole quarter of the sky rather than drawing a
// bright spot in it. Well over half of all discharges never leave the cloud at all.
const float FLASH_SEEN = 3.0;

// How tightly the glow gathers around where the discharge is.
//
// One is the whole hemisphere and large numbers are a spot. Four is a broad quarter of the sky: a
// storm cell is kilometres across and the deck spreads the light over all of it, so what an eye sees
// is a region brightening rather than a point.
const float FLASH_FOCUS = 4.0;

// What the flash adds to the sky in `direction`.
vec3 flash_sky(vec3 direction) {
    if (frame.flash_radiance == vec3(0.0)) {
        return vec3(0.0);
    }
    // Toward the discharge rather than at it: what is drawn is the cloud around the channel, and a
    // cloud is far too big and far too near for a parallax-free direction to be wrong about it.
    vec3 toward = normalize(frame.flash_source - frame.camera_position);
    float facing = max(dot(direction, toward), 0.0);
    return frame.flash_radiance * (FLASH_SEEN * pow(facing, FLASH_FOCUS));
}

// How many segments the channel is walked in.
//
// A bolt is self-similar, so the count decides the finest kink that shows rather than how accurate
// the shape is. Forty-eight over a kilometre of channel is a kink every twenty metres, which is
// where a real one stops being legible against the sky anyway.
const uint BOLT_STEPS = 48u;

// How many octaves of wander the channel carries, and how wide the first one is.
//
// **Midpoint displacement, summed rather than recursed.** The classic construction halves the
// displacement at each subdivision, which is the same thing as a sum of octaves doubling in
// frequency and halving in amplitude — and written that way it can be evaluated at any point of the
// channel without building the channel first, which is what lets a shader draw one with no buffer
// and no geometry. Lightning is the textbook case of it: the tortuosity is scale-free, so a bolt
// looks like a bolt at any distance.
const int BOLT_OCTAVES = 5;
const float BOLT_WANDER = 0.08;

// How thick the channel is drawn, and how far its halo reaches past that — both in *pixels*.
//
// **A real channel is centimetres across and drawing it that way puts nothing on the screen.** Five
// centimetres at half a kilometre is a ten-thousandth of a pixel; written in world units the core
// came out at eight thousandths of one and the whole bolt was a single lit texel. A photograph shows
// a line anyway, and not because the channel is wide: it is so far past saturation that the lens and
// the eye spread it themselves, and what is being looked at is that spread rather than the arc.
//
// So the width is angular and the distance sets the rest, which is the same argument
// `precipitation.glsl` makes about a drop — a bright thing smaller than the optics looking at it has
// the size of the optics, not its own.
const float BOLT_CORE = 1.1;
const float BOLT_HALO = 16.0;

// How much of an in-cloud channel survives the cloud it is buried in.
//
// **The deck is between it and the eye, and that is the whole difference between the two shapes that
// have a channel at all.** A discharge inside a cloud is lighting water from within: what leaves is
// scattered on the way out, so the arc reads as a bright seam in the deck rather than as the hard
// line a strike below the base draws. Nothing at all would be a sheet flash, which is the third
// shape and much the most common.
const float BOLT_VEILED = 0.12;

// How much brighter than the flash's own radiance the channel is.
//
// The return stroke is over ninety-nine percent of a flash's light, and it is concentrated into a
// line rather than spread over the sky the cloud scatters it across. This is that ratio, blunted:
// the true one is orders of magnitude and would leave nothing on the screen but a white line.
const float BOLT_ARC = 260.0;

// How many forks a channel throws, and how far down each runs.
const uint BOLT_FORKS = 3u;
const float BOLT_FORK_RUN = 0.35;

// A smooth wander along the channel, from one hashed value per unit of `along`.
float bolt_noise(float along, uint salt) {
    float cell = floor(along);
    float f = along - cell;
    f = f * f * (3.0 - 2.0 * f);
    uint at = uint(int(cell) + 4096);
    float here = float(hash(uvec4(at, salt, 0u, 0u)) & 0xFFFFu) / 65535.0 - 0.5;
    float next = float(hash(uvec4(at + 1u, salt, 0u, 0u)) & 0xFFFFu) / 65535.0 - 0.5;
    return mix(here, next, f);
}

// Where the channel between `from` and `to` has wandered to at `along`, from nought to one.
vec3 bolt_at(vec3 from, vec3 to, float along, uint salt) {
    float run = distance(from, to);
    vec2 wander = vec2(0.0);
    float amplitude = BOLT_WANDER * run;
    float frequency = 3.0;
    for (int octave = 0; octave < BOLT_OCTAVES; ++octave) {
        wander += amplitude * vec2(bolt_noise(along * frequency, salt + uint(octave) * 2u),
                                   bolt_noise(along * frequency, salt + uint(octave) * 2u + 1u));
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    // Pinned at both ends, because a channel starts where the discharge is and finishes where it
    // strikes: everything between them is free and neither end is.
    float free = sin(along * 3.1415927);
    return mix(from, to, along) + vec3(wander * free, 0.0);
}

// How near the ray passes the segment `a`..`b`, and how far along itself it does — `1e9` for a
// segment the ray has already gone past or that lies beyond what it hit.
float bolt_near(vec3 origin, vec3 direction, vec3 a, vec3 b, float span, out float depth) {
    vec3 ab = b - a;
    vec3 ao = a - origin;
    float along = dot(ab, direction);
    float spread = dot(ab, ab) - along * along;
    float toward = dot(direction, ao);
    // A segment end-on to the ray has no unique nearest point; either end will do.
    float at = spread > 1e-4 ? clamp((toward * along - dot(ab, ao)) / spread, 0.0, 1.0) : 0.0;
    depth = toward + at * along;
    if (depth <= 0.0 || depth >= span) {
        return 1e9;
    }
    return length(ao + ab * at - direction * depth);
}

// What the channel puts in front of a ray that got as far as `span`.
//
// **Drawn where it is rather than composited over everything**, which is the whole reason it is
// worth tracing at all: a bolt behind a headland is behind the headland, and one falling in front of
// a hillside lights the hillside and stands in front of it.
vec3 bolt_along(vec3 origin, vec3 direction, float span) {
    // A sheet flash has no channel: `flash_ground` is the source and there is nothing between them.
    if (frame.flash_radiance == vec3(0.0)
        || distance(frame.flash_source, frame.flash_ground) < 1.0) {
        return vec3(0.0);
    }
    // **One test against the straight line before forty-eight against the crooked one.** The wander
    // is bounded by twice its first octave, so a ray that misses the straight channel by more than
    // that plus the halo cannot touch the real one.
    float run = distance(frame.flash_source, frame.flash_ground);
    float depth;
    float straight = bolt_near(origin, direction, frame.flash_source, frame.flash_ground,
                               span, depth);
    if (straight > BOLT_WANDER * run * 2.0 + depth * frame.cone_spread * BOLT_HALO) {
        return vec3(0.0);
    }

    uint seed = frame.flash_seed;
    float glow = 0.0;
    vec3 last = bolt_at(frame.flash_source, frame.flash_ground, 0.0, seed);
    for (uint step = 1u; step <= BOLT_STEPS; ++step) {
        vec3 next = bolt_at(frame.flash_source, frame.flash_ground,
                            float(step) / float(BOLT_STEPS), seed);
        float near = bolt_near(origin, direction, last, next, span, depth);
        // A core that saturates and a halo that falls off around it, which is what a line source
        // photographs as: the channel itself is smaller than a pixel and what is seen is its air.
        float core = depth * frame.cone_spread * BOLT_CORE;
        float halo = depth * frame.cone_spread * BOLT_HALO;
        glow = max(glow, 1.0 / (1.0 + (near / core) * (near / core)));
        glow = max(glow, 0.22 / (1.0 + (near / halo) * (near / halo)));
        last = next;
    }

    // **Forks, because a channel that does not branch reads as a crack in the screen.** Each leaves
    // the main channel at its own height and runs a third of the way on its own heading, with the
    // same wander at a fraction of the brightness — a fork is a leader that never became a return
    // stroke, and Garg's twenty-to-a-hundred-fold ratio is the measured difference.
    for (uint fork = 0u; fork < BOLT_FORKS; ++fork) {
        uint salt = seed + 64u + fork * 8u;
        float from = 0.2 + 0.6 * float(hash(uvec4(salt, 0u, 0u, 0u)) & 0xFFFFu) / 65535.0;
        vec3 root = bolt_at(frame.flash_source, frame.flash_ground, from, seed);
        vec2 heading = unit_pair(hash(uvec4(salt + 1u, 0u, 0u, 0u))) * 2.0 - 1.0;
        vec3 tip = root + vec3(heading * run * BOLT_FORK_RUN,
                               -run * BOLT_FORK_RUN) * (1.0 - from);
        vec3 held = root;
        for (uint step = 1u; step <= BOLT_STEPS / 4u; ++step) {
            vec3 next = bolt_at(root, tip, float(step) / float(BOLT_STEPS / 4u), salt);
            float near = bolt_near(origin, direction, held, next, span, depth);
            float core = depth * frame.cone_spread * BOLT_CORE;
            float halo = depth * frame.cone_spread * BOLT_HALO;
            glow = max(glow, 0.35 / (1.0 + (near / core) * (near / core)));
            glow = max(glow, 0.08 / (1.0 + (near / halo) * (near / halo)));
            held = next;
        }
    }
    // A channel still inside the deck is seen through it — `flash_kind` is 1 for that one.
    float veiled = frame.flash_kind < 1.5 ? BOLT_VEILED : 1.0;
    return frame.flash_radiance * (glow * BOLT_ARC * veiled);
}
