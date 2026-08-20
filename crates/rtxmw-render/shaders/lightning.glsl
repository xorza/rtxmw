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
//
// **Kept low because the bay is a mirror.** Water reflects a lit dome one for one where a diffuse
// surface returns its albedo's share of it, so every unit added here arrives on the sea at several
// times what it is worth on the shore — a storm whose sky glowed came out as a bright sea with a
// dark landscape beside it, whatever the land was doing. What lights a landscape is `FLASH_LIT`,
// which the water sees on the same terms as everything else.
const float FLASH_SEEN = 1.8;

// How tightly the glow gathers around where the discharge is.
//
// One is the whole hemisphere and large numbers are a spot. Four is a broad quarter of the sky: a
// storm cell is kilometres across and the deck spreads the light over all of it, so what an eye sees
// is a region brightening rather than a point.
const float FLASH_FOCUS = 4.0;

// How far off a flash's brightness is quoted, in world units.
//
// The middle of `REACH`, so a strike at the near edge of the range throws about twice what one at
// the far edge does — which is the inverse square doing its own work over the band the schedule
// actually draws from.
const float FLASH_REFERENCE = 25000.0;

// What a flash at that distance throws on a surface facing it, **before the air takes its share**.
//
// A real channel is ten orders of magnitude above anything else in this world, and what reaches a
// landscape a few hundred metres off is still comparable to daylight — which is why a photograph
// taken by one is exposed like a photograph taken by the sun. This is that, blunted to where the
// haze cap above lands it somewhere a tone curve can hold.
//
// **Not folded into the ambient, which is where this started and why it looked wrong.** An ambient
// arrives from everywhere at once, so a flash written that way lit every face of every object
// equally and cast nothing — and against water, which reflects the lit sky at nearly full strength
// rather than through an albedo, the land came out dim beside a bay that had gone white. A discharge
// is a *place*; lighting from it is what puts the shadow under the eaves and the highlight on one
// side of a mast.
const float FLASH_LIT = 2600.0;

// How far a flash's shadow ray reaches, in world units.
//
// **Not to the channel, which is half a kilometre off across a whole cell of geometry.** A ray that
// long is most of a frame's traversal spent asking whether a headland stands between a deck and a
// storm — and what actually reads is the near end of it: the shadow a mast throws, the dark side of
// a wall, the porch the light does not get under. Two hundred metres catches all of that and costs a
// fraction of it; a hillside further out than this is one the flash lights through.
const float FLASH_SHADOW = 14000.0;

// The most the air may take off a flash, in nepers.
//
// See `flash_reaching`: the fog's own optical depth over the distance a strike stands off runs to
// dozens, and a light behind that is not attenuated, it is deleted. A channel outshines the haze in
// front of it by ten orders of magnitude and is dimmed by it, never hidden.
const float FLASH_HAZE = 2.0;

// What the channel delivers to `position` before anything there is asked which way it faces.
//
// Shared by the surfaces `flash_light` shades and the air `fog_light` scatters, because it is the
// same light arriving at the same place — and a flash that lit the ground but not the air between
// would be the one thing a storm never is.
vec3 flash_reaching(vec3 position) {
    if (frame.flash_radiance == vec3(0.0)) {
        return vec3(0.0);
    }
    vec3 at = mix(frame.flash_source, frame.flash_ground, 0.5);
    float away = distance(at, position);
    float fall = FLASH_REFERENCE / max(away, FLASH_REFERENCE * 0.2);
    // **Through the same air that hides the channel, and capped, because at this range that air is a
    // cliff rather than a gradient.** The bolt is drawn behind the fog's own transmittance and this
    // was not, so a storm thick enough to swallow the arc went on lighting the shore as though it
    // were clear. Putting the whole optical depth back is the other error: a thunderstorm's dial over
    // the four hundred metres a strike stands off comes to `exp(-81)`, which is not dim, it is
    // nothing — raising the constant two hundred and seventy fold moved not one pixel.
    //
    // What the cap says is that a channel is ten orders of magnitude above the haze in front of it,
    // so the air can take a stop or two off it and never take it away. Two nepers is a seventh, which
    // is the difference between a near strike and a far one without either being switched off.
    float haze = exp(-min(frame.fog_density * FOG_EXTINCTION * away, FLASH_HAZE));
    return frame.flash_radiance * (FLASH_LIT * fall * fall * haze);
}

// What the channel throws on `surface`, shadowed only where `samples` says this is a primary hit.
vec3 flash_light(Surface surface, uvec2 pixel, uint samples) {
    if (frame.flash_radiance == vec3(0.0)) {
        return vec3(0.0);
    }
    // **A point drawn along the channel, not the middle of it.** A discharge is kilometres of arc
    // rather than a lamp: it is the largest area light this world ever has, and an area light that
    // big throws almost no penumbra edge at all — everything it lights is half lit by some other
    // part of the channel. Testing one fixed point gave it a lamp's shadow, hard-edged and wrong for
    // a source a thousand times longer than the mast throwing it.
    //
    // One sample a frame from the frame's own stream, which is how the sun's disc is resolved here
    // too: Ray Reconstruction accumulates the softness that sixteen rays would otherwise have to buy.
    float along = unit_pair(hash(uvec4(sample_stream(pixel), STREAM_FLASH, 0u))).x;
    vec3 at = mix(frame.flash_source, frame.flash_ground, along);
    vec3 to = at - surface.position;
    float away = length(to);
    vec3 towards = to / max(away, 1e-3);
    float facing = lambert(surface, towards);
    if (facing <= 0.0) {
        return vec3(0.0);
    }
    // **One ray, and only where the eye is looking.** A bounce's share of a flash is a fill term
    // under an albedo under a cosine; buying it a shadow ray as long as this one costs more than
    // every other light in the frame put together and lands under a tenth of what the primary hit
    // already carries.
    if (samples >= SHADOW_SAMPLES
        && occluded(leaving(surface, towards), towards, min(away, FLASH_SHADOW))) {
        return vec3(0.0);
    }
    // And through whatever water stands over it, exactly as the sun is: a seabed under three metres
    // of bay is not lit as though the bay were not there, which is what this did before and what
    // made a shallow read brighter under a flash than the shore beside it.
    return flash_reaching(surface.position) * (facing * daylight_reaching(surface.position));
}

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
const float BOLT_CORE = 2.4;
const float BOLT_HALO = 45.0;

// How far past its own radius the bounding test reaches before it rejects a ray.
//
// **A hair over one, and the reason it may be one at all is the whole of the fix below.** The bound
// exists to skip the march for rays nowhere near the channel, and skipping is only sound where what
// is skipped is *nothing*. `bolt_falloff` reaches exactly zero at its radius, so a ray beyond it
// contributes exactly zero and rejecting it changes no pixel — the cut and the profile are the same
// number and cannot disagree.
//
// The margin is for the radius itself: it scales with the depth at the closest approach to the
// straight channel, and the crooked one wanders, so a segment can sit a little further out than the
// test measured. A quarter covers the wander that `BOLT_WANDER` already bounds.
const float BOLT_BOUND = 1.25;

// How much of an in-cloud channel survives the cloud it is buried in.
//
// **The deck is between it and the eye, and that is the whole difference between the two shapes that
// have a channel at all.** A discharge inside a cloud is lighting water from within: what leaves is
// scattered on the way out, so the arc reads as a bright seam in the deck rather than as the hard
// line a strike below the base draws. Nothing at all would be a sheet flash, which is the third
// shape and much the most common.
const float BOLT_VEILED = 0.12;

// How much brighter the channel is to look at than the light it throws on a surface a reference
// distance away.
//
// **Tied to `FLASH_LIT` rather than set beside it, because they are one discharge.** Held as
// independent numbers they drifted apart twice in opposite directions: first a channel bright enough
// to see over a landscape that stayed dark, then a landscape lit like a photograph with a faint
// smudge in the sky above it. Neither is a tuning error — a number that can disagree with another
// number eventually will. Move the flash and the bolt moves with it.
const float BOLT_SEEN = 14.0;

// How much brighter than the flash's own radiance the channel is.
//
// The return stroke is over ninety-nine percent of a flash's light, and it is concentrated into a
// line rather than spread over the sky the cloud scatters it across. This is that ratio, blunted:
// the true one is orders of magnitude and would leave nothing on the screen but a white line.
//
// **Blunted, but not so far that a storm can hide it.** The channel is drawn behind the fog like
// anything else at that distance, and a thunderstorm's air is thick — so at a quarter of this it was
// swallowed whole while the light it threw still lit the shore. A real channel is ten orders of
// magnitude above the haze in front of it and loses the argument with nothing.
const float BOLT_ARC = FLASH_LIT * BOLT_SEEN;

// How many forks a channel throws, and how far down each runs.
const uint BOLT_FORKS = 3u;
const float BOLT_FORK_RUN = 0.35;

// How steeply the glare falls away from the channel, as the denominator's weight at the radius.
//
// Twenty-five is a glow that has lost nine tenths of itself a fifth of the way out and is a faint
// wash for the rest — which is what a photograph of a bolt shows around it, rather than the even
// slab a flat profile draws.
const float BOLT_GLARE = 25.0;

// How much of the glow survives at `near` from a channel of radius `radius`.
//
// **Zero at the radius, and that is not a nicety — it is what makes the bound legal.** This was
// `1 / (1 + x^2)`, which is the right shape and has infinite support: it never reaches zero, so the
// cheap bounding test that skips distant rays was always discarding something. However far out the
// bound was pushed, the discarded value was only *small*, and the capsule the test describes stood
// in the sky as a hard-edged pill around the bolt. Pushing it further only made the step fainter —
// and every time the flash got brighter the step came back, because a fixed cut through a curve that
// never lands is a step whose visibility is a matter of exposure.
//
// **Glare's own shape, windowed to land.** The compact kernel `(1 - x^2)^2` fixed the support and
// got the profile wrong in the other direction: it is *flat* at the centre and falls at the edge,
// which is backwards for a glow. Scaled up until the channel read, it clipped to white across its
// whole radius and dropped in the last pixel — a hard tube rather than a bolt with air around it.
//
// What a bright line actually leaves on a lens or a retina falls as one over the square of the angle
// from it: steep near the core, and a long faint tail. That has infinite support, which is what
// started all this — so it is multiplied by the window, which is zero at the radius and zero-sloped
// there. Steep where glare is steep, gone where the bound cuts.
float bolt_falloff(float near, float radius) {
    float x = (near * near) / (radius * radius);
    float window = max(1.0 - x, 0.0);
    return window * window / (1.0 + BOLT_GLARE * x);
}

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
    if (straight > BOLT_WANDER * run * 2.0 + depth * frame.cone_spread * BOLT_HALO * BOLT_BOUND) {
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
        glow = max(glow, bolt_falloff(near, core));
        glow = max(glow, 0.0016 * bolt_falloff(near, halo));
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
            glow = max(glow, 0.35 * bolt_falloff(near, core));
            glow = max(glow, 0.0006 * bolt_falloff(near, halo));
            held = next;
        }
    }
    // A channel still inside the deck is seen through it — `flash_kind` is 1 for that one.
    float veiled = frame.flash_kind < 1.5 ? BOLT_VEILED : 1.0;
    // **Through the same capped air the light crosses**, which is the whole of what kept these two
    // from agreeing. The channel was drawn behind the fog's full marched transmittance while the
    // light it throws was capped at `FLASH_HAZE` — so a storm dense enough to delete the arc lit the
    // shore regardless, which is the complaint that started this in reverse. One expression now, and
    // the same one.
    float haze = exp(-min(frame.fog_density * FOG_EXTINCTION * depth, FLASH_HAZE));
    return frame.flash_radiance * (glow * BOLT_ARC * veiled * haze);
}
