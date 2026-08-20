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
//
// **Lowered again when the deck learned to light up.** This is a flat wash over a quarter of the sky
// and `CLOUD_FLASH` is the cloud the discharge is actually in; with both of them the sky near a
// strike is brighter than it was, and the part that grew is the part with shape in it.
const float FLASH_SEEN = 1.2;

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

// How near the channel the air may be asked about, in world units.
//
// **The arc has no thickness in this model and the integral says so**: the light from a line source
// goes as `1 / r` at the line, which is a real divergence rather than a numerical one, and the fog
// march walks straight through it. A floor is the whole of what stands in for a channel being a
// thing rather than a curve.
//
// **Thirty-six metres, which is a channel's luminous envelope and not much more.** It stood at
// seventy-one, and on a cloud deck a few thousand units overhead that is wider than the piece of sky
// being looked at — so the deck lit by a discharge inside it came out as a flat wash with no shape
// in it rather than a region of weather going bright. Tightening it is the more physical figure as
// well as the better-looking one.
//
// **This is what drew the bulb, when it floored a distance to the wrong place.** The discharge used
// to be a single point at the middle of the channel, and this floor made a ball of maximum
// in-scattering five thousand units across around that point — so an anvil crawler, whose arc runs a
// hundred thousand units through the deck, hung a glowing sphere in mid-air at its halfway mark with
// no bolt anywhere in it. Measured to the arc instead, the same floor is a soft tube along the
// channel, which is what a bolt does to the weather around it.
const float FLASH_NEAREST = 2500.0;

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

// How much air a flash's light crosses over `span`, in nepers.
//
// **The strength dial belongs here as well as in the march.** `fog_strength` fades the whole effect
// so that zero is the frame untouched however thick the cell says its fog is — and read without it,
// a scene asked for no fog at all still charged the cell's own density against every flash. Which
// showed up as the corona sitting at its storm height in clear air, the two conditions it exists to
// tell apart collapsed into one.
float fog_optical_depth(float span) {
    return frame.fog_density * frame.fog_strength * FOG_EXTINCTION * span;
}

// The share of a flash that survives `span` of the weather, capped — see `FLASH_HAZE`.
float flash_through_air(float span) {
    return exp(-min(fog_optical_depth(span), FLASH_HAZE));
}

// What the channel delivers to `position` before anything there is asked which way it faces.
//
// Shared by the surfaces `flash_light` shades and the air `fog_light` scatters, because it is the
// same light arriving at the same place — and a flash that lit the ground but not the air between
// would be the one thing a storm never is.
vec3 flash_irradiance(vec3 position, out float away) {
    away = 0.0;
    if (frame.flash_radiance == vec3(0.0)) {
        return vec3(0.0);
    }
    vec3 channel = frame.flash_ground - frame.flash_source;
    // A sheet has no channel at all — its source and its ground are one point — and the integral
    // below *is* the inverse square once the length goes to nothing, so it needs no branch of its
    // own. The floor only keeps the division alive.
    float run = max(length(channel), 1.0);
    vec3 heading = channel / run;
    vec3 from = position - frame.flash_source;
    // Where the perpendicular from `position` meets the channel's line, and how far out that is.
    float foot = dot(from, heading);
    float across = max(length(from - heading * foot), FLASH_NEAREST);
    // The inverse square summed along the arc, in closed form: an element `ds` at `s` from the foot
    // arrives over `across^2 + s^2`, and `integral ds / (r^2 + s^2)` is `atan(s / r) / r`. Divided by
    // the length so the whole channel carries one discharge's worth however long it is, which is what
    // makes this agree with the point source it replaces: far enough out that both endpoints subtend
    // almost nothing, the arctangents collapse to `run / across` and the whole thing to `1 / d^2`.
    float shaped = (atan((run - foot) / across) + atan(foot / across)) / (run * across);
    // How much air stands between, measured to the nearest point of the arc rather than to a place
    // on it chosen in advance — the same argument as above, applied to the extinction.
    away = length(from - heading * clamp(foot, 0.0, run));
    return frame.flash_radiance * (FLASH_LIT * FLASH_REFERENCE * FLASH_REFERENCE * shaped);
}

// The same, behind the air between — which is what anything standing in the weather is lit through.
//
// **Separate from the geometry above because not every caller owes the air.** The cloud deck is
// composited into the sky and then carried to the eye by the fog march like everything else in it,
// so charging it here as well would take the same haze out of it twice.
vec3 flash_reaching(vec3 position) {
    float away;
    vec3 arriving = flash_irradiance(position, away);
    // **Capped, because at this range the air is a cliff rather than a gradient.** The bolt is drawn
    // behind the fog's own transmittance and this was not, so a storm thick enough to swallow the arc
    // went on lighting the shore as though it were clear. Putting the whole optical depth back is the
    // other error: a thunderstorm's dial over the four hundred metres a strike stands off comes to
    // `exp(-81)`, which is not dim, it is nothing — raising the constant two hundred and seventy fold
    // moved not one pixel.
    //
    // What the cap says is that a channel is ten orders of magnitude above the haze in front of it,
    // so the air can take a stop or two off it and never take it away. Two nepers is a seventh, which
    // is the difference between a near strike and a far one without either being switched off.
    return arriving * flash_through_air(away);
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
const int BOLT_OCTAVES = 6;
const float BOLT_WANDER = 0.08;

// How narrow the channel is drawn at its far end, against its width where the discharge begins.
//
// **A channel splinters as it goes.** What leaves the cloud is one broad path and what arrives is the
// last and finest of the threads it broke into on the way — so the width tapers along it, and the
// forks that leave it are finer again for the same reason. Drawn at one width end to end it read as
// a tube laid over the sky rather than as something that had travelled.
const float BOLT_TAPER = 0.4;

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

// And the corona: the broad wash a very bright thing has in air that is full of water.
//
// **Three tiers because one curve cannot be both.** A channel is a hard line a couple of pixels
// across, and the glow around it runs out to a quarter of the frame — no single falloff spans that
// without being a slab at one end or invisible at the other, which is the whole history of this
// file. What the tiers approximate is the aerosol forward-scattering lobe around a bright source: a
// narrow Mie peak on the source itself, and a wide shallow skirt that a storm's own droplets make
// far broader than a clear night would.
const float BOLT_CORONA = 110.0;

// The widest tier, which is the one the bounding test has to clear.
//
// **Named rather than repeated**, because the bound and the profile being the same number is the
// invariant that keeps a ring from appearing at the cut — see `BOLT_BOUND`. Adding a tier wider than
// the bound would put the ring straight back.
const float BOLT_REACH = BOLT_CORONA;

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
// **The deck is between it and the eye, and that is what separates the shapes that stay in it from
// the one that leaves.** A discharge inside a cloud is lighting water from within: what leaves is
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

// How many forks a channel throws, how far down each runs, and where along it they leave.
//
// **Low down, because that is where a discharge is breaking up.** A leader forks harder the nearer it
// gets to what it is reaching for — the ground pulls threads out of it — so a channel that branched
// evenly along its length looked like a diagram of a tree rather than a bolt.
const uint BOLT_FORKS = 5u;
const float BOLT_FORK_RUN = 0.35;
const float BOLT_FORK_FROM = 0.3;

// How many times the white point the glow peaks at, for the channel and for a fork.
//
// **Written in display terms, because a glow is only a glow while it is on the curve.** These were
// ratios to `BOLT_ARC`, which is in the tens of thousands — so whether a given ratio came out above
// white or below it was invisible at the point of writing, and the one in use peaked at *fifty-eight
// times* white. Everything above one clips, so what was drawn was not a falloff at all: it was a
// solid slab out to seven tenths of the radius with a thin rim where the curve finally dropped
// through. On a crawler passing overhead, whose segments are long on screen, that slab is a fat
// white capsule with a rounded cap — which is the third time saturation has been mistaken for a
// shape in this file.
//
// Under `BOLT_HALO_GLARE` these peak just above white at the very centre and fall through it
// within a third of the radius, leaving the rest of the profile as an actual gradient. A fork
// stays under white throughout, so it is all gradient.
const float BOLT_GLOW_PEAK = 1.8;
const float BOLT_FORK_PEAK = 0.9;

// The corona's own, in two parts: what it is in still clear air, and what the weather adds to it.
//
// **Because the corona is the air, how much of it there is follows how much air there is.** A fixed
// figure cannot serve both ends of the range and the attempt showed it plainly: at the value that
// reads against a night sky it vanished inside a thunderstorm, and at the value that reads inside a
// thunderstorm it flattened a clear night into a pale wash. The two are not one number badly chosen.
// They are two conditions, and the medium is what tells them apart.
//
// Clear air still gets some — a bright source has a skirt through any atmosphere and through the
// optics looking at it — and a storm gets that plus what its own water throws back, which comes out
// near eight times as much.
const float BOLT_CORONA_PEAK = 0.5;
const float BOLT_CORONA_AIR = 2.5;

// How steeply the glare falls away from the channel, as the denominator's weight at the radius.
//
// Twenty-five is a glow that has lost nine tenths of itself a fifth of the way out and is a faint
// wash for the rest — which is what a photograph of a bolt shows around it, rather than the even
// slab a flat profile draws.
// The core is drawn far above white on purpose — a return stroke is a line the eye cannot look at
// — so its glare barely shows: it governs only the last percent of a radius that is already solid.
// The halo is the opposite. It is the part that has to *read* as light, and light falls off gently:
// at the core's twenty-five it was a spike with a rim, which is a shape rather than a glow. Six
// leaves a smooth gradient running the whole way out to `BOLT_HALO`.
const float BOLT_CORE_GLARE = 25.0;
const float BOLT_HALO_GLARE = 6.0;
// The corona is the gentlest of the three: a wash has no shoulder at all, and two makes it almost a
// plain window over the whole of its reach.
const float BOLT_CORONA_GLARE = 2.0;

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
float bolt_falloff(float near, float radius, float glare) {
    float x = (near * near) / (radius * radius);
    float window = max(1.0 - x, 0.0);
    return window * window / (1.0 + glare * x);
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
    float frequency = 4.0;
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
    if (straight > BOLT_WANDER * run * 2.0 + depth * frame.cone_spread * BOLT_REACH * BOLT_BOUND) {
        return vec3(0.0);
    }
    // **How far off the discharge is along this ray, and the only depth defined everywhere the
    // corona is drawn.** The tiers below each report the depth of the segment that won them, which is
    // the right answer for dimming that segment and no answer at all outside its reach — and the
    // corona reaches several times further than the halo does. Scaled by that instead, the wash lost
    // its whole weather term the moment the halo's support ended and stepped by five times over three
    // pixels, which is a hard edge drawn around the middle of a glow. This one comes off the straight
    // channel and is the same number for every ray that gets past the bound.
    float reach = depth;

    uint seed = frame.flash_seed;
    float glow = 0.0;
    // The corona is kept apart from the other two all the way to the end, because it is the only one
    // the air does not stand in front of — see below.
    float wash = 0.0;
    // **How far off the glow that won actually is.** `bolt_near` reports a depth per segment through
    // an out parameter, so reading it after the loops gives whatever the last fork happened to write
    // rather than the depth of the thing being drawn — and the last fork is the one furthest down a
    // branch nobody may be looking at. Carried with the maximum instead, so the air charged below is
    // the air in front of the part of the channel this ray can see.
    float lit = 0.0;
    vec3 last = bolt_at(frame.flash_source, frame.flash_ground, 0.0, seed);
    for (uint step = 1u; step <= BOLT_STEPS; ++step) {
        float along = float(step) / float(BOLT_STEPS);
        vec3 next = bolt_at(frame.flash_source, frame.flash_ground, along, seed);
        float near = bolt_near(origin, direction, last, next, span, depth);
        // A core that saturates and a halo that falls off around it, which is what a line source
        // photographs as: the channel itself is smaller than a pixel and what is seen is its air.
        // Thinner the further it has come — see `BOLT_TAPER`.
        float drawn = depth * frame.cone_spread * mix(1.0, BOLT_TAPER, along);
        float core = drawn * BOLT_CORE;
        float halo = drawn * BOLT_HALO;
        float here = max(bolt_falloff(near, core, BOLT_CORE_GLARE),
                         (BOLT_GLOW_PEAK / BOLT_ARC) * bolt_falloff(near, halo, BOLT_HALO_GLARE));
        if (here > glow) {
            glow = here;
            lit = depth;
        }
        // The shape only — the corona's height is settled once at the end, out of the weather.
        wash = max(wash, bolt_falloff(near, drawn * BOLT_CORONA, BOLT_CORONA_GLARE));
        last = next;
    }

    // **Forks, because a channel that does not branch reads as a crack in the screen.** Each leaves
    // the main channel at its own height and runs a third of the way on its own heading, with the
    // same wander at a fraction of the brightness — a fork is a leader that never became a return
    // stroke, and Garg's twenty-to-a-hundred-fold ratio is the measured difference.
    for (uint fork = 0u; fork < BOLT_FORKS; ++fork) {
        uint salt = seed + 64u + fork * 8u;
        float from = BOLT_FORK_FROM
                   + (1.0 - BOLT_FORK_FROM) * float(hash(uvec4(salt, 0u, 0u, 0u)) & 0xFFFFu)
                   / 65535.0;
        vec3 root = bolt_at(frame.flash_source, frame.flash_ground, from, seed);
        vec2 heading = unit_pair(hash(uvec4(salt + 1u, 0u, 0u, 0u))) * 2.0 - 1.0;
        vec3 tip = root + vec3(heading * run * BOLT_FORK_RUN,
                               -run * BOLT_FORK_RUN) * (1.0 - from);
        vec3 held = root;
        for (uint step = 1u; step <= BOLT_STEPS / 4u; ++step) {
            vec3 next = bolt_at(root, tip, float(step) / float(BOLT_STEPS / 4u), salt);
            float near = bolt_near(origin, direction, held, next, span, depth);
            // Finer than the channel it left, and finer still toward its own tip.
            float drawn = depth * frame.cone_spread * BOLT_TAPER
                        * mix(1.0, BOLT_TAPER, float(step) / float(BOLT_STEPS / 4u));
            float core = drawn * BOLT_CORE;
            float halo = drawn * BOLT_HALO;
            float here = max(0.35 * bolt_falloff(near, core, BOLT_CORE_GLARE),
                             (BOLT_FORK_PEAK / BOLT_ARC)
                                 * bolt_falloff(near, halo, BOLT_HALO_GLARE));
            if (here > glow) {
                glow = here;
                lit = depth;
            }
            held = next;
        }
    }
    // A channel still inside the deck is seen through it, which is every shape but the one that
    // reaches the ground — see the ordering in `FrameConstants`.
    float veiled = frame.flash_kind < 2.5 ? BOLT_VEILED : 1.0;
    // **Through the same capped air the light crosses**, which is the whole of what kept these two
    // from agreeing. The channel was drawn behind the fog's full marched transmittance while the
    // light it throws was capped at `FLASH_HAZE` — so a storm dense enough to delete the arc lit the
    // shore regardless, which is the complaint that started this in reverse. One expression now, and
    // the same one.
    float haze = flash_through_air(lit);
    // How much of the path to the channel scattered rather than reaching the eye, which is the air
    // that makes the corona — so the wash grows with exactly what dims the line inside it. Off
    // `reach` rather than `lit`, for the reason given where `reach` is taken.
    float scattered = 1.0 - flash_through_air(reach);
    // **But not the corona, because the corona is the air.** The narrow tiers are a distant object
    // seen through weather and are dimmed by it; the wash around them is that same weather scattering
    // the channel's light back toward the eye, so attenuating it charges the haze for hiding a thing
    // the haze is making. Drawn that way it was deleted exactly when it should have been at its
    // strongest: in clear air the halo was plain and in a thunderstorm — the only weather that has
    // lightning at all — the cap took it to a fifteenth and left a hard white line in a flat grey
    // sky. Which is the same mistake as `FLASH_HAZE`'s, one level down: an effect the medium causes
    // cannot also be occluded by it.
    return frame.flash_radiance * veiled
         * (glow * BOLT_ARC * haze
            + wash * (BOLT_CORONA_PEAK + BOLT_CORONA_AIR * scattered));
}
