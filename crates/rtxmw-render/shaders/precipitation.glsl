// Rain and snow, as a lattice of streaks the ray is tested against rather than geometry.
//
// **Not instances in the top level, and the reason is both cost and looks.** Rebuilding the
// acceleration structure every frame is machinery this engine does not have — `rebuild_top`
// allocates and creates a fresh structure, which is right for a cell arriving and wrong sixty times
// a second — and four hundred and fifty sub-pixel streaks would alias into a crawling mess that
// Ray Reconstruction would then smear. A lattice tested analytically has neither problem: coverage
// comes out as a fraction rather than a hit, so a drop finer than a pixel dims that pixel instead
// of flickering in and out of it.
//
// The lattice falls rather than the drops: subtracting the fall from the sampled position leaves a
// field that is static in its own frame, which costs one add and makes every drop's motion exact.

// How far apart the streaks stand across the fall, in world units.
//
// **A constant of the world, which is the whole point.** This was derived from the ray — the volume
// divided by the number of samples — and that is incoherent on its face: a drop is a thing in the
// air, and how big it is cannot depend on which ray happens to look at it or how far that ray
// travels before hitting something. Because `reach` is `min(surface distance, Rain Diameter)`, the
// lattice changed size per *pixel*: looking down at a deck two metres away gave one lattice and
// looking past it into the bay gave another, so neighbouring pixels disagreed about where the drops
// were. That drew a ring of disconnected blobs around the eye whose edge sat exactly where the
// floor's distance crossed the volume's, and it is why the horizon band survived every other fix —
// the same incoherence, seen along a different contour.
//
// How solid a drawn streak is, from which the lattice that carries it is solved.
//
// **The cell is derived and this is the knob, which is the other way round from how it began.** A
// fixed cell fixes how many streaks a ray meets, and the opacity each needs to reach the physical
// coverage then falls where it falls — which for both rain weathers was above one, so the clamp took
// the difference and they drew the same rain. Every count the game writes, and `RAINFALL` over them,
// decided nothing at all.
//
// Turned around it cannot happen. Coverage is `steps * pi r^2 / cell^2 * duty` with `steps` as
// `walked / cell`, so `alpha = wanted * cell^3 / (walked * pi r^2 * duty)`; `wanted` carries
// `walked` too, and it cancels. What is left solves for the cell in closed form:
//
//     cell = spacing * cbrt(PRECIP_OPACITY * r^2 * duty / (PRECIP_SPREAD * PRECIP_RADIUS^2))
//
// So the cell tracks the weather's own drop spacing — more drops in the air, a finer lattice, more
// streaks across a ray, which is the whole of how `Max Raindrops` reaches the picture — and it
// tracks the drawn radius as its two-thirds power, which is what holds a flake drawn three times a
// drop's width to the same opacity instead of a ninth of it. Four fifths for every weather, by
// construction, with the clamp a guard that cannot bind.
//
// **Per weather is not per pixel**, which is the distinction the horizon band turned on: `spacing`
// is one number for the whole frame, so every ray still agrees about where the drops are. What was
// incoherent was deriving the lattice from the *ray*.
const float PRECIP_OPACITY = 0.8;

// What the cell may not fall outside, in world units, whatever spacing it is handed.
//
// A guard on data rather than a tuning knob: a weather with an absurd count would otherwise divide
// the world into cells finer than the march can walk, or coarser than the volume it is drawn in.
const float PRECIP_CELL_MIN = 2.0;
const float PRECIP_CELL_MAX = 40.0;

// The most cells a ray walks before it stops asking.
//
// **The march steps one cell at a time**, which is what keeps consecutive samples in *adjacent*
// cells rather than skipping across dozens of them. That is the difference between walking a
// lattice and aliasing it, and aliasing it is what put a band across the horizon: the screen-space
// shape of aliasing is set by how fast the lattice coordinates move per pixel, a rate that peaks
// exactly where the ray runs perpendicular to the fall. Painting `abs(dot(direction, fall))` puts a
// black line straight through the middle of that band.
//
// The cell is the weather's now, so this is a ceiling rather than a count: sixty-four of the finest
// any weather asks for — a thunderstorm's seven units — covers four hundred, past the deepest volume
// the game names, and a ray that meets a surface sooner simply stops. What they actually walk is
// twenty-six for rain, twenty-nine for a thunderstorm, and twenty-two for snow, whose flakes are
// drawn three times as wide and so sit in a lattice twice as coarse.
const uint PRECIP_STEPS = 64u;

// How near the eye the march starts, in world units.
//
// **This, and not the cell size, is what sets how big a drop looks.** A streak's size on screen is
// its radius over its distance; `PRECIP_DRAWN` fixes the first in world units, so all the lattice
// decides is the second — and the nearest sample sits half a cell past this. At half a cell and
// nothing else the march began five units from the eye, where a streak subtends four degrees and
// covers seventy pixels.
//
// A hundred units is a metre and a half, and it is also honest: a drop that close is far outside
// the depth of field of any lens focused on a landscape, which is why Garg and Nayar photograph and
// render theirs at three metres.
const float PRECIP_NEAR = 100.0;


// How wide a real drop is, in world units.
//
// **The physical radius, which is what fixes how much of the air the rain blocks** — not what is
// drawn, which is `PRECIP_DRAWN`. Together with the host's spacing it says what fraction of a ray
// real drops would cover, and the shader solves the drawn streaks' opacity to match.
//
// **One radius for a whole distribution, and there is only one that is right.** A millimetre and a
// half stood here, which is a large drop — the kind Garg and Nayar photographed and rendered their
// streak database at. But what a ray meets is every drop in the air, most of them far smaller, and
// what decides how much of it they block is the *mean cross-section*, not the biggest one.
//
// Marshall and Palmer give the distribution in closed form and it comes out in one line. For
// `N(D) = N0 exp(-ΛD)` the mean square diameter is `2 / Λ^2`, so the mean cross-section per drop is
// `(π/4)<D^2> = π / (2Λ^2)`, and the radius of a disc with that area is
//
//     r = 1 / (Λ sqrt 2)
//
// At `Λ = 4.1 R^-0.21` for moderate rain — ten millimetres an hour — that is 2.528 per millimetre
// and `r` is 0.280 mm, a fifth of what stood here and a twenty-ninth of the area. Seventy units to
// the metre makes it a fiftieth of a unit.
const float PRECIP_RADIUS = 0.019577;

// How much more of a ray the drawn streaks cover than real drops block.
//
// **The one number here that physics does not give, and it is honest about which one it is.** Run
// the line above out and moderate rain blocks about two parts in a thousand of a three-metre ray:
// real rain is nearly transparent, and a photograph of it shows streaks anyway because a drop is a
// *lens* — see `PRECIP_LIT` — that gathers a hundred and sixty degrees of the world into a small
// solid angle and comes back many times brighter than what is behind it. Rain is a radiance, not an
// occlusion.
//
// A renderer cannot have that shape. Two parts in a thousand at fifty times the background is a
// scattering of sub-pixel specks, which alias into crawling static and which Ray Reconstruction then
// eats — §8.64 is the whole account of losing that argument. So the same signal is spread: more of
// the ray covered, each streak dimmer, and `coverage * radiance` is what is held.
//
// Thirty-five is the factor that holds it, and the check on it is that `PRECIP_SPREAD * PRECIP_LIT`
// comes to a lens gain near fifty — which is the order Garg and Nayar and Wang both put a drop's at,
// arrived at from the other end.
const float PRECIP_SPREAD = 35.0;

// How much wider a flake is drawn than a drop.
//
// Snow is loose crystal rather than a sphere, several millimetres across and often far more, and it
// is what makes a flake read as a flake where a drop reads as a smear.
const float PRECIP_FLAKE = 3.0;

// How wide a drawn streak is, in world units.
//
// **A drawing width, not the drop's own** — that is `PRECIP_RADIUS`, which still sets how much of
// the air the rain blocks. This only decides how a streak is spread across what the sampling can
// resolve: wide enough to be seen, narrow enough to leave air between one and the next.
//
// **In the world rather than in the cell**, which it was, and that was the whole of why the coverage
// solve saturated. A drop is a thing in the air and how wide it is drawn cannot depend on how finely
// the ray is sampled — the same argument the lattice below makes about where the drops *are*, which
// took four failed attempts at the horizon band to arrive at. It also made the cell useless as a
// knob: coverage goes as `r^2 / cell^2`, so a cell that set `r` could not move it at all.
//
// Five millimetres, and the same five millimetres a drop was always drawn at — what the change moved
// is how many of them a ray meets, not how big one is.
const float PRECIP_DRAWN = 0.35;

// How long the shutter is that a streak is smeared over, in seconds.
//
// **What makes a streak a streak rather than a dot**: a drop crosses fifteen centimetres while the
// shutter is open, and that is the shape drawn. A sixtieth is the exposure Garg and Nayar rendered
// their streak database at, and it is what a camera showing rain as streaks rather than as dots
// uses. What speed it multiplies is `PRECIP_REAL`'s business — the ini's own fifty-seven metres a
// second would put a whole metre on the screen, which is the thing that constant exists to refuse.
//
// Their closed form for a streak's *opacity* — `2r / (vT)`, the drop's diameter over the distance
// it smeared into, which comes to two tenths of a percent — is no longer what this shader applies,
// and the reason is worth keeping. It is the opacity of a *real* drop in a lattice at a real rain
// density, and that lattice cannot be sampled: see `across_side`. What the shader draws is a
// coarser lattice whose streaks each carry more, solved so the two block the same fraction of the
// ray. The closed form still sets what that total has to be; it is no longer what any one streak
// gets.
const float PRECIP_SHUTTER = 1.0 / 60.0;

// How much of the speed the ini writes down a drop really falls at.
//
// **What a streak's *length* is measured from, where the ini's own speed carries the drops.** A
// raindrop reaches about nine metres a second and goes no faster whatever the storm; Morrowind's
// `Precip Gravity` times rain's entrance speed is fifty-seven, six times that, because the original
// draws long sprites and they have to move to read. Taking the length from the game's figure makes
// a streak a full metre — which at half a metre from the eye subtends fifty-six degrees, half the
// screen, and is why the drops looked enormous however narrow they were drawn.
//
// **A ratio and not a speed, because every weather falls at its own.** Written as one it was rain's
// alone, and a flake drifting at 345 units a second came out smeared over the same fifteen
// centimetres as a drop doing 4,025 — a needle ten times longer than it was drawn wide, where a
// flake over a sixtieth of a second barely moves its own width. Nine metres a second is 630 units,
// against the 4,025 the file gives rain; everything it writes is that much too fast, and the ratios
// *between* weathers are close enough to keep — snow's 345 comes out at 0.77 m/s against a real
// flake's one to one and a half.
//
// The ini's speed still says how fast the field is carried past, which is the game's look; this says
// how far one of them smears while the shutter is open, which is physics.
const float PRECIP_REAL = 630.0 / 4025.0;

// What a drop shows of the world it is lit by, against that world's own radiance.
//
// **A raindrop is a lens with an enormous field of view**, so what leaves one toward the eye is a
// wide swathe of the environment gathered into a small solid angle — which is why a streak reads
// *brighter* than what is behind it rather than darker, and why rain is most visible against a dark
// background and around a lamp. Wang and colleagues compute the whole transfer with precomputed
// radiance transfer over an environment map; this takes the dome in the drop's mirror direction and
// scales it, which is that averaged down to one sample.
//
// **Coupled to `PRECIP_SPREAD`, and moving one alone breaks the other's argument.** A real drop's
// gain is tens, not 1.4; the rest of it was traded there for coverage, because a scattering of very
// bright sub-pixel specks is what the upscaler destroys. What the two hold between them is the
// product, so `PRECIP_SPREAD * PRECIP_LIT` is the gain actually being claimed — near fifty, which is
// the order the literature puts a drop's at, and the only check either number has.
const float PRECIP_LIT = 1.4;

// Two axes across the fall, so a drop can be placed in the plane it falls through.
void precip_basis(vec3 fall, out vec3 across, out vec3 along) {
    vec3 other = abs(fall.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    across = normalize(cross(fall, other));
    along = cross(fall, across);
}

// How much of a column the streak fills, the rest being the gap to the one behind it.
//
// **Without a gap there is no motion to see.** A streak is a metre long where the drops are seven
// centimetres apart, so a lattice with one cell per drop fuses every column into a continuous rod —
// and a rod of rain falling through itself looks exactly like a rod of rain standing still, which
// is what the first version of this drew. The cell along the fall is therefore the streak's own
// length over this, and what is left is empty air with the next drop behind it.
const float PRECIP_DUTY = 0.55;

// Where the streak whose cell `latticed` falls in sits, in that cell's own frame.
//
// **Three dimensions, not two.** Hashing only the two axes across the fall gives every drop in a
// column the same sideways offset, which fuses them into rods; indexing the along-fall axis too lets
// each be jittered independently, so a column is a broken line of streaks and stepping from cell to
// cell as the lattice falls is what carries them past the eye.
//
// Returns the distance from the axis across the fall; `down` comes back as how far past the head of
// the streak the sample sits, and `phase` as the drop's own seed for the highlights along it.
float precip_offset(vec3 latticed, vec2 cell_size, out float down, out float phase) {
    vec3 cell = floor(latticed / vec3(cell_size.x, cell_size.x, cell_size.y));
    uint noise = hash(uvec4(uvec3(ivec3(cell) + 4096), 0u));
    vec3 jitter = vec3(float(noise & 0x3FFu),
                       float((noise >> 10) & 0x3FFu),
                       float((noise >> 20) & 0x3FFu)) / 1023.0;
    // The drop's own seed, so the highlights along it differ from its neighbours' rather than
    // varying across its width — which is finer than a pixel and came out as noise.
    phase = float(noise >> 30) * 7.0;
    vec3 drop = (cell + jitter) * vec3(cell_size.x, cell_size.x, cell_size.y);
    // **Wrapped inside the cell rather than measured from the drop.** Taken as a plain difference
    // this runs negative wherever the jitter puts a drop near the top of its own cell, and a sample
    // below the head of a streak is one the cell below owns — so a streak came out cut to whatever
    // length fitted under the boundary, full where the jitter was kind and a round dot where it was
    // not. Modulo gives every cell one whole streak in its own frame, and since the streak is
    // shorter than the cell none of them straddles a boundary at all.
    down = mod(latticed.z - drop.z, cell_size.y);
    return length(latticed.xy - drop.xy);
}

// What the precipitation puts in front of a ray, and how much of it is left behind.
//
// Returns the radiance in `xyz` and the transmittance in `w`, the same shape `fog_along` uses and
// composited the same way — what falls is a thing in the air rather than a property of a surface,
// so it belongs in `emitted` and never behind the albedo the denoiser divides out.
vec4 precipitation_along(vec3 origin, vec3 direction, float span) {
    if (frame.precip_spacing <= 0.0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    // **Air only, and under a bay there is none**, which is the same thing `fog_density_at` says
    // about the fog and for a stronger reason: a drop that reached the water stopped being a drop.
    // Nothing in this lattice knows that, so a submerged eye had a downpour hanging in the bay with
    // it, lit by a sun the water had already taken most of.
    //
    // `water_level - z` is never positive for a dry cell — that is what the negative infinity in
    // the frame constants is for — so this is the same guard every other reader of the level
    // writes, and the crossing below comes out at infinity rather than biting.
    if (frame.water_level - origin.z > 0.0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    // How much air stands under the eye, which is what a downward ray spends before it reaches the
    // surface.
    float head = origin.z - frame.water_level;
    // Only as far as the weather says its own volume reaches, never past what the ray hit first,
    // and never past the surface a downward ray crosses — an opaque hit already ends the march, but
    // a ray that finds nothing would otherwise carry the rain down through open water. Beyond all
    // three a drop is finer than a pixel and what stands in for it is the fog.
    float reach = min(span, frame.precip_reach);
    if (direction.z < 0.0) {
        reach = min(reach, head / -direction.z);
    }
    // Both halves of the same vector, taken once: the direction the lattice is built on and the
    // speed a streak's length comes from.
    float falling = length(frame.precip_fall);
    vec3 fall = frame.precip_fall / max(falling, 1e-6);
    vec3 across, along;
    precip_basis(fall, across, along);

    // The lattice falls, so the field is static in its own frame and a drop's motion is exact.
    //
    // **Subtracted from the sample, not added to it**, and the difference is which way the rain
    // goes. A drop sits at a fixed place `L` in the falling frame and is seen wherever the sample
    // point maps to it, so adding the drift solves `p + d = L` and puts the drop at `L - d` — which
    // climbs, because `d` points down. Rain hid it: its streaks are long and near vertical and turn
    // over completely between frames, so up and down look alike. Snow drawn as a round flake
    // drifting slowly does not hide anything.
    vec3 drift = frame.precip_fall * frame.time;
    // Wide enough to be seen against the cell it sits in and narrow enough to leave air between one
    // streak and the next — and the cell is solved from it below, not the other way round.
    float radius = PRECIP_DRAWN * (frame.precip_snow > 0.0 ? PRECIP_FLAKE : 1.0);

    // How far this weather's own drop travels while the shutter is open — and never less than it is
    // drawn across, because below that a shape is not short, it is *pointed the wrong way*. `down`
    // runs from the head of the streak and `distance` from its axis, so a round one is twice the
    // radius long. A drop smears ten units and is drawn two thirds of one across, so it is a
    // streak; a flake barely moves its own width and comes out round, which is what a flake is.
    float streak = max(falling * PRECIP_REAL * PRECIP_SHUTTER, 2.0 * radius);
    // One streak to a cell, the cell as tall along the fall as the streak plus the gap behind it.
    //
    // **Tied to the streak so that shortening one costs no coverage.** The share of a column a
    // streak fills is `PRECIP_DUTY` whatever its length, and `chance` below is the disc it presents
    // across the fall — so a weather whose drops smear less gets more of them stacked in the same
    // air rather than a thinner shower.
    float column = streak / PRECIP_DUTY;

    // **The lattice is the weather's, and the march walks it a cell at a time.** Solved so a drawn
    // streak comes out at `PRECIP_OPACITY` — see that constant, which carries the whole derivation.
    // Both cross-sections are discs now, so the pi divides out and the cell is a ratio of radii.
    float cells_cubed = PRECIP_OPACITY * radius * radius * PRECIP_DUTY
                      / (PRECIP_SPREAD * PRECIP_RADIUS * PRECIP_RADIUS);
    float across_side = clamp(frame.precip_spacing * pow(cells_cubed, 1.0 / 3.0),
                              PRECIP_CELL_MIN, PRECIP_CELL_MAX);
    float walked = min(reach - PRECIP_NEAR, across_side * float(PRECIP_STEPS));
    if (walked <= 0.0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    uint steps = uint(max(walked / across_side, 1.0));

    // **And the coverage stays the air's, which is what keeps the picture honest.** A lattice this
    // coarse holds far fewer drops than the air does, so each has to carry more: the physical answer
    // is what a ray crossing `n * pi r^2 * L` of real drop cross-section blocks — a disc, which is
    // what a sphere presents to a ray, where this squared the diameter instead and ran a quarter
    // over — with `n` and `r` the ones the host's spacing and `PRECIP_RADIUS` give. `PRECIP_SPREAD`
    // then says how much more of the ray is drawn than blocked, and what is solved for below is the
    // opacity a drawn streak needs for this many cells of a coarse lattice to come to that total.
    // Nothing about how it looks moves how much of it there is.
    float density = 1.0 / max(frame.precip_spacing * frame.precip_spacing
                              * frame.precip_spacing, 1e-6);
    float wanted = PRECIP_SPREAD * (PRECIP_RADIUS * PRECIP_RADIUS / INV_PI) * walked * density;
    // The odds a sample lands on a streak: the disc one presents against the cell it sits in —
    // `pi r^2 / s^2`, with the reciprocal because `sampling.glsl` carries `INV_PI` and no `PI` —
    // times the share of the column it fills along the fall.
    //
    // **The duty belongs here and was missing**, which quietly overstated the odds by nearly two and
    // sent `alpha` to the clamp that much sooner. A sample inside the drop's radius but in the gap
    // behind the streak is thrown away by `lengthways` below, and a solve that does not know that is
    // not solving for what gets drawn.
    float chance = (radius * radius / INV_PI) / max(across_side * across_side, 1e-6)
                 * min(streak / column, 1.0);
    // **A guard rather than a limit, which it had quietly become.** An opacity above one is not a
    // thing, so it is clamped — but for both rain weathers `wanted` used to land above what this
    // lattice could carry at full opacity, and the clamp swallowed the difference. Rain and
    // thunderstorm came out of it byte for byte identical, which meant the ini's counts, `RAINFALL`
    // both decided nothing at all for the two weathers they were written for. With `PRECIP_DRAWN`
    // off the cell and the cell solved from the opacity, this cannot come out above one at all, and
    // the counts drive the picture again.
    float alpha = min(wanted / max(float(steps) * chance, 1e-6), 1.0);

    float through = 1.0;
    float speckle = 0.0;
    for (uint layer = 0u; layer < PRECIP_STEPS; ++layer) {
        if (layer >= steps) {
            break;
        }
        float at = PRECIP_NEAR + across_side * (float(layer) + 0.5);
        vec3 position = origin + direction * at - drift;
        vec3 latticed = vec3(dot(position, across), dot(position, along), dot(position, fall));
        float down;
        float phase;
        float distance = precip_offset(latticed, vec2(across_side, column), down, phase);
        // Across the fall it has to be within the drop; along it, within the streak the drop
        // smeared into rather than in the gap behind it.
        float sideways = 1.0 - smoothstep(radius * 0.5, radius, distance);
        float lengthways = 1.0 - smoothstep(streak * 0.7, streak, down);
        float caught = sideways * lengthways * alpha;
        // **Not one brightness the whole way down.** A falling drop oscillates between an oblate
        // and a transverse mode, and what that does to the light through it is speckles and smeared
        // highlights along the streak rather than an even line — which Garg and Nayar call out as
        // the thing a constant-brightness streak gets wrong close up.
        float bright = 0.4 + 1.6 * fract(sin(down * 0.7 + phase) * 43758.5453);
        speckle += through * caught * bright;
        through *= 1.0 - clamp(caught, 0.0, 1.0);
    }
    float covered = clamp(speckle, 0.0, 1.0);

    // **What a drop shows is the sky, not a surface.** It is a lens: nearly everything reaching the
    // eye from one is the dome refracted through it. A flake is the other case — loose crystal,
    // which scatters what lands on it in every direction, so it comes out at the ambient rather
    // than at a slice of the sky.
    vec3 shown = frame.precip_snow > 0.0
               ? frame.ambient
               : sky_lighting(reflect(direction, fall)) * PRECIP_LIT;
    return vec4(shown * covered, through);
}
