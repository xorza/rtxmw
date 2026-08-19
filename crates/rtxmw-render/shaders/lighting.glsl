// What a surface does with the light that reaches it.
//
// Next-event estimation throughout: every light is asked directly rather than found by chance,
// which is what makes one bounce affordable. The sun and the placed lamps share `leaving` for where
// a shadow ray starts and `lambert` for how much of what arrives comes back out.

// Defined by `water.glsl`, which needs what this file defines and so comes after it.
// The sun is dimmed on its way down through water, and so is the sky.
vec3 sun_through_water(vec3 position, float footprint);
vec3 daylight_reaching(vec3 position);

// How many shadow rays the sun gets at the primary hit.
//
// More than a lamp's, because its penumbra is the one every outdoor surface is judged by: a disc
// half a degree across throws an edge that stays sharp for centimetres and softens over metres, and
// too few samples turn that gradient into banding across a whole hillside.
const uint SUN_SAMPLES = 16u;

// How many shadow rays each light gets at the primary hit. A point light needs one and gives a hard
// edge; a light with real size needs several, and the penumbra is exactly the disagreement between
// them.
const uint SHADOW_SAMPLES = 8u;

// And how many at a bounce hit. One, because a bounce contributes a fraction of the pixel's
// radiance and is already being averaged over `bounce_samples` directions — resolving the penumbra
// of light that arrives indirectly would cost thirteen rays a bounce to change nothing visible.
const uint BOUNCE_SHADOW_SAMPLES = 1u;

// How much of the light landing on a sheet's far face comes out of the near one.
//
// Morrowind's cloth, banners and foliage are a single layer of triangles with no back side to
// occlude anything, so a sail with the sun behind it glows rather than turning black. Half is the
// figure for thin cloth; the exact value matters far less than that it is neither 0 nor 1.
const float TRANSMISSION = 0.5;

// How fast a bounce ray's cone widens, against `frame.cone_spread` for a primary ray.
//
// A diffuse bounce spreads over the whole hemisphere, and the indirect term wants the *average*
// albedo across that solid angle rather than a point sample of it — so a coarse mip is the correct
// answer here rather than a concession to cost. One unit of width per unit travelled is wide
// without collapsing every bounce to the top level.
const float BOUNCE_SPREAD = 1.0;

// How much of a light reaches a point `reach` units away.
//
// Inverse square, windowed so the contribution reaches exactly zero at the light's radius rather
// than being clipped there. Morrowind's radius is a hard cutoff in the original engine, and a
// clipped inverse square leaves a visible edge where the falloff jumps to nothing.
float attenuation(float reach, float radius) {
    float ratio = reach / radius;
    float window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
    // The `+ 1` keeps the singularity at zero distance finite; a light is not a point in practice.
    return window * window / (reach * reach + 1.0);
}

// How much of the light arriving along `towards` a surface turns back out, per unit irradiance.
//
// A solid is lit on the side the light is on and black on the other, because its own body is in the
// way. A sheet has no body: light landing on its far face comes through, dimmed by what the cloth
// absorbs on the way but never stopped. That is what a backlit sail is, and it is also what stops
// the handful of triangles Morrowind wound inside out from reading as holes — they now come back
// dimmer than their neighbours rather than black.
//
// View-independent, as a Lambertian sheet is: the same radiance leaves both faces.
float lambert(Surface surface, vec3 towards) {
    float facing = dot(surface.normal, towards);
    if (surface.thin) {
        return max(facing, 0.0) + TRANSMISSION * max(-facing, 0.0);
    }
    return max(facing, 0.0);
}

// The sun's contribution, before the surface's albedo.
//
// No attenuation and no distance: the sun is far enough away that its rays are parallel and its
// brightness is the same everywhere, so the only questions are which way the surface faces and how
// much of the disc it can see. That second one is the soft shadow — the same visibility fraction a
// lamp's sphere gives, over a cone rather than a sphere.
vec3 sun_light(Surface surface, vec3 origin, uvec2 pixel, uint salt, uint samples) {
    float facing = lambert(surface, -frame.sun_direction);
    // A black sun is how a cell with no sky says so, and a surface facing away needs no rays.
    if (facing <= 0.0 || frame.sun_colour == vec3(0.0)) {
        return vec3(0.0);
    }

    // The full disc at a primary hit, and whatever the caller asked for anywhere else — a bounce
    // wants one ray, not sixteen, and its penumbra is invisible under an albedo it is about to be
    // averaged with. Scaling by the ratio instead gave a bounce *two*, which the comment here used
    // to claim was one.
    uint taken = samples >= SHADOW_SAMPLES ? SUN_SAMPLES : samples;
    float visible = 0.0;
    for (uint s = 0u; s < taken; ++s) {
        vec2 u = unit_pair(hash(uvec4(sample_stream(pixel), salt + STREAM_SUN, s)));
        vec3 towards = cone_direction(-frame.sun_direction, frame.sun_cos_radius, u);
        if (!occluded(origin, towards, RAY_MAX)) {
            visible += 1.0;
        }
    }
    return frame.sun_colour * facing * (visible / float(taken))
         * sun_through_water(surface.position, surface.footprint);
}

// The lights whose reach can cover `position`, as a range into `light_grid_indices`.
//
// **What this exists to skip.** Every light was walked for every shading point, primary and bounce
// alike, and a light nowhere near the point still cost a fetch and a distance test: measured at
// 1920x1080, 0.031 ms per light per frame whether or not it contributes. Balmora's 53 spent 1.6 ms
// of a 7.3 ms trace being rejected one at a time.
//
// An empty range for a point outside the grid, which is also what an empty grid gives — the
// dimensions are zero then, so the bounds test below rejects everything without a case of its own.
uvec2 lights_reaching(vec3 position) {
    ivec3 dimensions = ivec3(frame.light_grid_dimensions);
    ivec3 cell = ivec3(floor((position - frame.light_grid_origin) * frame.light_grid_scale));
    if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, dimensions))) {
        return uvec2(0u);
    }
    uint slot = uint((cell.z * dimensions.y + cell.y) * dimensions.x + cell.x);
    return uvec2(light_grid_offsets[slot], light_grid_offsets[slot + 1u]);
}

// Radiance leaving a diffuse surface toward the viewer from the cell's placed lights.
//
// Shared by the primary hit and by every bounce hit, which is the whole reason next-event
// estimation is affordable: a bounce that had to *find* a light by chance would need hundreds of
// rays to stop being noise, and asking each light directly needs one.
//
// `salt` selects the hash stream so two calls for the same pixel do not draw the same sample
// pattern, and `samples` how finely each light's disc is resolved.
//
// **Returns light arriving, not light leaving** — the surface's own albedo is deliberately left
// out. That is what lets the primary hit's lighting be filtered on its own and multiplied back
// afterwards; a bounce hit, whose albedo belongs to a different surface, multiplies immediately in
// `shade`.
vec3 direct_light(Surface surface, uvec2 pixel, uint salt, uint samples) {
    vec3 total = sun_light(surface, leaving(surface, -frame.sun_direction), pixel, salt, samples);
    uvec2 near = lights_reaching(surface.position);
    for (uint k = near.x; k < near.y; ++k) {
        uint i = light_grid_indices[k];
        Light light = lights[i];
        vec3 to_light = light.position - surface.position;
        float to_light_length = length(to_light);
        if (to_light_length >= light.radius) {
            continue;
        }
        vec3 towards = to_light / to_light_length;
        float facing = lambert(surface, towards);
        if (facing <= 0.0) {
            continue;
        }

        // The light is a sphere, not a point, so visibility is the fraction of it that can be seen
        // rather than a yes or no. That fraction *is* the penumbra: near a blocker's edge some
        // samples reach the light and some do not, and the gradient between them is the soft
        // shadow.
        float visible = 0.0;
        vec3 origin = leaving(surface, towards);
        for (uint s = 0u; s < samples; ++s) {
            vec2 u = unit_pair(hash(uvec4(sample_stream(pixel), salt, i * samples + s)));
            vec3 offset = light.position + sphere_point(u) * light.source_radius - origin;
            float reach = length(offset);
            if (!occluded(origin, offset / reach, reach - SHADOW_BIAS)) {
                visible += 1.0;
            }
        }
        if (visible == 0.0) {
            continue;
        }
        total += light.colour * facing * attenuation(to_light_length, light.radius)
               * (visible / float(samples));
    }
    return INV_PI * total;
}

// Radiance leaving a hit toward whatever traced it.
//
// One statement of what a diffuse surface does with light, used at both depths: the primary hit and
// a bounce hit differ only in where `incoming` comes from — a gathered hemisphere for the first, the
// flat ambient that terminates the path for the second. Writing it twice is how the two would drift.
vec3 shade(Surface surface, vec3 incoming, uvec2 pixel, uint salt, uint samples) {
    return surface.emissive
         + surface.albedo * (incoming + direct_light(surface, pixel, salt, samples));
}

// What a ray that hits nothing sees.
//
// Derived from the cell's ambient rather than from constants of its own: outdoors the ambient *is*
// the sky, so the two cannot disagree, and indoors it stays the dark fill an enclosed room has,
// which is what a ray escaping through a doorway should find.
//
// The disc is drawn but is **not** energy-consistent with the directional term above: a real sun's
// radiance is its irradiance divided by the solid angle it subtends, which for half a degree is
// some sixty thousand times the value used here. Making the two agree means turning the directional
// light into an area light, which is a larger change than drawing a bright circle.
// The three of `sky.rs`'s constants the dome's *shape* needs, which are written down in two
// languages because the shape is: the Rayleigh spectrum normalised (`ZENITH_DEPTH` over its own
// sum), how far up the dome the horizon's paleness reaches in air masses, and what the sky away
// from a low sun keeps of what the sky toward it has.
//
// **What makes duplicating them bearable is that a test fails when they drift**, not that any of
// them is beyond argument — two are tuning dials. `tests/sky_dome.rs` renders a frame of nothing but
// sky and checks every pixel against `Sky::shape`, so a number changed on one side and not the other
// is a red suite rather than a slowly wrong horizon. Everything that varies with the hour arrives in
// the frame constants instead, where there is only one copy to change.
const vec3 RAYLEIGH_TINT = vec3(0.110687, 0.257634, 0.631679);
const float PALE_MASS = 6.0;
const float AWAY_SHARE = 0.25;

// How coarsely the sky is diced to place stars, and how many of the cells get one.
//
// **Procedural rather than the game's own texture**, which is the one place this file departs from
// vanilla content and does so deliberately: `tx_sky_*` is a painted star field at a resolution that
// was generous on a 2002 monitor and is four pixels a star now. Stars are points of light rather than
// art, so generating them costs nothing, gives a crisp one at any resolution, and lets the field be
// *stable in world space* — the same star sits in the same place as the camera turns, which a
// screen-space dither cannot do.
const float STAR_CELLS = 170.0;
const float STAR_SHARE = 0.0100;

// How wide a star is drawn, as a multiple of the ray's own footprint, and how wide it may ever get.
//
// **Never narrower than a pixel**, which is what keeps them from boiling under the upscaler: a point
// smaller than the sampling rate is aliasing whatever else is true about it, and the temporal
// accumulation downstream would turn that into a crawl. Sized against the cone the ray already
// carries, so it holds at any resolution.
//
// **And never wider than half a cell**, which is the other half of drawing them at all. A star is
// found by hashing the cell a direction falls in, so only that cell draws it — one that reaches past
// its own boundary is simply cut off there, because the neighbour hashes to a different star and
// knows nothing about this one. At 340 cells the radius came to 0.62 of a cell at 1080p and 0.93 at
// 720p, so most of the field was fragments. Halving the cell count doubles a cell's angular size and
// brings the radius under the cap at every resolution the renderer targets.
const float STAR_WIDTH = 1.5;
const float STAR_REACH = 0.45;

// What a star is worth against the night sky it sits on.
//
// The brightest come out some sixty times the moonless sky's own floor, which is far above the real
// ratio — a real star is a hundredth of the sky around it and invisible against anything else. This
// is a sky meant to be looked at rather than measured.
const float STAR_BRIGHTNESS = 0.9;

// The star field in one direction: how bright a star is there, if there is one.
//
// A hash per cell of a diced sphere gives each its own position and magnitude, so the field is fixed
// in the world rather than on the screen. The cells are cubes rather than anything spherical, which
// bunches them a little toward the axes and does not matter for stars.
float star_field(vec3 direction, float footprint) {
    vec3 scaled = direction * STAR_CELLS;
    ivec3 cell = ivec3(floor(scaled));
    uint seed = hash(uvec4(uvec3(cell + 8192), 0u));
    // Most cells hold nothing; the share that do are the field's density.
    if (float(seed & 0xFFFFu) / 65535.0 > STAR_SHARE) {
        return 0.0;
    }
    // **Dead centre of its cell**, and no jitter: a star placed anywhere else can reach past a
    // boundary its neighbour will not draw across, and comes out a fragment. The grid is already an
    // irregular lattice in angle — the cells are cubes projected onto a sphere — so it supplies all
    // the disorder a star field needs without moving anything inside one.
    vec3 at = vec3(cell) + 0.5;
    // The cell's own scale back to radians, so a star is as wide as the ray is however far the grid
    // has stretched it — up to the point where it would leave the cell.
    float radius = min(max(footprint, 1e-5) * STAR_WIDTH * STAR_CELLS, STAR_REACH);
    float away = length(scaled - at);
    // **Cubed, because star brightness is nothing like uniform.** A flat distribution gives a sky
    // of identical pinpricks that reads as static; the real thing is mostly faint with a scattering
    // of bright ones, and the eye finds the constellations in that rather than in the average.
    float pick = float((seed >> 16) & 0xFFFFu) / 65535.0;
    float magnitude = 0.08 + 0.92 * pick * pick * pick;
    return magnitude * (1.0 - smoothstep(0.0, radius, away));
}

// How many atmospheres a beam crosses to something `climb` above the horizon — Kasten and Young.
float air_mass(float climb) {
    float degrees = degrees(asin(clamp(climb, 0.0, 1.0)));
    return 1.0 / (climb + 0.50572 * pow(degrees + 6.07995, -1.6364));
}

vec3 sky_seen_through(vec3 direction, float lobe, bool stars) {
    // **The sky has a direction, and this is `Sky::shape` drawn per pixel.** Deep blue overhead,
    // pale at the horizon where the air is thick enough to have forgotten what colour it was, and
    // once the sun is low, orange on its side and dim blue on the other. Everything that knows what
    // hour it is arrives in the frame constants; what is here is only the part that varies with
    // where the ray is pointing.
    //
    // Two mixes and never a product, which is the mistake this cost twice: multiplying a rising
    // spectrum by a falling one peaks in the middle, and the middle is green.
    float cosine = dot(direction, -frame.sun_direction);
    float sunward = clamp(cosine * 0.5 + 0.5, 0.0, 1.0);
    float pale = 1.0 - exp(-(air_mass(max(direction.z, 0.0)) - 1.0) / PALE_MASS);
    vec3 tint = mix(mix(RAYLEIGH_TINT, vec3(1.0 / 3.0), pale), frame.sky_warm,
                    sunward * frame.sky_warmth);

    float phase = 0.75 * (1.0 + cosine * cosine);
    float side = AWAY_SHARE + (1.0 - AWAY_SHARE) * (1.0 - frame.sky_warmth * (1.0 - sunward));
    // **No branch for an interior**, which has no dome: the host hands over a scale of zero and its
    // own recorded ambient as the floor, so the same expression covers a room and a sky.
    vec3 colour = tint * (phase * (1.0 + pale) * side * frame.sky_scale) + frame.sky_floor;

    // **Behind the fog and under the sun**, which is why they are added here rather than composited
    // later: a star is a thing in the sky, so the fog attenuates it and the dawn drowns it exactly as
    // they do everything else the sky sends. The schedule is Morrowind's own — see
    // `TimeOfDay::starlight`.
    //
    // **Drawn, and lighting nothing**, which is why `stars` is a flag rather than always on. The
    // brightest here is sixty times the sky's floor, and a bounce ray finds one about once in eight
    // hundred — so with four samples a pixel, one in two hundred would come back with a sixteenfold
    // spike in it and the night would crawl with fireflies. Real starlight is a rounding error on a
    // moonless landscape, so the honest answer and the cheap one agree.
    if (stars && frame.sky_stars > 0.0) {
        colour += vec3(STAR_BRIGHTNESS * frame.sky_stars
                     * star_field(direction, frame.cone_spread + lobe));
    }

    // **`lobe` is how far a rough surface smears the sun**, in radians. Water too fine to resolve
    // is not flat — the slopes are still there, they are simply smaller than a pixel — and what
    // they do to a reflected sun is spread it. That spreading *is* the glitter path: a mirror shows
    // one hard dot, and a mile of ruffled water shows a shimmering road to the horizon. Cox and
    // Munk measured sea roughness by photographing exactly this in 1954.
    float widened = cos(acos(clamp(frame.sun_cos_radius, -1.0, 1.0)) + lobe);
    if (dot(direction, -frame.sun_direction) > widened) {
        // The same flux over a larger cap, so a broader glitter is a dimmer one and the total light
        // the sun contributes does not grow with the wind.
        float spread = max(1.0 - widened, 1e-6);
        colour = frame.sun_colour * min((1.0 - frame.sun_cos_radius) / spread, 1.0);
    }
    return colour;
}

// What a ray that hits nothing sees, from a surface sharp enough not to smear it.
vec3 sky(vec3 direction) {
    return sky_seen_through(direction, 0.0, true);
}

// The same sky as a *source of light* rather than a thing to look at, which is the same sky without
// its stars — see the note beside them in `sky_seen_through`.
vec3 sky_lighting(vec3 direction) {
    return sky_seen_through(direction, 0.0, false);
}

// The cosine-weighted mean radiance arriving at a surface over its hemisphere, one bounce deep.
//
// Multiplying this by albedo is the whole indirect term: sampling by the cosine cancels both the
// cosine in the rendering equation and the `1/pi` in the Lambertian BRDF, so no other factor
// belongs here.
//
// `frame.ambient` is the environment's radiance — what a ray that escapes the geometry sees. That
// reading is what makes zero samples meaningful rather than black: with no bounce rays every
// direction escapes by definition, the mean is the ambient itself, and the term collapses to the
// flat `albedo * ambient` fill the original engine applied unconditionally. With rays, geometry
// occludes that fill where it should and replaces it with what the surroundings actually reflect.
vec3 gather_indirect(Surface surface, uvec2 pixel) {
    // The sky as it arrives here rather than as it leaves the clouds, which under water is less of
    // it. Taken at the shading point for every ray of the gather: a bounce escaping from three
    // metres down has had to climb the same three metres.
    vec3 ambient = frame.ambient * daylight_reaching(surface.position);
    if (frame.bounce_samples == 0u) {
        return ambient;
    }
    vec3 total = vec3(0.0);
    for (uint b = 0u; b < frame.bounce_samples; ++b) {
        vec2 u = unit_pair(hash(uvec4(sample_stream(pixel), STREAM_BOUNCE, b)));
        vec3 towards = cosine_direction(surface.normal, u);
        Surface bounce = trace(leaving(surface, towards), towards, surface.footprint,
                               BOUNCE_SPREAD, MASK_ANY);
        // Water counts as an escape rather than a surface: it has no albedo of its own, so
        // shading it here would return black, and what a bounce ray actually finds there is the
        // sky reflected off it or the ground through it — both far closer to the ambient this
        // terminates with than to nothing at all.
        if (!bounce.hit || bounce.water) {
            // **What the ray actually found**, which is the sky in that direction rather than the
            // dome's average. The ray was traced either way, so this costs one evaluation of a
            // function the frame already runs on every pixel that sees sky — and it is what makes a
            // directional dome do any work: a wall facing a sunset takes its colour from the part of
            // the sky that is orange, not from the average of a sky that is orange on one side.
            total += sky_lighting(towards) * daylight_reaching(surface.position);
            continue;
        }
        // Terminated with a flat ambient rather than a second bounce: the next order is ambient
        // against an albedo below one against a cosine, so it lands under a tenth of this one, and
        // the traversal it costs buys nothing visible.
        total += shade(bounce, ambient, pixel, STREAM_INDIRECT + b, BOUNCE_SHADOW_SAMPLES);
    }
    return total / float(frame.bounce_samples);
}
