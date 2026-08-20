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

// And how many each moon gets.
//
// **Half the sun's, and the two of them together still cost more than it does**, because the hours
// they are up are the hours it is not: a night frame was casting no shadow rays at all before there
// were moons in it, so every one of these is new work. Traced at 1920x1080, sixteen apiece took a
// 5.88 ms night trace to 10.47 and eight apiece to 9.12 — so this halves 4.6 ms to 3.2.
//
// Eight is enough where sixteen is not for the sun because the two questions are different sizes:
// Masser subtends eighteen degrees against the sun's half, so its penumbra is spread over thirty
// times the distance, and a gradient resolved in eight steps across a room is smoother than one
// resolved in sixteen across a hand's width. The light is a fraction of the sun's besides, so what
// noise is left is a fraction of a fraction.
const uint MOON_SAMPLES = 8u;

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

// What one disc in the sky contributes, before the surface's albedo.
//
// **The sun and both moons, which are the same problem.** No attenuation and no distance: all three
// are far enough away that their rays are parallel and their brightness is the same everywhere, so
// the only questions are which way the surface faces and how much of the disc it can see. That
// second one is the soft shadow — the same visibility fraction a lamp's sphere gives, over a cone
// rather than a sphere — and it is why a moon eighteen degrees across throws an edge metres wide
// where the sun's half-degree throws one a few centimetres wide.
//
// `direction` is the way the light travels, so the direction *to* the disc is its negation.
vec3 disc_light(Surface surface, vec3 direction, vec3 colour, float cos_radius, vec3 origin,
                uvec2 pixel, uint salt, uint samples, uint at_primary) {
    float facing = lambert(surface, -direction);
    // A black disc is how a sky without one says so, and a surface facing away needs no rays.
    if (facing <= 0.0 || colour == vec3(0.0)) {
        return vec3(0.0);
    }

    // The full disc at a primary hit, and whatever the caller asked for anywhere else — a bounce
    // wants one ray, not sixteen, and its penumbra is invisible under an albedo it is about to be
    // averaged with. Scaling by the ratio instead gave a bounce *two*, which the comment here used
    // to claim was one.
    uint taken = samples >= SHADOW_SAMPLES ? at_primary : samples;
    float visible = 0.0;
    for (uint s = 0u; s < taken; ++s) {
        vec2 u = unit_pair(hash(uvec4(sample_stream(pixel), salt, s)));
        vec3 towards = cone_direction(-direction, cos_radius, u);
        if (!occluded(origin, towards, RAY_MAX)) {
            visible += 1.0;
        }
    }
    return colour * facing * (visible / float(taken))
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
    vec3 total = disc_light(surface, frame.sun_direction, frame.sun_colour, frame.sun_cos_radius,
                            leaving(surface, -frame.sun_direction), pixel, salt + STREAM_SUN,
                            samples, SUN_SAMPLES);
    // **And the two moons, on the same terms.** A moonlit night costs no more than a sunlit day:
    // the sun's rays are skipped where its colour is black, which is every hour the moons are worth
    // anything, and the moons' where theirs is — which is most of the day.
    total += disc_light(surface, frame.masser.direction, frame.masser.light,
                        frame.masser.cos_radius, leaving(surface, -frame.masser.direction), pixel,
                        salt + STREAM_MASSER, samples, MOON_SAMPLES);
    total += disc_light(surface, frame.secunda.direction, frame.secunda.light,
                        frame.secunda.cos_radius, leaving(surface, -frame.secunda.direction), pixel,
                        salt + STREAM_SECUNDA, samples, MOON_SAMPLES);
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

// Whether a ray lands on a moon's disc at all.
//
// **What says the sky behind it is hidden**, which a moon that is merely *added* to the sky does
// not: a moon is a rock, and the stars past it are past a rock. `Moon::NONE` has zero width, so its
// cosine is one and nothing is ever more aligned than that — a cell with no sky needs no case here.
bool moon_covers(vec3 direction, Moon moon) {
    // The same two conditions `moon_disc` draws under, so nothing is ever hidden behind a moon that
    // is not itself drawn — a moon faded out across the horizon would otherwise leave a disc of sky
    // with neither stars nor moon in it.
    return moon.colour != vec3(0.0) && dot(direction, -moon.direction) > moon.cos_radius;
}

// What a moon adds to the sky along `direction`, which is nothing unless the ray lands on its disc.
//
// **A lit sphere, not a sprite.** The disc is the orthographic silhouette of a ball eighteen degrees
// across, so the surface normal at a pixel is recoverable from where in the disc the pixel fell —
// and once there is a normal there is a terminator, drawn by asking whether the sun reaches it. That
// is the whole of the phase: nothing selects one of eight painted textures, nothing schedules a
// crescent, and a crescent points at the sun because there is no other direction it could point.
//
// **Lommel-Seeliger rather than Lambert**, which is the model astronomy uses on airless bodies and
// is one divide. A Lambertian sphere is brightest at the middle and falls off to its limb, so a full
// moon would read as a shaded ball; the real one reads as a flat disc, because a rough dusty surface
// scatters back the way the light came. `mu0 / (mu0 + mu)` is exactly that — at full phase the two
// cosines are equal everywhere and the disc comes out uniform, which is what the sky actually shows.
vec3 moon_disc(vec3 direction, Moon moon, float footprint) {
    float along = dot(direction, -moon.direction);
    if (along <= moon.cos_radius || moon.colour == vec3(0.0)) {
        return vec3(0.0);
    }

    // Where in the disc the ray landed, in units of its radius: the part of the ray perpendicular to
    // the moon's centre, over the sine of the angular radius. Unit length at the limb.
    float sin_radius = sqrt(max(1.0 - moon.cos_radius * moon.cos_radius, 0.0));
    vec3 offset = (direction + moon.direction * along) / max(sin_radius, 1e-6);
    float across = dot(offset, offset);
    if (across >= 1.0) {
        return vec3(0.0);
    }
    // **How much of the disc one ray covers**, and the finest its curvature can honestly be
    // resolved at. A sphere's emission cosine is `sqrt(1 - across)`, which falls to zero over the
    // outermost fraction of a pixel, and point-sampling something finer than the ray is aliasing
    // whatever else is true about it. Worth 24 of 255 on the limb pixels and nothing anywhere else —
    // which is invisible in a still and is the size of thing that crawls under a temporal filter.
    float pixel = clamp(footprint / max(sin_radius, 1e-6), 0.0, 1.0);

    // The sphere's own normal there: the offset, plus however far the surface bulges toward us. The
    // moon's light travels *from* it to us, so the way out of the moon toward the eye is
    // `moon.direction` itself — and at the middle of the disc, where the bulge is all of it, that is
    // the whole normal.
    float mu = sqrt(max(1.0 - across, pixel));
    vec3 normal = offset + moon.direction * mu;
    float mu0 = max(dot(normal, -frame.sun_direction), 0.0);
    // **McEwen's lunar-Lambert**, not Lommel-Seeliger alone. On its own that law puts the sunward
    // limb at exactly twice the disc's middle at every phase but full — its emission cosine goes to
    // zero there while the incidence cosine does not — so a *full* moon, which every photograph
    // shows flat, came out with a doubled rim. Blending toward a Lambertian term, whose cosine
    // *does* vanish at the limb, is the standard answer and the one planetary photometry uses: it
    // leaves full flat and keeps the 1.25x the limb really has. The doubling is still there so a
    // full moon comes out at the colour it was given rather than half of it.
    float lommel = 2.0 * moon.lunar_lambert / max(mu0 + mu, 1e-4);
    float shade = mu0 * (lommel + 1.0 - moon.lunar_lambert);

    vec3 face = vec3(1.0);
    if (moon.face != 0u) {
        // **The vanilla portrait**, mapped across the disc as the game's own art has it: the texture
        // is a square with the moon inscribed, so the offset in disc units is the texture coordinate
        // scaled and centred.
        //
        // **Held upright against the moon's own orbit rather than the horizon**, which is what a
        // tidally locked body does: it keeps its face toward us and its orientation toward its
        // orbit, so the face turns against the horizon as it crosses the sky — 106 degrees over one
        // of Masser's transits. Against the world's up instead it would be pinned all night, which
        // is what a billboard does and what this did before the swing was measured.
        vec3 right = normalize(cross(moon.pole, moon.direction));
        vec3 up = cross(moon.direction, right);
        vec2 uv = vec2(dot(offset, right), -dot(offset, up)) * 0.5 + 0.5;
        vec4 portrait = textureLod(textures[moon.face], uv, 0.0);

        // **Times its alpha, over its own mean** — and both halves of that were bugs once.
        //
        // The alpha is not decoration. These files are not premultiplied, so past the edge of the
        // painted disc the colour is whatever was left in the file: for Secunda it climbs back to
        // 0.39 of its mean where the disc just inside has fallen to 0.14. Sampling `.rgb` alone drew
        // a bright ring hugging every moon, two pixels at nearly twice the disc beside them — **that
        // was the outline**, not the shading law that the two notes above are about. It doubles as
        // the silhouette's antialiasing: the cut at `across` is a hard yes or no, while the painted
        // alpha ramps over a couple of texels.
        //
        // The mean is the portrait's, not each texel's own luminance. The faces were drawn to be
        // shown rather than lit — Masser's mean texel is a linear 0.033 — so what is wanted from the
        // picture is each texel's *ratio* to that mean, with `moon.colour` supplying the level it
        // multiplies. Dividing by the texel instead flattens every part of the disc to one
        // brightness and leaves the maria as a hue shift, where a moon's are darker.
        face = portrait.rgb * portrait.a / max(moon.face_mean, 1e-6);
    }
    return moon.colour * (shade * face);
}

// What the cloud layer puts in front of the sky along `direction`, and how much of it it hides.
//
// **A vanilla asset lit rather than shown.** `tx_sky_*.dds` is a painted photograph of a sky with
// 2002's lighting in it, so what is taken from it is the *shape* — the alpha its artist drew the
// clouds with, and the texel's own luminance against the sheet's mean where that alpha is flat, as
// it is for every overcast weather. The colour comes from `cloud_lit` and `cloud_shadowed`, which
// the host built from the sun and the dome. Compositing the painting itself would light every cloud
// twice, which is `docs/design.md` §5.1's whole subject.
//
// Returns the layer's radiance; `hiding` comes back as how much of the sky behind it is covered.
vec3 cloud_layer(vec3 direction, float footprint, out float hiding) {
    hiding = 0.0;
    if (frame.cloud_cover <= 0.0 || frame.cloud_sheet == 0u || direction.z <= 0.0) {
        return vec3(0.0);
    }

    // **Where the ray meets a shell over a curved world**, which is the whole of why a sky has
    // depth. With the eye on a world of radius `R` and the layer `h` above it, the shell is centred
    // `R` below and the positive root of the quadratic is the distance to it: `h` looking straight
    // up, and `sqrt(2Rh)` — a hundred and sixty kilometres — along the horizon. A shell centred on
    // the *eye* instead is the same distance in every direction, which piles the whole sheet into a
    // ring in the last few degrees.
    //
    // **Solved as `c / (b + sqrt(b*b + c))` rather than `-b + sqrt(b*b + c)`**, which is the same
    // root without the cancellation. `b` is the world's radius times the ray's climb — 4.5e8 in
    // game units — so `b*b` is 2e17 and subtracting `b` from its own square root throws away every
    // digit that mattered: straight up it came out 17 units from an answer that is exactly the
    // altitude. The conjugate form adds two positive numbers instead and is exact there.
    float b = frame.cloud_world_radius * direction.z;
    float c = frame.cloud_altitude * (2.0 * frame.cloud_world_radius + frame.cloud_altitude);
    float reach = c / (b + sqrt(b * b + c));
    vec2 where = direction.xy * (reach / frame.cloud_tile) + frame.cloud_drift;

    // **The level the ray can actually resolve**, the same argument `cone_lod` makes for a surface:
    // the cone is `footprint` wide per unit travelled and it has travelled `reach`, so it covers
    // that much of the layer — and near the horizon `reach` is a hundred times what it is overhead,
    // so a tile there is compressed past what its finest mip can answer for. Sampling with plain
    // `texture` takes level zero, because a compute shader has no derivatives to pick one from, and
    // the horizon crawls.
    float texels = float(textureSize(textures[frame.cloud_sheet], 0).x) / frame.cloud_tile;
    float lod = max(0.0, log2(max(footprint * reach * texels, 1e-6)));
    vec4 sheet = textureLod(textures[frame.cloud_sheet], where, lod);

    // The painting's structure, as a ratio to its own mean rather than as a level — so the sheet
    // says where a cloud is thick and the light says how bright that is. Where the alpha carries no
    // shape, which is every overcast sheet, this is all the shape there is.
    float luminance = dot(sheet.rgb, vec3(0.2126, 0.7152, 0.0722));
    float thickness = clamp(luminance / max(frame.cloud_mean, 1e-6), 0.0, 2.0);

    // How much sky the cloud hides: its own alpha, faded out toward the horizon where the shell is
    // seen so obliquely that a tile covers more sky than it has detail for.
    hiding = sheet.a * frame.cloud_cover * smoothstep(0.0, 0.12, direction.z);

    // **Thick where it is lit and thin where it is not.** A cloud's own body shadows it, so the
    // dense parts of the sheet keep the sun and the wisps are lit through; that is the difference
    // between the two colours the host handed over, and the sheet's structure is what picks between
    // them.
    return mix(frame.cloud_shadowed, frame.cloud_lit, clamp(thickness, 0.0, 1.0));
}

// How many atmospheres a beam crosses to something `climb` above the horizon — Kasten and Young.
float air_mass(float climb) {
    float degrees = degrees(asin(clamp(climb, 0.0, 1.0)));
    return 1.0 / (climb + 0.50572 * pow(degrees + 6.07995, -1.6364));
}

vec3 sky_seen_through(vec3 direction, float lobe, bool looking) {
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
    // `WorldTime::starlight`.
    //
    // **Drawn, and lighting nothing**, which is why `looking` is a flag rather than always on. The
    // brightest here is sixty times the sky's floor, and a bounce ray finds one about once in eight
    // hundred — so with four samples a pixel, one in two hundred would come back with a sixteenfold
    // spike in it and the night would crawl with fireflies. Real starlight is a rounding error on a
    // moonless landscape, so the honest answer and the cheap one agree.
    // **The moons, and for the same reason drawn rather than gathered.** Masser is eighteen degrees
    // across against the sun's half, so its disc covers thirteen hundred times the solid angle and a
    // bounce ray finds one about once in a hundred — at three hundred times the night sky's own
    // floor, which is a firefly in every other tile. The light they actually contribute arrives as a
    // directional term in `direct_light`, where it is resolved rather than sampled.
    bool eclipsed = looking && (moon_covers(direction, frame.masser)
                             || moon_covers(direction, frame.secunda));
    if (looking) {
        float footprint = frame.cone_spread + lobe;
        colour += moon_disc(direction, frame.masser, footprint)
                + moon_disc(direction, frame.secunda, footprint);
    }

    // **Not through a moon**, which is what `eclipsed` is for. The dome's own glow stays added
    // because that is air *in front* of the moon and it really does sit on top; a star does not, and
    // adding it anyway put the constellations across Masser's face.
    if (looking && !eclipsed && frame.sky_stars > 0.0) {
        colour += vec3(STAR_BRIGHTNESS * frame.sky_stars
                     * star_field(direction, frame.cone_spread + lobe));
    }

    // **The clouds, over everything the sky itself has.** They hide the dome, the stars and the
    // moons in proportion to their own alpha, because a cloud is opaque where it is thick — and they
    // are drawn only for a ray being looked along, for the same reason the moons are: a bounce ray
    // that found one would be sampling a light the layer's own contribution to `ambient` has already
    // accounted for.
    float hidden = 0.0;
    if (looking) {
        vec3 layer = cloud_layer(direction, frame.cone_spread + lobe, hidden);
        colour = mix(colour, layer, hidden);
    }

    // **`lobe` is how far a rough surface smears the sun**, in radians. Water too fine to resolve
    // is not flat — the slopes are still there, they are simply smaller than a pixel — and what
    // they do to a reflected sun is spread it. That spreading *is* the glitter path: a mirror shows
    // one hard dot, and a mile of ruffled water shows a shimmering road to the horizon. Cox and
    // Munk measured sea roughness by photographing exactly this in 1954.
    float widened = cos(acos(clamp(frame.sun_cos_radius, -1.0, 1.0)) + lobe);
    // **And a moon in the way hides the sun too**, which is what an eclipse is. Rare, because the
    // two arcs are inclined to the sun's — but Masser is eighteen degrees across and the sun half of
    // one, so when it happens it is total.
    if (!eclipsed && dot(direction, -frame.sun_direction) > widened) {
        // The same flux over a larger cap, so a broader glitter is a dimmer one and the total light
        // the sun contributes does not grow with the wind.
        float spread = max(1.0 - widened, 1e-6);
        vec3 disc = frame.sun_colour * min((1.0 - frame.sun_cos_radius) / spread, 1.0);
        // **Behind whatever cloud is in front of it**, which it was not: this block *replaces* the
        // colour, so a sun under solid overcast was coming through at full strength with the cloud
        // deck drawn round it. Blended by the same coverage everything else in the sky is.
        colour = mix(disc, colour, hidden);
    }
    return colour;
}

// What a ray that hits nothing sees, from a surface sharp enough not to smear it.
vec3 sky(vec3 direction) {
    return sky_seen_through(direction, 0.0, true);
}

// The same sky as a *source of light* rather than a thing to look at, which is the same sky without
// the two things in it small enough and bright enough to be fireflies: its stars and its moons. Both
// are resolved elsewhere — see the notes beside them in `sky_seen_through`.
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
