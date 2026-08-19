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
vec3 sky_seen_through(vec3 direction, float lobe) {
    // Brighter overhead than at the horizon, which is the one feature of a sky worth having before
    // there is a weather system to ask.
    float height = clamp(direction.z * 0.5 + 0.5, 0.0, 1.0);
    vec3 colour = frame.ambient * mix(0.75, 1.35, height);

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
    return sky_seen_through(direction, 0.0);
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
            total += ambient;
            continue;
        }
        // Terminated with a flat ambient rather than a second bounce: the next order is ambient
        // against an albedo below one against a cosine, so it lands under a tenth of this one, and
        // the traversal it costs buys nothing visible.
        total += shade(bounce, ambient, pixel, STREAM_INDIRECT + b, BOUNCE_SHADOW_SAMPLES);
    }
    return total / float(frame.bounce_samples);
}
