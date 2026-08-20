// What the rain does to what it lands on.
//
// **The darkening is the part that reads, and it is not a tint.** A wet stone is not stone with grey
// over it; it is stone seen through a film that keeps sending the light back for another pass. What
// makes a surface look rained-on is that it went *darker and more saturated* while the sky picked
// out a sheen on it, and neither half is worth much without the other.
//
// Rain only, and only where the rain can reach — see `wetness`. Snow is left out for the same reason
// `waves.glsl` leaves it out of the ripples: `Snow Ripples=0` in the ini, because a flake settles
// rather than breaking.
//
// **The darkening goes everywhere and the sheen stays at the primary hit.** `shade` calls `wetness`
// and `soaked` for every surface a ray lands on, so a reflection in a wet street, the bed of a
// shallow and the fill light off a rained-on wall are all shaded wet — that costs the sky-visibility
// ray once per hit and measured at 0.9 ms. What is not worth a second ray is the film's own
// reflection off a bounce hit, which would be a reflection of a reflection at a tenth of the light.
//
// **What it reads, which is most of the tree**: `WATER_F0` and `SHORE_FADE` from `water.glsl`,
// `sky_seen_through` from `lighting.glsl`, `fog_noise` from `fog.glsl`, `ripple_slope` from
// `waves.glsl`, and `trace`, `occluded` and `leaving` from `surface.glsl` on top of the samplers.
// So this is included last of them, which is the ordering `primary_visibility.comp`'s include block
// asks callers to keep — and `lighting.glsl` forward-declares the two names it needs back, because
// it comes first and cannot wait.

// How much of the light arriving at a wet surface the film's own top turns away before it reaches
// what lies under it.
//
// Fresnel's reflectance at an air-water boundary, averaged over the hemisphere it arrives from,
// which comes to about a twelfth at 1.333. The smallest of the three terms `soaked` multiplies
// together, and kept because it costs nothing.
const float FILM_ENTERING = 0.0918;

// How much of the light leaving the substrate the film sends straight back down into it.
//
// **This is where the darkening comes from, and it is nearly half.** Light scattered back out of
// stone leaves in every direction, and everything past the critical angle — 48.6 degrees for water,
// which is most of the hemisphere by solid angle — is turned round by total internal reflection at
// the top of the film and handed back for the stone to absorb its share of again. Egan and
// Hilgeman's fit to that average, the one Jensen's dipole carries for the same reason,
//
//     F = -1.440 / n^2 + 0.710 / n + 0.668 + 0.0636 n
//
// comes to 0.475 at water's 1.333. This is Lekner and Dorf's answer to why things are darker when
// wet, and it is a closed form rather than a look-up.
const float FILM_RETURNING = 0.475;

// How far off the sky is, as the specular guide counts distance.
//
// **The guide is asked how far the *reflection* is and not how far the surface is** — see `Guides` —
// because that is what sets how fast it crosses the screen when the camera moves. A sheen is the
// dome, which does not shift at all for a step sideways, so what belongs here is a distance past
// everything in the cell rather than a number from inside it. Water writes `WATER_MAX_PATH` in this
// same field, which is four thousand and chosen to exceed the deepest water for absorption's sake:
// the right shape of answer for a ray that might have hit something, and much too near for one that
// never could.
const float FILM_SKY_DISTANCE = 1.0e5;

// How rough the film is, on the scale the specular guide uses — nought being a mirror.
//
// **Not nought, because a film follows what is under it rather than levelling it.** A millimetre of
// water over Morrowind's stone is a millimetre of water shaped like stone. Low enough to gather the
// world into a sheen, nowhere near low enough to mirror it.
//
// Mud, sand, thatch and turf are most of what this world's ground is made of, and a porous surface
// drinks a film rather than holding one on top — it darkens hard and glosses little, which is the
// half of Lagarde's porosity that survives having no map to read it from.
const float FILM_ROUGHNESS = 0.55;

// How wide a cone the roughness opens the reflection into, as a multiple of it squared.
//
// **Roughness has to move the ray or it is not roughness.** Widening the *cone* the way water does
// picks a coarser mip at the far end and leaves the ray itself a perfect mirror, so a wet shore came
// back with the trees standing sharp in it — one continuous puddle, whatever number the guide
// carried. What spreads a real reflection is that the surface faces a slightly different way at
// every point of it, so the ray has to be spread too.
//
// GGX puts the lobe's width near `alpha = roughness^2` and a reflection turns by twice what its
// normal does, which is the two. Drawn from the frame's own stream, so Ray Reconstruction — which
// takes the roughness and the hit distance below for exactly this — accumulates it away.
const float FILM_LOBE = 2.0;

// How much of the environment a film this rough actually returns, at `cosine` off its normal.
//
// **There is no BRDF under any of this, and at a grazing angle that shows.** What the film had was
// Fresnel times a reflection: no microfacet distribution, no geometric term, nothing to say that the
// facets which would have to face the eye at a glancing angle are shadowed by the ones in front of
// them. Schlick alone runs to one whatever the surface, and a wet street came back with nearly half
// the sky laid over it — washed out, bright, and flatter than the dry stone it replaced.
//
// The reflection above is a cone sample of the environment, which is precisely what the split-sum
// approximation is written for, so the closed form for it applies unchanged: Lazarov's analytic fit
// to the environment BRDF, `F0 * A + B`. At this roughness it returns **0.107** at grazing where
// Schlick returned 0.45, and the two agree to a thousandth head on — the whole of the difference is
// the shadowing term that was missing.
float film_response(float cosine, float roughness) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * cosine)) * r.x + r.y;
    vec2 scale_bias = vec2(-1.04, 1.04) * a004 + r.zw;
    return WATER_F0 * scale_bias.x + scale_bias.y;
}

// How much of the water's own ring a film a millimetre deep can hold, against an open bay's.
//
// **Without this a wet plank is a moulded panel, and the reason is in the model rather than in the
// tuning.** Lagarde's account of wet surfaces turns on which normal the film reflects: a *thin* film
// reflects the disturbed normal of what is underneath it, and only a thick one — a puddle — reflects
// a flat one. Vanilla Morrowind carries no normal map and `NiSpecularProperty` is force-disabled at
// this NIF version, so what the film had to disturb was an interpolated vertex normal across a flat
// plank: the puddle case, on every surface in the world at once, which is exactly what plastic is.
//
// What is left to disturb it with is the thing already falling on it. The same lattice of impacts
// `waves.glsl` rings the bay with rings a wet deck, because it is the same rain — and a ring on a
// film is the one piece of surface detail in this world that needs no map to exist, since it is made
// by the weather rather than painted by an artist.
//
// **At the water's own strength, and scaled by how much water is there to ring.** A ring needs
// something to form in: where the film stands it rings like a bay, and where it has all but soaked
// away there is nothing to disturb. Held back to a fraction everywhere instead, the rings were the
// same faint everywhere too — which is the one thing they cannot be, since what makes them read is
// that some of the ground is puddled and the rest of it merely damp.
const float FILM_RIPPLE = 1.2;

// How much of the film sits on top of a surface rather than soaking into it.
//
// **Lagarde's porosity, and the half of it that survives having no map to read.** Water that has
// gone *into* stone still darkens it — that is what `soaked` is about, and it needs no film on the
// surface at all — but only water left standing on top can reflect anything. Mud, thatch, sand and
// dressed tuff are what this world is built from and every one of them drinks; a pane of glass in
// the rain is the other extreme and there is none of it here.
//
// Applied to the sheen alone, which is why a wet street can go dark without going white. Left out,
// every surface came back looking gelled: the whole of a film's reflection over a world that would
// have absorbed most of it.
//
// **And gated on where the water is deep enough to stand, not spread evenly over everything.** A
// flat share everywhere is the same mistake in the other direction: it gives the whole world a
// little gloss and nowhere a puddle, and the rings — which are the one thing rain writes on the
// ground — came out at a hundredth of a hundredth, invisible on every surface at once. Water does
// not lie in an even coat; it runs to the low places and soaks away from the high ones. So the
// thickest of the grain reflects like the standing water it is and the rest of it stays dark, and
// what the rings ring in is the puddles.
const float FILM_HELD = 0.55;
const float FILM_STANDS = 0.45;

// How rough standing water is, against the `FILM_ROUGHNESS` of water that has soaked in.
//
// **A puddle is smooth and a damp flagstone is not, and that is the difference between seeing a ring
// and seeing a bright patch.** Water gone into stone takes the stone's own shape; water standing on
// it takes its own, which is flat except for whatever is falling on it. Held at one roughness for
// both, the reflection was a thirty-degree cone average — near enough uniform that tilting the
// normal moved the sample from one part of a flat field to another and changed nothing. All a ring
// could do then was turn the brightness up, which is a white blob rather than a ripple.
//
// Sharp enough that a ring bends what it is reflecting, which is the only way a ring is ever seen.
const float FILM_PUDDLE = 0.12;

// How far apart the film's own thickness varies, in world units, and how thin it gets between.
//
// **A film is not a coat of paint.** Water beads on some of a surface and pools on the rest of it
// according to what the surface is made of and where it dips, and a wetness that is one number
// everywhere is the other half of what reads as moulded. There is no map that says where, so the
// grain is world-space noise — the same `fog_noise` the haze is built from, at a scale a little
// coarser than a plank is wide.
const float FILM_PATCH = 90.0;

// How far off the fall's own line the rain arrives from, in radians.
//
// Wind gusts, and drops that bounce off whatever they land on first. A sixth of a radian is ten
// degrees, which over the height of a Balmora eave smears the edge of its shelter across a couple of
// metres of street — a boundary rather than an outline.
const float FILM_GUST = 0.17;
const float FILM_THINNEST = 0.35;

// How wet `surface` is, from dry to a film across all of it.
float wetness(Surface surface, uvec2 pixel) {
    if (frame.precip_spacing <= 0.0 || frame.precip_snow > 0.0) {
        return 0.0;
    }
    // How much air is under the surface, which decides whether the rain reaches it at all and how
    // much of a film it is allowed to keep — see the fade this returns through.
    float above = surface.position.z - frame.water_level;
    if (above < 0.0) {
        return 0.0;
    }
    // **Where the rain comes from, which is not straight up.** `precip_fall` carries the wind's
    // slant, so the weather side of a wall takes some of it and the lee side takes none — and the
    // dry patch under an eave sits offset from the eave rather than under it, which is what a rain
    // shadow looks like.
    vec3 falling = normalize(frame.precip_fall);
    float caught = dot(surface.normal, -falling);
    if (caught <= 0.0) {
        return 0.0;
    }
    // **Back along the fall, and not along one line of it.** A surface the rain cannot reach stays
    // dry however level it is, which is what makes a porch read as shelter — but rain does not
    // arrive as a beam. Gusts swing it, drops splash off what they hit, and the edge of a rain
    // shadow is a metre of half-wet ground rather than a line. Testing one direction drew that edge
    // as the *silhouette of the building*, polygon by polygon, which is the sharpest thing a picture
    // can have and never what shelter looks like.
    //
    // One sample of the spread, taken from the frame's own stream so Ray Reconstruction resolves it
    // the way it resolves every other estimator here.
    vec2 u = unit_pair(hash(uvec4(sample_stream(pixel), STREAM_FILM, 1u)));
    vec3 arriving = cone_direction(-falling, cos(FILM_GUST), u);
    if (occluded(leaving(surface, arriving), arriving, RAY_MAX)) {
        return 0.0;
    }
    // **How much stays, not how much arrives, and the two run opposite ways with slope.** The cosine
    // is the rate the rain lands at; what a surface *holds* is that against how fast it runs off,
    // and a tilted face sheds what a level one keeps. This took the square root of the cosine at
    // first, on an argument about a surface saturating once it is covered — which is true and is
    // the wrong end of it, because it made a slope wetter than the rate alone would and left every
    // face in the world within a whisker of soaked. Squared runs the right way: level ground pools,
    // a roof holds less than it catches, and a trunk is barely damp.
    //
    // Thinner in places on top of that, because water beads and pools rather than coating — see
    // `FILM_PATCH`.
    //
    // **And a shore stops being wet before it goes under, which is a debt rather than a claim.**
    // Drowned ground is soaked — a lakebed is wet sand and wet sand is dark — and this returned one
    // for it briefly. What that broke is §8.20's rule that the water plane must not cut the terrain
    // along a visible line: it put the bed *darker* than the shore beside it, in weather with no
    // rain in it, and `the_waterline_leaves_no_seam_where_the_ground_meets_the_surface` said so at
    // 0.205 against 0.269. Wet sand really is darker than dry sand, and the seam that admits it is
    // worse than the one that does not, until the shore above the line is wet too.
    //
    // So the film goes to nothing over the same `SHORE_FADE` the waterline already dissolves across,
    // and the two sides meet at dry. That hides the step without being right about it.
    return caught * caught
         * mix(FILM_THINNEST, 1.0, fog_noise(surface.position / FILM_PATCH))
         * smoothstep(0.0, SHORE_FADE, above);
}

// What is left of `albedo` under a film of water.
//
// The whole path summed: in through the top, off the substrate, and back down however many times
// the film returns it. A geometric series in `albedo * FILM_RETURNING`, which is why a dark surface
// loses proportionally more of itself than a bright one — there is less coming back each time round.
vec3 soaked(vec3 albedo, float wet) {
    vec3 drowned = (1.0 - FILM_ENTERING) * albedo * (1.0 - FILM_RETURNING)
                 / (1.0 - albedo * FILM_RETURNING);
    return mix(albedo, drowned, wet);
}

// What a film does to the surface under it: what is left of its own response, and what it adds.
//
// **Both halves or neither.** A surface cannot reflect most of the light specularly and still be as
// diffuse as it was — that is more light leaving than arrived, and at a grazing angle it is a great
// deal more. The first version added the reflection on top and left the diffuse alone; what came
// back was a deck lit like wood with a sheet of sky summed over it.
struct Film {
    // How much of the surface's own diffuse response survives the film's reflection, to scale the
    // albedo the composite multiplies back in.
    float kept;
    // What the film returns, already weighted by what it took.
    vec3 reflected;
};

// The world in the film's own reflection, and the guide that describes it.
//
// **Traced rather than the sky alone, which is the difference between wet and plastic.** A wide lobe
// does average a large part of the hemisphere, and that argument held right up until the eye came
// down to the surface: at a grazing angle the lobe compresses onto the mirror direction, the
// reflection is whatever lies along it, and answering "the sky" there paints a featureless sheet
// over the one view where a wet floor has the most to show. `water_ray` already does this — a cone
// widened by the lobe, shaded where it lands and the dome where it does not — and it is only in
// `water.glsl` because water was the only thing that needed it.
Film filmed(Surface surface, vec3 direction, float wet, uvec2 pixel, out Guides guides) {
    guides = matte();
    // Nothing on top of a drowned surface either: `wetness` returns nothing under a bay, because
    // what stands between it and the sky is water rather than air and `water_shade` is already
    // drawing that interface.
    if (wet <= 0.0) {
        return Film(1.0, vec3(0.0));
    }
    // How much of the water here is standing rather than soaked in, which decides how much the film
    // reflects, how sharply, and how far a ring can raise itself in it.
    float stands = smoothstep(FILM_STANDS, 1.0, wet);

    // **The rings the rain is making on it, which is the whole of the film's own shape.** Solved in
    // the plane the rain falls through and tilted into the surface from there: for anything level
    // enough to hold water that plane *is* the surface, and the expression below reduces to
    // `water_normal`'s exactly when the normal is up.
    float unresolved;
    vec2 slope = ripple_slope(surface.position.xy, frame.time, surface.footprint, unresolved)
               * FILM_RIPPLE * stands;
    vec3 filmed_normal = normalize(surface.normal + vec3(-slope, 0.0));
    // What the cone could not resolve of those rings is not gone, it is rough — `water_normal`'s own
    // argument, and it is what keeps a wet street from turning back into a mirror with distance.
    float lobe = clamp(mix(FILM_ROUGHNESS, FILM_PUDDLE, stands)
                     + 2.0 * sqrt(unresolved) * FILM_RIPPLE * stands, 0.0, 1.0);

    float cosine = clamp(dot(-direction, filmed_normal), 0.0, 1.0);
    float fresnel = film_response(cosine, lobe) * FILM_HELD * stands;

    // Spread across the lobe rather than sent down its axis — see `FILM_LOBE`.
    vec2 u = unit_pair(hash(uvec4(sample_stream(pixel), STREAM_FILM, 0u)));
    vec3 spread = cone_direction(reflect(direction, filmed_normal),
                                 cos(FILM_LOBE * lobe * lobe), u);
    // A facet that ended up facing away is one the surface would have hidden behind the roughness
    // in front of it; the axis is the nearest thing that still leaves.
    vec3 mirrored = dot(spread, surface.geometric) * dot(-direction, surface.geometric) > 0.0
                  ? spread
                  : reflect(direction, filmed_normal);
    Surface hit = trace(leaving(surface, mirrored), mirrored, surface.footprint,
                        frame.cone_spread + lobe, MASK_SOLID);
    float travelled = hit.hit ? hit.t : FILM_SKY_DISTANCE;
    // **Flat-lit rather than shaded, which is the whole of what makes this affordable.** A lobe this
    // wide smears whatever it lands on, so what the reflection has to carry is *variation* — the
    // difference between a plank and the sea behind it — and not a second opinion about the light
    // falling on it. Shading it properly, which is what `water_ray` does, costs a shadow ray per
    // light and measured at more than twice the whole trace — 16.7 ms against 7.9. What is drawn
    // instead is emissive raw over a flat ambient, which is `shade`'s own answer for a bounce.
    vec3 seen = hit.hit ? hit.emissive + hit.albedo * frame.ambient
                        : sky_seen_through(mirrored, lobe, true);
    guides = Guides(vec3(fresnel), lobe, travelled);
    return Film(1.0 - fresnel, seen * fresnel);
}
