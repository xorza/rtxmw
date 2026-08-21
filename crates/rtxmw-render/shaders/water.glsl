// What happens to light at the water and under it.
//
// Fresnel at the surface, Beer-Lambert and single scattering along the path, and the caustics —
// which are ray density, and so a determinant of the map from where light met the surface to where
// it landed. All of it analytic, because the surface is a closed-form height field: no photons, no
// buffer, no splatting.

// Defined by `lightning.glsl`, which needs `Surface` and so comes after this file.
vec3 bolt_along(vec3 origin, vec3 direction, float span);

// Defined by `lighting.glsl`, which comes before this file — the sky it lights is added in front of
// the fog, so a reflection of it has to be added here rather than found in `sky_seen_through`.
vec3 flash_on_deck(vec3 direction, float footprint);

// Water's index of refraction, and the reflectance it gives at normal incidence: `((1.33 - 1) /
// (1.33 + 1))^2`, which is why water is a window head-on and a mirror at a glancing angle.
const float WATER_IOR = 1.333;
const float WATER_F0 = 0.02;

// Extinction per world unit, per channel — how fast water swallows light along a path through it.
//
// Red goes first and the rest goes slowly: every water absorbs red within a metre or two, which is
// the one thing about water's colour that is not a matter of taste. Green and blue are held close
// together here and both low, which is what keeps the water clear and only faintly tinted rather
// than a coloured filter over the seabed. Quoted per metre — 0.32, 0.05 and 0.08 — and divided by
// the game's 69.99 units to the metre.
const vec3 WATER_EXTINCTION = vec3(0.004572, 0.000714, 0.001143);

// The single-scattering albedo: the fraction of extinction that was scattering rather than
// absorption, and so the part the water hands back as its own colour instead of swallowing.
//
// **This is what decides whether deep water is dark.** Absorption takes light out of the scene;
// scattering returns it, and a channel whose scattering albedo approaches one settles at a bright
// colour however deep it gets — a milky sheet rather than a channel. Clear tropical water really
// does behave that way, because molecular scattering dominates its blue. A tannin-stained coastal
// swamp does not: dissolved matter absorbs far more than the water scatters, so it stays
// transparent close in and goes dark where the bottom falls away.
//
// Provisional in the way `INTENSITY` is — the shape is physical, the magnitudes are a water type
// chosen by eye, and §5.1 could move them.
const vec3 WATER_SCATTER = vec3(0.012, 0.042, 0.040);

// How squarely a wave facet has to face the ray that found it before it is tilted back toward the
// plane. Small: this is a guard against a facet turning away entirely, not a limit on the waves.
const float MIN_FACING = 0.03;

// How far refraction deflects a ray, per unit of surface slope.
//
// A tilted surface bends light toward its normal, and the deflection is the difference between the
// angle of incidence and the angle of refraction — which for small angles is the slope times this.
// Multiplied by depth it is how far the light has wandered by the time it reaches the bottom, which
// is what makes caustics; multiplied by a lobe it is how much less a refraction is blurred by rough
// water than a reflection is. Derived from the index of refraction rather than written out, so
// nothing here can come to disagree about what water is.
const float REFRACTION_BEND = 1.0 - 1.0 / WATER_IOR;

// The same bend, per channel, because water's index is not one number.
//
// **Blue light turns harder than red**, so the three colours are focused by slightly different
// amounts and a caustic has coloured edges — the prism fringing a real pool shows against a white
// bottom. Cauchy's fit for distilled water, `n = 1.32403 + 3080 / lambda^2` with the wavelength in
// nanometres, at 600, 550 and 450: indices of 1.3326, 1.3342 and 1.3392, so the bends spread by
// about a part in seventy.
const vec3 REFRACTION_BEND_RGB = vec3(0.249579, 0.250494, 0.253308);

// A ceiling on how bright a focus is allowed to get.
//
// Where the refracted bundle collapses to a line the Jacobian goes to zero and the intensity to
// infinity — a real caustic *cusp*, and the reason a pool's bright lines are as sharp as they are.
// A renderer that let it through would produce a pixel no exposure could hold, so the focus is
// capped and what it would have contributed beyond that is simply not drawn.
const float CAUSTIC_MAX = 3.0;

// The depth past which the pattern stops sharpening, in world units — about two metres.
//
// **The honest edge of the approximation.** `q = p - bend * grad(h)` holds while the refracted
// bundle has not yet crossed itself; past the first focus the rays have folded over one another and
// a single Jacobian no longer describes what is there. Evaluated at the seabed rather than at the
// surface it came from, the model then starts *making* light — measured, three quarters more of it
// at four hundred units. Holding the depth here keeps it inside the regime where it conserves, and
// says the true thing anyway: caustics are sharp in a shallow pool and washed out in deep water.
const float CAUSTIC_MAX_DEPTH = 140.0;

// How little water there has to be before the surface stops being drawn at all, in world units.
//
// **The waterline.** Where the ground rises to meet the surface, the depth between them goes to
// zero, and a pixel of water with no water under it has to come out as exactly the shore beside it
// — otherwise the plane cuts the terrain along a hard line, which is the classic tell of a water
// plane and is on screen in 533 of the game's 1,292 land cells. Half a metre of fade is enough to
// hide the intersection without making the shallows look thin.
//
// **Measured straight down, and it was measured along the refraction before.** Those are the same
// number only where the bed is flat. At Seyda Neen's shore the terrain runs within a few units of
// sea level for hundreds of units, so the two planes are very nearly parallel and their crossing
// inside a single 128-unit facet is a straight line hundreds of units long — while the refracted
// ray, leaving at forty-three degrees, lands far enough out to find a bed a hundred units down. It
// reported deep water at a pixel with none, the fade never engaged, and what was left was exactly
// the hard line this constant exists to prevent. `docs/design.md` §8.101.
const float SHORE_FADE = 35.0;

// What a refraction ray that hits nothing has travelled through. Longer than the deepest water in
// the game — the lowest terrain vertex is 2,152 units down — so nothing is ever *less* absorbed for
// having missed the seabed.
const float WATER_MAX_PATH = 4096.0;

// How much the sun's light is concentrated or spread by the time it reaches `depth` below the
// surface, as a multiplier around one.
//
// **Caustics are ray density**, and density change is the determinant of the Jacobian of the map
// from where light met the surface to where it landed. For small slopes that map is
// `q = p - bend * depth * grad(h)`, so its Jacobian is `I - bend * depth * H` with `H` the Hessian
// of the same height field the normals come from — and because that field is five sinusoids, `H`
// is written out here rather than sampled, filtered or splatted. No photons, no buffer, no noise:
// the light is where the arithmetic says it is.
//
// The approximation is the small-angle one, and it is the right one for this game: Vvardenfell's
// water is thirty metres deep at its very worst and a few at the shore, with slopes under a
// seventh, so the exact refraction and its linearisation differ by less than the sun's own width.
//
// Waves too small for the ray cone are left out of the curvature exactly as they are left out of
// the normal, and for the same reason — a caustic finer than a pixel is a sparkle, not a pattern.
vec3 caustic(vec2 p, float depth, float time, float footprint) {
    // Second derivatives of `sum(A * sin(k * dot(d, p) - w * t))`, which are `-A k^2 d_i d_j sin`.
    //
    // Taken with respect to the drifted position rather than the true one, which drops the chain
    // rule's contribution from the drift itself. That field turns over six hundred units against a
    // curvature set by ten, so its Jacobian is within a fifth of the identity — and what the
    // omission costs is a slow variation in how strong the caustics are, which is indistinguishable
    // from the variation real water already has.
    vec2 here = drifted(p, time);
    // The Hessian of the height, and the Jacobian of the horizontal displacement beside it. They
    // are the same outer products: differentiating `A cos(phase)` once gives `A k sin(phase)` where
    // differentiating `A sin(phase)` twice gives `-A k^2 sin(phase)`, so the second is the first
    // scaled by `-1/k`. One loop produces both.
    vec3 hessian = vec3(0.0);
    vec3 gathering = vec3(0.0);
    for (int i = 0; i < WAVE_COUNT; ++i) {
        WaveSample wave = sample_wave(i, here, time, footprint);
        if (wave.detail <= 0.0) {
            continue;
        }
        float second = -wave.detail * wave.amplitude * wave.wavenumber * wave.wavenumber
                     * sin(wave.phase);
        vec3 axes = vec3(wave.direction.x * wave.direction.x,
                         wave.direction.y * wave.direction.y,
                         wave.direction.x * wave.direction.y);
        hessian += second * axes;
        gathering += (-second / wave.wavenumber) * axes;
    }

    // **The surface is stretched before the light ever leaves it**, so the caustic is the ratio of
    // two determinants rather than one. A patch of parameter space covers `det(I + dD)` of world
    // surface and lands on `det(I + dD - bend*H)` of seabed; the light it carries is spread over the
    // second having arrived across the first. Written as one determinant instead, a choppy surface
    // would brighten the bottom of a puddle with no depth for the light to converge over.
    vec3 bend = REFRACTION_BEND_RGB * min(depth, CAUSTIC_MAX_DEPTH);
    vec3 spread = vec3(1.0, 1.0, 0.0) + WAVE_CHOPPINESS * gathering;
    float surface = spread.x * spread.y - spread.z * spread.z;
    // One determinant per channel, over a Hessian that does not depend on the channel — three
    // multiply-adds of numbers already in registers, not three passes over the spectrum.
    vec3 seabed = (spread.x - bend * hessian.x) * (spread.y - bend * hessian.y)
                - (spread.z - bend * hessian.z) * (spread.z - bend * hessian.z);
    return min(abs(surface) / max(abs(seabed), vec3(abs(surface) / CAUSTIC_MAX)), vec3(CAUSTIC_MAX));
}

// How much of the sky's own light survives the trip down to a point.
//
// **The sun was attenuated on its way down and the sky was not.** An underwater surface was being
// lit by a full-strength sky alongside a dimmed sun — brighter than either would allow and
// inconsistent between them, which is most of why the water read as clearer from above than from
// within it. Outdoors the sky is the larger of the two terms, so this is the one that decides how
// dark it gets down there.
vec3 daylight_reaching(vec3 position) {
    float depth = frame.water_level - position.z;
    if (!(depth > 0.0)) {
        return vec3(1.0);
    }
    return exp(-WATER_EXTINCTION * depth);
}

// What the sun's light has left, and how it is gathered, by the time it reaches a point under water.
//
// Two things happen on the way down and neither was modelled before: the water absorbs along the
// path, and the surface acts as a lens that pools the light into the moving lines on a seabed. The
// shadow ray already passes through the surface — water carries a mask bit that keeps it out of
// occlusion — so this is the whole of what the surface does to sunlight.
//
// Returns white above the water, and for a cell with no water at all, where `water_level` is
// negative infinity and the depth is never positive.
vec3 sun_through_water(vec3 position, float footprint) {
    float depth = frame.water_level - position.z;
    if (!(depth > 0.0)) {
        return vec3(1.0);
    }
    // The sun refracted at a flat surface: the waves' contribution to *direction* averages out over
    // the path, and what they do to the light's distribution is the caustic term below.
    vec3 downward = refract(frame.sun_direction, vec3(0.0, 0.0, 1.0), 1.0 / WATER_IOR);
    float path = depth / max(-downward.z, 0.05);
    return exp(-WATER_EXTINCTION * path) * caustic(position.xy, depth, frame.time, footprint);
}

// What a ray spawned at the water surface brings back.
//
// Shaded but not bounced, terminated with the flat ambient a diffuse bounce ends with — the light
// reaching the seabed through two more surfaces is not what anyone is looking at.
//
// **Solid geometry only.** GLSL has no recursion, so water behind water could not be shaded anyway
// — and culling it is what lets both rays start on the *viewer's* side of the plane. That matters
// more than it sounds: pushed to the far side instead, a refraction ray at a shore begins below
// ground wherever the bed is nearer the surface than the offset itself, then travels down through
// open air and reports water of infinite depth. On a gentle slope that band is metres wide, and it
// drew a flat ribbon of scattering colour along the whole waterline.
vec3 water_ray(vec3 origin, vec3 direction, float footprint, float lobe, uvec2 pixel, uint salt,
               out float travelled) {
    // The pixel's own cone, not a bounce's. **A reflection and a refraction are specular**: they
    // carry the footprint the primary ray had, where a diffuse bounce spreads over a hemisphere and
    // wants the coarse mip that goes with it. Tracing these at the bounce rate put a seabed a
    // hundred units down under a hundred-unit footprint — every texture at its top mip, and every
    // wave averaged out of the caustics that the same footprint governs.
    // The cone widens by the lobe as well as by the pixel: a reflection off water too fine to
    // resolve is blurred by the slopes that were averaged away, and what it reflects should blur
    // with it rather than staying mirror-sharp against a matt sea.
    Surface hit = trace(origin, direction, footprint, frame.cone_spread + lobe, MASK_SOLID);
    travelled = hit.hit ? hit.t : WATER_MAX_PATH;

    // **The channel is in the reflection too.** The sky's flash already came back through
    // `sky_seen_through`, so a strike lit the sea — off a sea with nothing on it that could have
    // thrown the light. A bolt over a bay is as much a line on the water as it is in the air, and it
    // is the half anyone looks at.
    //
    // **The pixel's own cone and not this ray's widened one**, which is the difference between a
    // reflected channel and a white sheet over half a bay. `lobe` is an angle; `cone_spread` is an
    // angle *per pixel*, and `BOLT_CORE` and `BOLT_HALO` are counts of pixels — so adding the two
    // put the drawn radius at a couple of times the distance to the channel instead of a couple of
    // pixels across it. That is a blob large enough to defeat the bounding test as well as to fill
    // the frame, which is why it also landed nowhere near the bolt.
    //
    // Nothing is lost by leaving the lobe out: what breaks a reflection up on water is the wave
    // normal each pixel reflects about, and that is already in `direction`. A rough sea gives back a
    // shivering broken line, which is what one does.
    //
    // Its own span, too — `travelled` is clamped to `WATER_MAX_PATH` for the absorption that reads
    // it, four thousand units, and a strike stands at fifteen to thirty-five thousand.
    vec3 arc = bolt_along(origin, direction, hit.hit ? hit.t : RAY_MAX);
    if (!hit.hit) {
        // Stars kept: a reflection is something looked *at*, so the sea shows them the way it
        // shows the sun. It is the lighting path that leaves them out.
        // **The lit deck as well as the channel.** The cloud a discharge is inside is added in
        // front of the fog rather than composited into the sky — see `flash_on_deck` — so a
        // reflection that only asks `sky_seen_through` gets the cloud's fair-weather colour and
        // none of the light in it. A bay under a bolt was giving back the bolt and not the cloud it
        // came out of, which is the brighter half of the two.
        // **The lit deck as well as the channel.** The cloud a discharge is inside is added in
        // front of the fog rather than composited into the sky — see `flash_on_deck` — so a
        // reflection that only asks `sky_seen_through` gets the cloud's fair-weather colour and
        // none of the light in it. A bay under a bolt was giving back the bolt and not the cloud it
        // came out of, which is the brighter half of the two.
        // **The lit deck as well as the channel.** The cloud a discharge is inside is added in
        // front of the fog rather than composited into the sky — see `flash_on_deck` — so a
        // reflection that only asks `sky_seen_through` gets the cloud's fair-weather colour and
        // none of the light in it. A bay under a bolt was giving back the bolt and not the cloud it
        // came out of, which is the brighter half of the two.
        return arc + flash_on_deck(direction, frame.cone_spread + lobe)
             + sky_seen_through(direction, lobe, true);
    }
    return arc + shade(hit, frame.ambient * daylight_reaching(hit.position), pixel, salt,
                       BOUNCE_SHADOW_SAMPLES);
}

// What is left of `radiance` after `path` units of water, plus what the water glows back.
//
// Beer-Lambert one way and single scattering the other: what the water takes out of a beam it
// partly returns as its own colour, which is why a deep channel is not simply darker but bluer.
vec3 absorbed_by_water(vec3 radiance, float path) {
    vec3 transmittance = exp(-WATER_EXTINCTION * path);
    // **Light that scatters toward the eye had to get down there first.** Attenuating only the way
    // back — `1 - T` — lets deep water asymptote to the scattering colour at full sky brightness,
    // which is the milky sheet a real channel is not. Integrating both legs along the path turns
    // that into `(1 - T^2) / 2`: the same answer in the shallows, half as bright where it settles,
    // and markedly less red, because squaring the transmittance costs red twice over.
    vec3 in_scattered = (1.0 - transmittance * transmittance) * 0.5;
    return radiance * transmittance + WATER_SCATTER * in_scattered * frame.ambient;
}

// Radiance leaving a water surface toward whatever traced it.
//
// Two rays and no sampling, which is the point: a mirror reflection and a refraction are
// *deterministic*, so water carries no noise and needs no denoising — which is what lets it write
// straight to the output past a filter that has no albedo to demodulate it by.
//
// The normal is flipped to face the ray so one path serves both sides of the surface; `eta` is what
// distinguishes them, and total internal reflection then falls out of `refract` returning zero
// rather than needing the critical angle spelled out.
vec3 water_shade(Surface surface, vec3 incident, uvec2 pixel, out Guides guides) {
    // Which side of the water the ray is on is a question about the *plane*, not about a wave:
    // at a glancing angle a facet can tilt far enough to face away from the ray, and reading that
    // as "the camera is underwater" sends the reflection down into the seabed and turns the far
    // water white.
    // Water is the one surface whose sides have absolute names: it is a horizontal plane, so a ray
    // travelling upward into it came from underneath, whatever the quad's winding says.
    bool from_below = incident.z > 0.0;
    vec3 flat_normal = from_below ? vec3(0.0, 0.0, -1.0) : vec3(0.0, 0.0, 1.0);

    // **How much water is under *this pixel*, which is the whole of what the shore is.** Straight
    // down rather than along anything, and it is worth a ray of its own: the two rays cast below
    // both leave at an angle, so what they measure is how far *they* travelled, which at a shore is
    // a question about the slope beyond rather than about the depth here. See `SHORE_FADE`.
    //
    // From underneath there is no shore: the distance to the bed says nothing about a surface seen
    // from below it, and the ray is spared.
    float shore = 1.0;
    if (!from_below) {
        Surface bed = trace(surface.position + flat_normal * SHADOW_BIAS, -flat_normal,
                            surface.footprint, frame.cone_spread, MASK_SOLID);
        shore = smoothstep(0.0, SHORE_FADE, bed.hit ? bed.t : WATER_MAX_PATH);
    }

    // The wave normal replaces the quad's flat one. Keyed off world position rather than anything
    // interpolated, so one cell's surface continues into the next without a seam at the boundary.
    float unresolved;
    vec3 waves = water_normal(surface.position.xy, frame.time, surface.footprint, unresolved);
    // A normal tilting by an angle turns its reflection by twice that, so the lobe the lost slopes
    // leave behind is twice their root mean square.
    float lobe = 2.0 * sqrt(unresolved);
    vec3 n = from_below ? -waves : waves;

    // A facet that still faces away is one the surface would have hidden behind the wave in front
    // of it. Tilting it back toward the plane until it faces the ray is the cheap stand-in for the
    // self-occlusion that is missing, and it is what keeps a glancing reflection finite.
    float facing = dot(-incident, n);
    float flat_facing = dot(-incident, flat_normal);
    if (facing < MIN_FACING) {
        // How far toward the plane the facet has to come back. The dot is linear in the blend, so
        // this is the exact fraction that brings it to `MIN_FACING` — solved rather than iterated.
        float t = (MIN_FACING - facing) / max(flat_facing - facing, 1e-4);
        n = normalize(mix(n, flat_normal, clamp(t, 0.0, 1.0)));
    }
    float eta = from_below ? WATER_IOR : 1.0 / WATER_IOR;

    float cosine = clamp(dot(-incident, n), 0.0, 1.0);
    float fresnel = WATER_F0 + (1.0 - WATER_F0) * pow(1.0 - cosine, 5.0);

    // Water is the whole of this world's specular response. The lobe is the spread the waves too
    // small to resolve leave behind, which is exactly what roughness means; the reflection's
    // distance is filled in below, once the ray has been cast.
    guides = Guides(vec3(fresnel), clamp(lobe, 0.0, 1.0), 0.0);

    // The refraction's distance drives absorption, and the reflection's is the specular guide — it
    // was discarded here until an upscaler needed to know how fast the reflection moves.
    // Offset along the *plane*, not the facet: what a ray has to clear to avoid finding this
    // surface again is the quad, and only the plane's normal is guaranteed to take it off that.
    // **Absorption follows whichever ray went into the water, and that flips with the side.** Seen
    // from above, the refraction dives into it and the reflection leaves into air; from below, the
    // reflection stays under and the refraction is the sky through Snell's window, which has
    // travelled no water at all. Attenuating the wrong one turns that window green.
    float mirrored;
    vec3 reflected = water_ray(surface.position + flat_normal * SHADOW_BIAS, reflect(incident, n),
                               surface.footprint, lobe, pixel, STREAM_WATER_REFLECT, mirrored);
    guides.specular_distance = mirrored;
    if (from_below) {
        reflected = absorbed_by_water(reflected, mirrored);
    }

    // **And the bending goes with it.** Water that is not there cannot refract: at the waterline
    // the ray has to leave straight, or the last pixel of water shows a piece of ground displaced
    // by a fifth of a radian from the dry pixel beside it — which is a hard line however faint the
    // surface over it has been made. Straightening it is what makes the two sides meet.
    vec3 through = normalize(mix(incident, refract(incident, n, eta), shore));
    if (dot(through, through) < 1e-6) {
        // Past the critical angle looking up from underwater, where the surface is a mirror and
        // there is nothing behind it to see.
        return reflected;
    }

    float depth;
    // Refraction bends by a third of what reflection does, so what is seen *through* the surface
    // is blurred correspondingly less by the same lost slopes.
    vec3 behind = water_ray(surface.position + flat_normal * SHADOW_BIAS, through,
                            surface.footprint, lobe * REFRACTION_BEND, pixel,
                            STREAM_WATER_REFRACT, depth);
    vec3 refracted = from_below ? behind : absorbed_by_water(behind, depth);

    // With no water left between the surface and the ground, this is the ground.
    return mix(behind, mix(refracted, reflected, fresnel), shore);
}
