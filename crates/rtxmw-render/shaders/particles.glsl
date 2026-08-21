// Flames, steam and smoke, drawn out of their emitters rather than out of geometry.
//
// **A particle system carries no triangles at all.** The sprites are the whole of the drawing, so
// there is nothing to put in the acceleration structure even if it were worth putting there — a
// candle's twenty-two flames are twenty-two sub-pixel quads that would have to be rebuilt every
// frame, in a structure whose build is per *cell*. They go into the transparency layer instead,
// beside the rain, and for the same three reasons: an upscaler composites that layer rather than
// denoising it, coverage comes out as a fraction so a sprite finer than a pixel dims it rather than
// flickering in and out of it, and the whole thing costs no state.
//
// **Nothing is simulated and nothing is stored.** Slot `i` of an emitter is a closed form in its
// own hash and the clock: where it was born, which way it left, how fast, how long it lives, and a
// constant acceleration on top. That is exact rather than approximate — a stepped simulation of a
// constant acceleration is the same parabola with rounding error — and it means a frame allocates
// nothing, keeps nothing between frames, and gives the same picture from any starting time.

// Most emitters a ray walks, whatever the host uploaded.
//
// A guard on the buffer rather than a budget: the fullest interior measured is Seyda Neen's census
// office at fifty-five, and a cell would have to be twenty times that to bind.
const uint EMITTER_LIMIT = 1024u;

// Distinct hash streams, so a slot's direction and its birth point are not the same number twice.
const uint PARTICLE_STREAM_DIRECTION = 0u;
const uint PARTICLE_STREAM_BIRTH = 1u;
const uint PARTICLE_STREAM_LIFE = 2u;

// What a slot's hash gives, unpacked.
struct Spawn {
    // Where it started, in world units from the emitter's origin.
    vec3 offset;
    // Which way it left, and how fast.
    vec3 direction;
    float speed;
    // How far through its life it is, and how long that life is.
    float age;
    float life;
};

// Four unit values out of one slot, one generation and one stream.
//
// **The generation is in the hash and that is what keeps it from repeating.** A slot cycles: it is
// born, it lives, it is born again. Hashed on the slot alone every rebirth would leave in exactly
// the same direction, and twenty-two of them would read as a fountain of identical sparks rather
// than as a flame.
vec4 particle_random(uint seed, uint slot, uint generation, uint stream) {
    uint low = hash(uvec4(seed, slot, generation, stream));
    uint high = hash(uvec4(low, stream, slot, seed));
    return vec4(unit_pair(low), unit_pair(high));
}

// Where slot `slot` of `emitter` is, and how old.
Spawn particle_spawn(Emitter emitter, uint slot, float time) {
    Spawn spawn;
    vec4 chance = particle_random(emitter.seed, slot, 0u, PARTICLE_STREAM_LIFE);
    spawn.life = max(emitter.lifetime + emitter.lifetime_variation * (2.0 * chance.x - 1.0), 1e-3);
    // **The phase is the slot's own, so the emitter is full from the first frame.** A rate and a
    // capacity would have to be integrated from a start time to say which slots exist yet; a
    // scattered phase gives the same steady state with no start time at all, and a candle that has
    // been burning since before the cell loaded is what a candle is.
    float cycles = time / spawn.life + chance.y;
    uint generation = uint(max(floor(cycles), 0.0));
    spawn.age = fract(cycles) * spawn.life;

    vec4 aim = particle_random(emitter.seed, slot, generation, PARTICLE_STREAM_DIRECTION);
    // A polar angle off the emitter's own `+Z` and an azimuth about it — zero declination is
    // straight up, which is what 113 of the game's emitters are and why they are called flames.
    float declination = emitter.declination + emitter.declination_variation * (2.0 * aim.x - 1.0);
    float azimuth = emitter.azimuth + emitter.azimuth_variation * (2.0 * aim.y - 1.0);
    vec3 local = vec3(sin(declination) * cos(azimuth), sin(declination) * sin(azimuth),
                      cos(declination));
    spawn.direction = local.x * emitter.axis_x + local.y * emitter.axis_y
                    + local.z * emitter.axis_z;
    spawn.speed = emitter.speed + emitter.speed_variation * (2.0 * aim.z - 1.0);

    vec4 born = particle_random(emitter.seed, slot, generation, PARTICLE_STREAM_BIRTH);
    spawn.offset = emitter.spread_x * (2.0 * born.x - 1.0) * emitter.axis_x
                 + emitter.spread_y * (2.0 * born.y - 1.0) * emitter.axis_y
                 + emitter.spread_z * (2.0 * born.z - 1.0) * emitter.axis_z;
    return spawn;
}

// How wide a particle is drawn at `age`, as a fraction of the emitter's own size.
//
// `NiParticleGrowFade`, which 245 of the 272 files that emit anything carry: in over `grow`
// seconds, out over `fade`, and full size in between. The twenty-seven without it are full size for
// their whole life, which is what a grow and a fade of zero come out as here.
float particle_ramp(Emitter emitter, float age, float life) {
    float rising = emitter.grow > 0.0 ? clamp(age / emitter.grow, 0.0, 1.0) : 1.0;
    float falling = emitter.fade > 0.0 ? clamp((life - age) / emitter.fade, 0.0, 1.0) : 1.0;
    return rising * falling;
}

// The emitter's colour ramp at `fraction` of a life, which is where a puff of smoke goes out.
//
// Two linear segments, which reproduces the shipped ramps exactly rather than approximating them:
// every one of the game's 85 is three linear keys, and their middle key sits anywhere from a
// twentieth of a life to nine tenths of one.
vec4 particle_tint(Emitter emitter, float fraction) {
    if (fraction < emitter.ramp_mid) {
        return mix(emitter.ramp[0], emitter.ramp[1], fraction / max(emitter.ramp_mid, 1e-4));
    }
    return mix(emitter.ramp[1], emitter.ramp[2],
               (fraction - emitter.ramp_mid) / max(1.0 - emitter.ramp_mid, 1e-4));
}

// Everything one emitter puts along the ray, added into `colour` and taken out of `through`.
void emitter_along(Emitter emitter, vec3 origin, vec3 direction, float span, vec3 right, vec3 up,
                   inout vec3 colour, inout float through) {
    vec3 to = emitter.origin - origin;
    // The emitter's whole reach against the segment the ray actually covers — which is why this is
    // clamped rather than taken at the unconstrained nearest approach: an emitter behind the eye,
    // or behind the surface the ray already hit, is not on the ray at all.
    float nearest = clamp(dot(to, direction), 0.0, span);
    if (distance(to, direction * nearest) > emitter.reach) {
        return;
    }

    uint id = materials[emitter.material].base_colour;
    if (id == NO_TEXTURE) {
        return;
    }
    uint slot = colour_slot(id);
    float texels = float(textureSize(textures[nonuniformEXT(slot)], 0).x);

    for (uint index = 0u; index < emitter.count; ++index) {
        Spawn spawn = particle_spawn(emitter, index, frame.time);
        float radius = 0.5 * emitter.size * particle_ramp(emitter, spawn.age, spawn.life);
        if (radius <= 0.0) {
            continue;
        }
        // A parabola, taken whole rather than stepped: constant acceleration has a closed form and
        // integrating it per frame would only add drift.
        vec3 centre = emitter.origin + spawn.offset
                    + spawn.direction * (spawn.speed * spawn.age)
                    + 0.5 * emitter.gravity * spawn.age * spawn.age;

        vec3 relative = centre - origin;
        float distance_along = dot(relative, direction);
        if (distance_along <= 0.0 || distance_along >= span) {
            continue;
        }
        vec3 offset = relative - direction * distance_along;
        // A square rather than a disc: the sprite is a quad and its texture is painted to the
        // corners, so a circular test would throw away the parts of a flame that are furthest out.
        float turn = emitter.spin * spawn.age;
        vec2 local = vec2(dot(offset, right), dot(offset, up));
        local = vec2(local.x * cos(turn) - local.y * sin(turn),
                     local.x * sin(turn) + local.y * cos(turn));
        if (max(abs(local.x), abs(local.y)) >= radius) {
            continue;
        }

        // The sprite covers `2 * radius` across at `distance_along`, and a pixel covers
        // `cone_spread * distance_along` — so the ratio is how many texels fall in a pixel.
        float footprint = frame.cone_spread * distance_along;
        float lod = log2(max(footprint / (2.0 * radius) * texels, 1.0));
        vec2 uv = 0.5 + 0.5 * local / radius;
        vec4 texel = textureLod(textures[nonuniformEXT(slot)], uv, lod);
        vec4 over = emitter.colour * particle_tint(emitter, spawn.age / spawn.life);
        float alpha = clamp(texel.a * over.a * materials[emitter.material].opacity, 0.0, 1.0);
        if (alpha <= 0.0) {
            continue;
        }
        vec3 tint = texel.rgb * over.rgb;

        // **What adds does not occlude, and that is the whole reason it needs no sorting.** An
        // additive sprite leaves the frame behind it whole and piles its own light on top, so the
        // order these arrive in cannot change the sum — which is what makes an unsorted walk of a
        // flame exact rather than approximate. A puff of smoke is the other case: it covers what is
        // behind it, so it takes the transmittance down and shows the room's own light back.
        //
        // **No gain on either, deliberately.** The blend the file asks for is `SRC_ALPHA, ONE` on
        // 473 of the game's 678 emitters, which says exactly this much and no more — the texel *is*
        // the radiance, so a flame comes out sixty times the mean of the room it stands in, because
        // that is what a flame is, and the exposure downstream decides where it lands. A puff is
        // the same argument with the ramp's own colour standing in for an albedo, the way §8.87
        // gives ash a seventh: what a grain sends back is what the room sent it.
        if (emitter.additive > 0.0) {
            colour += tint * alpha;
        } else {
            colour += tint * frame.ambient * (alpha * through);
            through *= 1.0 - alpha;
        }
    }
}

// Everything every emitter puts along the ray: premultiplied colour, and what is left of the ray.
//
// Matches `precipitation_along`, which the caller sums with — see the note there.
vec4 particles_along(vec3 origin, vec3 direction, float span) {
    vec3 colour = vec3(0.0);
    float through = 1.0;
    if (frame.emitter_count == 0u) {
        return vec4(colour, through);
    }
    // One basis for the whole ray, so a sprite is square to the screen rather than to the world.
    // Built off the ray instead of the camera because the two differ by less than a pixel of turn
    // and this needs no matrix.
    vec3 right = normalize(cross(direction, abs(direction.z) < 0.99 ? vec3(0.0, 0.0, 1.0)
                                                                   : vec3(1.0, 0.0, 0.0)));
    vec3 up = cross(right, direction);

    uint count = min(frame.emitter_count, EMITTER_LIMIT);
    for (uint index = 0u; index < count; ++index) {
        emitter_along(emitters[index], origin, direction, span, right, up, colour, through);
    }
    return vec4(colour, through);
}
