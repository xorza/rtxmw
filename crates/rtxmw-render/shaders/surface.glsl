// What a ray finds, and what it costs to find it.
//
// Fetching a hit's attributes, choosing the mip its cone can resolve, deciding whether a cutout
// keeps it, and the two traversals themselves — `occluded` for a shadow ray that only asks whether
// anything is in the way, and `trace` for one that has to come back with a surface.

// A raytracer has no near plane — that is a rasterization artefact. This is only large enough to
// keep a ray from hitting the surface it starts on, and is deliberately far below the camera's
// near plane so nothing is clipped just for being close.
const float RAY_MIN = 0.01;

// Far enough to cross an exterior cell, which is 8192 world units.
const float RAY_MAX = 1.0e6;

// How far along the normal a shadow ray starts. Morrowind's unit is about 1.4 cm, so this is a
// couple of centimetres — enough to clear the interpolation error on a large triangle without
// letting light leak under a wall.
const float SHADOW_BIAS = 1.5;

// The three shared-stream vertex indices of a hit triangle.
//
// `first_index` locates the run inside the shared index buffer and `first_vertex` rebases the
// mesh-local index values onto the shared attribute and position streams — the same two offsets the
// acceleration structure was built with.
uvec3 triangle_vertices(Geometry geom, uint primitive) {
    uint triangle = geom.first_index + 3u * primitive;
    return uvec3(geom.first_vertex + indices[triangle],
                 geom.first_vertex + indices[triangle + 1u],
                 geom.first_vertex + indices[triangle + 2u]);
}

// A hit's three corner weights. Barycentrics come back as the second and third; the first is what
// is left.
vec3 corner_weights(vec2 bary) {
    return vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
}

vec2 interpolate_uv(uvec3 verts, vec3 weights) {
    return attributes[verts.x].uv * weights.x
         + attributes[verts.y].uv * weights.y
         + attributes[verts.z].uv * weights.z;
}

vec3 interpolate_normal(uvec3 verts, vec3 weights) {
    return attributes[verts.x].normal * weights.x
         + attributes[verts.y].normal * weights.y
         + attributes[verts.z].normal * weights.z;
}

// Mip level for a hit, from the width of the ray's cone where it landed.
//
// A rasterizer takes this from screen-space derivatives, which a compute shader does not have, so
// the level is reconstructed instead: `footprint` is how wide the cone is at the hit, spread
// further by the angle the surface is seen at, and the mip whose texels match that width is the one
// to sample. Without it every surface samples mip zero, and anything dense or grazing turns to
// static.
//
// This is Akenine-Möller's ray-cone method, minus the curvature term, which needs a second
// derivative the primary ray has no cheap way to get.
float cone_lod(uvec3 verts, mat4x3 to_world, vec3 direction, float footprint,
               uint texture_slot) {
    // World space throughout. The footprint is a world-space length, so an object-space area here
    // would be wrong by the instance's scale — silently, and only for scaled instances.
    vec3 p0 = to_world * vec4(positions[verts.x], 1.0);
    vec3 p1 = to_world * vec4(positions[verts.y], 1.0);
    vec3 p2 = to_world * vec4(positions[verts.z], 1.0);
    vec3 face = cross(p1 - p0, p2 - p0);
    float world_area = length(face);
    vec2 t0 = attributes[verts.x].uv;
    vec2 t1 = attributes[verts.y].uv;
    vec2 t2 = attributes[verts.z].uv;
    float uv_area = abs((t1.x - t0.x) * (t2.y - t0.y) - (t2.x - t0.x) * (t1.y - t0.y));
    if (world_area <= 0.0 || uv_area <= 0.0) {
        return 0.0;
    }

    // How much of the texture one world unit covers, in texels.
    vec2 size = vec2(textureSize(textures[nonuniformEXT(texture_slot)], 0));
    float texels_per_unit = sqrt(uv_area / world_area) * max(size.x, size.y);

    // Widened as the surface tilts away: at grazing incidence a pixel covers far more surface,
    // which is exactly where the aliasing is worst. Measured against the triangle's own plane,
    // which is what the footprint actually lands on — and which spares every caller a normal.
    float tilted = footprint / max(abs(dot(face / world_area, direction)), 0.02);
    return max(0.0, log2(tilted * texels_per_unit));
}

// What the map's byte range spans. Declared again as the top of `RANGE` in `shading_map.rs`, which
// pins the two with a test, because a GLSL shader cannot see a Rust constant.
const float SHADING_SCALE = 2.0;

// What a texture had painted into it, at the point `uv` names.
//
// **Scaled back up, because the map is stored divided down**: its values run past one and an sRGB
// byte only holds `0..1`. Neutral is one, so `frame.delight` of zero leaves the texture exactly as
// it was.
//
// Always mip zero: the map is a single level of thirty-two squared, and what it describes varies
// across the whole texture rather than between neighbouring texels.
float baked_shading(uint id, vec2 uv) {
    if (frame.delight <= 0.0) {
        return 1.0;
    }
    float shading =
        textureLod(textures[nonuniformEXT(shading_slot(id))], uv, 0.0).r * SHADING_SCALE;
    return mix(1.0, shading, frame.delight);
}

// The material's base colour texel at `uv`, with the lighting painted into it divided out.
//
// **Vanilla textures have shading, ambient occlusion and directional light painted in**, authored
// for a renderer with none of its own — see `docs/design.md` §5.1. Tracing over that lights every
// surface twice. Dividing by the estimate removes the low-frequency part of it, which is the part
// that came from lighting rather than from the artist's detail.
//
// `nonuniformEXT` because neighbouring rays in a warp land on different materials, and an index
// that varies across the warp is undefined without it.
vec4 base_colour(Material material, vec2 uv, float lod) {
    vec4 texel = textureLod(textures[nonuniformEXT(colour_slot(material.base_colour))], uv, lod);
    texel.rgb /= baked_shading(material.base_colour, uv);
    return texel;
}

// The four ground textures a `KIND_TERRAIN` material names, as slots in the bindless array.
//
// Sixteen bits apiece in the two words `material_buffers.rs` packs them into. Returned as ids
// rather than slots, because each one addresses a colour and a shading map — see `colour_slot`.
uint[4] terrain_layers(Material material) {
    return uint[4](material.terrain_layers0 & 0xFFFFu,
                   material.terrain_layers0 >> 16,
                   material.terrain_layers1 & 0xFFFFu,
                   material.terrain_layers1 >> 16);
}

// How much of each of the four tiles nearest a point on the ground it draws.
//
// **A cell names one texture per 512-unit tile and nothing in between.** Drawn as it is written,
// each tile meets its neighbours along a straight line, and a coast comes out as a patchwork of
// squares. Real ground has no such edges, and neither did the original engine's — it blended.
//
// Bilinear between tile *centres*, which is why the half-tile offset is here: a point exactly on a
// centre draws that tile alone, and one on the seam between two draws half of each. The cell's own
// origin never enters it — a cell is a whole number of tiles across, so it falls out of the
// fractional part and the ground is continuous across a cell boundary for free.
//
// **Not a ramp across the whole tile.** Interpolating centre to centre leaves no point on the map
// drawing a single texture — every one is a mix of four, and a tile reads as a translucent square
// laid over its neighbours rather than as ground. The original engine blends through a map of two
// texels per tile, each tile's own pair at full weight, so bilinear filtering confines the
// transition to the 256 units straddling the tile boundary and leaves the middle half of every tile
// pure. `components/esmterrain/storage.cpp:497` is where OpenMW says so — "upscale the blendmap 2x
// with nearest neighbor sampling to look like Vanilla".
//
// The ramp is that transition, written directly: flat for the first quarter, across over the middle
// half, flat again for the last quarter.
vec4 tile_weights(vec2 world) {
    vec2 weight = clamp(fract(world / TERRAIN_TILE - 0.5) * 2.0 - 0.5, 0.0, 1.0);
    return vec4((1.0 - weight.x) * (1.0 - weight.y), weight.x * (1.0 - weight.y),
                (1.0 - weight.x) * weight.y, weight.x * weight.y);
}

// The ground's colour, blended across those four.
//
// Four samples where an unblended ground took one. They are of the same texture more often than
// not: most of a cell is interior to a tile whose neighbours match it, and the sampler's cache is
// what makes that cheap.
vec3 ground_colour(Material material, vec2 world, vec2 uv, float lod) {
    uint layers[4] = terrain_layers(material);
    vec4 across = tile_weights(world);
    vec3 colour = vec3(0.0);
    for (int layer = 0; layer < 4; ++layer) {
        uint id = layers[layer];
        vec3 tile = textureLod(textures[nonuniformEXT(colour_slot(id))], uv, lod).rgb;
        colour += across[layer] * tile / baked_shading(id, uv);
    }
    return colour;
}

// The ground's painted relief, blended across the same four tiles the colour is.
//
// The gradients are blended rather than the heights: a height is only defined up to the constant
// each tile was painted around, so a mix of four of them steps wherever the constants differ, and a
// derivative of that mix is a wall along every tile boundary. A mix of the four derivatives has no
// such term in it.
//
// **Terrain never carries a cutout**, so no alpha gate — the ground is opaque everywhere.
vec2 ground_relief(Material material, vec2 world, vec2 uv, float lod) {
    uint layers[4] = terrain_layers(material);
    vec4 across = tile_weights(world);
    vec2 gradient = vec2(0.0);
    for (int layer = 0; layer < 4; ++layer) {
        gradient += across[layer] * relief_gradient(colour_slot(layers[layer]), uv, lod, 0.0);
    }
    return gradient;
}

// What an upscaler needs about a surface beyond its colour.
//
// Separate from `Surface` because it is a different question: `Surface` is what the geometry and the
// material say, and this is what a temporal filter needs in order to know how a *reflection* on that
// surface moves — which is not the same as how the surface itself moves. DLSS Ray Reconstruction
// wants all three; §5.2 records that the guide is easy to forget and awkward to add late, which is
// why it is written now, before there is anything reading it.
struct Guides {
    // The mirror-like part of the surface's response. Zero across vanilla Morrowind, whose materials
    // carry no specular at all — `NiSpecularProperty` is force-disabled at this NIF version — so
    // water is the only thing in the world that fills this in.
    vec3 specular_albedo;
    // How sharp that part is: one for a surface that reflects nothing coherently, toward zero for a
    // mirror.
    float roughness;
    // How far the specular ray travelled. **This is the guide proper**: a reflection moves across
    // the screen at a rate set by the distance to what is reflected rather than to the surface, so a
    // filter given only depth reprojects every mirror wrongly. Zero where nothing was reflected.
    float specular_distance;
};

// What the guides say a pixel reflects where nothing was hit.
//
// Mid grey rather than black: an upscaler demodulates by this, and zero would claim the sky absorbs
// everything reaching it. `DLSS-RR Integration Guide` §3.4.2 names the value.
#define SKY_GUIDE_ALBEDO 0.5

// What every surface in a diffuse world says: nothing reflects, and nothing is reflected.
Guides matte() {
    return Guides(vec3(0.0), 1.0, 0.0);
}

// Whether a candidate intersection survives its material's alpha cutout.
//
// Only non-opaque geometry reaches this: the build marks a run `OPAQUE` when its material is, and
// traversal then commits without ever asking.
bool alpha_passes(uint slot, uint primitive, vec2 bary, mat4x3 to_world, vec3 direction,
                  float footprint) {
    Geometry geom = geometries[slot];
    Material material = materials[geom.material];
    if (material.alpha_cutoff <= 0.0 || material.base_colour == NO_TEXTURE) {
        return true;
    }
    uvec3 verts = triangle_vertices(geom, primitive);
    vec2 uv = interpolate_uv(verts, corner_weights(bary));
    // **The level the cone can actually resolve.** A mask point-sampled at its finest mip answers
    // for one texel out of the hundreds a distant pixel covers, and a binary test on that is a coin
    // toss per pixel: the fringe of a rug comes out as crawling sparks and a tree as speckle.
    // Averaging first costs the edge some of its bite — the alpha a leaf was authored with softens
    // — and that is the better of the two errors by a long way.
    //
    // A footprint of zero means the caller has no cone to resolve with, which is every shadow ray;
    // those keep the finest level, and skip the work of asking for it.
    float lod = footprint > 0.0
              ? cone_lod(verts, to_world, direction, footprint, colour_slot(material.base_colour))
              : 0.0;
    float alpha = base_colour(material, uv, lod).a;
    return alpha >= material.alpha_cutoff;
}

// Whether anything blocks the segment from `origin` along `dir` within `reach` units.
//
// `TerminateOnFirstHit` because a shadow ray only asks whether *something* is in the way — the
// nearest blocker is no more useful than any other, and stopping at the first is most of why a
// shadow ray is cheaper than a visibility ray.
bool occluded(vec3 origin, vec3 dir, float reach) {
    rayQueryEXT query;
    // Solid geometry only. Water carries a mask bit of its own and is skipped here in hardware:
    // sunlight reaching a seabed has passed through the surface, and a sea that occluded would
    // black out every shallow in the game. `MASK_SOLID` in `scene_acceleration.rs`, pinned there
    // by a test, because a shader cannot see a Rust constant.
    rayQueryInitializeEXT(query, scene, gl_RayFlagsTerminateOnFirstHitEXT, MASK_SOLID,
                          origin, RAY_MIN, dir, reach);
    // The same candidate loop as `trace`, and it cannot be shared: `glslc` rejects `rayQueryEXT` as
    // an `out` or `inout` parameter, so a traversal cannot be handed to a function. Any change to
    // the cutout here has to be made in both places.
    while (rayQueryProceedEXT(query)) {
        if (rayQueryGetIntersectionTypeEXT(query, false)
            == gl_RayQueryCandidateIntersectionTriangleEXT) {
            uint slot = rayQueryGetIntersectionInstanceCustomIndexEXT(query, false)
                      + rayQueryGetIntersectionGeometryIndexEXT(query, false);
            // A cutout casts the shadow of what survives it, not of its bounding quad — a grate
            // has to throw bars rather than a rectangle.
            //
            // No cone here: a shadow ray carries no footprint, so its cutout is still decided at
            // the finest level. A leaf's shadow is a place aliasing is worth far less than it is
            // on the leaf itself.
            if (alpha_passes(slot,
                             rayQueryGetIntersectionPrimitiveIndexEXT(query, false),
                             rayQueryGetIntersectionBarycentricsEXT(query, false),
                             rayQueryGetIntersectionObjectToWorldEXT(query, false),
                             dir,
                             0.0)) {
                rayQueryConfirmIntersectionEXT(query);
            }
        }
    }
    return rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

// What a ray found, resolved down to the shading inputs at the hit.
struct Surface {
    vec3 position;
    vec3 normal;
    vec3 albedo;
    vec3 emissive;
    // How wide the ray's cone was where it landed, so a ray spawned here can carry it on.
    float footprint;
    // Distance from the ray's origin, which for a primary ray is the depth a filter compares.
    float t;
    bool hit;
    // Water shades through `water_shade` rather than `shade`, and has no albedo of its own.
    bool water;
    // The vertex normal the mesh interpolated, before any relief a texture painted into it tilted
    // it — the same vector as `normal` on an untextured surface.
    //
    // **What an upscaler is guided by.** A guide normal answers "which surface is this pixel on",
    // which is what history is reprojected and rejected against, and painted relief is not a
    // different surface: it is detail inside one, and it is already in the albedo the upscaler has.
    // Handing it the tilted normal costs Ray Reconstruction most of its temporal accumulation — the
    // settled error at DLAA goes from 0.0093 to 0.0126 against a converged reference that has the
    // relief in it, so the frame is measurably *less* like the truth for having described the
    // surface in more detail. See `docs/design.md` §8.90.
    vec3 interpolated;
    // The plane the triangle actually lies in, as wound — with no side chosen for it.
    //
    // **Every ray leaving a surface is offset along this and not along `normal`.** A normal comes
    // from the vertices and is interpolated across the face, which is what a surface should be
    // *shaded* by; on Morrowind's low-poly rocks it can point tens of degrees away from the
    // triangle it belongs to. Offsetting along it puts the origin under the surface on some
    // triangles and over it on others, and a shadow ray that starts underneath finds the surface
    // it started from — which is the speckle that covers a smooth-shaded rock and the sparkle
    // along the edge of a rug.
    //
    // Which side a given ray should leave from is not a property of the surface, so it is not
    // decided here — see `leaving`.
    vec3 geometric;
    // A sheet rather than the skin of something solid, which is what makes it lit from both sides.
    bool thin;
};

// Where a ray leaving `surface` toward `towards` should start.
//
// Off the triangle's own plane, on the side the ray is *going* — which is not always the side the
// viewer is on. Morrowind hangs single-sided planes everywhere, and a shadow ray from the back of a
// sail heads for a sun on the far side: started on the viewer's side it would be stopped by the
// sail itself, and every such surface goes black. Deciding this per ray rather than per surface is
// what lets the shading normal face the viewer without taking the lighting with it.
vec3 leaving(Surface surface, vec3 towards) {
    vec3 side = faceforward(surface.geometric, -towards, surface.geometric);
    return surface.position + side * SHADOW_BIAS;
}

// Traces one ray and resolves the material at whatever it hits.
//
// The ray query lives entirely inside here, which is what lets the primary ray and a bounce ray
// share one traversal: GLSL forbids passing a query to a function, but not owning one in a function
// that returns a result.
//
// `cone_width` is the footprint the ray already carries at its origin and `cone_spread` how fast it
// widens per unit travelled; together they pick the mip a hit samples.
Surface trace(vec3 origin, vec3 direction, float cone_width, float cone_spread, uint mask) {
    // Every field has to hold something before the miss returns below, and on a miss none of them
    // is read — `hit` is what a caller branches on. So these are zeroes rather than plausible
    // stand-ins, which would invite someone to use one.
    Surface surface;
    surface.position = vec3(0.0);
    surface.normal = vec3(0.0);
    surface.interpolated = vec3(0.0);
    surface.albedo = vec3(0.0);
    surface.emissive = vec3(0.0);
    surface.footprint = 0.0;
    surface.t = 0.0;
    surface.hit = false;
    surface.water = false;
    surface.geometric = vec3(0.0, 0.0, 1.0);
    surface.thin = false;

    // No blanket opacity flag: the per-geometry `OPAQUE` bits the build set from each material are
    // what decide this. Forcing it here would override them and put the cutout back.
    rayQueryEXT query;
    rayQueryInitializeEXT(query, scene, gl_RayFlagsNoneEXT, mask,
                          origin, RAY_MIN, direction, RAY_MAX);
    // Duplicated in `occluded`, which the language forces — see the note there.
    while (rayQueryProceedEXT(query)) {
        if (rayQueryGetIntersectionTypeEXT(query, false)
            == gl_RayQueryCandidateIntersectionTriangleEXT) {
            uint slot = rayQueryGetIntersectionInstanceCustomIndexEXT(query, false)
                      + rayQueryGetIntersectionGeometryIndexEXT(query, false);
            // The cone as it is at *this* candidate, which is what decides how much of the mask
            // one pixel is looking at.
            float reach = rayQueryGetIntersectionTEXT(query, false);
            if (alpha_passes(slot,
                             rayQueryGetIntersectionPrimitiveIndexEXT(query, false),
                             rayQueryGetIntersectionBarycentricsEXT(query, false),
                             rayQueryGetIntersectionObjectToWorldEXT(query, false),
                             direction,
                             cone_width + reach * cone_spread)) {
                rayQueryConfirmIntersectionEXT(query);
            }
        }
    }
    if (rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        return surface;
    }
    surface.hit = true;

    // The custom index is where this instance's mesh starts in the flat geometry table, and the
    // geometry index is which of its runs was hit. Their sum is the entry, with no indirection.
    uint slot = rayQueryGetIntersectionInstanceCustomIndexEXT(query, true)
              + rayQueryGetIntersectionGeometryIndexEXT(query, true);
    Geometry geom = geometries[slot];
    Material material = materials[geom.material];
    float hit_t = rayQueryGetIntersectionTEXT(query, true);
    // Fetched once and shared: the texture level needs the same three vertices the shading
    // attributes come from.
    uvec3 verts = triangle_vertices(geom, rayQueryGetIntersectionPrimitiveIndexEXT(query, true));
    vec3 weights = corner_weights(rayQueryGetIntersectionBarycentricsEXT(query, true));

    // The normal is interpolated in object space, so it needs the instance's own transform.
    // Morrowind scales uniformly, which is why the plain matrix works here — a non-uniform scale
    // would need the inverse transpose instead.
    mat4x3 to_world = rayQueryGetIntersectionObjectToWorldEXT(query, true);
    surface.normal = normalize(mat3(to_world) * interpolate_normal(verts, weights));
    // The triangle's own plane, turned to face the ray. Morrowind scales uniformly, so the plain
    // matrix carries a cross product correctly here for the same reason it carries the normal.
    surface.geometric = normalize(mat3(to_world) * cross(positions[verts.y] - positions[verts.x],
                                                        positions[verts.z] - positions[verts.x]));
    // Turned to the side the ray came from, so a surface met from behind is shaded as its back
    // rather than reporting the light landing on its front. Morrowind's foliage is thousands of
    // single cards wound every which way, and without this two neighbouring pixels landing on
    // oppositely wound cards come back at opposite brightnesses — the dust over every tree.
    //
    // **Decided by the triangle's plane, not by the normal being turned.** An interpolated normal
    // near a silhouette can point away from a face that is squarely toward the viewer, so testing
    // it against the ray flips part of a surface and not the rest, along a seam that slides as the
    // camera moves. A rug seen at a grazing angle gets a hard band of wrong shading across it.
    // The plane cannot disagree with itself that way: either the ray met the front of the triangle
    // or it met the back.
    surface.normal = faceforward(surface.normal, direction, surface.geometric);
    surface.interpolated = surface.normal;
    surface.position = origin + direction * hit_t;
    surface.footprint = cone_width + hit_t * cone_spread;
    surface.t = hit_t;
    surface.thin = (geom.flags & GEOMETRY_THIN) != 0u;
    surface.water = material.kind == KIND_WATER;
    if (surface.water) {
        // Nothing to demodulate by and nothing to light: `water_shade` produces the whole of it,
        // and neither the material's colour nor its texture has any say.
        surface.albedo = vec3(0.0);
        surface.emissive = vec3(0.0);
        return surface;
    }

    surface.emissive = material.emissive;
    surface.albedo = material.diffuse;
    // The relief the texture has painted into it, applied to the shading normal — see
    // `relief.glsl`. Nothing to read it from where the material has no texture, and nothing to
    // apply it to where the triangle's uvs are degenerate.
    if (material.kind == KIND_TERRAIN) {
        // One lod for all four, chosen from the first: they are sampled over the same footprint
        // with the same uv, and every land texture the game ships is 256 square.
        float lod = cone_lod(verts, to_world, direction, surface.footprint,
                             colour_slot(terrain_layers(material)[0]));
        vec2 uv = interpolate_uv(verts, weights);
        surface.albedo *= ground_colour(material, surface.position.xy, uv, lod);
        if (frame.relief > 0.0) {
            surface.normal = relief_normal(surface.normal, surface.geometric, direction,
                                           surface_tangents(verts, to_world),
                                           ground_relief(material, surface.position.xy, uv, lod));
        }
    } else if (material.base_colour != NO_TEXTURE) {
        float lod = cone_lod(verts, to_world, direction, surface.footprint,
                             colour_slot(material.base_colour));
        vec2 uv = interpolate_uv(verts, weights);
        surface.albedo *= base_colour(material, uv, lod).rgb;
        if (frame.relief > 0.0) {
            vec2 gradient = relief_gradient(colour_slot(material.base_colour), uv, lod,
                                            material.alpha_cutoff);
            surface.normal = relief_normal(surface.normal, surface.geometric, direction,
                                           surface_tangents(verts, to_world), gradient);
        }
    }
    return surface;
}
