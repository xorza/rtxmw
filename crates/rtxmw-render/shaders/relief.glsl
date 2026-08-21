// The relief an artist painted into a texture, read back as a tilt on the normal it is shaded by.
//
// **Vanilla Morrowind ships no normal map and cannot carry one** — `docs/design.md` §5.1 — so a
// brick wall and a sheet of glass are the same shape to a tracer, and every one of them is lit as
// the flat triangle it is. What the wall does have is a painter's account of its own relief: the
// mortar between the stones was drawn dark because it is deep, and the stone faces light because
// they stand proud. Read as a height field, that account is the map the format never had.
//
// **The height is the log of luminance, and the gradient of it is the answer.** Log because a
// painted shadow multiplies the pigment under it rather than being added to it, so a ratio is what
// carries the shape and a difference of logs is invariant to how bright or dark the texture was
// painted overall. And a gradient rather than an integrated height, because a normal map is not
// required to be integrable and solving for one would only throw the answer away again.
//
// **The chromaticity gate `docs/design.md` §5.1 planned for is refused, by measurement.** Retinex
// separates shading from pigment on the premise that painted relief is achromatic while a change of
// pigment is not. Across fifty of the shipped textures the mean chromaticity step *rises* with the
// luminance step it accompanies — 0.003 where the luminance barely moves, 0.059 across the
// strongest edges, measured between neighbours both above five percent grey. The strongest edges
// are the most chromatic ones, so a gate keyed on colour suppresses exactly the mortar lines and
// carved mouldings that carry the shape. Two causes compound: BC1 quantises a block to a segment
// between two 5-6-5 endpoints, which cannot darken without moving off grey unless that segment
// happens to lie along it, and the art shadows with a cooler pigment rather than with less of the
// same one. Gating on it flattened the stonework and left noise where the relief had been.

// Linear luminance added before the log, so a black texel does not send the height to minus
// infinity and take the whole gradient with it.
//
// One sRGB step above nothing: BC1's endpoints quantise hardest at the bottom of the range, where a
// pure ratio between two nearly-black texels is enormous and means nothing.
const float RELIEF_BLACK = 1.0 / 255.0;

// Tangent-space slope per unit of log-luminance gradient, in texel widths.
//
// The frame the slope is applied in is normalised to one texel, so this is the whole of the height
// scale: relief on a texture stretched twice as wide across a wall comes out twice as tall and
// twice as broad, which leaves the slope where it was. That is what a height field scaled in three
// axes at once does, and it is why nothing here needs the surface's world size.
const float RELIEF_SLOPE = 0.7;

// Steepest tilt the slope is compressed toward, as a tangent — a little over fifty degrees.
//
// The gradient has no bound of its own: a cutout's silhouette or a single quantised black texel
// against a lit one produces a step the log turns into several units at once, and a normal laid
// flat against the surface both shades wrongly and disagrees with the plane every secondary ray
// leaves along.
//
// **Compressed rather than clamped.** A clamp is flat past its knee, so every texel of a strongly
// painted edge comes back with the same slope and the edge renders as one facet with a crease down
// it. `tanh` has the same bound and no knee — linear where the gradient is ordinary, which is where
// nearly all of a texture lives, and asymptotic only where it would have run away.
const float RELIEF_MAX_SLOPE = 1.2;

// How far above the triangle's own plane a perturbed normal is held.
//
// A tilt large enough to push the shading normal below the surface lights a face by a source behind
// it, while the shadow ray leaving along the plane finds the surface in the way — one goes bright
// and the other black, along whichever contour the two disagree on.
const float RELIEF_HORIZON = 0.05;

// The world-space directions the texture's two axes run in across a hit triangle.
//
// From the triangle's own winding rather than from a stored tangent: the NIF format at this version
// carries none, and three positions and three uvs determine the frame exactly. Zero where the uvs
// are degenerate — a triangle mapped to a point or a line has no such directions, and no relief.
struct SurfaceTangents {
    vec3 along_u;
    vec3 along_v;
};

SurfaceTangents surface_tangents(uvec3 verts, mat4x3 to_world) {
    // The w of zero is what makes these directions rather than points: the translation drops out
    // and what is left is the instance's rotation and scale.
    vec3 edge1 = to_world * vec4(positions[verts.y] - positions[verts.x], 0.0);
    vec3 edge2 = to_world * vec4(positions[verts.z] - positions[verts.x], 0.0);
    vec2 duv1 = attributes[verts.y].uv - attributes[verts.x].uv;
    vec2 duv2 = attributes[verts.z].uv - attributes[verts.x].uv;
    float det = duv1.x * duv2.y - duv2.x * duv1.y;
    if (abs(det) < 1.0e-12) {
        return SurfaceTangents(vec3(0.0), vec3(0.0));
    }
    float inverse = 1.0 / det;
    return SurfaceTangents((edge1 * duv2.y - edge2 * duv1.y) * inverse,
                           (edge2 * duv1.x - edge1 * duv2.x) * inverse);
}

float relief_height(vec4 texel) {
    return log(dot(texel.rgb, LUMA) + RELIEF_BLACK);
}

// How the painted height changes across one texel of `slot`, in texels, at the point `uv` names.
//
// **Four taps, half a texel out on each diagonal, which is a three-by-three Sobel exactly.** A
// bilinear tap centred on a corner returns the mean of the four texels around it, and differencing
// those four means gives the same weights Sobel does — the eight neighbours at one, two, one — for
// half the fetches and with the averaging done by the sampler. A plain central difference of two
// taps is cheaper still and measurably noisier: these textures are 256 square and grainy, and a
// derivative taken from a single pair of texels carries all of that grain into the normal.
//
// The offsets are a texel of the *level being read*, so relief coarsens with distance exactly as
// the colour it is derived from does. That is what a normal map's own mip chain is for, and it
// comes free here because there is no second map to build. Held to the levels the texture actually
// has, because `textureLod` clamps to the last one and offsets measured past it would straddle a
// level that was never filtered — a wide stencil over unsmoothed texels, which is noise.
//
// **The texture as shipped, not the de-lit one.** What `baked_shading` divides out is a thirty-two
// square estimate of the light the texture was painted under, which is flat across any four texels
// a gradient is taken over — so dividing first would cancel out of the difference and cost four
// more fetches to do it.
//
// `cutoff` gates on the material's rather than reading alpha unconditionally: a texture whose alpha
// is unused may store nothing at all in it, and the darkest tap deciding is only right where alpha
// decides anything. Where it does, the taps straddling a leaf's silhouette meet whatever colour was
// left in the transparent texels, which is a step of no relevance to the leaf's shape.
vec2 relief_gradient(uint slot, vec2 uv, float lod, float cutoff) {
    float level = min(lod, float(textureQueryLevels(textures[nonuniformEXT(slot)]) - 1));
    vec2 texel = exp2(level) / vec2(textureSize(textures[nonuniformEXT(slot)], 0));
    vec4 pp = textureLod(textures[nonuniformEXT(slot)], uv + texel * vec2(0.5, 0.5), lod);
    vec4 pn = textureLod(textures[nonuniformEXT(slot)], uv + texel * vec2(0.5, -0.5), lod);
    vec4 np = textureLod(textures[nonuniformEXT(slot)], uv + texel * vec2(-0.5, 0.5), lod);
    vec4 nn = textureLod(textures[nonuniformEXT(slot)], uv + texel * vec2(-0.5, -0.5), lod);
    float hpp = relief_height(pp);
    float hpn = relief_height(pn);
    float hnp = relief_height(np);
    float hnn = relief_height(nn);
    vec2 gradient = 0.5 * vec2((hpp + hpn) - (hnp + hnn), (hpp + hnp) - (hpn + hnn));
    if (cutoff > 0.0) {
        gradient *= min(min(pp.a, pn.a), min(np.a, nn.a));
    }
    return gradient;
}

// `normal` tilted by a painted gradient, kept on the side of the plane the ray came from.
//
// The frame is orthonormalised against the normal rather than used as it comes out of the triangle,
// so a mesh whose uvs are sheared does not shear the relief with them.
vec3 relief_normal(vec3 normal, vec3 geometric, vec3 direction, SurfaceTangents axes,
                   vec2 gradient) {
    vec3 tangent = axes.along_u - normal * dot(normal, axes.along_u);
    if (dot(tangent, tangent) < 1.0e-12) {
        return normal;
    }
    tangent = normalize(tangent);
    vec3 bitangent = axes.along_v - normal * dot(normal, axes.along_v)
                   - tangent * dot(tangent, axes.along_v);
    if (dot(bitangent, bitangent) < 1.0e-12) {
        return normal;
    }
    bitangent = normalize(bitangent);

    vec2 slope = -RELIEF_SLOPE * frame.relief * gradient;
    float steepness = length(slope);
    if (steepness > 1.0e-6) {
        // Compressed by magnitude rather than per axis, which would turn the direction the surface
        // tilts in as well as how far.
        slope *= RELIEF_MAX_SLOPE * tanh(steepness / RELIEF_MAX_SLOPE) / steepness;
    }
    vec3 tilted = normalize(normal + tangent * slope.x + bitangent * slope.y);
    vec3 side = faceforward(geometric, direction, geometric);
    float above = dot(tilted, side);
    return above >= RELIEF_HORIZON ? tilted
                                   : normalize(tilted + side * (RELIEF_HORIZON - above));
}
