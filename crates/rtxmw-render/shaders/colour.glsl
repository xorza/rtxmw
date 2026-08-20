// What the eye makes of a linear colour, which is the only sense in which one has a brightness.
//
// **Its own file because two shaders that share no descriptors share this.** `bindings.glsl` is
// where everything else common lives, but `luminance_histogram.comp` cannot include it: that pass
// binds an image at `set = 0, binding = 0` where the tracer binds its acceleration structure, so
// the two declarations collide. A constant that both need therefore has to sit somewhere neither
// pass's layout reaches, which is here.
//
// `rtxmw_texture::LUMA` is the same three numbers on the host, and `sky_dome.rs`'s veil test is
// what holds the two together: it measures a shader-side luminance-preserving operation with the
// host's weights, so a typo in either set moves the picture.
//
// **Guarded, which nothing else under `shaders/` is**, because nothing else is a leaf two
// independent roots reach: `bindings.glsl` brings it in for the tracer and the histogram includes
// it directly, so the first pass to want both would redeclare `LUMA` and fail to compile.
#ifndef COLOUR_GLSL
#define COLOUR_GLSL

// Rec. 709, matching the primaries every texture is decoded to.
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

#endif
