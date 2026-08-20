# Open issues

- `luminance_histogram.comp` defines its own `LUMA` rather than including `bindings.glsl`, so Rec.
  709 is written down three times across the tree — there, in `bindings.glsl`, and as `srgb::LUMA`.
