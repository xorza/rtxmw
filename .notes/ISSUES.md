# Open issues

- `NiStencilProperty` is read as flags, one bool and five `u32`s, but the format lays out flags, a
  byte for enabled, and seven `u32`s — test function, ref, mask, three actions and the draw mode.
