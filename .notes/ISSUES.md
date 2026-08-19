# Open issues

- `NiStencilProperty` is read as flags, one bool and five `u32`s, but the format lays out flags, a
  byte for enabled, and seven `u32`s — test function, ref, mask, three actions and the draw mode.
- The visibility shader evaluates every light in the scene for every pixel, with no bound on how
  many or how far away they are. Measured on the exterior at Seyda Neen: 229 point lights across the
  loaded cells cost 3.8 ms of a 5-6 ms trace at 1920x1080.
