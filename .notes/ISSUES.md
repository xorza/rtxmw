# Open issues

- The coverage solve in `precipitation.glsl` saturates for the two rain weathers: `alpha` comes out
  at 1.8 for rain and 4.6 for thunderstorm and is clamped to 1, so neither reaches the physical
  coverage the solve is written to match. Snow, at 0.68, is the only weather the clamp does not bind.
