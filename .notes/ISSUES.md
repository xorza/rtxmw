# Open issues

- `moved()` in `crates/rtxmw-render/tests/precipitation.rs` counts any pixel whose bytes differ at
  all. Auto-exposure follows the frame's overall brightness, so a change that shifts it nudges every
  pixel by a level and the count saturates at the whole frame rather than measuring what moved.
