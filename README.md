# rtxmw

A Morrowind engine in Rust with a hardware-raytraced renderer, built on raw Vulkan via `ash`.

It is not a port of OpenMW's rasterizer — it is a new renderer against the same game data. Ray
tracing is the primary rendering mode, not an effect layered on a rasterizer.

**Status: early. Nothing is playable.** The current tree brings up a Vulkan device with the ray
tracing extensions, a window and a camera, and an offscreen golden-image test harness. No game data
is loaded yet. `docs/design.md` records the architecture and the decisions behind it.

## Building

Needs a recent Rust toolchain, an Ada-class NVIDIA GPU with the ray tracing extensions, and `glslc`
plus `spirv-val` on `PATH`. Running against game data additionally needs an existing Morrowind GOTY
install, pointed at by a hand-written `.env` in the repo root:

```
MORROWIND_DIR=/path/to/Morrowind
MORROWIND_DATA_DIR=/path/to/Morrowind/Data Files
```

```sh
cargo run -p rtxmw
```

No game assets are included or redistributed.

## Licence

`MIT OR Apache-2.0`, at your option. See [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE).
