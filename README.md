# rtxmw

A Morrowind engine in Rust with a hardware-raytraced renderer, built on raw Vulkan via `ash`.

It is not a port of OpenMW's rasterizer — it is a new renderer against the same game data. Ray
tracing is the primary rendering mode, not an effect layered on a rasterizer.

**Status: a renderer, not a game.** The tree reads an unmodified Morrowind install and ray-traces
it — interiors and streamed exteriors, bindless materials de-lit from the vanilla textures, direct
and indirect light, water, a day-night cycle with both moons, and the game's own ten weathers —
through DLSS Ray Reconstruction. Nothing is playable: there is no animation, and no creatures or
NPCs are placed. `docs/design.md` records the architecture and the decisions behind it.

## Building

Needs a recent Rust toolchain, an Ada-class NVIDIA GPU with the ray tracing extensions, and `glslc`
plus `spirv-val` on `PATH`. Running against game data additionally needs an existing Morrowind GOTY
install, pointed at by a hand-written `.env` in the repo root:

```
MORROWIND_DIR="/path/to/Morrowind"
MORROWIND_DATA_DIR="/path/to/Morrowind/Data Files"
```

A value containing a space has to be quoted, or the dotenv parser drops the line.

```sh
cargo run                          # a window on the deck of the ship the game starts on
cargo run -- --screenshot out.png  # one frame, no window, no surface extensions
```

No game assets are included or redistributed.

## Licence

`MIT OR Apache-2.0`, at your option. See [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE).
