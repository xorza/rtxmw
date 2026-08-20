## What this is

A Morrowind engine in Rust with a hardware-raytraced renderer — a new renderer against the original
game data, not a port of OpenMW's rasterizer. OpenMW is a reference for file formats and game rules
only: it is GPLv3, so reading it to learn a format is fine and copying or transliterating its code
is not. Notes on it are in `docs/openmw.md`; decisions, milestones and findings in `docs/design.md`;
issues noticed in passing in `.notes/ISSUES.md`. **Keep status out of this file** so it stays worth
trusting.

## Posture

A 2002 game made to look astonishing on current hardware — ray-traced visibility, path-traced
indirect light, materials recovered from pre-lit vanilla textures, DLSS Ray Reconstruction, opacity
micromaps, SER. Vanilla content, new light transport.

Priorities, in order:

1. **How it looks.** Trading image quality for simplicity or convenience is the wrong trade.
2. **Performance.** 1920×1080 internal → 3840×2160 at 60 fps (`docs/design.md` §5.3).

Nothing else ranks: no mod compatibility, no configurability for its own sake, no portability layer,
no abstraction over hardware this does not target.

Sports programming — strongest technique over safest, fast path first, delete what stopped earning
its place, settle arguments by measuring. Nothing here is published, so rewriting beats working
around.

## Layout

Cargo workspace, edition 2024, resolver 3. One crate per layer in `crates/`, named `rtxmw-<layer>`;
`rtxmw` is the binary. Hard seam: nothing below `rtxmw-scene` knows Vulkan, nothing above it knows
ESM records (`docs/design.md` §2).

Test helpers other crates need are exported from `lib.rs` under
`#[cfg(any(test, feature = "internals"))]` — the gate on the re-export must match the gate on the
module, or `unreachable_pub` rejects it.

`[profile.dev]` is `opt-level = 1` with dependencies at `3`; BSA/NIF/DDS decoding is unusable
otherwise. `.refs/`, `/target` and `.env` are gitignored.

## Toolchain

Raw Vulkan via `ash`, GLSL compiled by `glslc` from `build.rs` and validated with `spirv-val`,
licensed MIT OR Apache-2.0. Do not reintroduce wgpu without reading `docs/design.md` §1 — refit,
opacity micromaps, SER and RT pipelines are all unreachable there, and each is a certainty here.

Dependencies are declared once in `[workspace.dependencies]` and referenced as
`name.workspace = true`. Propose and wait for approval before adding one.

`unsafe_op_in_unsafe_fn`, `missing_debug_implementations`, `unreachable_pub` and the rustdoc link
lints are denied workspace-wide; `pub` in the binary crate is always a lie, so it is `pub(crate)`
throughout. Verification, **`cargo doc` included** — the rustdoc lints fire nowhere else:

```
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo doc --workspace --all-features --no-deps
cargo test --workspace --all-features
```

The assumed GPU is Ada-class NVIDIA with `VK_KHR_ray_tracing_pipeline`,
`VK_KHR_acceleration_structure`, `VK_KHR_ray_query`, `VK_KHR_ray_tracing_position_fetch`,
`VK_KHR_ray_tracing_maintenance1`, `VK_EXT_opacity_micromap` and
`VK_EXT/NV_ray_tracing_invocation_reorder` (real SER, not a no-op);
`VK_NV_cluster_acceleration_structure` and `VK_NV_partitioned_acceleration_structure` are used if
present. Tests need the Vulkan validation layers.

## Zero heap allocations per frame

Steady-state frames must not allocate. `crates/rtxmw-render/tests/frame_allocations.rs` measures a
whole frame of the real renderer — constants, recording, submit, wait — under `dhat` and fails if
the window allocates at all. `ALLOCATION_BUDGET` is `0`; raising it needs a reason in the commit,
and "the test started failing" is not one. The concern is jitter, not throughput: at 60 fps one
2 ms slow-path stall is a dropped frame, and averages hide it.

So on anything the frame path reaches: persistent scratch buffers refilled with `clear()`, results
into a `&mut Vec<T>` out-param, no `format!`, no `collect()`. Debug and logging code either follows
the same rules or compiles out. On failure the test writes `dhat-heap.json` — open it with
<https://nnethercote.github.io/dh_view/dh_view.html> and sort by "Total (blocks)".

## Game data

An unmodified Morrowind GOTY install — the three `.esm` and their `.bsa`. Point `.env` at it by hand
(gitignored); **quote any value containing a space** or the dotenv parser rejects the line:

```
MORROWIND_DIR="/path/to/Morrowind"
MORROWIND_DATA_DIR="/path/to/Morrowind/Data Files"
```

`rtxmw_vfs::morrowind_data_dir()` prefers the process environment over `.env`. Without the game
those tests **skip**; with a path set but wrong they **panic**, because a silent skip looks like a
pass.

## Running it

**Do not open the windowed binary to check a change** — it costs tens of seconds of the user's
screen and confirms almost nothing the headless path does not. `SceneRenderer` knows nothing about
surfaces, so `crates/rtxmw-render/tests/primary_visibility.rs` drives the real one headlessly in
under a second; only swapchain acquire and present need a surface.

```
cargo run -- --screenshot out.png                    # one frame, no window, ~0.6 s warm
cargo run -- --screenshot out.png 1920x1080 -2,-9    # §5.3 internal size, outdoors
cargo run -- [CELL] --frames N                       # N frames, then a clean shutdown
```

`--screenshot` brings up a device with no surface extensions, so it works over ssh, and it reports
the fraction of rays that hit geometry — enough to tell "the cell rendered" from "the camera faced
nothing" without opening the file. A trailing cell argument is addressed the way Morrowind does:
**a pair of integers is an exterior, anything else is an interior's name** — `-2,-9` is Seyda Neen's
shore, `"Balmora, Guild of Mages"` a building. Without one it is `-2,-9`, which stands the camera on
the deck of the ship the game itself starts on.
