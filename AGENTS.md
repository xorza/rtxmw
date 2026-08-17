## What this is

A Morrowind game engine in Rust with a hardware-raytraced renderer. It is not a port of OpenMW's
rasterizer — it is a new renderer against the same game data. OpenMW is a reference for data formats
and game rules, not an architecture to copy.

Status: **M0 complete.** A winit window presents a cleared swapchain with a free noclip camera, on
`ash` with the ray tracing device features enabled and validation aborting on error in debug builds.
There is an offscreen test harness with golden-image comparison. Nothing is read from Morrowind data
yet — M1 (VFS, BSA, ESM cell enumeration) is next; see `docs/design.md` for the milestone plan.

Everything under "OpenMW reference" below is research about `.refs/openmw`, not about code here.

## Repository structure

A Cargo workspace, edition 2024, resolver 3. Members are `crates/*`: `rtxmw` is the binary (window,
input, noclip camera) and `rtxmw-gpu` owns everything Vulkan, including the `internals`-gated
offscreen test harness.

`[profile.dev]` is `opt-level = 1` with all dependencies at `opt-level = 3` — BSA/NIF/DDS decoding is
unusably slow in an unoptimized build.

`.refs/`, `/target` and `.env` are gitignored.

### Dependencies and toolchain

Every dependency is declared once in `[workspace.dependencies]` in the root `Cargo.toml`; crate
manifests reference them as `name.workspace = true`, never with an inline version.

The renderer targets **raw Vulkan via `ash`**, with shaders in **GLSL** compiled by `glslc` from
`build.rs` and validated with `spirv-val`. Licence is MIT OR Apache-2.0.

Runtime: `ash`, `ash-window`, `raw-window-handle`, `gpu-allocator`, `winit`, `glam`, `bytemuck`.
Behind the `internals` feature, for the test harness only: `png`, `half`. Dev-only: `dhat`.

**wgpu was evaluated and rejected** — see `docs/design.md` §1. Short version: inline ray queries work
well there, but BLAS refit is silently ignored, opacity micromaps and SER are unreachable, and ray
tracing *pipelines* do not exist at the safe API level despite `EXPERIMENTAL_RAY_TRACING_PIPELINES`
being advertised. Those gaps are certainties for a Morrowind engine, not risks. Do not reintroduce
wgpu without reading that section.

Propose and wait for approval before adding any further dependency.

### Hardware and tooling requirements

The renderer assumes an Ada-class NVIDIA GPU exposing `VK_KHR_ray_tracing_pipeline`,
`VK_KHR_acceleration_structure`, `VK_KHR_ray_query`, `VK_KHR_ray_tracing_position_fetch`,
`VK_KHR_ray_tracing_maintenance1`, `VK_EXT_opacity_micromap` and
`VK_EXT/NV_ray_tracing_invocation_reorder` (real SER, not a no-op).
`VK_NV_cluster_acceleration_structure` and `VK_NV_partitioned_acceleration_structure` are used if
present. Building shaders needs `glslc` and `spirv-val`; running the test suite needs the Vulkan
validation layers.

The performance target is **1920×1080 internal → 3840×2160 output at 60 fps**; see
`docs/design.md` §5.3 for the frame budget that follows from it.

### Per-frame heap allocations are a tracked metric

**Steady-state frames must allocate zero times on the heap.** This is enforced, not aspirational:
`crates/rtxmw-gpu/tests/frame_allocations.rs` runs the frame path under the `dhat` heap profiler and
fails if the measured window allocates at all. It is a `#[test]`, not a benchmark — the property is
pass/fail rather than a wall-clock measurement, and keeping it out of criterion keeps compile times
down.

The reason is jitter, not throughput. An allocation per frame is individually cheap but occasionally
takes a slow path, and at 60 fps one 2 ms stall is a dropped frame. Averages hide this; a hard zero
does not.

The usual discipline applies to anything on that path — persistent scratch buffers refilled with
`clear()`, results written into a `&mut Vec<T>` out-param, no `format!` or `collect()`. Debug and
logging code is not exempt; it either follows the same rules or compiles out.

`ALLOCATION_BUDGET` in that test is currently `0` and observed to hold. Raising it requires a reason
recorded in the commit — a driver that provably allocates on some path is a reason, "the test
started failing" is not.

When it fails, the test writes `dhat-heap.json`; open it with
<https://nnethercote.github.io/dh_view/dh_view.html> and sort by "Total (blocks)" to find the call
site. Note the test currently covers record-and-submit only; swapchain acquire and present need a
window, so fold them in once there is a headless path through the real frame loop.

### Game data

The engine reads an unmodified Morrowind GOTY install — `Morrowind.esm`, `Tribunal.esm`,
`Bloodmoon.esm` and the three matching `.bsa`. Its location belongs in `.env` at the repo root, which
is gitignored and must be created by hand:

```
MORROWIND_DIR=/path/to/Morrowind
MORROWIND_DATA_DIR=/path/to/Morrowind/Data Files
```

### Cross-checking a format implementation

OpenMW's CLI tools decode the same files and can be diffed against. Build them from `.refs/openmw`,
or install `openmw`, when a format needs verification.

- `esmtool dump [-t TYPE] [-n NAME] [-C] file.esm` — record dump; also `--raw`, `--quiet`
- `bsatool list|extract|extractall archive.bsa`
- `niftest` — parse a NIF, or a directory of them, and report failures
- `navmeshtool`, `bulletobjecttool` — collision/navmesh generation from content files

## OpenMW reference: `.refs/openmw`

OpenMW master at `openmw-51-rc2-590-g5ad8e95c78` (v0.52.0 dev), GPLv3, C++. Gitignored, read-only.

OpenMW is GPLv3. Reading it to learn a file format is fine; copying its code or transliterating a
function line-for-line makes this project a derivative work.

`components/` is engine-agnostic library code — formats, VFS, terrain, resource cache.
`apps/openmw/mw*/` is the game itself. `docs/source/reference/` documents the sky system, texture
conventions, animation blending, and every setting.

### Coordinate system and units

Morrowind is Z-up, right-handed; +X = east, +Y = north. From `components/misc/constants.hpp`:

| Value | Constant |
|---|---|
| 69.99125109 units/metre (64 units per yard) | `UnitsPerMeter:10` |
| gravity 8.96 m/s² (9.8 *yards*/s² — an original-engine bug) | `GravityConst:22` |
| exterior cell = 8192 units (ESM4: 4096) | `CellSizeInUnits:25` |
| active grid radius 1 → 3×3 cells (ESM4: 2 → 5×5) | `CellGridRadius:29` |
| step height 34 units, max slope 46° | `:43`, `:44` |

Terrain (`components/esm3/loadland.hpp`): 65×65 vertices per cell, 128 units apart; 16×16 texture
tiles of 512 units; height quantum 8 units; default height with no LAND record −2048. World→cell is
`floor(x / cellSize)`; cell→world is `cellSize * index`.

NIF matrices are stored row-major and are transposed on load (`components/nif/niftypes.hpp:62,81`).
`ESM::Position` rotations are radians (`components/esm/position.hpp:11`).

### Asset pipeline

**VFS** (`components/vfs`) — an ordered archive list flattened into one index; later archives win,
and loose files beat BSAs. Paths are lowercased with `\` → `/`, no leading separator, no duplicate
separators (`components/vfs/pathutil.hpp:18-71`). Morrowind-specific path fixups — force `.dds`,
prepend `meshes/`/`sound/`, prefer `xfoo.nif` when `xfoo.kf` exists — are in
`components/misc/resourcehelpers.hpp:26-63`. These are quirks of the original data, not conventions.

**BSA** (`components/bsa`) — four formats behind `detectVersion()`. Morrowind's is documented in full
in the comment at `components/bsa/bsafile.cpp:79-108`; the others (TES4-era compressed, BA2 GNRL, BA2
DX10) apply only to non-Morrowind content.

**ESM3** (`components/esm3`) — a flat record stream, `NAME(4) | u32 size | u32 unused | u32 flags`.
The reader state machine is `components/esm3/esmreader.cpp`; the canonical per-record `load()` shape
is `components/esm3/loadstat.cpp:8-37`. Two non-obvious behaviours:

- **CELL does not contain its refs.** The record stores file offsets; refs are streamed later via
  `getNextRef()` (`loadcell.hpp:188`). LAND is lazy in the same way (`loadland.cpp:310`).
- **LAND VHGT is delta-coded** (running row and column offsets, × 8) and **VTEX is transposed** in
  4×4 blocks of 4×4. Texture index 0 means "default"; every other index is offset by one into the
  per-plugin LTEX palette. `components/esm3/loadland.cpp:31-39, 315-345`.

`RefId` is a 6-way variant — interned string, ESM4 FormId, generated, index, exterior-cell coords, or
empty (`components/esm/refid.hpp:46`). Exterior cells are identified by their `(x, y)`.

**NIF** (`components/nif`, ~12k lines) — a header plus a flat array of numbered blocks; cross-refs are
block indices resolved in a second `post()` pass. Header layout at
`components/nif/niffile.cpp:551-662`. Morrowind is `VER_MW = 0x04000002` and uses roughly 40 block
types: `NiNode`, `NiTriShape(Data)`, `NiTriStrips(Data)`, `NiTexturingProperty`, `NiSourceTexture`,
`NiMaterialProperty`, `NiAlphaProperty`, `NiSkinInstance/Data`, the controller family,
`NiTextKeyExtraData`, `RootCollisionNode`, `NiBillboardNode`, `AvoidNode`, particles. The `bhk*`
Havok records exist in the format but are unused for Morrowind, and OpenMW ignores them for Bethesda
meshes too (`components/nifbullet/bulletnifloader.cpp:132`).

Collision-mesh selection is pure format logic at `components/nifbullet/bulletnifloader.cpp:108-262` —
the `MRK` / `NC…` / `NCC` / `RCN` string markers, `RootCollisionNode` lookup, fallback to visible
geometry, and first-child-only handling for switch nodes.

**Resource cache** (`components/resource`) — per-type caches with a timestamp sweep, documented as
callable from any thread. `SceneManager::getTemplate` / `getInstance` is a one-shared-template,
N-instances split.

### Renderer

OpenMW's renderer is OSG-based and forward-rendered. Locations of the data a renderer needs:

- **Frame constants** — `components/fx/stateupdater.hpp:284-288`, a complete std140 block: matrices,
  eye, fog, sun, weather id and transition, water height, underwater/interior flags, simulation time,
  wind.
- **Sun** — `apps/openmw/mwworld/weather.cpp:866-904` computes a linear east→west orbit and produces
  the hardcoded-from-Morrowind `sunDir = (-400·orbit, 75, -100)`. `renderingmanager.cpp:564-580` then
  splits the visible sun disc from the light direction — the disc follows a `z = 400 - |x|` arc. The
  sun is directional, with no angular diameter anywhere in the codebase.
- **Weather → colors** — `MWRender::WeatherResult` (`mwrender/skyutil.hpp:30-84`) carries per-frame
  fog/ambient/sky/sun colors, fog depth, glare, night fade and precipitation, interpolated per
  time-of-day band in `weather.cpp:1280-1436`.
- **Materials from NIF** — `components/nifosg/nifloader.cpp:2727-2910`. `NiSpecularProperty` is
  force-disabled for `VER_MW` (`:2892-2898`), which is why vanilla meshes are diffuse-only.
- **Texture slots and the `_n` / `_spec` filename conventions** —
  `components/shader/shadervisitor.cpp:255-256` and `docs/source/reference/modding/texture-modding/`.
- **Water** — `files/shaders/compatibility/water.frag:13-51` holds the tuned constant set
  (visibility 2500, wave scale 75, scatter color, sun extinction). Water level is per-cell,
  `ESM::Cell::mWater`.
- **Cell ambient and the interior brightness floor** — `renderingmanager.cpp:507-547`. Morrowind
  interiors ship ambient values authored against an engine with no global illumination.
- **Light semantics** — `components/sceneutil/lightutil.cpp:45-91` and
  `files/shaders/lib/light/util.glsl` for the attenuation and radius conventions derived from `LIGH`
  records; `components/sceneutil/clusteredlighting.hpp:14-24` is a GPU-shaped light struct.

### Animation and skinning

Skinning is entirely CPU-side, run every frame during the cull traversal
(`components/sceneutil/riggeometry.cpp:138-233`): it rebuilds positions, normals and tangents and
re-uploads the whole VBO. Two details of the data layout:

- `RigGeometry` groups vertices that share identical bone/weight sets and applies the accumulated
  matrix once per group. `mData->mBones` is `{name, boundSphere, invBindMatrix}`.
- `SceneUtil::Skeleton` has `Inactive` / `SemiActive` / `Active` states
  (`components/sceneutil/skeleton.cpp:131-144`): inactive skeletons skip the update traversal
  entirely, semi-active ones skip it after 3 frames without being culled in.

OpenMW runs `SceneUtil::Optimizer` with `MERGE_GEOMETRY` on loaded models
(`components/resource/scenemanager.cpp:863`) and merges statics into batched chunks in
`objectpaging.cpp`. Both discard per-instance identity.

### Game architecture

`Engine::frame()` (`apps/openmw/engine.cpp:191`) runs, in order: Lua GC join → input → sound → apply
the previous frame's queued Lua mutations → state → scripts → mechanics → physics → world (weather,
navigator, player, cell streaming, rendering update) → GUI → OSG update traversal → focus raycast →
release Lua thread → cull+draw → join Lua thread. The OSG update traversal runs after game logic, and
cull/draw overlaps the Lua thread.

Three things are concurrent with gameplay:

1. **Physics** — a POD frame-data snapshot (`ActorFrameData`) is double-buffered and handed to a
   worker pool; frame N's physics is joined at the start of frame N+1. Workers never touch game
   objects. `apps/openmw/mwphysics/mtphysics.cpp`.
2. **Lua** — its own thread, queueing `DelayedAction`s applied at one main-thread sync point.
3. **Background loading** — resource, terrain and navmesh work producing detached immutable results.

Everything else is single-threaded and freely re-entrant through the `MWBase::Environment` service
locator: `Scene::unloadCell` alone re-enters mechanics, GUI, sound, Lua, physics and the navigator.

Subsystem map: `mwworld` (record store, cell state, streaming, weather, time), `mwmechanics`
(stats/combat/magic/AI — `character.cpp` is the per-actor state machine), `mwphysics` (Bullet
collision only, no rigid-body dynamics; character movement is a custom sweep solver), `mwscript` (the
original Morrowind VM), `mwlua` (modern scripting API), `mwgui`, `mwdialogue` (`filter.cpp` holds the
INFO selection logic), `mwsound`, `mwstate` (saves), `mwinput`, `mwclass` (one behavior table per
record type, dispatched through an abstract singleton per type).

Cell streaming is in `apps/openmw/mwworld/scene.cpp`: `changeCellGrid` (`:615`) unloads outside a
Chebyshev radius, loads nearest-first, and batches navmesh invalidation under a single guard;
`playerMoved` (`:572`) applies hysteresis so the grid does not thrash on a cell boundary.
