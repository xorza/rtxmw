# OpenMW reference: `.refs/openmw`

Research notes about OpenMW master at `openmw-51-rc2-590-g5ad8e95c78` (v0.52.0 dev), GPLv3, C++,
checked out read-only under `.refs/` and gitignored. **Nothing here describes code in this
repository** — it is a map of where the file formats and game rules are documented in a codebase
that already decodes them.

OpenMW is GPLv3. Reading it to learn a file format is fine; copying its code or transliterating a
function line-for-line makes this project a derivative work.

`components/` is engine-agnostic library code — formats, VFS, terrain, resource cache.
`apps/openmw/mw*/` is the game itself. `docs/source/reference/` documents the sky system, texture
conventions, animation blending, and every setting.

## Cross-checking a format implementation

OpenMW's CLI tools decode the same files and can be diffed against. Build them from `.refs/openmw`,
or install `openmw`, when a format needs verification.

- `esmtool dump [-t TYPE] [-n NAME] [-C] file.esm` — record dump; also `--raw`, `--quiet`
- `bsatool list|extract|extractall archive.bsa`
- `niftest` — parse a NIF, or a directory of them, and report failures
- `navmeshtool`, `bulletobjecttool` — collision/navmesh generation from content files

## Coordinate system and units

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

## Asset pipeline

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

## Renderer

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

## Animation and skinning

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

## Game architecture

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
