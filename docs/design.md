# rtxmw — renderer design proposal

Status: decisions settled, nothing built yet. Written 2026-08-18.

Immediate goal: **render static Morrowind locations with a free noclip camera**, with hardware ray
tracing as the primary rendering mode rather than an effect layered on a rasterizer.

## Decisions

| Question | Decision | Consequence |
|---|---|---|
| Graphics API | **`ash` (raw Vulkan) + GLSL via `glslc`** | wgpu removed. Nothing is a dead end — refit, opacity micromaps, SER and RT pipelines all reachable. Costs 1–2 weeks of plumbing at M0 |
| License | **MIT OR Apache-2.0** | DLSS Ray Reconstruction is available. M7 becomes an integration rather than writing a denoiser |
| Material data | **De-light vanilla textures offline** | The most ambitious option; needs its own spike. See §5.1 — de-lighting recovers albedo only |
| Perf target | **1920×1080 internal → 3840×2160 output @ 60 fps** | ~8–9 ms denoiser, ~7 ms for everything else. 16:9 output, not native ultrawide |
| First scene | **Interiors through M8; exteriors at M9** | Fastest path to a correct image; terrain and streaming stay unvalidated until late |
| Windowing | **`winit`** | Gamepad and audio need separate crates when they arrive |

---

## 1. Graphics API: decided

**`ash` (raw Vulkan) with GLSL shaders compiled by `glslc`. wgpu removed.**

The reasoning is below; §1.3 keeps the counter-case on record, because if this decision is ever
revisited that is the argument to re-read.

### 1.1 What wgpu 30 can and cannot do

Audited against the vendored crate sources, not documentation.

**Works, and works well:**

- Ray queries from compute/fragment with a complete WGSL surface: candidate *and* committed
  intersections, `RayIntersection` carrying `t`, barycentrics, instance/geometry/primitive index,
  `instance_custom_data`, front-face, and both object↔world matrices; AABB
  `rayQueryGenerateIntersection`; `ray_query<vertex_return>` position fetch.
- Bindless, **including non-uniform indexing of a `binding_array` by a value read at a ray-query
  hit** — naga auto-emits the `NonUniform` decoration. This is the capability a bindless material
  system depends on, and it is present.
- BLAS/TLAS creation, batched builds, AABB geometry, per-geometry transforms, working async BLAS
  compaction, TLAS binding arrays.
- `as_hal` gives `ash::Device`, `vk::Image`, `vk::Buffer`, queue, semaphores — enough for
  native-SDK interop.

**Cannot do, and these are not edge cases for this project:**

| Gap | Consequence for a Morrowind engine |
|---|---|
| **No BLAS refit.** `ALLOW_UPDATE`/`PreferUpdate` are accepted and silently ignored; wgpu-core hardcodes `Build` and logs *"only rebuild is implemented"* | Every skinned NPC and creature is a **full BLAS rebuild every frame** |
| **No opacity micromaps** (`VK_EXT_opacity_micromap` is exposed by the target hardware, unreachable through wgpu) | Morrowind is saturated with alpha-tested foliage, grates, lanterns, banners. This is the single biggest RT cost multiplier in the game, and the hardware fix is off the table |
| **No RT pipelines at the safe API level** — zero mentions in wgpu-core; the draft PR is untouched and will not land in v31. No `as_hal` on `BindGroup`/`PipelineLayout`/`ShaderModule`, so the wgpu-hal implementation is unreachable without a parallel descriptor system | No SBT-driven material dispatch, no intersection or callable shaders |
| **No shader execution reordering** (the target hardware reports real `REORDER` mode) | Leaves substantial performance unclaimed on divergent hit shading |
| **TLAS instances are a CPU `Vec`**, serialized and staged on every build; the raw-instance escape hatch was removed in v30 | Measured upstream at ~1 ms per 1,000 instances, with instance marshalling at 98.7% of encode time. Morrowind exteriors are thousands of small statics |
| No async/second queue for AS builds; no per-instance geometry flags or SBT offset; no buffer device address on ordinary buffers | Assorted, each individually survivable |

### 1.2 The decisive evidence

**bevy_solari** is the flagship wgpu ray-tracing project, written by wgpu's principal RT consumer.
Its own tracking issue still lists as *not supported*: transparent and alpha-masked materials,
skinned and morphed meshes, point/spot lights, environment lighting, LODs, mipmaps. **That list is
approximately the Morrowind feature set.** And it has already dropped to `wgpu-hal` plus a compute
shader for TLAS building specifically to escape wgpu's overhead.

Meanwhile wgpu's RT surface has had breaking changes in **every release since 23.0.0**, one release
required adding a new `enable` directive to every existing shader, the feature is authored almost
entirely by one person, and using it requires an `unsafe { ExperimentalFeatures::enabled() }` token
documented as *"inherently bugs in our implementation that we will eventually fix."* There is also a
live codegen bug in the current release: ray queries hoisted out of loops emit invalid SPIR-V.

Set against that, `ash` 0.38 exposes the full `VK_KHR_acceleration_structure` /
`ray_tracing_pipeline` / `ray_query` / `ray_tracing_position_fetch` surface plus
`VK_EXT_opacity_micromap` and invocation reorder, and `glslc` + `spirv-val` were verified compiling
`.rgen`/`.rchit`/`.rmiss` and ray-query compute against the target hardware.

### 1.3 The honest counter-case

For **the stated immediate goal alone**, wgpu is sufficient. Static geometry means BLASes are built
once and compacted, and the TLAS is built once — refit and GPU-driven instance updates never fire.
Ray-queries-in-compute is a mainstream 2026 architecture, not a compromise. wgpu would also save
perhaps one to two weeks of allocator/descriptor/barrier/SBT plumbing, and `dlss_wgpu` 5.0.0 pins
exactly to wgpu 30, making DLSS Ray Reconstruction almost free.

Two things defeat that argument:

1. The gaps are **certainties, not risks**. A Morrowind engine has skinned actors, pervasive
   alpha-tested geometry, and torch point lights. The question is *when* the wall arrives, not *if*.
2. The DLSS advantage is smaller than it looks. `dlss_wgpu` exists precisely to work around wgpu's
   abstraction — it has to inject Vulkan extensions via `init_with_callback` before device creation.
   Under `ash` the `VkDevice` is ours and NGX is called directly, which is *simpler*, not harder.

Trading a bounded backend rewrite later for a Morrowind interior on screen two weeks sooner is a
defensible choice, and §2's module split is designed so the backend is replaceable either way. But
"RTX as a first-class feature" should not be designed on an API that cannot reach opacity micromaps,
refit, or SER.

### 1.4 Shader language

**GLSL**, via `glslc` from `build.rs`, with `spirv-val` in the verification chain.

`GLSL_EXT_ray_tracing` and `GLSL_EXT_ray_query` are complete and frozen-because-done. Critically,
**almost every readable open-source RT reference is GLSL**: Lumen (MIT, ReSTIR DI+GI+PT, Vulkan,
Linux), caldera, Q2RTX's A-SVGF, Godot's RT plumbing, NVIDIA's Godot RTX fork, `adrien-ben`'s
ash-0.38/edition-2024 examples. For a first RT renderer, reference-code availability outweighs
language quality.

**Slang** is the better language, has the most complete RT surface of any option (all six stages,
`HitObject`/SER, autodiff), is Apache-2.0 under Khronos governance, and is where new NVIDIA and
Khronos samples are going — nvpro's ray tracing tutorial has been rewritten Slang-only. It costs an
AUR install and `slangc` shelled from `build.rs`. Both compile to SPIR-V, so this is **not a
one-way door**: shaders can migrate individually. Revisit once the renderer's shape is settled.

Rejected: WGSL (wgpu-proprietary RT dialect, no WebGPU RT spec exists, dead end off wgpu); HLSL/DXC
(hard ceiling — DXC knows only three RT extensions, no position fetch, no micromaps, no SER);
rust-gpu (zero RT examples, incomplete buffer-device-address, and even kajiya used HLSL for its ray
tracing).

---

## 2. Module structure

Workspace crates, in dependency order. Each is a hard boundary; nothing below the line knows about
Vulkan, nothing above it knows about ESM records.

```
crates/
  rtxmw-vfs      path normalization, archive layering, BSA readers
  rtxmw-esm      ESM3 reader, record types, RefId, record store
  rtxmw-nif      NIF block reader → geometry, material, node graph
  rtxmw-scene    format-neutral scene: meshes, materials, instances, lights
  rtxmw-gpu      Vulkan: instance/device/queues/allocator/descriptors/swapchain
  rtxmw-render   acceleration structures, passes, denoise, post
  rtxmw          binary: window, input, noclip camera, wiring
```

`rtxmw-scene` is the seam that matters. Everything above it is Morrowind-specific and testable
headlessly; everything below is a renderer that would work for any scene. It is also what makes the
API decision in §1 revisable — swapping `rtxmw-gpu` and `rtxmw-render` leaves four crates untouched.

Within crates, the existing style rules apply: directory modules, one major struct per file named
after it, `impl` blocks only in the struct's own file, no re-exports except `lib.rs`, gated test mods
at the end of the production file.

---

## 3. Core data types

Sketches, not final signatures. Flat storage throughout, per the project's collection rules — a
Morrowind cell is thousands of small objects and `Vec<Vec<_>>` would allocate per instance.

### Scene (format-neutral, `rtxmw-scene`)

```rust
/// One loaded location, ready to hand to the renderer.
struct StaticScene {
    meshes: Vec<Mesh>,
    materials: Vec<Material>,
    textures: Vec<TextureId>,
    instances: Vec<Instance>,
    lights: Vec<Light>,
    ambient: CellAmbient,
}

/// Geometry for one NIF, flattened. Submeshes index into the shared buffers.
struct Mesh {
    positions: Vec<[f32; 3]>,
    normals:   Vec<[f32; 3]>,
    uvs:       Vec<[f32; 2]>,
    colors:    Vec<[u8; 4]>,        // empty when the NIF has none
    indices:   Vec<u32>,
    submeshes: Vec<SubMesh>,        // small and fixed after load
}

/// An index range plus the material it draws with. One BLAS geometry each.
struct SubMesh {
    indices: Range<u32>,
    base_vertex: u32,
    material: MaterialId,
}

/// A placed reference. `transform` already folds in position, rotation and scale.
struct Instance {
    mesh: MeshId,
    transform: Affine3A,
    /// Packed into the TLAS instance's 24-bit custom index.
    geometry_base: u32,
}

struct Material {
    base_color: TextureId,
    /// Only ever populated by replacer texture packs — vanilla NIFs carry none.
    normal: Option<TextureId>,
    specular: Option<TextureId>,
    emissive: [f32; 3],
    alpha: AlphaMode,               // Opaque | Masked { threshold } | Blended
    two_sided: bool,
}

/// From an ESM `LIGH` record plus its placement.
struct Light {
    position: Vec3,
    color: [f32; 3],
    /// Morrowind units. Needs a physical mapping — see §5.4.
    radius: f32,
    flags: LightFlags,              // flicker, pulse, negative, fire
}
```

### GPU-side (`rtxmw-render`)

The shading pass is bindless: one storage buffer of geometry descriptors, one of materials, one
texture `binding_array`. A hit gives `instance_custom_data + geometry_index`, which indexes
`GeometryRef`, which gives the material and the vertex/index offsets needed to fetch and interpolate
attributes at the barycentrics.

```rust
/// One per BLAS geometry, indexed by `instance_custom_data + geometry_index`.
#[repr(C)]
struct GeometryRef {
    index_offset: u32,
    vertex_offset: u32,
    material: u32,
    flags: u32,                     // has_colors, two_sided, alpha_mode
}

#[repr(C)]
struct GpuMaterial {
    base_color: u32,                // index into the bindless texture array
    normal: u32,                    // sentinel when absent
    specular: u32,
    emissive: [f32; 3],
    alpha_cutoff: f32,
    flags: u32,
}
```

`FrameConstants` should be lifted field-for-field from OpenMW's
`components/fx/stateupdater.hpp:284-288` — current and previous view/projection and their inverses,
eye position and vector, fog, ambient, sky and sun colors, sun position and vector, resolution,
near/far/fov, game hour, water height, underwater and interior flags, simulation time, frame number,
wind. It is a complete, battle-tested list, and the previous-frame matrices are exactly what a
denoiser and any temporal pass need.

### Vertex layout for BLAS

Position must be a separate, tightly packed `Float32x3` stream — acceleration structure builds read
positions with their own stride and ignore everything else. Keep shading attributes (normal, uv,
color, tangent) in a parallel buffer indexed by the same vertex id. This is not an optimization; it
is what the build API wants.

---

## 4. Implementation plan

Ten milestones. Each names a **sub-goal**, a **done-when** that is observable rather than "it
compiles", and the risk it retires. Milestones 1–2 and 3–4 are independent and can interleave.

### M0 — Foundations — **done**
winit window, Vulkan instance/device with the RT extension set, swapchain, a cleared frame,
validation layers on in debug. Noclip camera with mouse-look and WASD, frame timing in the title bar.
**Done when:** a cleared window presents at the target resolution with validation silent, and the
camera reports plausible world coordinates. ✔
**Retired:** the Vulkan plumbing risk, and the `ash` decision is now confirmed on real hardware —
position fetch, ray tracing maintenance1 and **opacity micromap** all report available.

Two corrections from building it:

- **Vulkan 1.3, not 1.4.** `ash` 0.38 ships 1.3.281 headers, so `API_VERSION_1_4` does not exist.
  1.3 plus the KHR ray tracing extensions covers everything needed; revisit at ash 0.39.
- **The swapchain cannot be a storage image.** sRGB formats do not expose
  `VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT`, so no compute or ray tracing shader can write it directly.
  Ray tracing output must go to an offscreen HDR image and reach the swapchain through the tonemap
  blit — which is the wanted shape anyway, but it means the offscreen target is required from M3
  rather than being an M8 concern.

Hardware limits worth having recorded: shader group handle size 32, base alignment 64, max ray
recursion depth 31, max BLAS geometries and TLAS instances both 16,777,215.

The memory allocator (`gpu-allocator`) is wired as a dependency but not yet used — nothing allocates
device memory until M2 uploads geometry.

### M1 — Data: VFS + BSA + ESM enumeration — **done**
`rtxmw-vfs` path normalization and archive layering; the Morrowind BSA reader; enough of `rtxmw-esm`
to read `CELL`, stream its refs, and resolve `STAT`/`DOOR`/`CONT`/`ACTI`/`LIGH`/`MISC` to model
paths.
**Done when:** given an interior cell name, it prints the ref list — refid, model path, position,
rotation, scale — and the ref count matches `esmtool dump -t CELL` for that cell. ✔
**Retired:** the format-decode risk.

`esmtool` is not installed and building it from `.refs/openmw` was not worth it, so the cross-check
is against the file's **own header record count** instead: walking Morrowind.esm yields exactly the
48,295 records the header declares. That catches the failure that matters — a mis-sized record
shifting every subsequent offset — at least as well as an external diff would.

Measured: 20,952 VFS paths across the three BSAs plus loose files (7,319 meshes, 6,256 textures);
1,134 interiors and 1,404 exteriors holding 316,116 references; Seyda Neen's Census and Excise
Office resolves 261 of its 268 references to meshes, with **zero** model paths missing from the VFS.

### M2 — Geometry: NIF — **done**
`NiNode` graph traversal with accumulated transforms, `NiTriShape`/`NiTriStrips` → indexed
triangles, `NiTexturingProperty` base slot, `NiMaterialProperty`, `NiAlphaProperty`, marker and
`RootCollisionNode` filtering.
**Done when:** every NIF in `Morrowind.bsa` parses without error or panic (a `niftest` equivalent),
and triangle/vertex counts for a sample of meshes match `niftest`'s. ✔
**Retired:** the largest single format risk.

All **7,319** shipped meshes parse: 4,579,361 triangles and 4,631,142 vertices, with 41,702 geometry
blocks passing index-bounds validation. As with `esmtool`, `niftest` is unavailable, so the
cross-check is self-consistency: every triangle index inside its vertex buffer, every UV set and
normal array matching the vertex count, and the block walk landing exactly on the root list.

**Carried into M3, deliberately:** node-graph traversal with accumulated transforms, and marker /
`RootCollisionNode` filtering. Both are listed in the scope above, and both are about *placing*
geometry rather than decoding it — they belong with scene assembly, not with the reader.

What made this milestone exacting is worth recording: blocks carry no size at version 4.0.0.2, so a
parser off by one byte shifts every subsequent block and the failure surfaces far from its cause.
Four bugs, three of them the same mistake — `bool` is four bytes at this version, but several fields
that look like booleans are declared `char`/`uint8_t` and are one. The thing that made them findable
was wrapping every block failure in an error carrying the block's index and type; before that the
report was "6,601 × read past the end", after it was "6,601 × NiSourceTexture", which named the bug.

### M3 — First light: RT primary visibility
One BLAS per mesh, compacted; one TLAS over the cell's instances; a raygen (or ray-query compute)
pass writing barycentric or normal-visualized hits.
**Done when:** an interior cell is recognizable on screen in false color, and the camera flies
through it.
**Retires:** the whole acceleration-structure pipeline. **This is the milestone that proves the
project.**

Split into steps, since it is the largest milestone: **M3a** scene assembly ✔, **M3b** the shader
build and its ray-query pass ✔, **M3c** geometry upload ✔, **M3d** BLAS and TLAS, **M3e** the
compute pass into an offscreen HDR target and the blit to the swapchain, **M3f** camera into frame
constants.

**M3c — done.** `Memory`, `Buffer`, `Uploader` in `rtxmw-gpu`; `GeometryBuffers` in `rtxmw-render`.
Seyda Neen's Census and Excise Office uploads its 104 distinct meshes — 21,113 vertices, 17,835
triangles — into 0.85 MiB across three buffers. Four things settled here:

- **Positions get their own tightly packed stream** and shading attributes a parallel one, as §3
  requires. `GeometryBuffers::POSITION_STRIDE` is asserted to be 12, because the stride is a number
  the build is *told* rather than one it derives: padding the vertex would not fail, it would
  misplace every triangle.
- **Indices stay mesh-local**, with `MeshRange::first_vertex` passed to the build as its
  `firstVertex` and added in the shader before an attribute fetch. Rebasing them into the shared
  buffer would work equally well today, but it ties a mesh's index data to where it landed, and
  cells will relocate meshes at M9.
- **A mesh that flattened to nothing keeps its slot** as a zero-length range, so a `MeshId` stays a
  direct index.
- **`gpu-allocator` pads an allocation to the memory requirement's alignment**, so the raw mapped
  slice is longer than the buffer — an 8-byte buffer maps 16. `Buffer::mapped` trims to the
  requested size; without that, every readback would return padding as data.

`TestGpu` was rebuilt on the same three types rather than keeping its own allocator and submitter,
so the upload path the engine uses is the one the tests exercise. The one-shot `Commands` type stays
`pub(crate)`: uploads and acceleration structure builds happen at load time, where blocking on a
fence is the simplest correct thing, and nothing outside the crate should reach for that.

**M3d — done.** `AccelerationStructure` and `SceneAcceleration` in `rtxmw-render`. The same interior
builds **104 BLAS over 17,835 triangles and a 261-instance TLAS**; compaction takes the bottom level
from **1.31 MiB to 0.58 MiB, a 56% saving**, which is why `ALLOW_COMPACTION` is on by default rather
than being an option. Top level is 79 KiB.

- **One `cmd_build_acceleration_structures` call for all 104**, each with its own scratch region cut
  from a single buffer, and the compaction-size query rides in the same submission behind a barrier.
  The alternative — one shared scratch plus a barrier between builds — is simpler to write and
  serialises work that is independent by construction.
- **Scratch alignment is `minAccelerationStructureScratchOffsetAlignment`, 128 on this device**, and
  it is *not* satisfied by the buffer's natural memory requirement. `Buffer::with_alignment` raises
  the requirement and then asserts the resulting address, because raising a requirement is an
  indirect way to align an address and a heap that satisfies it by luck would stop under
  fragmentation.
- **Instances use `TRIANGLE_FACING_CULL_DISABLE`.** Morrowind authors single-sided planes and relies
  on seeing them from both faces, and winding is inconsistent across the mesh library.
- **`instance_custom_index` carries the mesh index**, which is what the material lookup at M4 is
  built on.
- **Geometry flag is `OPAQUE`, pairing with `gl_RayFlagsOpaqueEXT` in the shader.** Both change
  together at M4; either one alone is wrong.

What M3d does *not* prove is that the triangles inside a structure are the right ones. The build
range triple — `first_vertex`, `primitive_offset`, `max_vertex` — could be wrong in a way that still
builds and still validates, because validation cannot inspect index contents on the device. If the
first image at M3e is recognisable-but-scrambled, that triple is where to look.

Not yet wired into `Renderer` — nothing in the frame loop needs any of this until M3e traces it.
Note when it is: `Memory` hands out clones that must all drop before the `Device`, so it belongs
*before* the device in `Renderer`'s field order.

### M4 — Textures and bindless materials
DDS/TGA decode, BCn transcode where needed, mip generation, bindless texture array, `GeometryRef`
and material buffers, attribute interpolation at the hit, alpha-test in the candidate loop.
**Done when:** the cell renders with correct albedo and alpha-tested geometry reads correctly, no
unbound-descriptor validation errors.

### M5 — Direct lighting
Sun as a directional light with a real angular diameter (so shadows are soft), shadow rays, cell
ambient, `LIGH` point lights with shadow rays and a defensible attenuation model, emissive surfaces.
**Done when:** a torch casts a soft shadow that moves correctly, and an interior without lights is
lit only by its ambient term.
**Retires:** the "does this look like Morrowind or like a tech demo" question — see §5.1.

### M6 — Indirect lighting
One-bounce diffuse GI with next-event estimation and cosine-weighted sampling. Escalate to ReSTIR DI
then GI only if variance demands it; do not start there.
**Done when:** an interior lit through a doorway shows plausible bounce, and per-pixel variance is
measured and recorded so M7 has a baseline.

### M7 — Denoise and upscale
DLSS Ray Reconstruction via NGX — denoise, antialias and upscale in one pass, 1920×1080 → 3840×2160.
No separate TAA. Requires the full G-buffer from §5.2 including the specular guide, and the NGX
Vulkan extensions requested back at M0.
**Done when:** a still frame at 1 sample per pixel is comparable to a 1024-sample reference by a
numeric metric, and the frame holds 60 fps at the §5.3 target.
**Retires:** the biggest performance unknown. Worth spiking early against a placeholder scene rather
than waiting until M7 — the G-buffer requirements reach back into M3.

### M8 — Filters and output
Exposure/auto-exposure, tonemap, bloom, color grading, sharpening; correct sRGB↔linear discipline
end to end; optional HDR output.
**Done when:** a Morrowind screenshot and an rtxmw render of the same viewpoint can be compared
side by side without gamma mismatch.

### M9 — Exteriors: terrain and streaming
`LAND` decode (delta-coded VHGT, transposed VTEX), terrain as BLAS geometry, texture layer blending,
the 3×3 active cell grid, distant statics.
**Done when:** the camera can fly from Seyda Neen to Balmora with the grid streaming and no hitching.
**Note:** this is where TLAS instance counts get large — the point at which wgpu's CPU marshalling
would have become the bottleneck.

### M10 — Water
Per-cell water plane, RT reflection and refraction, absorption and scattering. OpenMW's tuned
constants (`files/shaders/compatibility/water.frag:13-51`) are a good starting point.

---

## 5. Decisions still open

Ordered by how much damage they do if deferred. The first two outrank the API choice.

### 5.1 Vanilla Morrowind has no material data, and its albedo is pre-lit

OpenMW's own documentation states it plainly: *"Morrowind format NIF files do not support normal
maps or specular maps."* Vanilla assets are 256²-era diffuse textures with **lighting, shading and
ambient occlusion painted into the albedo**, authored for a fixed-function renderer with no
per-pixel lighting.

Physically-based ray tracing on top of pre-lit albedo double-lights everything. Surfaces read flat
and muddy, ambient occlusion appears twice, and no amount of denoiser tuning fixes it. This is the
single largest threat to "looks great", and it is an art-pipeline problem that no renderer
architecture solves.

Four options, not mutually exclusive:

1. **Accept it and tune.** Cheapest. Reduce GI contribution, lean on direct lighting. Will look
   better than vanilla but will not look like a modern RT title.
2. **Support replacer texture packs.** Morrowind has a mature ecosystem of HD/PBR replacers, and
   OpenMW's `_n` / `_nh` / `_spec` filename conventions are the de-facto standard for wiring them
   in. Cheap to implement, and shifts the quality ceiling onto the pack. **Probably the best
   value.**
3. **Synthesize normal and roughness from diffuse.** Classic height-from-luminance is unreliable;
   a small image model would do better. Medium effort, uncertain payoff.
4. **De-light the vanilla textures offline** — estimate and divide out baked shading. Research-grade,
   high effort, but it is the only path that makes *vanilla* assets look physically correct.

**Decided: option 4, de-light the vanilla textures offline.**

One implication to plan for: **de-lighting recovers base colour only.** It does nothing about the
missing normal and roughness maps, which vanilla NIFs simply do not have. So the material pipeline is
really two problems, and choosing 4 solves the harder half:

- *Albedo* — de-lighting. Estimate and divide out baked shading and AO.
- *Normal / roughness* — still needs option 2 (replacer packs) or option 3 (synthesis). Since
  `Material` now carries these slots as first-class either way, supporting the `_n` / `_nh` / `_spec`
  filename conventions is nearly free and worth doing alongside.

De-lighting deserves its own spike **before M4 commits to a texture format**, run offline against a
sample of a few dozen textures and judged by eye against the same surfaces under flat lighting. The
failure mode to watch for is over-correction: flat, washed-out output where the algorithm removed
genuine painted detail along with the shading. Keep the vanilla textures as a fallback path so a
regression in the bake is always visible as an A/B.

### 5.2 Licensing — settled: MIT OR Apache-2.0

**Decided.** The workspace is permissive, so the NVIDIA RTX SDK conflict below does not apply and
**DLSS Ray Reconstruction is available**. M7 becomes an integration task rather than writing a
denoiser: DLSS-RR does denoise, antialias and upscale in one pass, which also means **no separate TAA
pass** — running TAA on top of a temporally-accumulated denoiser double-blurs and compounds ghosting.

The cost is a fat G-buffer. DLSS-RR requires diffuse albedo, specular albedo, normals, roughness,
colour, depth, motion vectors, and a specular guide (hit distance or specular motion vectors), plus
jitter offset and a reset flag. **Design the G-buffer for this from M3 onward** rather than
retrofitting it at M7 — the specular guide in particular is easy to forget and awkward to add late.

Practical notes for when M7 arrives: NGX ships real Linux `.so` files (not Proton stubs) and needs
specific Vulkan instance and device extensions **at creation time**, so M0's device setup must leave
room for them. OTA model updates are broken on Linux, so the model baked into the shipped library is
the one that gets used. Streamline is Windows-only — call NGX directly.

The original analysis is kept below, since reversing the licence would reinstate every word of it.

#### Why GPL-3 would have conflicted

Kept for the record: this section describes the position the workspace would have been in under
`GPL-3.0-or-later`, the licence considered before `MIT OR Apache-2.0` was settled on. NRD, DLSS,
SHARC and RTXDI all ship under the NVIDIA RTX SDK license, whose §4(e) forbids using the SDK in any
way that would subject it to a license requiring source disclosure, derivative works, or free
redistribution — i.e. exactly what GPL-3 requires.

Building and running locally is fine; the obligation triggers on **distribution**. But:

- No OpenMW code is being copied, so **GPL-3 here would be a choice, not an inheritance.** For
  distributing binaries with DLSS-RR, a permissive license removes the conflict outright.
- Under GPL-3 with distribution, the denoiser would have to be hand-rolled SVGF from the BSD-3
  references (Falcor's `SVGFPass`, Schied's A-SVGF, chrismile's Vulkan/GLSL implementation) —
  roughly 700 lines of shader to work, 2,000 to be good.

(Resolved by going permissive.)

### 5.3 Output resolution is the hardest constraint in the project

Denoising scales with *output* pixels, not internal ones. Measured upstream, DLSS Ray Reconstruction
alone costs ~6.1 ms at 3200×1800 on a 3080. Even scaling generously for Ada, anything above 4K output
spends 10–14 ms on the denoiser before a single ray is cast — the whole frame, at 60 fps. Driving a
high-resolution ultrawide at its native mode is therefore off the table.

**Decided: 1920×1080 internal → 3840×2160 output at 60 fps** on Ada-class hardware — DLSS Performance
mode. That is 8.3 M output pixels, so budget roughly **8–9 ms of denoiser**, leaving about **7 ms of
a 16.6 ms frame** for acceleration structures, rays, GI and post.

Treat that 7 ms as the real budget from M3 onward and measure against it, rather than discovering at
M7 that the frame is already full. If the renderer turns out efficient enough, the internal
resolution moves up to 2560×1440 (Quality mode) at no architectural cost. Ray tracing at native
ultrawide resolution is not a target; it is a wish.

The good news is Morrowind-shaped: ~1.1 GB of game data, 256² textures, very low-poly meshes. BLAS
memory and build time will be a non-issue in a 16 GB budget. The pressure is **TLAS instance count**
and **alpha-tested foliage**, not geometry volume.

### 5.4 Smaller decisions — settled by default

These are recorded as chosen rather than left open. Each is cheap to revisit; none warrants blocking.

- **Light units.** Work in physical units throughout, with 69.99125109 units per metre as the single
  conversion constant. `LIGH` radius maps to an inverse-square falloff with a radius-derived cutoff,
  not Morrowind's original curve — matching vanilla attenuation exactly would fight the renderer.
- **Emissive vs analytic.** A Morrowind torch is *both* a `LIGH` record and a glowing mesh. Use the
  `LIGH` record as an analytic light for direct lighting, and mark the mesh emissive **but excluded
  from light sampling**, so it appears bright to primary and reflection rays without being counted
  twice. This is the standard resolution and the double-count is very visible when got wrong.
- **Sky and environment.** Ambient light needs an environment map. Morrowind's sky is procedural
  (`WeatherResult` carries the colors); render it to a small cubemap once per frame and sample it.
- **Color management.** sRGB textures, linear working space, tonemap at the end. Getting this wrong
  is the most common reason RT renders look washed out, and it is nearly free to get right on day
  one.
- **Asset cache.** Re-parsing 500 MB of BSA and every NIF on each launch will dominate iteration
  time. Plan a converted-asset cache keyed by content hash before it becomes painful.
- **Debug affordances.** A debug-view selector (albedo / normal / instance id / material id /
  variance / ray count), shader hot reload, and a headless golden-image mode. All three pay for
  themselves within a week. `ash` also behaves better than wgpu under RenderDoc and Nsight.
- **Testing.** Format layers get round-trip and cross-check tests against `esmtool`/`niftest`. The
  renderer gets golden-image tests via the headless mode. Neither is optional at this scale.
- **Scope boundary for "static".** M0–M8 render the **bind pose only, no particles, no animation of
  any kind** — animated doors and activators appear in their rest state, NIF controllers are parsed
  but not evaluated. Creatures and NPCs are not placed at all. This is a deliberate cut, not an
  omission to discover later; it is also what keeps the missing-BLAS-refit problem out of scope until
  after the renderer is proven.

---

## 6. Dependencies this proposal implies

Approved. `wgpu` is removed.

| Crate | Purpose | Note |
|---|---|---|
| `ash` | Vulkan | 0.38 on crates.io is two years stale but carries the full RT surface plus opacity micromap and invocation reorder. It lacks only cluster/partitioned AS, which this project does not need. 0.39 will be a large breaking rewrite — pin and revisit |
| `gpu-allocator` | device memory | `ash` provides none |
| `winit` | window + input | Gamepad and audio will need separate crates later |
| `glam` | math | `Affine3A` matches the 3×4 transforms the AS API wants |
| `bytemuck` | `#[repr(C)]` GPU structs | |
| build-time: `glslc` | GLSL → SPIR-V | external tool; shelled out to from `build.rs` |

Not yet added, needed later: an NGX binding for DLSS-RR at M7 (hand-written FFI against the C
header), and an image decoder for DDS/TGA at M4.
