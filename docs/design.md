# rtxmw — renderer design proposal

Written 2026-08-18. The decisions below are settled; §4 carries what is built and what each
milestone measured.

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
builds and still validates, because validation cannot inspect index contents on the device. M3e
settles it.

**M3e — done.** `Image` and `image_blit` in `rtxmw-gpu`, `VisibilityPass` and `FrameConstants` in
`rtxmw-render`, and the whole chain wired into `Renderer`. `cargo run` opens Seyda Neen's Census and
Excise Office and traces it: 1920×1080 offscreen `R16G16B16A16_SFLOAT`, blitted to the swapchain.
A debug build, whose validation layer aborts on error, runs the frame loop silently.

The milestone's real value is that the pass is verified **headlessly, pixel by pixel**, which
retires every "silently wrong" risk carried since M3c:

- A wall 100 units ahead covers the centre and misses the corners, with the covered fraction
  matching what 75° of field of view predicts.
- Geometry to the **north appears on the left**, geometry **above appears at the top**. That pair
  pins the whole handedness chain — instance transform, projection convention, Vulkan's Y-down NDC,
  and the shader's unprojection. A mirror anywhere in it still draws a wall, just in the wrong
  place, and nothing before M3e could tell.
- Two meshes in one buffer both render at their own offsets, which is the direct test of the build
  range triple M3d could not check.
- An instance behind the camera is not drawn, so instance transforms are applied rather than
  ignored.
- The real cell renders with 48% of pixels hitting geometry across 11,854 distinct barycentric
  shades — structure, not a single wrongly-placed triangle filling the view.

Two things worth recording:

- **`RenderTarget`'s readback used to lock the shared uploader internally**, so a caller holding the
  guard deadlocked — which is exactly what the first run of the new test did. The methods now take
  `&mut Uploader`, making it a compile error rather than a hang. The doc comment warning about it,
  added a milestone earlier, did not prevent it.
- **Locating the game install is production behaviour**, not a test helper, so `morrowind_data_dir`
  and friends came out from behind the `internals` feature and `dotenvy` became a normal dependency.

The camera's projection uses the *offscreen* aspect ratio, not the window's — the trace happens at a
fixed internal resolution and is stretched to whatever the window is, so a projection built for the
window would distort as soon as the two differed.

### M4 — Textures and bindless materials
DDS/TGA decode, bindless texture array, `GeometryRef` and material buffers, attribute interpolation
at the hit, alpha-test in the candidate loop.
**Done when:** the cell renders with correct albedo and alpha-tested geometry reads correctly, no
unbound-descriptor validation errors.

**M4a — done.** `rtxmw-texture`, a new crate beside `rtxmw-nif` on the same one-format-one-crate
rule. All **6,256** shipped textures decode: 190.8 MiB across 4,181 BC1, 1,971 BC2, 93 uncompressed
BGRA8 and 11 TGA.

A survey of the shipped data cut the scope before any of it was written, and is worth recording
because the plan above was wrong in three ways:

- **There is no DXT5.** The library is DXT1 and DXT3 only, so BC3 never appears. The corpus test
  asserts that, because a replacer pack introducing it would otherwise render as noise.
- **"BCn transcode where needed" was not needed at all.** DXT1 and DXT3 *are* BC1 and BC2, which
  every target GPU samples natively — decoding is parsing a header and handing the blocks on. The
  scope item is struck from the milestone above. Likewise "mip generation": the files already carry
  their chains, up to eleven levels deep.
- **TGA is 11 files**, all uncompressed 24-bit, against 6,245 DDS. Run-length encoding and colour
  maps are rejected rather than implemented, since no shipped asset exercises them and untested
  format code is worse than an error.

Three decisions worth keeping:

- **`TextureFormat` is not a `VkFormat`,** and says nothing about colour space. The same BC1 bytes
  are sRGB as albedo and UNORM as a replacer pack's normal map, so the consumer chooses and the
  decoder only reports what the bytes are.
- **DXT1 maps to BC1 *with* alpha.** Morrowind uses its one-bit alpha for exactly the foliage and
  grates the alpha test at M4d needs; reading it as RGB-only would discard that.
- **Mip levels share one buffer with a range table beside it**, never `Vec<Vec<u8>>` — eleven-deep
  chains across 6,256 textures would be an allocation each, and it is already the shape
  `vkCmdCopyBufferToImage` wants: one staging buffer, one region per level.

The corpus test initially had a hole worth noting: it checked that the level table tiled the buffer,
which a decoder that *drops* a level still satisfies. Deleting one mip passed cleanly. It now also
asserts that header plus data accounts for the whole file, which ties the decode back to its source
and catches it.

**M4b (host half) — done.** `Material`, `AlphaMode` and `MaterialTable` in `rtxmw-scene`; `Mesh`
gains `submeshes`. Across the whole library: **7,319 meshes flatten into 26,869 submeshes drawing
4,593 distinct materials over 4,311 distinct textures**. Seyda Neen's office alone resolves 119
materials, 118 of them textured, across 309 submeshes.

- **A model is one mesh but rarely one surface.** A lantern is glass and metal, so flattening now
  keeps runs of indices tagged by material. Adjacent blocks sharing a material merge; non-adjacent
  ones do not, because collapsing those would need reordering and the index ranges are what a build
  reads. Each run becomes a geometry within the model's BLAS, which is how a hit names its material.
- **NIF properties are inherited**, so resolution carries a property stack down the node graph
  alongside the transform — a whole building shares one texture property set on its root.
- **The material table is scene-wide, not per mesh**, because that is the granularity the GPU wants:
  one bindless array and one material buffer per cell. Meshes intern into it as they are built, so
  `Submesh::material` is already the index the shader will use.
- **Two path fixups the original data needs**, both quirks rather than conventions: a texture name is
  relative to `textures/`, and it routinely claims an extension the shipped file does not have —
  the art was converted to DDS and the references were never updated.

**45 of 4,311 texture references resolve to nothing.** They are dangling in the shipped data, not a
decoding error: `tx_moon`, `tx_lavacrust00` and `tx_hlaalu_wall1_02` exist nowhere in the archives.
So the upload needs a fallback texture rather than treating a miss as fatal, and the corpus test
asserts a *rate* (under 2%) instead of perfection — a broken path fixup pushes it far past that,
which is what the assertion is really for.

**M4b (device half) — done**, minus the texture array. One acceleration structure geometry per
submesh, a flat `GpuGeometry` table and a `GpuMaterial` table, and a shader that resolves a hit to
its material. The interior renders with surfaces coloured by material identity.

- **`instance_custom_index` carries the mesh's `first_submesh`,** not the mesh index. A hit adds its
  own `geometry_index` and lands directly on the geometry entry — one indexed read, no per-mesh
  indirection to chase first. That is the whole reason the submesh table is flat.
- **Geometry opacity now follows the material.** `OPAQUE` lets traversal commit without invoking a
  shader, which is right for a wall and catastrophic for a grate; the flag is decided per submesh
  from its `AlphaMode` rather than set once for everything.
- **The shader declares `scalar` block layout.** Under default std430 a `vec3` pads to sixteen bytes
  and every table entry after the first is misread — the kind of failure that renders as plausible
  nonsense.
- **Alpha carries an explicit hit flag.** Tests used to infer "did this ray hit" from brightness,
  which worked only while the output was barycentric; a material can legitimately be dark, so the
  shader writes the flag rather than leaving it to be guessed at.

The test that matters is `two_materials_in_one_mesh_shade_differently`: both halves live in the
*same* mesh, so they share an instance and a custom index and only `geometry_index` separates them.
Dropping that term from the shader still fills exactly the same pixels — and fails only this test.

**M4 — done.** `TextureArray` in `rtxmw-render`, UV interpolation at the hit, and the cell renders
with its own albedo. Seyda Neen's office samples **118 textures across 3,133 distinct shades**, with
none of its references missing.

- **The bindless array is the set's last binding.** Vulkan permits a variable descriptor count only
  on the final element, which validation caught the moment the array sat at binding 4 with two
  buffers after it. Adding any binding after it moves it.
- **Slot zero is a magenta fallback**, and a material's texture id addresses `id + 1`. Missing
  textures are a normal case — 45 of the library's 4,311 references name files that were removed —
  so the array absorbs them rather than the caller special-casing a hole.
- **Every format maps to an sRGB view.** These are all albedo; sampling them as UNORM feeds
  gamma-encoded values to a linear renderer, which darkens midtones by about half and cannot be
  tuned out afterwards. Pinned by a test, because it looks merely "a bit dark" rather than wrong.
- **`textureLod`, never `texture`.** Implicit LOD needs screen-space derivatives, which a compute
  shader does not have. Level zero aliases at distance; the real fix is ray differentials, which
  need the pixel footprint carried along the ray.
- **`spirv-val` needs `--scalar-block-layout`.** The shaders declare scalar layout and the device
  enables the feature, but the validator defaults to std430 and rejects a `vec3`-carrying struct
  array for a stride that is correct under the rules actually in force.
- **Anisotropy is deliberately off.** It is a rasterizer's answer to a footprint problem a ray
  tracer should solve with differentials, and enabling it would paper over their absence.

One diagnostic lesson worth keeping: alpha carries the hit flag, and the debug PNG preserved it — so
every missed ray wrote a transparent pixel that a viewer composited as white. Half the frame looked
blown out and I went looking for a sampling bug that did not exist. The pixel values were
`[13, 18, 25, 0]`, the background, all along. The dump now forces alpha opaque.

**M4d — done.** The alpha cutout runs in the candidate loop. `gl_RayFlagsOpaqueEXT` is gone from the
ray query, so the per-geometry `OPAQUE` bits the build sets from each material are what decide
whether traversal asks at all — forcing the flag at the query overrode them and put the rectangles
back regardless of what the build said.

The survey that shaped this is worth recording, because the obvious reading of the data is wrong.
Across the 4,593 materials in the shipped library:

| mode | count |
|---|---|
| Opaque | 3,982 |
| **Blend** | **539** |
| Mask | 72 |

Only 72 materials are *explicitly* alpha-tested, which looks like the cutout barely matters. It is
the 539 blended ones that carry Morrowind's foliage, grates and banners: the game draws them with
`NiAlphaProperty` blending over a texture whose alpha is very nearly binary, not with a threshold.
Treating blend as opaque — which is faster, and which the flag would honestly describe today —
renders every tree as a rectangle. So blended materials get a stand-in cutoff of 0.5 and run the
same cutout path until ordered transparency replaces it. That takes 611 of 4,593 materials off the
fast path, which is the right shape; marking only the 72 would have looked correct and been wrong.

Seyda Neen's office runs the cutout on 14 of its 119 materials. The mask thresholds the data
actually uses are 100 and 192 out of 255, plus two materials at zero.

### M5 — interior direct lighting — **done**
`LightRecord` in `rtxmw-esm`, `Light`/`Ambient` in `rtxmw-scene`, `LightBuffer` in `rtxmw-render`,
and shadow rays in the visibility shader. Seyda Neen's office places **13 lights** — warm torch
orange, radii 64 to 128 units — under an ambient of `(0.038, 0.026, 0.026)`.

- **Both the light colours and the cell ambient are sRGB-encoded** in the file, like everything
  authored for a fixed-function renderer. Using them unconverted makes every light too bright and
  too washed out, in exactly the way an unconverted albedo texture does. Decoded on the way in and
  pinned by tests at mid grey, where the two spaces diverge most.
- **Morrowind stores no intensity.** A `LIGH` record carries a colour and a radius and nothing else,
  because the original renderer's fixed attenuation curve supplied brightness. Radiant intensity is
  therefore derived as `radius² × INTENSITY`, which makes reach the only control the data needs to
  give and keeps a lamp and a candle differing by their size rather than by an invented number.
- **Attenuation is inverse square, windowed to reach exactly zero at the radius.** Morrowind's
  radius is a hard cutoff, and a clipped inverse square leaves a visible edge where the falloff
  jumps to nothing.
- **Carried, negative and off-by-default lights are not placed.** A carried light belongs to whoever
  holds it; a negative one *subtracts* illumination, which is a trick for a renderer accumulating
  into a framebuffer and meaningless for one tracing paths.
- **Shadow rays run the alpha cutout too**, so a grate throws bars rather than a rectangle, and they
  use `TerminateOnFirstHit` — a shadow ray asks only whether something is in the way.

**The interior comes out dark, and that is the honest result rather than a bug.** The ambient is
0.038 and the lights reach 64–128 units inside a room spanning 1,757 × 2,559. Away from a torch,
almost nothing is lit. This is §5.1 arriving from the other direction: the original engine leaned on
pre-lit albedo *and* a flat ambient to carry a room's illumination, so lighting that albedo correctly
leaves it looking underlit rather than double-lit. `INTENSITY` is tuned by eye and provisional —
the de-lighting spike is what would make any value here mean something.

**Soft shadows.** The milestone's done-when is a torch casting a *soft* shadow, so each light is
sampled as a sphere over eight shadow rays rather than as a point over one — visibility becomes the
fraction of the emitter that can be seen, and that fraction is the penumbra. The emitter size is
invented, because Morrowind records none: a light there is a point with a falloff curve, and its
shadows were a decal. It is derived as 8% of reach with a 10-unit floor, so a lantern's penumbra
stays wider than a candle's while the smallest lights do not collapse back to points.

The sample pattern is a **stable** per-pixel hash rather than a per-frame one. Without temporal
accumulation, reseeding each frame turns the penumbra into crawling noise; a fixed pattern dithers
instead, and holds still. That changes at M7, where a denoiser wants the opposite.

The test for this took two attempts, and the first was wrong in an instructive way. Counting distinct
brightness levels across the shadow boundary *passed with a point light*: with no ambient, the lit
wall already varies row to row through attenuation and the cosine term, so a hard shadow produces
plenty of levels too. Only the fault injection showed it. The measure that actually isolates
visibility is the ratio against the same scene with the occluder removed, which cancels the shading
and leaves partial occlusion alone — a point emitter gives zero partly-lit rows, an area one gives a
band.

### Test harness: the renderer is the thing under test

`Renderer` used to own both the scene and the swapchain, which put it in the binary crate where no
test could reach it — so `primary_visibility` assembled its own copy of the load-and-trace sequence.
That replica was a **parallel abstraction**: every assertion was about a reconstruction of the
engine rather than the engine.

Split along the seam §2 already describes. `rtxmw_render::SceneRenderer` owns the pass, the target
and the loaded cell; `rtxmw::Renderer` adds surface, swapchain, frame ring and present. The tests
now drive `SceneRenderer` directly and the replica is gone.

Two things fell out of doing it:

- **The uploader is borrowed, never owned.** Giving each `SceneRenderer` its own made twelve tests
  submit to one queue concurrently, which Vulkan requires external synchronisation for — every
  parallel test failed and every serial one passed. An uploader wraps one command pool on one
  queue, so it is a device-wide resource; the renderer takes `&mut Uploader` everywhere instead.
- **Image readback moved off `RenderTarget`** and onto `readback::image_to_rgba8`, because the image
  worth reading back is the renderer's own output, which no test owns.

**`cargo run -- --screenshot <path>`** renders one frame through the same path on a device brought
up with no surface extensions — no window, works headless, ~0.6 s warm against tens of seconds for
the windowed binary. Cost of the whole verification loop: **2.9 s of tests plus 0.6 s for a picture**,
against ~60 s and a window appearing on the user's screen.

### M6 — indirect lighting — **done**

One diffuse bounce per pixel, cosine-weighted, with next-event estimation at the bounce hit. The
shader was restructured to get it: `trace` now owns a ray query and returns a resolved `Surface`, so
the primary ray and a bounce ray share one traversal rather than two copies of a candidate loop, and
`direct_light` is a function both hits call. `occluded` still keeps its own copy — `glslc` rejects
`rayQueryEXT` as a parameter, so a traversal genuinely cannot be handed around.

**Ambient became the environment's radiance rather than an unconditional fill.** This is the
decision the milestone turns on, and it is what makes zero bounce samples meaningful rather than
black: a bounce ray that escapes the geometry returns the cell's ambient, so with no bounce rays
every direction escapes by definition, the estimator's mean is the ambient itself, and the term
collapses *exactly* to the `albedo * ambient` the renderer applied before. With rays, geometry
occludes that fill where it should. Ambient occlusion is therefore not a separate effect here — it
is the same integral, sampled.

The consequence is that **interiors get darker, and correctly so**. Seyda Neen's office is sealed,
so almost every bounce ray finds a wall instead of escaping; the mean frame brightness falls 9%
(8.36 to 7.60 in 8-bit units) and the loss is concentrated in corners and under furniture, which is
where a flat fill was most obviously wrong. §5.1 again: the albedo already has ambient occlusion
painted into it, so this is the second one. That is a content problem, not a lighting-model one.

**The Lambertian `1/pi` moved into the shader.** It had been folded into `LightBuffer::INTENSITY`,
where a single scale on the only lighting term was unobservable. With a second term integrating over
the hemisphere the ratio between them became real, so the factor went where it belongs and
`INTENSITY` was multiplied by pi to compensate. The direct-lit image is unchanged, which the M5
tests passing untouched is the evidence for.

**Sampling.** Cosine-weighted by Malley's method — a uniform sphere point added to the normal, which
reuses the sphere sampler the soft shadows already had and is exact rather than an approximation.
Because the pdf cancels both the cosine and the `1/pi`, the estimator is albedo times the mean
radiance and carries no other factor. Shadow rays at a bounce hit are cut from eight to one: a bounce
is a fraction of the pixel's radiance and is already being averaged over four directions, so
resolving *its* penumbra would cost thirteen rays a bounce to change nothing visible. Four bounce
samples by default rather than the one the frame budget eventually wants, because nothing accumulates
or denoises yet; that drops to one at M7.

**The variance baseline M7 needs**, measured as RMSE against a 256-sample reference of the same
frame, over the pixels both renders hit:

| samples per pixel | RMSE |
|---|---|
| 1 | 0.0713 |
| 4 | 0.0355 |
| 16 | 0.0174 |

Ratios of 2.01 and 2.04 for each quadrupling — textbook `1/sqrt(N)`, which says the error is Monte
Carlo noise with no bias underneath it. That is the property the test asserts, and a stuck sample
index fails it flat (0.0000227 at every count) rather than merely degrading.

The synthetic scenes are worth stating because they are hand-checkable: a white wall with a coloured
floor at its base, lit by one white light nearly overhead. Every bit of colour on the wall arrived by
reflection, so the red-minus-blue gap at a pixel *is* the indirect term with no reference render to
compare against. The predictions were computed before the trace ran and came back within 2% —
direct 0.188 against 0.188 predicted, indirect 0.151 against 0.156, ambient occlusion 0.323 against
0.313. Both tests read a 17x17 patch rather than a pixel, because at four samples a pixel holds one
of five levels and means nothing on its own.

**Cost is not measured yet.** The bounce work is below run-to-run noise in a single headless frame at
1280x720, which bounds it usefully at neither end. The renderer has no GPU timers, and adding them
belongs with M7, where the frame budget stops being background and becomes the milestone's own
done-when.

**Not done, deliberately:** stratifying the bounce directions (the hash is white noise per pixel, and
a low-discrepancy sequence would cut the same error for the same cost), and sampling one light rather
than looping all of them — thirteen is cheap and lower-variance, but it is `O(lights)` per bounce and
does not survive exteriors. ReSTIR is the answer to the second one when it stops being cheap, and the
plan is still to escalate only when variance demands it.

### Spawning: where a cell puts the player

The camera used to start at the cell's geometry centroid, which for Seyda Neen's census office is
near the ceiling and pointed out through the roof — 47% of primary rays escaped the room. The fix is
not a better guess, because the game already records the answer.

**Morrowind stores the arrival point on the door you leave through, not in the cell you enter.** A
door reference carries `DODT`, a position *in its destination cell*, and `DNAM`, that cell's name.
Two consequences follow, and they pull in opposite directions:

- **Travelling through a door is a local question.** The reference is already in the cell being
  played, so `Door::from_ref` answers it with no search. That is the whole mechanism door travel
  will need.
- **Arriving in a cell without walking through anything is not.** It means finding a door elsewhere
  that leads there — `Door::leading_to`, one pass over all 316,116 references in `Morrowind.esm`,
  about 25 ms. There is no cheaper route, because a cell does not record how it is entered.

`DNAM` is absent for a door to an exterior, where the arrival position identifies the cell on its
own: `CellId::containing` floors the world coordinates by the 8,192-unit grid. Flooring rather than
truncating matters — truncation puts everything between −8192 and 8192 in cell zero and mirrors the
western and southern half of the map onto the eastern and northern one.

**Editor markers place no geometry either.** The same six meshes are the original engine's
placement aids and it never drew them, yet 1,145 references to them are placed across the shipped
game — one `NorthMarker` in the census office alone, a solid 160-unit arrow standing in the room.
`StaticScene` now skips them on the way in, which is one line and one integration test comparing the
cell's meshes against the marker's own vertices rather than against a guess at its shape.

**Editor markers are excluded from the door search too, and finding that out is what made this
work.** `PrisonMarker` is
filed as a `DOOR`, carries a destination into the census office, and is the *first* such door in
file order — so the obvious rule picked it, and the camera started inside the furniture. It is where
the character-generation script drops the player, not a door in a wall. Morrowind names its
placement aids `meshes/Marker_*.nif` and ships exactly six: the north arrow, the door and travel
arrows, the temple and divine intervention targets, and the prison marker. Filtering on that leaves
four real doors into the cell, and a traveller arriving through any of them stands in the room
facing into it.

The remaining pieces are conventions worth writing down because neither is guessable:

- **Yaw is a compass bearing.** The stored rotation turns about the **negated** Z axis, so zero
  faces `+Y` (north) and a quarter turn faces `+X` (east) — the opposite handedness to what a maths
  library gives by default. OpenMW spells the same rotation out at `mwmechanics/combat.cpp:695`.
- **Only the arrival's horizontal position is authored data. Its height is a hint.** Measured
  against the floor directly beneath it — sixteen arrivals across twelve interiors — it sits a
  median of 89 units up and ranges from 22 to 144. The original engine throws the stored height
  away: `World::adjustPosition` (`mwworld/worldimp.cpp:1207`) raises the actor 20 units, traces
  down, and takes whichever is lower. So this drops too, through `StaticScene::ground_below`, and
  stands the traveller on what it finds. That is what makes the camera the same height above the
  floor in every cell, when the authored heights vary by 120 units across the game.

  `ground_below` walks every placed triangle — 0.3 ms over the census office's 46,251, which is
  nothing once per cell and hopeless per frame. It also queries the *visible* geometry, while
  Morrowind ships separate collision meshes this does not read yet. Both are fine for the question
  it answers and neither survives an actor that has to ask every frame; that wants the collision
  meshes and an acceleration structure over them.

- **Standing eye height is 160 units.** An actor's height is twice the median arrival height above
  the floor, since the arrival marks roughly an actor's centre, and eyes sit about nine tenths of
  the way up — the ratio a human has, close to where OpenMW measures line of sight from
  (`mwphysics/mtphysics.cpp:767`). That lands 83% of the way up a 194-unit door, which is where a
  person's eyes are in a doorway.

  Two wrong answers came before that one, and the second is the instructive one. The first used the
  geometry centroid. The second used `MWRender::Camera::mHeight`, 124, on the assumption that the
  arrival was a standing position — but that constant is the *third-person* orbit pivot, applied at
  `camera.cpp:97` only `if (mMode != Mode::FirstPerson)`, and adding a body height to a marker that
  already contained one put the camera at 394 in a room whose ceiling is at 420. A constant lifted
  from a reference implementation without reading the branch it sits in is not a citation.

**What this exposed:** with the camera inside the room rather than looking out through the roof, the
frame is nearly black. That is not a regression — it is the §5.1 darkness the old viewpoint was
hiding behind the miss colour, now unavoidable. The interior renders at roughly 3% of full scale and
there is no exposure or tone mapping anywhere in the pipeline yet, which is M8.

**Not done:** `StaticScene` does not carry the doors a cell places. That is the one line travel
needs and nothing consumes it yet, so it waits for the thing that activates a door.

### M8 — exposure, tone curve and sRGB — **core done**

Everything before this wrote linear radiance and then clamped it to `0..1` on the way out. An
interior traces at about three hundredths of mid grey, so the whole cell arrived at a display as
near-black — and the frame that finally showed the room is the same trace, unchanged, with only the
output path fixed. Nothing about the lighting moved.

**Auto-exposure is measured from a log histogram, not an average.** The scene that makes the case is
the one being rendered: thirteen candle flames at a luminance above one, in a room sitting at 0.02.
A linear mean of that is dominated by whichever population has more pixels — for a frame that is
five sixths dark and one sixth bright, the mean is 1.7 and exposing for it puts the room at 0.002,
which is zero after encoding. The mean of the *logs* is 2⁻⁴·¹, which exposes the room to something
visible and lets the flames roll off. Both numbers are asserted in `tests/output.rs`.

Two dispatches, because a reduction cannot see every pixel's contribution until every workgroup has
finished writing it and a dispatch boundary is the only barrier that wide:

- `luminance_histogram.comp` bins log luminance into 256 bins spanning 2⁻¹⁰ to 2⁶. It tallies into
  shared memory first, so the global buffer sees one atomic per bin per workgroup rather than one
  per pixel. The buffer is cleared with `vkCmdFillBuffer` rather than by the shader — anything
  zeroing it from inside the same dispatch would race the accumulation.
- **Bin zero is reserved for pixels with no light on them at all**, and they are excluded from the
  divisor as well as the sum. Counting them halves the mean bin, which reads as two stops darker
  than the truth and opens the exposure until everything lit is white. Measured on a half-black
  frame: 103 correct, 230 with them counted.
- `exposure.comp` reduces the bins in one workgroup of exactly 256 threads and divides mid grey by
  the result.

**The tone curve is Khronos PBR Neutral**, chosen against the milestone's own done-when rather than
by taste: the test is that an original Morrowind screenshot and a render of the same viewpoint
compare without a gamma mismatch, and the original applied no tone curve at all. This one is the
identity below 0.76 and only rolls off above it, so the midtones the comparison rests on are
untouched — where a filmic curve like ACES would darken and twist every one of them. Its
desaturation term is what keeps a torch flame going white instead of clipping channel by channel
through yellow.

**sRGB is encoded in the shader, and the swapchain was changed to `UNORM` to stop it happening
twice.** sRGB formats expose no storage capability, so a compute shader cannot write one; the
alternatives were to encode in the shader or to leave a *linear* 8-bit intermediate for the
presentation engine to encode, which bands badly in exactly the near-black range an interior lives
in. Encoding here has a second payoff that is worth more than it sounds: **the screenshot is now
byte-identical to what the window shows**, which matters when the entire verification loop is
looking at a PNG and believing it.

The whole chain is pinned by one hand-computable number. A flat frame of *any* radiance should reach
the file at 103: auto-exposure puts it on the 0.18 key, the curve's shadow lift takes 0.04 off it
leaving 0.14, which is below the compression threshold and passes through, and sRGB-encoding 0.14
gives 0.404. Measured: 103, and unchanged across a hundredfold change in scene radiance. The same
assertion catches a missing encode (36), a double one (172) and a missing tone curve (118).

`rtxmw-gpu` gained `ComputePipeline` on the way. Three new passes would each have repeated the
eighty lines of descriptor layout, pool, set, pipeline layout and pipeline that `VisibilityPass`
already had; that pass keeps its own because its bindless texture array needs a variable descriptor
count, which is the one thing the helper deliberately does not do.

**What it exposed:** with the room visible, the dominant artefact is now unmistakably the indirect
lighting's noise — the thing M7 exists to remove, and which was invisible while the frame was black.
That is the argument for having done this first.

**Not done:** bloom, colour grading and sharpening; exposure adaptation over time, which needs a
frame delta this renderer does not yet plumb through and which is only observable once the camera
moves between a lit and an unlit space; and HDR output.

### M7 groundwork — a G-buffer and a denoiser over it

With the frame finally visible, the dominant artefact was the indirect lighting's noise. This is the
half of M7 that needs no external SDK, and none of it is throwaway: DLSS Ray Reconstruction consumes
the same G-buffer, so only the filter itself is replaced.

**The trace no longer writes a picture. It writes what a surface is, separately from what light
reaches it.** That split is the whole idea: all the noise is in the lighting, while the albedo a ray
reads from a texture is exact. Filtering the lighting alone smooths the noise without touching a
texel of surface detail, and recombining is one multiply. A filter run on the composed frame cannot
tell the two apart and blurs both.

- `albedo` — base colour, **half float, not the eight bits a reflectance in `0..1` appears to
  need.** The error does not stay small: the composite multiplies albedo by unbounded illumination,
  so a quantisation step is scaled by however bright the light is. Measured against the same frame
  rendered without the split, eight bits moved the mean pixel by 0.32 of 255 and the worst by 37;
  half float moves the worst by 1. Exact recombination is what the split rests on.
- `normal_depth` — world normal and distance from the eye, together because they are the pair an
  edge-stopping filter tests to decide whether two pixels are the same surface.
- `illumination` — light arriving, albedo divided out, double-buffered for the ping-pong.
- Emissive surfaces and the sky stay in the existing target, which the composite adds on top of.
  They are the one part of the frame that is neither noisy nor demodulated, so they need neither the
  filter nor an albedo to divide by, and keeping them there costs no image at all.

A miss now writes **zeroes** across the G-buffer rather than being skipped. Not tidiness: the filter
reads a neighbourhood, and an untouched pixel holds whatever the allocator left there.

**The filter is an edge-stopping à-trous wavelet**, four passes with the tap spacing doubling each
time — 5×5 taps reaching sixteen pixels either way, where a direct blur of that radius would need
nearly a thousand. Weights are a normal term and a *relative* depth term; relative because
Morrowind's units put a room's walls tens apart and a hillside thousands, and one absolute threshold
cannot serve both. Two pipelines over one shader, differing only in which way their descriptor sets
point, because a pass reads a whole image and writes a whole image and cannot work in place.

Measured on a fixture built to hold three different edges at once:

| | unfiltered | filtered |
|---|---|---|
| neighbour-to-neighbour roughness on a flat lit surface | 8.8 | 0.07 |
| width of an albedo step, in columns | 0 | 0 |

A hundredfold drop in noise with the albedo step still one pixel wide. Fault injection covers each
term separately, and the fixture had to grow twice to earn that: with only a wall and a floor, whose
normals are perpendicular, deleting the *depth* term changed nothing, so a panel parallel to the
wall and identically surfaced was added — two surfaces nothing but depth distinguishes. Composing the
lighting before filtering, the regression this design exists to prevent, spreads the albedo step
over fifteen columns.

**What it cost the test suite:** the M6 convergence test measures the estimator's Monte Carlo error,
which is precisely what a denoiser removes, so it started failing. Every assertion in
`primary_visibility.rs` is about what the *trace* computes — hand-worked radiance, `1/sqrt(N)`
convergence, the fraction of rays that hit — so that file now traces unfiltered, and the denoiser
has a test file of its own.

**Not built: motion vectors.** They need the previous frame's view-projection, and the frame
constants are already at 100 of Vulkan's guaranteed 128 push-constant bytes — a second matrix does
not fit, so they need a small per-frame buffer written with `vkCmdUpdateBuffer`. That is
straightforward, but nothing reads a motion vector until temporal reuse exists, which is M7 proper.

### Frame timing — the first measurement against §5.3

Every performance decision so far had been made blind. There were no GPU timers at all: the only
number anywhere was a frames-per-second counter in the window title, which requires opening a window
— and twice during M6 and M8 a question about what something cost had to be answered with "below
run-to-run noise", because nothing could resolve it.

A timestamp query pool now brackets each stage. **Device time, not wall clock**: a timestamp is
written when everything recorded before it has completed, so the gap between two is what the GPU
spent, which is what a budget is written in. It means anything only because the stages are already
separated by full barriers — without them the device would overlap adjacent work and a per-stage
figure would be a fiction.

Measured on the shipped cell, an RTX 4090 Laptop, with four bounce samples and four à-trous passes.
Medians of five runs, because single figures are not trustworthy here — see the note on spread below:

| internal resolution | total (median) | range | trace | denoise | composite | exposure | tonemap |
|---|---|---|---|---|---|---|---|
| 1280×720 | 1.40 ms | 1.39–1.41 | 0.96 | 0.39 | 0.02 | 0.02 | 0.01 |
| **1920×1080** | **3.43 ms** | 3.31–4.69 | 2.41 | 0.83 | 0.04 | 0.03 | 0.02 |
| 2560×1440 | 5.96 ms | 4.85–9.35 | 4.35 | 1.62 | 0.09 | 0.04 | 0.04 |

**Against the §5.3 budget there is room, though less than a single lucky run suggests.** The target
is 1920×1080 internal upscaled to 3840×2160 at 60 fps — 16.7 ms a frame — with 8 to 9 of those
reserved for DLSS Ray Reconstruction. That leaves roughly 8 ms for everything else, and an interior
uses about 3.4 of it. The à-trous filter's 0.83 ms is also returned when DLSS-RR replaces it.

**Take the spread seriously; it is the measurement's own caveat.** This is a laptop GPU whose clocks
move with thermals and recent load. At 720p five runs agree to within 0.02 ms, at 1080p they span
1.4 ms, and at 1440p nearly two to one. A single run taken straight after the test suite read 5.77
ms at 1080p — 1.7 times the median — so anything measured cold, or immediately after other GPU work,
is not comparable to anything else.

Scaling is close to linear in pixels but not exactly: 2.25 times the pixels from 720p to 1080p cost
2.45 times the time, and 1.78 times again to 1440p cost 1.74. Near enough that there is no hidden
fixed cost to grow into, not so exact that a figure can be extrapolated rather than measured.

And it is an *interior*: 260 instances and 13 lights, every ray terminating on a wall a few metres
away. Exteriors are the case this budget will actually be decided by, and M9 is where that is
tested.

`--screenshot <path> [WIDTHxHEIGHT]` takes a resolution now, because the budget cannot be checked at
the 720p the flag defaulted to.

What the tests can assert about timings is narrow, and deliberately so: no wall-clock threshold,
which would be flaky, but that four filter passes are timed at something and zero passes at nothing,
that the trace outweighs the composite, and — as a pure function, since this machine's queue reports
all 64 bits valid and so never exercises it — that a duration masks off the bits a queue does not
promise, including across a wrap.

### M9 — exteriors: the ground under the world

The first half of the milestone: a named exterior cell loads with its terrain and its objects, and
renders. Streaming, the active grid and distant statics are not built.

**`LAND` is two encodings that both look plausible when decoded wrongly**, which is what the tests
are shaped around:

- **`VHGT` stores gradients, not heights.** Each row's first column steps from the row above it and
  every other column steps from its left-hand neighbour, over a float offset for the whole cell. So
  one cell is a running total down its western edge and 65 running totals across. Read as absolute
  values it still produces a surface — just not this one.
- **`VTEX` is sixteen 4×4 blocks, not a 16×16 grid.** Read flat it scrambles the texturing into the
  right textures in the wrong places.
- **A `VTEX` index is one past its `LTEX` index**, because zero means the region's default rather
  than the palette's first entry. OpenMW resolves the same way at
  `components/esmterrain/storage.cpp:376`, and the default is `_land_default.dds`.

Verified against every `LAND` record in `Morrowind.esm`: 1,292 cells of terrain from −2,152 to
18,952 units, neighbouring vertices never more than 1,016 apart, 548 cells centred below sea level.
Those are Vvardenfell's shape — sea level at zero, Red Mountain at eighteen thousand, an island —
and they are what a corpus catches that a round trip cannot. A further **98 `LAND` records carry no
terrain at all**, only coordinates and a flag word; they are cells that exist for their objects, and
treating them as parse failures would discard 7% of the world's records silently.

**Terrain is placed rather than instanced.** A NIF is authored about its own origin and an instance
transform puts it somewhere; a heightmap belongs to exactly one cell and could never appear
elsewhere, so `Mesh::from_land` writes world coordinates and the instance carries no transform. Its
65×65 grid shares its last row and column with the neighbouring cell, which is what makes adjacent
terrain meet without a seam and why 65 vertices span 64 quads.

Texture tiles become submeshes: the quads of every tile sharing a material are emitted together, so
a cell of marsh, sand and rock is a handful of contiguous runs rather than 256 of them. Seyda Neen
comes out as **eight submeshes over 84 meshes and 289 instances, loading in 19 ms**.

The test that matters names the eight textures exactly. Asserting they belong to the Bitter Coast —
Seyda Neen's region — was not enough, because **the palette turns out to be grouped by region**, so
dropping the index offset still lands on Bitter Coast art: `Tx_BC_rock_01` where `Tx_BC_rock_03`
belongs. Every wrong answer looks as plausible as the right one, so the set has to be spelled out.

**A fault-injection harness reported four false negatives** before that was noticed. It treated a
compile failure as "the test passed", and `mesh.rs`'s unit tests had stopped compiling when
`from_land` changed signature. It now distinguishes *did not build* from *not caught*, which is the
second time in this project that a verification step quietly reported success for work it never ran.

**Not built, and needed before this milestone closes:** the 3×3 active grid and streaming; distant
statics; blending between neighbouring texture layers, which the original engine did and this
replaces with a hard edge per tile; and **the sun**. An exterior carries no `AMBI`, so out of doors
there is nothing lighting the world at all — the placeholder is a fixed overcast daylight standing
in for a sky the weather system will eventually drive.

### The sun — M5's last piece, and what an exterior needed

An exterior carries no `AMBI`, so until now the outdoors was lit by a flat placeholder ambient with
no direction and no shadows. The sun closes that, and closes the one part of M5 that was never
built.

**The direction is Morrowind's own hardcoded constant**, `(-400 · orbit, 75, -100)` from
`apps/openmw/mwworld/weather.cpp:900`, with `orbit` running 1 at sunrise to −1 at dusk. It is not an
astronomical model: the sun runs east to west at a fixed angle from overhead and sits to the south.
It is the direction light *travels*, which is why the shader negates it to point at the disc.

**The angular diameter is ours, not Morrowind's.** OpenMW's sun is a pure direction with no size
anywhere in its codebase, and its shadows are as sharp as the shadow map. Half a degree across is
the real sun, and it is the only reason a shadow has a penumbra rather than an edge. Shadow rays
sample the cone it subtends, uniformly over solid angle rather than over the disc, which keeps the
gradient even instead of bunching samples at its centre.

Measured on a fixture built to show it:

| | penumbra |
|---|---|
| sun with a real half-degree disc | **38 rows** |
| the same light from a point | **0 rows** |

The fixture is an odd shape for a reason worth writing down: **a real sun's penumbra is about one
pixel wide** when the camera and the blocker are the same distance from the surface. Its angular
width is `2·tan(r)·d/D`, so seeing it at all means putting the blocker far and the camera close —
here fifty times closer, which turns one pixel into about seventy-five. Sharp sun shadows are the
correct answer, and the fixture exists to make the softness measurable rather than to make it large.

The lit value is the whole chain in one number: albedo 0.5, the Lambertian `1/pi`, a sun of 2.0
straight on — 0.318, or 81 of 255. Four injections are caught, including the sun's direction not
being negated.

**The push constant block is now exactly 128 bytes — Vulkan's guarantee, in full.** The field order
is load-bearing rather than cosmetic: std430 aligns a `vec3` to sixteen bytes, so the sun's radius
sits *between* its direction and its colour to keep both on a boundary. A unit test reads the actual
field offsets and asserts each is a multiple of sixteen. Anything further has to move the block into
a buffer, which is also what motion vectors will need.

**The sky comes from the ambient rather than from constants of its own**, brighter overhead than at
the horizon. That ties the two together so they cannot disagree, and it removes the hardcoded
blue-grey a missed ray used to return — indoors it now yields the room's own dark fill, which is
what a ray escaping through a doorway should find. The disc is drawn but is **not** energy-consistent
with the directional term: a real sun's radiance is its irradiance over the solid angle it subtends,
some sixty thousand times the value used, and reconciling them means making the directional light an
area light.

**Cost:** the exterior went from 1.75 ms to 3.5 ms at 1920×1080, nearly all of it in the trace —
sixteen shadow rays a pixel across a scene where most of them reach the sky unobstructed. That is
generous, and the first thing to spend down if the budget tightens; temporal reuse at M7 is what
makes one sample a frame viable. Against the ~8 ms §5.3 allowance there is still room.

`Sun::at(orbit)` takes the time of day the engine does not yet track, so exteriors use a fixed
mid-morning. The colour is a warm white scaled by a constant chosen for its *ratio* to the sky —
about five to one, as on a clear day — and is provisional in exactly the way `INTENSITY` is.

### M9 — streaming cells one at a time

The camera flies across a 7×7 window of exterior cells and the window follows it, a cell at a time,
without a stall. Getting there meant throwing away the first answer.

**The window is nearly free past 5×5.** Measured at 1920×1080 on the shipped cell, with four bounce
samples, four à-trous passes and sixteen sun rays:

| radius | window | cells | instances | frame | trace |
|---|---|---|---|---|---|
| 0 | 1×1 | 1 | 289 | 3.12 ms | 2.52 |
| 1 | 3×3 | 9 | 1,700 | 6.93 ms | 6.13 |
| 2 | 5×5 | 25 | 3,888 | 8.03 ms | 7.15 |
| 3 | 7×7 | 49 | 6,406 | 8.16 ms | 7.30 |
| 4 | 9×9 | 81 | 10,545 | 8.15 ms | 7.33 |

Tripling the cells from 5×5 to 9×9 — 2.7 times the instances — costs **0.12 ms**. The extra cells are
beyond anything a ray reaches: the top-level structure dismisses them in log time and the rays that
would have visited them terminated long before. The expensive step is the *first* neighbour, 1×1 to
3×3, which nearly doubles the frame — not because of instance count but because rays that used to
escape to the sky now hit terrain and spawn shadow and bounce rays of their own.

So draw distance is not what this budget is spent on. **Ray termination is.** That reframes what to
tune: sixteen sun samples a pixel, not the size of the world.

#### What the first attempt cost, and why

Streaming began as a *block* reload: cross a boundary, load the 7×7 around the new centre, replace
everything. It worked and it stalled for a fifth of a second. Broken down, one crossing:

| | dev | release |
|---|---|---|
| read `Morrowind.esm` | 20.8 ms | 21.6 |
| build the model index — a pass over the file | 7.9 | 6.6 |
| build the scene — a second pass | 46.8 | 34.4 |
| find the doors leading in — a third pass | 45.4 | 25.7 |
| decode textures | 2.2 | 2.3 |
| upload and build structures | 71.9 | 43.1 |
| **total** | **195 ms** | **133 ms** |

The shape is what matters, and it is not "loading is slow". **Three passes over the same 79 MB
file** to find records whose location never changes, and then a re-upload of a scene that is 90%
identical to the one already resident — 49 neighbouring cells share 378 meshes between them, so the
block arriving after a one-cell step is almost entirely the block that was already on the device.

#### Identity, and two lifetimes

The fix is to stop rebuilding what has not changed, which requires knowing what "the same" means.
Every mesh carries the path it was loaded from (`StaticScene::mesh_sources`; terrain is keyed by its
cell, being the one mesh no other cell can share), and textures already carried theirs. On that
identity the renderer keeps two tiers, split by lifetime rather than by kind:

- **Assets are grow-only** — mesh data, bottom-level structures, textures, materials. Keyed by
  source, uploaded by the first cell to name them, and kept for the life of the renderer. The
  ceiling is the shipped library: 4,311 distinct textures against the array's 8,192 slots, and some
  three thousand meshes. A session that visited every cell in the game would hold all of it, in the
  region of half a gigabyte — so nothing is freed, no slot is renumbered, and a mesh's
  `first_submesh`, which is what an instance carries as its custom index, is stable forever.
- **Cells own placements and nothing else** — instances and lights. Evicting one is dropping two
  lists.

A commit therefore rebuilds only what the resident *set* determines: the geometry and material
tables (~100 KB together), the lights, the instance buffer and the top level. Everything there is
proportional to what is placed, not to what is loaded.

This is why the arena is grow-only rather than free-listed. With eviction the arena would fragment
and mesh slots would have to be reused, which means renumbering, which means rewriting the tables
that name them. Bounding the problem by the game's own art removes all of it. A mod with 2K textures
would blow the ceiling; the path-keyed map is where a refcount and a free list would go, and until
one exists the ceiling is asserted rather than assumed.

#### One walk, then two reads

The file is walked once, on the streamer's thread, into a `CellIndex`: the offset of every cell's
`CELL` and `LAND` record, plus the `LTEX` palette. After that, loading a cell is
`esm.record_at(offset)` twice. The palette has to live there for the same reason the offsets do —
`LTEX` records are scattered with no relation to the cells that use them, so a cell loaded on its
own would otherwise resolve its ground against however much of the palette happened to precede it.

The door search is simply not run for a streamed cell. It was a third of the cost and it answers
"where would someone arriving here appear", which a cell the camera is walking into has already
answered.

#### Hysteresis is two radii

Cells load within Chebyshev radius 3 and are evicted only past 4. Crossing a boundary pulls in a
column and pushes the far column *out of the load window but not out of the kept one*, so stepping
back over the same edge finds it still resident and loads nothing. One crossing settles it; no
timers, no distance-from-edge margin, and one mechanism rather than two.

#### What it costs now

Per cell, with its neighbours already resident: **0.7–3.7 ms to add, 0.6–2.5 ms to commit.** One
cell per frame, deliberately — a boundary crossing wants a whole column, and taking them together
would rebuild the same structure seven times inside one frame. Filling the whole 7×7 window from
cold is 48 cells in **136 ms**, against 195 ms for the single block reload it replaces.

In the windowed engine at 1920×1080: **81 fps while a cell arrives every frame**, 108–112 fps once
the window is full. The stall is gone — the frame that takes a cell is a slower frame, not a
dropped one.

M9's remaining pieces are distant statics and terrain layer blending, which is still a hard edge per
512-unit tile.


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
Per-cell water plane, RT reflection and refraction, absorption and scattering.
**Done when:** standing on Seyda Neen's shore, the seabed is visible through moving water, the
surface reflects the sky and the coast, and the frame still fits §5.3.
**Proposal and plan:** §7.

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

---

## 7. M10 water: proposal and plan

Not yet built. This is the design to argue with before any of it is written.

### 7.1 What the data says

Measured over `Morrowind.esm`, because the shape of the water decides the technique:

| | |
|---|---|
| exterior cells | 1,404, **none carrying a `WHGT`** |
| interiors | 1,134, of which **193 are flagged as having water** |
| interiors carrying a water height | **all 1,134** |
| land cells whose terrain crosses z = 0 | **533 of 1,292** |
| lowest terrain vertex in the game | **−2,152 units**, about 31 m |

Four things follow, and they are the whole brief:

1. **Sea level is z = 0 everywhere outdoors.** No per-cell lookup, no interpolation across a
   boundary, one constant. Vvardenfell is one body of water.
2. **The flag is the gate, not the value.** Every interior carries a water height whether or not it
   has water, so reading `WHGT` and testing for presence would flood 941 dry rooms. `has_water()`
   is `(flags & 0x02) || is_exterior()`, matching `loadcell.hpp:164`.
3. **Water is shallow.** Thirty-one metres at the deepest point in the game, and the Bitter Coast is
   a fraction of that. The seabed is *visible through the surface almost everywhere*, which is what
   makes caustics worth having rather than a detail nobody sees.
4. **The shoreline is the most-seen feature.** 41% of land cells contain one. Whatever happens where
   water meets sand is on screen more than anything else about the water.

### 7.2 What that rules out

**Not an FFT ocean.** Tessendorf's spectrum is the right answer for deep water at kilometre scale
with a horizon: it sums thousands of components for free by doing the sum in frequency space. We
have coastal shallows and interior pools. It would cost a compute pass and three textures a frame to
simulate a spectrum whose defining feature — long deep-water swell — does not belong in a swamp.

**A sum of trochoidal (Gerstner) waves instead**, four to six of them, plus a procedural detail
normal. Analytic, stateless, no texture. Critically it is also **differentiable in closed form**,
which §7.5 needs: caustics come out of the derivative of the surface, so a surface we can
differentiate on paper is worth more here than one that is merely prettier.

### 7.3 The surface, as geometry

One unit quad, instanced once per water cell with the level in its transform. This falls out of the
residency work: water is the *ideal* shared asset — every cell's water is the same mesh, unlike
terrain, which is per-cell by nature. One mesh, N instances, keyed `water`.

It goes in the acceleration structure rather than being intersected analytically in the shader so
that reflections, shadows and the refraction ray all see it without a second code path. A ray that
happens to hit water inside a bounce is then handled by the same branch as a primary hit.

### 7.4 Shading it

On a water hit, with `n` the wave normal and `η = 1/1.333`:

- **Fresnel**, Schlick with `F0 = 0.02`. At grazing angles water is a mirror; head-on it is a
  window. This one term is most of what reads as "water".
- **A reflection ray**, `reflect(d, n)`, traced and shaded — sky above the horizon, the coast below
  it.
- **A refraction ray**, `refract(d, n, η)`, traced to the seabed and shaded, then attenuated.
- **Beer–Lambert absorption** along the underwater path: `T = exp(−σ · depth)`, with σ from a
  **Jerlov coastal water type** rather than a hand-picked blue. Water absorbs red within a metre or
  two and passes green-blue furthest, which is *why* shallow water reads green and deep water blue —
  falling out of the physics rather than being painted in.
- **Single scattering** added back as `σs/σ · (1 − T) · L`, so what the water absorbs it partly
  returns as its own glow.
- **Sun glint**, GGX against the wave normal, with the sun's real angular radius already in the
  frame constants from M5.

**Why this dodges the denoiser.** The à-trous filter is demodulated by albedo — it filters light
arriving at a surface, and water has no albedo. But a mirror reflection and a refraction are
*deterministic*: one ray each, no sampling, no noise. So water shades into the emissive/sky channel
that already bypasses the denoiser, and needs no filtering at all. This is the argument for
perfectly specular water first: it is not a simplification to be undone later, it is the thing that
makes water compose with the denoiser we have.

### 7.5 Caustics, two ways

The seabed is visible nearly everywhere (§7.1), so caustics are not a garnish here.

**First: analytic, from the Jacobian.** Caustics are ray-density change. With an analytic height
field, the refracted direction at any surface point is known in closed form, so the convergence of
the refracted bundle at depth `d` is the determinant of the Jacobian of that map, and intensity is
`1/|det J|`. No photons, no buffer, no filtering — a few ALU per underwater hit, evaluated where the
seabed is already being shaded. This is the same quantity Evan Wallace's light-front mesh computes
with `dFdx`/`dFdy`, and the same stationarity condition the optics literature defines caustics by;
the difference is that a single analytic layer lets it be evaluated pointwise instead of meshed.

It is exact for what we have — **one water layer, one sun, a height field we wrote ourselves** — and
that is precisely the case where the expensive method buys nothing.

**Second, if that disappoints: photon-splatted caustic maps**, the *Ray Tracing Gems II* chapter 30
approach as shipped in UE4's RTX branch. Generate photons through the surface toward the light,
trace them, splat each as a decal sized by photon differentials, filter, composite. Reported at
0.5–2 ms on RTX hardware, roughly halved at half resolution. Our version would generate photons with
ray queries rather than rasterising a caustic map, since we already have the structure to trace
against. Held in reserve: it is the general answer, and generality is exactly what we do not need
yet.

### 7.6 Underwater

Below the surface the same model runs inverted: Beer–Lambert on every primary ray rather than on the
refraction ray, total internal reflection looking up past the critical angle (48.6°, and the reason
the surface from below is a mirror ringed by a bright disc of sky), and the sun's colour filtered by
depth before it lights anything. Sun shafts are a later want, not part of this.

### 7.7 Plan

Four stages, each independently verifiable, each leaving the tree green.

**Stage 1 — the plane exists — done.** `has_water` reaches `StaticScene`, one shared quad is
instanced per water cell, `MaterialKind::Water` selects the model, and the shader branch does
Fresnel, one reflection ray, one refraction ray and Beer–Lambert into the emissive channel. Flat —
no waves.

`Cell::has_water` now encodes the format's rule rather than the flag alone: an exterior has water
whatever the record says. Every shipped exterior sets the flag anyway, so this only matters for
content that does not — which is exactly the case where trusting the flag leaves a hole in the sea.

**It made the frame faster.** At Seyda Neen's shore, 900 frames under sustained load: **134 fps with
water against 108–116 without**. Water was expected to cost about double; instead a water pixel
*replaces* a diffuse one, and a diffuse pixel is the expensive kind — sixteen shadow rays and four
bounce rays, against water's two deterministic rays and their cheap shading. Screenshot timings are
useless for this comparison on a desktop that keeps the GPU at its idle 315 MHz: five runs of the
same frame spanned 6.2 to 11.0 ms, which is why the number above comes from a sustained run.

**Water must not cast a shadow**, and the way it is told not to matters. A surface in the
acceleration structure blocks shadow rays, so the first version of this put every seabed in the
shade of its own sea — measured, the bottom kept 3.6% of its sunlight instead of 85%. Building the
water non-opaque so the any-hit loop could wave shadow rays past *works* and **costs half the frame
rate**: 68 fps against 134, because every shadow ray crossing the sea then invokes a shader where
traversal alone had been enough. Water carries a mask bit of its own instead, and `occluded` asks
only for solid geometry — free, and back to 134 fps. The light is not attenuated on the way *down*
either, which the caustics stage replaces rather than a claim about water.

Verified by four headless tests, each catching one thing a plausible implementation gets wrong —
proven by injection: collapsing the extinction to one channel, dropping the angle from Fresnel,
shading water as an ordinary surface, and letting water occlude each fail exactly one test and no
others.

**Stage 2 — waves.** Four to six trochoidal components plus a detail normal.
*Blocked on one thing:* the push-constant block is **exactly 128 bytes and full**, and waves need
`time`. That forces the frame constants into a uniform buffer — which M7 needs anyway for motion
vectors, so the blocker is work already owed rather than new work.
*Tests:* height at t = 0 reproducible; normals average to vertical over a tile; two amplitudes
produce measurably different surfaces.

**Stage 3 — caustics.** The Jacobian term on the refraction path.
*Tests:* a flat surface gives exactly 1, and the caustic term averages to ~1 over a tile — energy
conservation, which is the property that catches a wrong derivative.

**Stage 4 — shore and underwater.** Depth fade where water meets land, underwater fog, total
internal reflection, optionally foam. Last because it is refinement, first in visibility.

### 7.8 Costs and risks

- **Water pixels cost roughly twice.** Two rays instead of one, each spawning its own shadow rays.
  On a coastal view perhaps 30% of the frame is water, so budget 1–2 ms and measure before tuning
  anything else. The 16-sun-sample count identified in M9 is the lever if it overruns.
- **No new dependency.** Everything here is shader work plus one mesh.
- **The 128-byte block is a hard stop for stage 2**, and nothing before it.
- **The largest unknown is the shoreline**, not the deep water: 533 cells have one, and a hard
  intersection line between a water plane and terrain is the classic tell. Depth-fading the
  absorption toward zero at the intersection is the standard answer and is stage 4 for that reason.
