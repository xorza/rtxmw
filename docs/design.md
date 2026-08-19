# rtxmw — renderer design

Written 2026-08-18. The decisions here are settled; §4 records what is built and what each milestone
measured.

Immediate goal: **render static Morrowind locations with a free noclip camera**, with hardware ray
tracing as the primary rendering mode rather than an effect layered on a rasterizer.

## Decisions

| Question | Decision | Consequence |
|---|---|---|
| Graphics API | **`ash` (raw Vulkan) + GLSL via `glslc`** | wgpu removed. Refit, opacity micromaps, SER and RT pipelines all stay reachable |
| License | **MIT OR Apache-2.0** | DLSS Ray Reconstruction is available; M7 is an integration rather than writing a denoiser |
| Material data | **De-light vanilla textures offline** | Needs its own spike. See §5.1 — de-lighting recovers albedo only |
| Perf target | **1920×1080 internal → 3840×2160 output @ 60 fps** | ~8–9 ms denoiser, ~7 ms for everything else. 16:9 output, not native ultrawide |
| First scene | **Interiors through M8; exteriors at M9** | Fastest path to a correct image; terrain and streaming stay unvalidated until late |
| Windowing | **`winit`** | Gamepad and audio need separate crates when they arrive |

---

## 1. Graphics API: `ash` and GLSL

wgpu 30 was audited against its vendored sources, not its documentation, and removed.

**It does well:** ray queries from compute with a complete WGSL surface — candidate *and* committed
intersections, barycentrics, instance and geometry indices, both object↔world matrices, position
fetch; bindless *including* non-uniform indexing of a `binding_array` by a value read at a hit;
BLAS/TLAS creation with working async compaction; `as_hal` down to `ash::Device` for SDK interop.

**It cannot do these, and none of them is an edge case here:**

| Gap | Consequence for a Morrowind engine |
|---|---|
| **No BLAS refit** — `ALLOW_UPDATE`/`PreferUpdate` are accepted and silently ignored | Every skinned actor is a full BLAS rebuild every frame |
| **No opacity micromaps**, though the hardware exposes `VK_EXT_opacity_micromap` | Morrowind is saturated with alpha-tested foliage, grates and banners — the single biggest RT cost multiplier in the game |
| **No RT pipelines** at the safe API level, and `as_hal` does not reach `BindGroup`/`PipelineLayout`/`ShaderModule` | No SBT-driven material dispatch, no intersection or callable shaders |
| **No shader execution reordering**, though the hardware reports real `REORDER` | Substantial performance unclaimed on divergent hit shading |
| **TLAS instances are a CPU `Vec`**, re-serialized every build | ~1 ms per 1,000 instances upstream, 98.7% of encode time. Exteriors are thousands of small statics |

The corroborating evidence is bevy_solari, wgpu's flagship RT consumer: its tracking issue still
lists transparent and alpha-masked materials, skinned meshes, point lights, environment lighting,
LODs and mipmaps as unsupported — approximately the Morrowind feature set — and it has already
dropped to `wgpu-hal` to escape TLAS-build overhead. wgpu's RT surface has also had breaking changes
in every release since 23.0.0 and sits behind an `unsafe` token documented as *"inherently bugs in
our implementation"*.

**The counter-case, kept because this is the argument to re-read if the decision is revisited.** For
the stated goal alone wgpu is sufficient: static geometry means BLASes and the TLAS are built once,
so refit never fires, and ray-queries-in-compute is a mainstream architecture rather than a
compromise. It would have saved one to two weeks of allocator, descriptor, barrier and SBT plumbing,
and `dlss_wgpu` pins exactly to wgpu 30. Two things defeat it: the gaps above are certainties rather
than risks, and the DLSS advantage is illusory — `dlss_wgpu` exists to inject Vulkan extensions
before device creation, which under `ash` is just calling NGX with a device we own.

**GLSL**, compiled by `glslc` from `build.rs` and validated with `spirv-val`. `GLSL_EXT_ray_tracing`
and `GLSL_EXT_ray_query` are complete, and almost every readable open-source RT reference is GLSL —
Lumen, caldera, Q2RTX's A-SVGF, Godot's RT plumbing, `adrien-ben`'s ash-0.38 examples. For a first
RT renderer, reference availability outweighs language quality. Slang is the better language, has
the most complete RT surface of any option, and both compile to SPIR-V, so shaders can migrate
individually; revisit once the renderer's shape is settled. Rejected: WGSL (a wgpu-proprietary
dialect, dead end off wgpu), HLSL/DXC (knows three RT extensions, no position fetch, micromaps or
SER), rust-gpu (no RT examples, incomplete buffer device address).

---

## 2. Module structure

Workspace crates, in dependency order. Each is a hard boundary; nothing below `rtxmw-scene` knows
about Vulkan, nothing above it knows about ESM records.

```
crates/
  rtxmw-vfs      path normalization, archive layering, BSA readers
  rtxmw-esm      ESM3 reader, record types, RefId, record store
  rtxmw-nif      NIF block reader → geometry, material, node graph
  rtxmw-texture  DDS/TGA decode
  rtxmw-scene    format-neutral scene: meshes, materials, instances, lights
  rtxmw-gpu      Vulkan: instance/device/queues/allocator/descriptors/swapchain
  rtxmw-render   acceleration structures, passes, denoise, post
  rtxmw          binary: window, input, noclip camera, wiring
```

Everything above `rtxmw-scene` is Morrowind-specific and testable headlessly; everything below is a
renderer that would work for any scene. That seam is also what keeps §1 revisable — swapping
`rtxmw-gpu` and `rtxmw-render` leaves the format crates untouched.

---

## 3. Core data types

Flat storage throughout: a Morrowind cell is thousands of small objects and `Vec<Vec<_>>` would
allocate per instance. `StaticScene` is parallel vectors of meshes, materials, textures, instances
and lights plus the cell ambient; a mesh is one flat vertex and index buffer with a `submeshes`
table of index ranges tagged by material.

**A hit resolves through one indexed read.** The TLAS instance's 24-bit custom index carries the
mesh's `first_submesh`; adding the hit's own `geometry_index` lands directly on a `GpuGeometry`
entry, which names the material and the vertex/index offsets needed to interpolate attributes at the
barycentrics. That is the whole reason the submesh table is flat rather than per-mesh.

**Positions get their own tightly packed `Float32x3` stream**, with shading attributes in a parallel
buffer indexed by the same vertex id. Acceleration structure builds read positions with their own
stride and ignore everything else; this is what the build API wants, not an optimization.

**Frame constants** were lifted field-for-field from OpenMW's `components/fx/stateupdater.hpp` —
current and previous matrices and their inverses, eye, fog, ambient, sky and sun, resolution,
near/far/fov, game hour, water height, underwater and interior flags, time, frame number, wind. They
started in push constants, reached Vulkan's guaranteed 128 bytes exactly, and now live in a storage
buffer read with `scalar` layout so a `vec3` packs at four-byte alignment and matches the `repr(C)`
struct field for field.

---

## 4. Implementation plan

Each milestone names a sub-goal, an observable done-when, and the risk it retires.

### M0 — Foundations — **done**
winit window, Vulkan instance/device with the RT extension set, swapchain, a cleared frame,
validation on in debug, noclip camera. **Retired:** the Vulkan plumbing risk. Position fetch, ray
tracing maintenance1 and opacity micromap all report available on the target hardware.

- **Vulkan 1.3, not 1.4.** `ash` 0.38 ships 1.3.281 headers, so `API_VERSION_1_4` does not exist.
- **The swapchain cannot be a storage image.** sRGB formats expose no
  `VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT`, so ray tracing output must go to an offscreen HDR image and
  reach the swapchain through a blit — which makes the offscreen target an M3 requirement rather
  than an M8 concern.

Hardware limits recorded: shader group handle size 32, base alignment 64, max ray recursion depth
31, max BLAS geometries and TLAS instances both 16,777,215.

### M1 — Data: VFS + BSA + ESM enumeration — **done**
Path normalization and archive layering, the Morrowind BSA reader, and enough ESM to read `CELL`,
stream its refs and resolve them to model paths. **Retired:** the format-decode risk.

`esmtool` is not installed, so the cross-check is the file's **own header record count**: walking
Morrowind.esm yields exactly the 48,295 records the header declares. That catches the failure that
matters — a mis-sized record shifting every subsequent offset.

Measured: 20,952 VFS paths across the three BSAs plus loose files (7,319 meshes, 6,256 textures);
1,134 interiors and 1,404 exteriors holding 316,116 references; Seyda Neen's Census and Excise
Office resolves 261 of 268 references to meshes, with zero model paths missing from the VFS.

### M2 — Geometry: NIF — **done**
`NiNode` traversal, `NiTriShape`/`NiTriStrips`, the texturing, material and alpha properties, marker
and `RootCollisionNode` filtering. **Retired:** the largest single format risk.

All 7,319 shipped meshes parse: 4,579,361 triangles, 4,631,142 vertices, 41,702 geometry blocks
passing index-bounds validation. `niftest` is unavailable too, so the cross-check is
self-consistency: every triangle index inside its vertex buffer, every UV set and normal array
matching the vertex count, the block walk landing exactly on the root list.

**Blocks carry no size at version 4.0.0.2**, so a parser off by one byte shifts every subsequent
block and the failure surfaces far from its cause. Four bugs, three of them the same mistake: `bool`
is four bytes at this version, but several fields that look like booleans are declared `char` and
are one. What made them findable was wrapping every block failure in an error carrying the block's
index and type — "6,601 × read past the end" became "6,601 × NiSourceTexture", which names the bug.

### M3 — First light: RT primary visibility — **done**
One BLAS per mesh, compacted; one TLAS over the cell's instances; a ray-query compute pass into an
offscreen HDR target. **Retired:** the whole acceleration-structure pipeline.

**Upload.** Seyda Neen's office uploads 104 distinct meshes — 21,113 vertices, 17,835 triangles —
into 0.85 MiB across three buffers.

- `GeometryBuffers::POSITION_STRIDE` is asserted to be 12, because the stride is a number the build
  is *told* rather than one it derives: padding the vertex would not fail, it would misplace every
  triangle.
- **Indices stay mesh-local**, with `MeshRange::first_vertex` passed to the build as its
  `firstVertex`. Rebasing them into the shared buffer ties a mesh's index data to where it landed,
  and cells relocate meshes at M9.
- **A mesh that flattened to nothing keeps its slot** as a zero-length range, so a `MeshId` stays a
  direct index.
- **`gpu-allocator` pads an allocation to the memory requirement's alignment**, so an 8-byte buffer
  maps 16. `Buffer::mapped` trims to the requested size; without it every readback returns padding.

**Structures.** 104 BLAS and a 261-instance TLAS; compaction takes the bottom level from 1.31 MiB to
0.58 MiB, a 56% saving, which is why `ALLOW_COMPACTION` is on by default rather than optional.

- **One `cmd_build_acceleration_structures` call for all 104**, each with its own scratch region cut
  from a single buffer, with the compaction-size query in the same submission behind a barrier. One
  shared scratch plus a barrier between builds serialises work that is independent by construction.
- **Scratch alignment is `minAccelerationStructureScratchOffsetAlignment`, 128 here**, and is *not*
  satisfied by the buffer's natural requirement. `Buffer::with_alignment` raises the requirement and
  then asserts the resulting address, because a heap that satisfies it by luck stops under
  fragmentation.
- **Instances use `TRIANGLE_FACING_CULL_DISABLE`.** Morrowind authors single-sided planes and relies
  on seeing them from both faces, and winding is inconsistent across the mesh library.

**Verified headlessly, pixel by pixel**, which is what retires the "silently wrong" risks: a wall
100 units ahead covers the centre and misses the corners in the fraction 75° of field of view
predicts; geometry to the **north appears on the left** and **above appears at the top**, which pins
the whole handedness chain — instance transform, projection convention, Vulkan's Y-down NDC, the
shader's unprojection — where a mirror anywhere in it still draws a wall in the wrong place; two
meshes in one buffer both render at their own offsets, the direct test of the build range triple;
and the real cell renders with 48% of pixels hitting geometry across 11,854 barycentric shades.

The camera's projection uses the *offscreen* aspect ratio, not the window's — the trace happens at a
fixed internal resolution and is stretched to whatever the window is.

### M4 — Textures and bindless materials — **done**
All 6,256 shipped textures decode: 190.8 MiB across 4,181 BC1, 1,971 BC2, 93 uncompressed BGRA8 and
11 TGA. A survey of the shipped data cut the scope before any of it was written:

- **There is no DXT5.** The library is DXT1 and DXT3 only. The corpus test asserts that, because a
  replacer pack introducing BC3 would otherwise render as noise.
- **No transcode is needed at all.** DXT1 and DXT3 *are* BC1 and BC2, which every target GPU samples
  natively, and the files already carry mip chains up to eleven levels deep.
- **TGA is 11 files**, all uncompressed 24-bit. RLE and colour maps are rejected rather than
  implemented, since no shipped asset exercises them.
- **`TextureFormat` says nothing about colour space.** The same BC1 bytes are sRGB as albedo and
  UNORM as a replacer's normal map, so the consumer chooses.
- **DXT1 maps to BC1 *with* alpha** — that one bit is exactly the foliage and grates the cutout
  needs.
- **Mip levels share one buffer with a range table beside it**, which is also the shape
  `vkCmdCopyBufferToImage` wants: one staging buffer, one region per level.

The corpus test initially checked only that the level table tiled the buffer, which a decoder that
*drops* a level still satisfies. It now asserts that header plus data accounts for the whole file.

**Materials.** 7,319 meshes flatten into 26,869 submeshes drawing 4,593 distinct materials over
4,311 distinct textures.

- **A model is one mesh but rarely one surface** — a lantern is glass and metal — so flattening
  keeps runs of indices tagged by material. Adjacent runs sharing a material merge; non-adjacent
  ones do not, because collapsing those needs reordering and the index ranges are what a build
  reads.
- **NIF properties are inherited**, so resolution carries a property stack down the node graph
  alongside the transform.
- **The material table is scene-wide**, because that is the granularity the GPU wants: one bindless
  array and one material buffer per cell.
- **Two path fixups the original data needs**: a texture name is relative to `textures/`, and it
  routinely claims an extension the shipped file does not have.
- **45 of 4,311 texture references resolve to nothing** — dangling in the shipped data, not a
  decoding error. The corpus test asserts a *rate* under 2% rather than perfection.

**Device side.** A flat `GpuGeometry` table, a `GpuMaterial` table, UV interpolation at the hit, and
the cell renders with its own albedo — 118 textures across 3,133 distinct shades.

- **Geometry opacity follows the material.** `OPAQUE` lets traversal commit without invoking a
  shader, which is right for a wall and catastrophic for a grate.
- **The shader declares `scalar` block layout.** Under default std430 a `vec3` pads to sixteen bytes
  and every table entry after the first is misread. `spirv-val` needs `--scalar-block-layout` to
  agree.
- **The bindless array is the set's last binding.** Vulkan permits a variable descriptor count only
  on the final element.
- **Slot zero is a magenta fallback** and a material's texture id addresses `id + 1`, so missing
  textures are absorbed rather than special-cased.
- **Every format maps to an sRGB view.** Sampling gamma-encoded albedo as UNORM darkens midtones by
  about half and cannot be tuned out afterwards. Pinned by a test, because it looks merely "a bit
  dark".
- **`textureLod`, never `texture`.** Implicit LOD needs derivatives a compute shader does not have.
  Anisotropy stays off: it is a rasterizer's answer to a footprint problem a ray tracer solves with
  ray differentials, and enabling it would paper over their absence.

The test that matters is `two_materials_in_one_mesh_shade_differently`: both halves share an
instance and a custom index, so only `geometry_index` separates them. Dropping that term fills
exactly the same pixels and fails only this test.

**The alpha cutout runs in the candidate loop**, with `gl_RayFlagsOpaqueEXT` gone from the query so
the per-geometry bits decide whether traversal asks at all. The survey that shaped it reads the
opposite of the obvious way:

| mode | count |
|---|---|
| Opaque | 3,982 |
| **Blend** | **539** |
| Mask | 72 |

Only 72 materials are *explicitly* alpha-tested. It is the 539 blended ones that carry Morrowind's
foliage, grates and banners — the game draws them with `NiAlphaProperty` over a texture whose alpha
is very nearly binary. Treating blend as opaque renders every tree as a rectangle, so blended
materials get a stand-in cutoff of 0.5 and run the same cutout path until ordered transparency
replaces it. Marking only the 72 would have looked correct and been wrong.

### M5 — Direct lighting — **done**
`LIGH` point lights with shadow rays and a defensible attenuation model, cell ambient, and the sun.
Seyda Neen's office places 13 lights, radii 64 to 128 units, under an ambient of `(0.038, 0.026,
0.026)`. **Retires:** the "does this look like Morrowind or like a tech demo" question — see §5.1.

- **Light colours and cell ambient are sRGB-encoded** in the file, like everything authored for a
  fixed-function renderer. Decoded on the way in and pinned by tests at mid grey, where the two
  spaces diverge most.
- **Morrowind stores no intensity** — a `LIGH` record carries a colour and a radius, because the
  original renderer's fixed attenuation curve supplied brightness. Radiant intensity is derived as
  `radius² × INTENSITY`, so reach is the only control the data gives and a lamp differs from a
  candle by its size.
- **Attenuation is inverse square, windowed to reach exactly zero at the radius.** Morrowind's
  radius is a hard cutoff and a clipped inverse square leaves a visible edge.
- **Carried, negative and off-by-default lights are not placed.** A negative light *subtracts*
  illumination, which is a trick for a renderer accumulating into a framebuffer.
- **Shadow rays run the cutout too**, so a grate throws bars rather than a rectangle, and they use
  `TerminateOnFirstHit`.

**Soft shadows.** Each light is sampled as a sphere over eight shadow rays, so visibility is the
fraction of the emitter that can be seen. The emitter size is invented, because Morrowind records
none: 8% of reach with a 10-unit floor, so a lantern's penumbra stays wider than a candle's while
the smallest lights do not collapse to points. The sample pattern is a **stable** per-pixel hash
rather than a per-frame one — without temporal accumulation, reseeding each frame turns the penumbra
into crawling noise.

The test took two attempts. Counting distinct brightness levels across the shadow boundary *passed
with a point light*: with no ambient the lit wall already varies row to row through attenuation and
the cosine term. The measure that isolates visibility is the ratio against the same scene with the
occluder removed, which cancels the shading — a point emitter gives zero partly-lit rows, an area
one gives a band.

**The sun.** Direction is Morrowind's own hardcoded `(-400 · orbit, 75, -100)`, with `orbit` running
1 at sunrise to −1 at dusk — not an astronomical model, and it is the direction light *travels*, so
the shader negates it. The **angular diameter is ours**: OpenMW's sun is a pure direction with no
size anywhere in its codebase. Half a degree is the real sun, and it is the only reason a shadow has
a penumbra; shadow rays sample the cone uniformly over solid angle rather than over the disc.

| | penumbra |
|---|---|
| sun with a real half-degree disc | **38 rows** |
| the same light from a point | **0 rows** |

The fixture is deliberately odd: a real sun's penumbra is about **one pixel wide** when camera and
blocker are the same distance from the surface, since its angular width is `2·tan(r)·d/D`. Sharp sun
shadows are the correct answer, and the fixture exists to make the softness measurable.

**The sky comes from the ambient** rather than constants of its own, brighter overhead than at the
horizon, which ties the two together so they cannot disagree — indoors a missed ray now yields the
room's own dark fill. The disc is drawn but is **not** energy-consistent with the directional term:
a real sun's radiance is its irradiance over the solid angle it subtends, some sixty thousand times
the value used, and reconciling them means making the directional light an area light.

**Cost:** the exterior went from 1.75 ms to 3.5 ms at 1920×1080, nearly all of it in the trace —
sixteen shadow rays a pixel where most reach the sky unobstructed. That is the first thing to spend
down if the budget tightens. `Sun::at(orbit)` takes a time of day the engine does not track yet, so
exteriors use a fixed mid-morning, and the colour is a warm white chosen for its *ratio* to the sky,
about five to one.

**The interior comes out dark, and that is the honest result rather than a bug.** The ambient is
0.038 and the lights reach 64–128 units inside a room spanning 1,757 × 2,559. This is §5.1 from the
other direction: the original engine leaned on pre-lit albedo *and* a flat ambient to carry a room's
illumination, so lighting that albedo correctly leaves it underlit. `INTENSITY` is tuned by eye.

### M6 — Indirect lighting — **done**
One diffuse bounce per pixel, cosine-weighted, with next-event estimation at the bounce hit. `trace`
owns a ray query and returns a resolved `Surface`, so primary and bounce rays share one traversal;
`occluded` keeps its own copy because `glslc` rejects `rayQueryEXT` as a parameter.

**Ambient became the environment's radiance rather than an unconditional fill.** That is the
decision the milestone turns on: a bounce ray that escapes returns the cell's ambient, so with zero
bounce samples every direction escapes by definition, the estimator's mean *is* the ambient, and the
term collapses exactly to the `albedo * ambient` the renderer applied before. With rays, geometry
occludes that fill where it should — ambient occlusion is not a separate effect here, it is the same
integral, sampled. Seyda Neen's office is sealed, so mean frame brightness falls 9% and the loss
concentrates in corners and under furniture, which is §5.1 again: the albedo already has AO painted
into it.

- **The Lambertian `1/pi` moved into the shader.** It had been folded into `INTENSITY`, where a
  single scale on the only lighting term was unobservable; with a second term integrating over the
  hemisphere the ratio became real. The direct-lit image is unchanged, which the M5 tests passing
  untouched is the evidence for.
- **Cosine-weighted by Malley's method** — a uniform sphere point added to the normal, exact rather
  than an approximation, reusing the sphere sampler the soft shadows already had. The pdf cancels
  both the cosine and the `1/pi`, so the estimator is albedo times mean radiance and nothing else.
- **Shadow rays at a bounce hit are cut from eight to one.** A bounce is already being averaged over
  four directions; resolving *its* penumbra would cost thirteen rays a bounce to change nothing.

**The variance baseline M7 needs**, RMSE against a 256-sample reference of the same frame:

| samples per pixel | RMSE |
|---|---|
| 1 | 0.0713 |
| 4 | 0.0355 |
| 16 | 0.0174 |

Ratios of 2.01 and 2.04 per quadrupling — textbook `1/sqrt(N)`, which says the error is Monte Carlo
noise with no bias underneath it. A stuck sample index fails that flat (0.0000227 at every count)
rather than merely degrading.

The synthetic scenes are hand-checkable: a white wall with a coloured floor at its base, lit by one
white light nearly overhead, so the red-minus-blue gap at a pixel *is* the indirect term with no
reference render needed. Predictions computed before the trace ran came back within 2% — direct
0.188 against 0.188, indirect 0.151 against 0.156, AO 0.323 against 0.313. Both tests read a 17×17
patch rather than a pixel, because at four samples a pixel holds one of five levels.

**Not done, deliberately:** stratifying the bounce directions, and sampling one light rather than
looping all of them — thirteen is cheap and lower-variance but is `O(lights)` per bounce. §8.10 is
what that became.

### Spawning: where a cell puts the player

The camera used to start at the cell's geometry centroid, which for Seyda Neen's census office is
near the ceiling pointing out through the roof. The game already records the answer.

**Morrowind stores the arrival point on the door you leave through, not in the cell you enter.** A
door reference carries `DODT`, a position *in its destination cell*, and `DNAM`, that cell's name.
So travelling through a door is a local question answered with no search, while arriving without
walking through anything means finding a door elsewhere that leads there — one pass over all 316,116
references, about 25 ms. There is no cheaper route, because a cell does not record how it is
entered.

`DNAM` is absent for a door to an exterior, where `CellId::containing` floors the world coordinates
by the 8,192-unit grid. **Flooring rather than truncating matters**: truncation puts everything
between −8192 and 8192 in cell zero and mirrors half the map onto the other half.

**Editor markers place no geometry, and are excluded from the door search too.** Morrowind names its
placement aids `meshes/Marker_*.nif` and ships exactly six; 1,145 references to them are placed
across the game, including a solid 160-unit `NorthMarker` standing in the census office.
`PrisonMarker` is filed as a `DOOR`, carries a destination into that office, and is the *first* such
door in file order — so the obvious rule picked it and the camera started inside the furniture.
Filtering on the name leaves four real doors into the cell.

- **Yaw is a compass bearing.** The stored rotation turns about the **negated** Z axis, so zero
  faces +Y (north) and a quarter turn faces +X (east) — the opposite handedness to what a maths
  library gives by default.
- **Only the arrival's horizontal position is authored data; its height is a hint.** Measured across
  sixteen arrivals in twelve interiors it sits a median of 89 units above the floor and ranges from
  22 to 144. The original engine throws it away, raising the actor 20 units and tracing down, so
  this does too, through `StaticScene::ground_below`. That walks every placed triangle — 0.3 ms over
  the office's 46,251, which is nothing once per cell and hopeless per frame — and queries the
  *visible* geometry, while Morrowind ships separate collision meshes this does not read yet.
- **Standing eye height is 160 units**: twice the median arrival height, times nine tenths. That
  lands 83% of the way up a 194-unit door. Two wrong answers came first, and the second is
  instructive — `MWRender::Camera::mHeight`, 124, is the *third-person* orbit pivot, applied only
  `if (mMode != Mode::FirstPerson)`, so adding it to a marker that already contained a body height
  put the camera at 394 in a room whose ceiling is at 420. A constant lifted from a reference
  implementation without reading the branch it sits in is not a citation.

### Test harness: the renderer is the thing under test

`Renderer` used to own both the scene and the swapchain, which put it in the binary crate where no
test could reach it, so `primary_visibility` assembled its own copy of the load-and-trace sequence —
a parallel abstraction whose every assertion was about a reconstruction of the engine.

Split along the seam §2 describes: `rtxmw_render::SceneRenderer` owns the pass, the target and the
loaded cell; `rtxmw::Renderer` adds surface, swapchain, frame ring and present.

- **The uploader is borrowed, never owned.** Giving each `SceneRenderer` its own made twelve tests
  submit to one queue concurrently — every parallel test failed and every serial one passed. An
  uploader wraps one command pool on one queue, so it is a device-wide resource.
- **Image readback moved off `RenderTarget`** onto `readback::image_to_rgba8`, because the image
  worth reading back is the renderer's own output, which no test owns.

**`cargo run -- --screenshot <path> [WIDTHxHEIGHT] [CELL]`** renders one frame through the same path
on a device brought up with no surface extensions, ~0.6 s warm against tens of seconds for the
windowed binary. The whole verification loop is 2.9 s of tests plus 0.6 s for a picture.

### M8 — Exposure, tone curve and sRGB — **core done**
Everything before this wrote linear radiance and clamped it to `0..1` on the way out, and an
interior traces at about three hundredths of mid grey. The frame that finally showed the room is the
same trace with only the output path fixed.

**Auto-exposure is measured from a log histogram, not an average.** Thirteen candle flames above
luminance 1 in a room at 0.02: a linear mean is dominated by whichever population has more pixels —
for a frame five sixths dark, the mean is 1.7 and exposing for it puts the room at 0.002, which is
zero after encoding. The mean of the *logs* is 2⁻⁴·¹.

Two dispatches, because a reduction cannot see every pixel's contribution until every workgroup has
finished writing it and a dispatch boundary is the only barrier that wide:

- `luminance_histogram.comp` bins log luminance into 256 bins spanning 2⁻¹⁰ to 2⁶, tallying into
  shared memory first so the global buffer sees one atomic per bin per workgroup. The buffer is
  cleared with `vkCmdFillBuffer` — anything zeroing it inside the same dispatch would race.
- **Bin zero is reserved for pixels with no light on them**, excluded from the divisor as well as
  the sum. Counting them halves the mean bin, two stops darker than the truth: measured on a
  half-black frame, 103 correct against 230.
- `exposure.comp` reduces the bins in one workgroup of exactly 256 threads.

**The tone curve is Khronos PBR Neutral**, chosen against the milestone's own done-when: the test is
that an original Morrowind screenshot and a render of the same viewpoint compare without a gamma
mismatch, and the original applied no tone curve at all. PBR Neutral is the identity below 0.76, so
the midtones the comparison rests on are untouched where ACES would darken and twist every one. Its
desaturation term keeps a torch flame going white instead of clipping channel by channel.

**sRGB is encoded in the shader and the swapchain is `UNORM`** to stop it happening twice. sRGB
formats expose no storage capability; the alternative, a linear 8-bit intermediate for the
presentation engine to encode, bands badly in exactly the near-black range an interior lives in. The
payoff is that **the screenshot is byte-identical to what the window shows**.

The chain is pinned by one hand-computable number. A flat frame of *any* radiance reaches the file
at 103: auto-exposure puts it on the 0.18 key, the curve's shadow lift leaves 0.14, which is below
the compression threshold, and sRGB-encoding 0.14 gives 0.404 → 103. Unchanged across a hundredfold
change in scene radiance, and the same assertion catches a missing encode (36), a double one (172)
and a missing tone curve (118).

`rtxmw-gpu` gained `ComputePipeline` on the way; `VisibilityPass` keeps its own because its bindless
array needs a variable descriptor count, which is the one thing the helper does not do.

**Not done:** bloom, colour grading, sharpening; exposure adaptation over time, which needs a frame
delta and is only observable once the camera moves between a lit and an unlit space; HDR output.

### M7 groundwork — a G-buffer and a denoiser over it

With the frame visible the dominant artefact was the indirect lighting's noise. None of this is
throwaway: DLSS Ray Reconstruction consumes the same G-buffer, so only the filter is replaced.

**The trace no longer writes a picture. It writes what a surface is, separately from what light
reaches it.** All the noise is in the lighting, while the albedo a ray reads from a texture is
exact, so filtering the lighting alone smooths the noise without touching a texel of surface detail
and recombining is one multiply.

- `albedo` — **half float, not the eight bits a reflectance in `0..1` appears to need.** The
  composite multiplies albedo by unbounded illumination, so a quantisation step is scaled by however
  bright the light is: eight bits moved the mean pixel by 0.32 of 255 and the worst by 37, half
  float moves the worst by 1.
- `normal_depth` — world normal and distance from the eye, together because they are the pair an
  edge-stopping filter tests to decide whether two pixels are the same surface.
- `illumination` — light arriving, albedo divided out, double-buffered for the ping-pong.
- Emissive surfaces and the sky stay in the existing target, which the composite adds on top. They
  are neither noisy nor demodulated, so they need neither the filter nor an albedo to divide by.
- A miss writes **zeroes** across the G-buffer rather than being skipped: the filter reads a
  neighbourhood, and an untouched pixel holds whatever the allocator left there.

**The filter is an edge-stopping à-trous wavelet**, four passes with tap spacing doubling each time
— 5×5 taps reaching sixteen pixels either way, where a direct blur of that radius needs nearly a
thousand. Weights are a normal term and a *relative* depth term; relative because Morrowind's units
put a room's walls tens apart and a hillside thousands.

| | unfiltered | filtered |
|---|---|---|
| neighbour-to-neighbour roughness on a flat lit surface | 8.8 | 0.07 |
| width of an albedo step, in columns | 0 | 0 |

A hundredfold drop in noise with the albedo step still one pixel wide. The fixture had to grow twice
to earn that: with only a wall and a floor, whose normals are perpendicular, deleting the *depth*
term changed nothing, so a panel parallel to the wall and identically surfaced was added — two
surfaces nothing but depth distinguishes. Composing the lighting before filtering, the regression
this design exists to prevent, spreads the albedo step over fifteen columns.

The M6 convergence test measures the estimator's Monte Carlo error, which is precisely what a
denoiser removes, so `primary_visibility.rs` now traces unfiltered and the denoiser has its own
file.

### M7 — Denoise and upscale — **not built**
DLSS Ray Reconstruction via NGX: denoise, antialias and upscale in one pass, 1920×1080 → 3840×2160,
with no separate TAA — running TAA over a temporally-accumulated denoiser double-blurs and compounds
ghosting. Needs the full G-buffer including the specular guide. **Done when:** a still frame at 1
spp is comparable to a 1024-sample reference by a numeric metric, and the frame holds 60 fps at the
§5.3 target. **Retires:** the biggest performance unknown.

Practical notes: NGX ships real Linux `.so` files and needs specific Vulkan instance and device
extensions **at creation time**; OTA model updates are broken on Linux, so the baked-in model is the
one that gets used; Streamline is Windows-only, so call NGX directly.

**Motion vectors are the missing piece.** They need the previous frame's view-projection, which is
what pushed the frame constants out of push constants into a storage buffer.

### Frame timing — the first measurement against §5.3

A timestamp query pool brackets each stage. **Device time, not wall clock**: a timestamp is written
when everything recorded before it has completed, so the gap between two is what the GPU spent. It
means anything only because the stages are already separated by full barriers.

Measured on the shipped cell, RTX 4090 Laptop, four bounce samples, four à-trous passes. Medians of
five runs:

| internal resolution | total (median) | range | trace | denoise | composite | exposure | tonemap |
|---|---|---|---|---|---|---|---|
| 1280×720 | 1.40 ms | 1.39–1.41 | 0.96 | 0.39 | 0.02 | 0.02 | 0.01 |
| **1920×1080** | **3.43 ms** | 3.31–4.69 | 2.41 | 0.83 | 0.04 | 0.03 | 0.02 |
| 2560×1440 | 5.96 ms | 4.85–9.35 | 4.35 | 1.62 | 0.09 | 0.04 | 0.04 |

**Against the §5.3 budget there is room.** 16.7 ms a frame with 8–9 reserved for DLSS-RR leaves
about 8 ms, and an interior uses 3.4 of it; the à-trous filter's 0.83 ms is also returned when
DLSS-RR replaces it.

**Take the spread seriously.** This is a laptop GPU whose clocks move with thermals and recent load.
At 720p five runs agree to within 0.02 ms, at 1440p they span nearly two to one, and a single run
taken straight after the test suite read 5.77 ms at 1080p — 1.7× the median. Scaling is close to
linear in pixels but not exactly: 2.25× the pixels from 720p to 1080p cost 2.45× the time. Near
enough that there is no hidden fixed cost, not so exact that a figure can be extrapolated.

And it is an *interior*: 260 instances, 13 lights, every ray terminating on a wall a few metres
away.

### M9 — Exteriors: terrain and streaming — **done**
`LAND` decode, terrain as BLAS geometry, texture layer blending, the active cell grid, distant
statics. **Note:** this is where TLAS instance counts get large — the point at which wgpu's CPU
marshalling would have become the bottleneck. It did not become one here: the 477 cells resident
from a hilltop rebuild the top level once a frame without showing up in the budget.

**`LAND` is two encodings that both look plausible when decoded wrongly**, which is what the tests
are shaped around:

- **`VHGT` stores gradients, not heights.** Each row's first column steps from the row above and
  every other column from its left-hand neighbour, over a float offset for the whole cell. Read as
  absolute values it still produces a surface — just not this one.
- **`VTEX` is sixteen 4×4 blocks, not a 16×16 grid.** Read flat it scrambles the texturing into the
  right textures in the wrong places.
- **A `VTEX` index is one past its `LTEX` index**, because zero means the region's default.

Verified against every `LAND` record: 1,292 cells from −2,152 to 18,952 units, neighbouring vertices
never more than 1,016 apart, 548 centred below sea level. Those are Vvardenfell's shape — sea level
at zero, Red Mountain at eighteen thousand, an island — and they are what a corpus catches that a
round trip cannot. A further **98 `LAND` records carry no terrain at all**, only coordinates and a
flag word; they exist for their objects, and treating them as parse failures would discard 7% of the
world's records silently.

**Terrain is placed rather than instanced.** A heightmap belongs to exactly one cell and could never
appear elsewhere, so `Mesh::from_land` writes world coordinates and the instance carries no
transform. Its 65×65 grid shares its last row and column with the neighbouring cell, which is what
makes adjacent terrain meet without a seam and why 65 vertices span 64 quads. Texture tiles become
submeshes — the quads of every tile sharing a material emitted together — so Seyda Neen comes out as
eight submeshes over 84 meshes and 289 instances, loading in 19 ms.

The test names the eight textures exactly. Asserting they belong to the Bitter Coast was not enough,
because **the palette is grouped by region**, so dropping the index offset still lands on Bitter
Coast art: `Tx_BC_rock_01` where `Tx_BC_rock_03` belongs.

**A fault-injection harness reported four false negatives** before that was noticed: it treated a
compile failure as "the test passed", and `mesh.rs`'s unit tests had stopped compiling when
`from_land` changed signature. It now distinguishes *did not build* from *not caught* — the second
time in this project that a verification step reported success for work it never ran.

**Not done:** the layer blend is bilinear between tile *centres* (§8.8), not the per-vertex blend
map the original engine used, and distant terrain is one stride rather than a hierarchy (§8.9).

### M9 — Streaming cells one at a time

**The window is nearly free past 5×5.** Measured at 1920×1080 with four bounce samples, four à-trous
passes and sixteen sun rays:

| radius | window | cells | instances | frame | trace |
|---|---|---|---|---|---|
| 0 | 1×1 | 1 | 289 | 3.12 ms | 2.52 |
| 1 | 3×3 | 9 | 1,700 | 6.93 ms | 6.13 |
| 2 | 5×5 | 25 | 3,888 | 8.03 ms | 7.15 |
| 3 | 7×7 | 49 | 6,406 | 8.16 ms | 7.30 |
| 4 | 9×9 | 81 | 10,545 | 8.15 ms | 7.33 |

Tripling the cells from 5×5 to 9×9 costs **0.12 ms** — the extra cells are beyond anything a ray
reaches, the top level dismisses them in log time, and the rays that would have visited them
terminated long before. The expensive step is the *first* neighbour, 1×1 to 3×3, and not because of
instance count: rays that used to escape to the sky now hit terrain and spawn shadow and bounce
rays. **Draw distance is not what this budget is spent on. Ray termination is.**

**The first attempt was a block reload** — cross a boundary, load the 7×7 around the new centre,
replace everything. It worked and stalled for a fifth of a second:

| | dev | release |
|---|---|---|
| read `Morrowind.esm` | 20.8 ms | 21.6 |
| build the model index — a pass over the file | 7.9 | 6.6 |
| build the scene — a second pass | 46.8 | 34.4 |
| find the doors leading in — a third pass | 45.4 | 25.7 |
| decode textures | 2.2 | 2.3 |
| upload and build structures | 71.9 | 43.1 |
| **total** | **195 ms** | **133 ms** |

The shape is what matters: **three passes over the same 79 MB file** to find records whose location
never changes, then a re-upload of a scene 90% identical to the resident one — 49 neighbouring cells
share 378 meshes.

**Identity, and two lifetimes.** Every mesh carries the path it was loaded from (terrain is keyed by
its cell, being the one mesh no other cell can share), and on that identity the renderer keeps two
tiers split by lifetime rather than by kind:

- **Assets are grow-only** — mesh data, bottom-level structures, textures, materials — keyed by
  source, uploaded by the first cell to name them, kept for the life of the renderer. The ceiling is
  the shipped library: 4,311 distinct textures against the array's 8,192 slots, some three thousand
  meshes, in the region of half a gigabyte if a session visited every cell. So nothing is freed, no
  slot is renumbered, and a mesh's `first_submesh` is stable forever. With eviction the arena would
  fragment and slots would be reused, which means renumbering, which means rewriting the tables that
  name them. A mod with 2K textures would blow the ceiling; until a refcount exists, the ceiling is
  asserted rather than assumed.
- **Cells own placements and nothing else** — instances and lights. Evicting one is dropping two
  lists.

A commit therefore rebuilds only what the resident *set* determines: the geometry and material
tables (~100 KB), the lights, the instance buffer and the top level — all proportional to what is
placed, not to what is loaded.

**One walk, then two reads.** The file is walked once into a `CellIndex`: the offset of every cell's
`CELL` and `LAND` record, plus the `LTEX` palette. The palette has to live there for the same reason
the offsets do — `LTEX` records are scattered with no relation to the cells that use them, so a cell
loaded on its own would resolve its ground against however much of the palette happened to precede
it. The door search is simply not run for a streamed cell: it was a third of the cost and answers
"where would someone arriving here appear", which a cell the camera walks into has already answered.

**Hysteresis is two radii.** Cells load within Chebyshev radius 3 and are evicted only past 4, so
crossing a boundary pushes the far column out of the load window but not out of the kept one and
stepping back finds it resident. One crossing settles it; no timers, no margin, one mechanism.

**Cost:** per cell with neighbours resident, **0.7–3.7 ms to add, 0.6–2.5 ms to commit**. Filling
the whole 7×7 window from cold is 48 cells in **136 ms** against 195 for the block reload. In the
windowed engine at 1920×1080: **81 fps while a cell arrives every frame**, 108–112 once full. The
frame that takes a cell is a slower frame, not a dropped one.

### M10 — Water — **done**
Per-cell water plane, RT reflection and refraction, absorption and scattering, analytic caustics.
Design and findings: §7. **Not done:** foam at the shoreline, and sun shafts underwater.

---

## 5. Decisions still open

### 5.1 Vanilla Morrowind has no material data, and its albedo is pre-lit

OpenMW states it plainly: *"Morrowind format NIF files do not support normal maps or specular
maps."* Vanilla assets are 256²-era diffuse textures with **lighting, shading and ambient occlusion
painted into the albedo**, authored for a fixed-function renderer with no per-pixel lighting.

Physically-based ray tracing on top of pre-lit albedo double-lights everything: surfaces read flat
and muddy, ambient occlusion appears twice, and no denoiser tuning fixes it. This is the single
largest threat to "looks great", and it is an art-pipeline problem no renderer architecture solves.

Four options, not mutually exclusive:

1. **Accept it and tune.** Cheapest. Will beat vanilla, will not look like a modern RT title.
2. **Support replacer texture packs.** Morrowind has a mature HD/PBR replacer ecosystem and OpenMW's
   `_n` / `_nh` / `_spec` filename conventions are the de-facto standard. Cheap, and shifts the
   quality ceiling onto the pack.
3. **Synthesize normal and roughness from diffuse.** Height-from-luminance is unreliable; a small
   image model would do better. Medium effort, uncertain payoff.
4. **De-light the vanilla textures offline** — estimate and divide out baked shading.
   Research-grade, and the only path that makes *vanilla* assets physically correct.

**Decided: option 4.** One implication: **de-lighting recovers base colour only.** Normal and
roughness still need option 2 or 3, and since `Material` carries those slots as first-class either
way, supporting the filename conventions is nearly free and worth doing alongside.

The spike should run offline against a few dozen textures, judged by eye against the same surfaces
under flat lighting. The failure mode to watch is over-correction — flat, washed-out output where
the algorithm removed genuine painted detail. Keep the vanilla textures as a fallback so a
regression in the bake is always visible as an A/B.

### 5.2 Licensing — settled: MIT OR Apache-2.0

Permissive, so the NVIDIA RTX SDK licence's source-disclosure prohibition does not apply and **DLSS
Ray Reconstruction is available**. Under GPL-3 the denoiser would have had to be hand-rolled SVGF
from the BSD-3 references — roughly 700 lines of shader to work, 2,000 to be good.

The cost is a fat G-buffer: DLSS-RR requires diffuse albedo, specular albedo, normals, roughness,
colour, depth, motion vectors and a specular guide, plus jitter offset and a reset flag. The
specular guide in particular is easy to forget and awkward to add late.

### 5.3 Output resolution is the hardest constraint in the project

**Confirmed against DLSS itself (§8.19):** asked what to render at for a 3840×2160 output under
Performance, it answers **1920×1080** — the number this section assumed and the whole frame budget is
measured at.

Denoising scales with *output* pixels, not internal ones. DLSS-RR alone costs ~6.1 ms at 3200×1800
on a 3080; even scaling generously for Ada, anything above 4K output spends 10–14 ms on the denoiser
before a single ray is cast. Driving a high-resolution ultrawide at its native mode is off the
table.

**Decided: 1920×1080 internal → 3840×2160 output at 60 fps** — DLSS Performance mode. 8.3 M output
pixels, so roughly **8–9 ms of denoiser**, leaving about **7 ms of a 16.6 ms frame** for
acceleration structures, rays, GI and post. If the renderer turns out efficient enough the internal
resolution moves up to 2560×1440 at no architectural cost.

The good news is Morrowind-shaped: ~1.1 GB of game data, 256² textures, very low-poly meshes. BLAS
memory and build time are a non-issue in a 16 GB budget. The pressure is **TLAS instance count** and
**alpha-tested foliage**.

### 5.4 Smaller decisions — settled by default

- **Light units.** Physical units throughout, 69.99125109 units per metre as the single conversion
  constant. `LIGH` radius maps to inverse-square with a radius-derived cutoff, not Morrowind's
  original curve — matching vanilla attenuation exactly would fight the renderer.
- **Emissive vs analytic.** A torch is *both* a `LIGH` record and a glowing mesh. Use the record as
  an analytic light and mark the mesh emissive **but excluded from light sampling**, so it appears
  bright to primary and reflection rays without being counted twice.
- **Sky and environment.** Morrowind's sky is procedural; render it to a small cubemap once per
  frame and sample it.
- **Colour management.** sRGB textures, linear working space, tonemap at the end. The most common
  reason RT renders look washed out, and nearly free to get right on day one.
- **Asset cache.** Re-parsing 500 MB of BSA and every NIF each launch will dominate iteration time.
  Plan a converted-asset cache keyed by content hash before it becomes painful.
- **Debug affordances.** A debug-view selector, shader hot reload, a headless golden-image mode.
- **Scope boundary for "static".** M0–M8 render the **bind pose only, no particles, no animation** —
  animated doors appear in their rest state, NIF controllers are parsed but not evaluated, creatures
  and NPCs are not placed. A deliberate cut, and what keeps the missing-BLAS-refit problem out of
  scope until the renderer is proven.

---

## 6. Dependencies

| Crate | Purpose | Note |
|---|---|---|
| `ash` | Vulkan | 0.38 is stale but carries the full RT surface plus opacity micromap and invocation reorder. 0.39 will be a large breaking rewrite — pin and revisit |
| `gpu-allocator` | device memory | `ash` provides none |
| `winit` | window + input | gamepad and audio will need separate crates |
| `glam` | math | `Affine3A` matches the 3×4 transforms the AS API wants |
| `bytemuck` | `#[repr(C)]` GPU structs | |
| `dotenvy` | locating the game install | production behaviour, not a test helper |
| build-time: `glslc` | GLSL → SPIR-V | external tool, shelled from `build.rs` |
| build-time: NGX SDK | DLSS Ray Reconstruction | **not a crate and not in this repository** — see below |

**The NGX SDK is fetched, not vendored.** It is NVIDIA's, under the RTX SDK licence, so it lives in
`.refs/dlss` beside the OpenMW checkout and is gitignored like it:

```
git clone --depth 1 https://github.com/NVIDIA/DLSS.git .refs/dlss
```

`DLSS_SDK_DIR` overrides the location. The binding is hand-written FFI against the C header — no
generator and no crate — and the whole path sits behind the `dlss` feature.

**The feature requires the SDK**, and `build.rs` fails with the command to fetch it if it is absent.
The first version warned and compiled the feature out so that `--all-features` would build without
it — and that put "is it really here" in a `cfg` only `rtxmw-render` can see, which the binary then
had to gate on and could not. `--all-features` therefore needs the SDK fetched, the way the tests
need the game installed.

---

## 7. Water

### 7.1 What the data says

| | |
|---|---|
| exterior cells | 1,404, **none carrying a `WHGT`** |
| interiors | 1,134, of which **193 are flagged as having water** |
| interiors carrying a water height | **all 1,134** |
| land cells whose terrain crosses z = 0 | **533 of 1,292** |
| lowest terrain vertex in the game | **−2,152 units**, about 31 m |

1. **Sea level is z = 0 everywhere outdoors.** No per-cell lookup, no interpolation across a
   boundary. Vvardenfell is one body of water.
2. **The flag is the gate, not the value.** Every interior carries a water height whether or not it
   has water, so reading `WHGT` and testing for presence would flood 941 dry rooms. `has_water()` is
   `(flags & 0x02) || is_exterior()` — an exterior has water whatever the record says, which only
   matters for content that does not set the flag, and that is exactly where trusting it leaves a
   hole in the sea.
3. **Water is shallow**, so the seabed is visible through the surface almost everywhere, which is
   what makes caustics worth having.
4. **The shoreline is the most-seen feature** — 41% of land cells contain one.

### 7.2 A sum of trochoidal waves, not an FFT ocean

Tessendorf's spectrum is the right answer for deep water at kilometre scale with a horizon: it sums
thousands of components for free by doing the sum in frequency space. We have coastal shallows and
interior pools, and it would cost a compute pass and three textures a frame to simulate a spectrum
whose defining feature — long deep-water swell — does not belong in a swamp.

A direct sum of sinusoids is **differentiable in closed form**, which is what makes the analytic
caustic possible at all: a closed-form height field can be differentiated twice, and an FFT texture
cannot without another pass.

### 7.3 The surface and its shading

One unit quad, instanced per water cell with the level in its transform — water is the *ideal*
shared asset, unlike terrain, which is per-cell by nature. It goes in the acceleration structure
rather than being intersected analytically so that reflections, shadows and refraction rays all see
it without a second code path.

On a water hit, with `n` the wave normal and `η = 1/1.333`: **Fresnel** (Schlick, `F0 = 0.02`),
which is most of what reads as "water"; **a reflection ray** and **a refraction ray**, each traced
and shaded; **Beer–Lambert absorption** along the underwater path with σ from a Jerlov coastal water
type rather than a hand-picked blue, which is *why* shallow water reads green and deep water blue;
**single scattering** added back so what the water absorbs it partly returns; and **sun glint**, GGX
against the wave normal.

**This dodges the denoiser by construction.** The à-trous filter is demodulated by albedo and water
has none — but a mirror reflection and a refraction are *deterministic*, one ray each, no sampling.
Water shades into the emissive/sky channel that already bypasses the denoiser. Perfectly specular
water is not a simplification to be undone later, it is what makes water compose with the filter.

### 7.4 Caustics from the Jacobian

Caustics are ray-density change. With an analytic height field the refracted direction is known in
closed form, so the convergence of the refracted bundle at depth `d` is the determinant of the
Jacobian of that map and intensity is `1/|det J|` — a few ALU per underwater hit, evaluated where
the seabed is already being shaded. No photons, no buffer, no filtering. This is the same quantity
image-space methods (wavefront meshes, caustic maps, photon splats) estimate by splatting; a single
analytic layer lets it be evaluated pointwise instead.

Held in reserve if that ever disappoints: photon-splatted caustic maps, the *Ray Tracing Gems II*
chapter 30 approach, reported at 0.5–2 ms on RTX hardware.

### 7.5 Underwater

The same model inverted: Beer–Lambert on every primary ray rather than on the refraction ray, total
internal reflection looking up past the critical angle of 48.6° — which is why the surface from
below is a mirror ringed by a bright disc of sky — and the sun's colour filtered by depth before it
lights anything.

### 7.6 What was built

**Stage 1, the flat plane, made the frame faster.** At Seyda Neen's shore over 900 frames under
sustained load: **134 fps with water against 108–116 without.** A water pixel *replaces* a diffuse
one, and a diffuse pixel is the expensive kind — sixteen shadow rays and four bounce rays, against
water's two deterministic rays.

**Water must not cast a shadow, and how it is told matters.** The first version put every seabed in
the shade of its own sea, keeping 3.6% of its sunlight instead of 85%. Building the water non-opaque
so the any-hit loop can wave shadow rays past *works* and **costs half the frame rate** — 68 fps
against 134 — because every shadow ray crossing the sea then invokes a shader where traversal alone
had been enough. Water carries a mask bit instead and `occluded` asks only for solid geometry: free.

**Waves.** Trochoidal components with real dispersion, `sqrt(gk)` with Morrowind's own gravity, so
the long waves outrun the short ones and the pattern never sets into one rigid shape. The quad stays
flat and only the normal moves: displacing two triangles buys nothing a normal does not, and the
silhouette against a shore comes from the terrain behind it. **Waves cost nothing measurable** —
flattening the surface over two thousand frames gives 131–134 fps against 131–133 with them.

- **A wave shorter than the pixel looking at it is averaged away rather than drawn**, using the ray
  cone footprint already carried for texture LOD. A cone a wavelength wide covers a crest and a
  trough whose slopes cancel; picking one of them instead is what makes distant water a field of
  crawling white sparks.
- **What a ray cone cannot resolve is not gone, it is rough.** A surface that lost its slope
  reflects like polished plastic. The variance of the discarded octaves comes back as a widened
  specular lobe — LEAN mapping's argument in the one dimension it needs. Its most visible
  consequence is the sun: a mirror shows one hard dot, a mile of ruffled water shows a shimmering
  road, because the glitter path **is** the wave-slope distribution made visible. The disc is
  widened by the lobe and dimmed by the same factor, so a rougher sea spreads the sun without adding
  light.
- **Which side of the surface a ray is on is a question about the plane, not about a wave.** Taking
  it from the wave normal reads a facet tilted away at a glancing angle as "the camera is
  underwater", sends the reflection into the seabed, and turns the far water white. A facet still
  facing away after that is standing in for self-occlusion a height field does not model.

**Caustics.** `J = I - bend·depth·H` with `H` the Hessian of the same sinusoids the normals come
from — written out, not sampled or splatted. The finding that made it work was not about caustics:
`water_ray` traced reflection and refraction at the *bounce* cone spread, one unit of width per unit
travelled, where a coarse mip is correct for a diffuse bounce. A reflection and a refraction are
specular and carry the pixel's own cone; at the bounce rate a seabed a hundred units down was
sampled with a hundred-unit footprint — every texture at its top mip and every wave averaged out of
the caustics the same footprint governs. The caustic term was varying by 25% on its own and arriving
at the frame as 4%. Fixing the cone turned the pattern on and sharpened every reflection in the
game.

**Where the model stops.** `q = p - bend·grad(h)` holds while the refracted bundle has not crossed
itself. Past the first focus the rays have folded, and because the term is evaluated at the seabed
rather than at the surface it came from it starts *making* light — three quarters more at four
hundred units. The depth fed to the lens is capped at 140 units, which holds the error under 6% at
every depth and says something true anyway: caustics are sharp in a shallow pool and washed out in
deep water.

**Chromatic dispersion is kept and worth almost nothing.** Cauchy's fit gives 1.3326, 1.3342 and
1.3392 at 600, 550 and 450 nm, so three determinants over a Hessian that does not depend on the
channel — two extra multiply-adds of numbers already in registers. **Twelve pixels in ninety
thousand differ by more than one level, none by more than two.** Kept because it is right and free;
if the sea ever gets steep enough for the determinant to approach zero, this is what puts prism
edges on cusps.

**Shore and underwater.** The waterline fades over the last thirty-five units of depth, and a camera
below the surface fogs every primary ray. Total internal reflection came free out of `refract`
returning zero past the critical angle.

- **The seam is a grazing-angle artefact.** From above, three units of water is almost invisible
  whatever the shader does — the first test compared views straight down, passed, and went on
  passing with the fade deleted entirely. Edge-on, Fresnel turns that same water into a mirror, so
  the pixel where the ground all but touches the surface reflects sky while the shore beside it
  shows sand.
- **Underwater the *albedo* is dimmed, not the lighting.** The filter divides lighting by albedo, so
  dimming both would put the water straight back, and what the depth took is a property of the path.
- **A ribbon of flat colour along every waterline**, found from a screenshot rather than a test. The
  refraction ray was offset to the *far* side of the plane, so wherever the bed sat nearer the
  surface than the 1.5-unit offset, the ray began under the ground, travelled down through open air
  and reported water of unbounded depth — pure scattering colour, metres wide on a gentle shore.
  Both water rays now leave from the viewer's side and trace against solid geometry only; culling
  water from its own reflection and refraction removes the self-intersection the offset existed to
  avoid. Every test passed throughout, because the waterline test used three units of water, twice
  the offset, and the artefact lives below it.
- **Deep water was a milky sheet, and the cause was the scattering albedo rather than the colour.**
  Absorption takes light out of the scene; scattering hands it back, so a channel whose scattering
  albedo approaches one settles at a bright colour however deep it gets. Clear tropical water does
  behave that way; a tannin-stained coastal swamp does not. Halving the albedo makes the deep go
  dark while lowering the extinction keeps the shallows transparent — the two complaints pull on
  different terms.
- **The in-scattering integral was wrong in the direction that made it worse.** Light scattered
  toward the eye has to reach the point it scattered from, and only the return leg was attenuated.
  Integrating both replaces `1 - T` with `(1 - T²) / 2`: identical in the shallows, half as bright
  where it settles, and markedly less red, because squaring the transmittance costs red twice.
- **The sun was attenuated on its way down and the sky was not**, so an underwater surface was lit
  by a dimmed sun alongside a full-strength sky — inconsistent, and flat with depth. Found by
  measuring an invariant rather than by looking: the same column of water seen from ten units above
  and ten below has to agree, and does, to 3%. At a *slant* they legitimately differ by 11% —
  entering at 53° a ray bends to 37° and reaches a floor 200 units down in 250 units of water, where
  the same look from below costs 317. **Water really is clearer from a boat than from under it**,
  and that asymmetry is now pinned by a test rather than mistaken for a bug.

The extinction and scattering coefficients are art direction resting on physics. The tests derive
every expectation from a single `EXTINCTION` constant so a tuning pass is one line rather than five
pieces of arithmetic that quietly stop describing the shader.

### 7.7 The spectrum is empirical, and its short end is a limit in time

**Caustics tiled when the octaves were spaced too widely.** The light on the seabed came out as a
lattice of near-identical cells while the surface itself looked fine, and the reason is in the
derivative: **curvature weights an octave by `A k²`** where slope weights it by `A k`. A hand-tuned
geometric series with gain 0.55 and lacunarity 0.618 gives a curvature ratio of **1.44 — above one**
— so however many octaves are summed the finest one or two own the Hessian entirely, and two plane
waves crossing is a grid. Slope's ratio was 0.89, so every octave contributed there.

Two fixes, and the second alone still leaves a visible grain:

- **Space the octaves closely** so five or six components land at comparable short scales pointing
  in different directions — broad in direction where the swell is narrow.
- **Carry the ripples on the swell.** A low-frequency displacement field applied to the sample
  position before the waves are evaluated: physically, short waves riding the orbital motion of long
  ones; in practice it bends the crests so the pattern wanders instead of tiling. Thirteen units of
  drift is most of a wavelength to the shortest waves and a rounding error to the longest. The
  Hessian is taken with respect to the drifted position, dropping the chain rule's contribution from
  the drift itself — that field turns over six hundred units against a curvature set by ten, so its
  Jacobian is within a fifth of the identity and the omission shows up as a slow variation in
  caustic strength indistinguishable from what real water does.

**The series is now the TMA spectrum** — JONSWAP under Kitaigorodskii's shallow-water attenuation —
spread over directions by **Donelan-Banner**, which is Horvath's pairing and the one the real-time
literature follows. Thirty-two components: eight wavenumber bands, four directions each, sampled by
*quantile* of the directional spread so every component carries the same energy and the spread's
shape is exact however few are taken.

- **The depth term is the coastal correction this game needs.** A six-metre swell over half a metre
  of water travels at two thirds of its open-sea speed; over three metres it travels at full speed.
  That is why this is TMA rather than JONSWAP.
- **The spread is frequency-dependent by construction** — the lowest band fans across 68° and the
  highest past 120° — which is the empirical form of what the tiling symptom pointed at.
- **The maths is testable in Rust**, where the old constants were three numbers in a shader with
  nothing to check them against.
- `alpha` never appears: it is a constant multiplier on the whole spectrum, so it cancels, and the
  table is scaled instead to a **significant wave height** — the one number about a sea a person can
  picture.

**The short cutoff is a limit in time, not in space.** Carrying waves down to four units produced
dense per-pixel noise: TMA's tail is the Phillips saturation range, where steepness is *constant*
with wavenumber, so `A k²` climbs without bound. Raising the cut to eighteen units produced the best
caustics this renderer has drawn — and made them **tear**, changing by 73% of their own contrast
every twelfth of a second, which reads as stripes ripping across the bottom rather than as water.
That is a trade to choose, not a bug to fix: **a wave's period falls with its length, so the waves
that focus hardest are the ones that move fastest. They are the same waves.**

| shortest wave | caustic contrast | change per twelfth of a second |
|---|---|---|
| the old hand-tuned series | 17.7 | 49% |
| 18 units | 24.6 | 73% |
| **32 units** | **18.5** | **51%** |
| 50 units | 16.5 | 33% |

**Choppiness is in and changes almost nothing to look at** — 788 pixels of a shore by at most a
twentieth of their value, contrast 18.07 to 18.28. The displacement's Jacobian contributes the
steepness `A k`, which sums to 0.28 across the spectrum, while the refraction term contributes
`bend·depth·A k²`, ten times that at a few metres of water. Choppiness would matter on a surface
steep enough to fold, and these waves cannot reach the trochoid limit at any choppiness. One thing
it buys outright: with displacement the map from surface to seabed is a *ratio* of determinants — a
patch covers `det(I + dD)` of surface and lands on `det(I + dD − bend·H)` of bottom — which is 1 at
zero depth by construction, where a single determinant would brighten the bottom of a depthless
puddle.

**This surface does not focus light; it modulates it by tens of percent**, and soft cells are the
correct answer for a 15° sea over a coastal shelf. Focus needs `bend·depth·A k² ≈ 1`; summed over
the whole spectrum with every octave aligned in phase and direction, which never happens, that
reaches 1.21 at the deepest water the term is allowed. Three plausible culprits were cleared:

| suspected | measured |
|---|---|
| the brightness ceiling clipping cusps | raised from 3 to 8: pixel-identical |
| the denoiser blurring the filaments | filtering off: pixel-identical |
| too little curvature in the spectrum | 14 octaves to 18, shortest wave 8.4 units to 2.5: finer and noisier, not bolder |

That last is the useful negative: curvature grows with wavenumber, but the cells it focuses are the
size of those waves, so the pattern gets *finer* rather than sharper, and below a pixel it is noise.

**A wind-chop band was built, measured and reverted.** Bold caustics need roughly nine times the
energy a swell-shaped spectrum puts in the metre band, and a second log-normal peak at the scale of
local wind chop supplies it — a bimodal sea of swell plus locally raised wind waves is ordinary
oceanography, not a fudge. **It does what it was supposed to**: the determinant reaches zero and the
seabed gets bright filaments with loops and cusps, contrast over a shore patch rising from 20.0 to
24.2 with a narrow band and 27.7 with a wide one. It costs more than that is worth:

| cost | measured |
|---|---|
| the whole sea roughens | distant water's pixel-to-pixel variation 14.0 → 23.1, the exact failure mode the ray-cone filtering exists to avoid |
| caustics alias where they sharpen | stipple 9.1 → 21.9 wide, 9.8 narrow |
| **the water stops obeying Beer–Lambert** | looking up from below, transmission fell to 0.649 of the near view where the analytic answer is 0.807 |

The third settled it. A surface rough enough to focus light that hard refracts the view through it
far enough that straight-line attenuation no longer describes what comes out — and absorption,
scattering and the sun's path to the seabed are all built on that attenuation. A band-limited
widening of the curvature cone was tried against the aliasing and works (stipple below baseline at
3.5×) but is too blunt: the same widening that cleans a close view erases the pattern at a middling
one. Something that widened with the *sharpness* rather than the distance would be the right shape.

**Vvardenfell's water is a sheltered coastal shelf, and a sheltered coastal shelf does not throw
pool caustics.** Soft cells are the honest answer.

### 7.8 Not built

Foam at the shoreline and sun shafts underwater. Foam has a natural source already computed: the
Jacobian determinant's **sign** detects surface self-intersection, and where a surface folds is
exactly where whitecaps belong — the shader currently takes its absolute value.

**No absolute frame rate from this machine is worth quoting.** Measurements across the water work
ranged from 116 to 382 fps for the same scene, because the GPU idles at 315 MHz and only ramps to
2,280 under sustained load, so a short run measures the ramp. Back-to-back A/B pairs — flat against
wavy, guard on against off — are the only comparisons that survived.

---

## 8. Findings

Things that were wrong, or were measured and turned out not to be worth it. Each is here because the
mistake was not visible from the code and something else would have made it again.

### 8.1 Ray offsets follow the triangle's plane, not the shading normal

Rugs sparkled along their edges and smooth-shaded rocks came out under salt-and-pepper, and **the
noise changed strength from triangle to triangle**, which is what made the diagnosis. Every ray left
a surface offset along the *interpolated* normal, which on geometry this coarse can point tens of
degrees away from the triangle it belongs to — so the ray origin lands under the surface on some
triangles and over it on others, and a shadow ray that starts underneath is stopped by the surface
it started from.

Rays now leave along the triangle's own plane, from the cross product of its edges. **Which way that
plane faces is the subtlety, and getting it wrong blacks out half an interior:** turned toward the
*ray*, a shadow ray leaving the back of a tapestry sets off for the sun, meets the tapestry, and
reports shadow. Morrowind hangs single-sided planes everywhere, so the plane has to face the side
the surface is *lit* from — the side its shading normal points at. Every leaf on every tree is a
back-facing hit.

### 8.2 The shading normal faces the ray

Dark dust over every tree, speckle on interior tapestries, sparkle along a rug's edge: one defect,
and neither the denoiser nor the alpha cutout the shape of it kept suggesting. **The shading normal
was whichever way the vertices were authored, so a surface hit from its far side reported the light
landing on its near one** — lit through its own body. Morrowind's foliage is thousands of single
cards packed below a pixel apiece, wound every which way, so neighbouring pixels landing on cards
facing opposite ways came back at opposite brightnesses.

The normal now faces the ray, which is what `gl_FrontFacing` does for every rasteriser. It could not
be done before §8.1: while offsets were taken along the shading normal, turning it toward the viewer
sent shadow rays out from under the surface and blacked out the census office — nine distinct shades
where there were hundreds.

**Which side the ray met is decided by the triangle's plane, not by the normal being turned.** The
obvious reading is wrong: an interpolated normal near a silhouette can point away from a face the
camera is looking straight at, so the test flips part of a surface and not the rest, along a seam
that slides across the floor as the camera moves. A plane cannot disagree with itself that way.

That rests on the winding agreeing with the authored normals, which nothing in the format enforces,
so it was measured: **77 of 60,215 triangles** across a furnished interior and a stretch of shore
are wound against their own normals, a fifth of a percent. Three of this repo's own *fixtures* were
— quads written normals-first with the indices copied from a neighbour — and had never been wrong
before, because nothing consulted a winding until now.

Five suspects were ruled out first, each by one render:

| suspected | measured |
|---|---|
| alpha cutout coin-flipping per pixel | cutoff swept 0.15/0.5/0.85: 17.8 / 19.0 / 17.1 — real but a tenth of it |
| the mip level the cutout is tested at | +3 levels of bias: 19.0 → 17.3, visually unchanged |
| alpha coverage drifting across mip levels | corrected to hold at 44%: 19.0 → 19.8, *worse* |
| the indirect gather under-sampled | 4 → 64 bounce samples: dust unchanged, so not stochastic at all |
| shadow rays through the canopy | sun forced fully visible: dust unchanged |

Rendering **albedo alone** came back clean, which put it in the lighting rather than the surface.

Two things found along the way and deliberately not acted on: Morrowind's cutout art is black
wherever it is transparent — 1,449 of the canopy texture's 1,635 fully transparent blocks against
one of its 955 opaque ones — so a filtering sampler mixes black into every leaf edge, and dividing
it back out by the alpha changed nothing measurable. And `NiStencilProperty` would name two-sided
surfaces outright, except that the three shipped archives contain **no** stencil property at all.

### 8.3 Sheets are lit from both sides

**Morrowind hangs single layers of triangles and the renderer treated every one as the skin of a
solid.** A layer has no inside: lit from the far side it should glow, and shaded as a solid it goes
black, taking its whole neighbourhood with it wherever a triangle is wound backwards. A run carries
a `thin` bit into `GpuGeometry`, and the shading term for one is `max(N·L, 0) + T·max(−N·L, 0)` with
`T = 0.5` — a Lambertian sheet, view-independent. The indirect gather takes the hemisphere the ray
came from rather than the one the normal names, so an inside-out triangle no longer gathers the
inside of the hull it is nailed to.

**Deciding which runs are sheets is the whole difficulty, and two plausible tests each fail on real
data.** Asking what a run *encloses* works for a flat rug and fails for a curved sail, because an
open surface encloses the cone it subtends. Asking whether it has a *border* — an edge with one
triangle on it — is exact for closedness but cannot be asked of a run: a run is a material boundary,
so a wine bottle with three textures arrives as three open patches. Measured that way, 268 of the
census office's 308 runs classified as cloth, chests and bottles included.

The test that holds asks both, at the level each is meaningful: the border of the whole *mesh*, the
enclosed volume of the *run*. Edges are keyed by quantised **position**, not by vertex index —
Morrowind splits vertices at every texture seam, so two triangles sharing an edge routinely name
four different vertices for it, and counted by index every seam reads as a border and every solid as
cloth. A room's shell is open at the ends where it meets the next section but wraps the room's air,
so it stays solid and a lamp next door does not shine through the wall: the wall's brightness moves
by 0.02% while the tapestries' speckle drops by a quarter.

**The shape test finds a rug and a sail and cannot find a tree.** A canopy is hundreds of leaf cards
joined at the branches, and the cupped cluster wraps as much air as a shell around a room —
`Flora_BC_Tree_02` scores 0.031 against a cube's 0.068, on the solid side of any threshold that
keeps a room solid, and splitting the run into connected pieces does not help because the cards
really are connected. **The material knows.** A run whose alpha is anything but opaque is a cutout,
and Morrowind has no solid cutouts: the mode is set on foliage, thatch, banners, grates and glass,
every one a single layer with nothing behind it. Together the two signals mark 47 of a shore's 419
runs and 50 of the office's 308; the tree comes out 3 runs of 5, the other two being trunk and
boughs. Backlit foliage is 27% brighter with no measurable change in noise.

### 8.4 A model's outermost node transform is discarded

Seyda Neen's fireplace presented its back to the room with the fire burning behind the stack, while
every other object around it was right — which is what made it hard, since a wrong *convention*
would have turned the whole room.

`in_nord_fireplace_01.nif` carries a half turn about Z on its outermost node, and **the original
engine ignores that transform** — for block zero only, only when that block is a node, and never for
one named `bip01`. 455 of the 7,317 shipped models carry a turn there and 423 a translation or
scale; 259 are rooted at `bip01`, 255 of those with a real transform, so the exception is as
load-bearing as the rule: discarding a rig's animation root would flatten every piece of armour in
the game.

The anchor that needed no screenshot: **a hearth and the fire inside it are placed as separate
references**, so the fire says which way the stack faces. Measured from the fireplace's own origin
the hearth slab pointed at −0.99 against the fire, and at +0.99 once the rule was in.

### 8.5 A reference's Euler angles apply Z first

A book standing through the side of its cupboard, a rock lying at an angle no one placed it at.
OpenMW composes the rotation as `Quat(z, -Z) * Quat(y, -Y) * Quat(x, -X)`, which reads as X-first
and was transcribed that way. It is not: **OSG writes its quaternion product in the opposite order
to everyone else**, so that expression means Z first and X last. The tell is a page away in OpenMW's
own source — `Misc::toEulerAnglesZYX` recovers the angles `makeOsgQuat` was given, and only inverts
it under the reversed reading. Transcribed both ways over four thousand random angles, reversed
round-trips to zero and the ordinary reading is out by up to two units of a unit vector.

The argument that delayed this is worth naming so it is not made again: OpenMW writes the *same*
order for Bullet a few lines below, Bullet's product is the ordinary one, and the two were assumed
to agree because physics ought to match graphics. They do not agree — that is a latent inconsistency
in OpenMW, not evidence about OSG.

It moves only references that turn about more than one axis, 22 of one interior's 268, which is why
a room looks broadly right either way, and the obvious measurements are blind to it: a plate's tilt
out of horizontal is *identical* under both orders whenever the Y angle is zero. What separates them
is that things stop resting where they were put — the book's base sits 8 units below the board its
cups stand on, which is the test.

### 8.6 `NiStencilProperty` was five bytes short, and nothing could have noticed

It read flags, a versioned `bool` and five words. The format is flags, a **one-byte** enabled flag —
a `char`, not the four-byte `bool` this NIF version uses elsewhere — and seven words. Twenty-six
bytes read against thirty-one.

**Blocks in this version carry no size**, so a property that reads one field too few leaves the file
pointing into its own tail and everything after it decodes as garbage, silently. No shipped mesh
could catch it — **0 of 7,319 name the block** — which is exactly why the corpus test passed and why
this needed pinning per block rather than by parsing more files. Every fixed-size property now has
its byte count asserted directly: a synthetic block of exactly the documented length must be
consumed whole, and one byte shorter must fail rather than quietly stop. The same mistake in the
other direction is annotated three blocks away at `NiSourceTexture`, where a `char` read as a `bool`
over-consumed by three.

### 8.7 The unprojection lost the world, and it looked like jitter

Reported as the camera shaking when it moved, everywhere except near the world origin. That last
detail is the whole diagnosis. The shader turned a pixel into a ray like this:

```glsl
vec4 target = frame.inverse_view_projection * vec4(ndc, 1.0, 1.0);
vec3 direction = normalize(target.xyz / target.w - frame.camera_position);
```

The unprojection lands a **world-space point on the near plane**, 0.05 units from the eye, and the
subtraction recovers the direction. Both operands are the size of the world; their difference is
0.05. At Seyda Neen's 75,000 units the gap between representable `f32` values is 0.024, so the
answer had two or three bits left in it.

| camera | aim error |
|---|---|
| the world origin | 0.00 px |
| 1,000 units out | 2.1 px |
| Seyda Neen | **127 px** |
| the far corner of Vvardenfell | **377 px** |

It read as jitter rather than as a broken projection because the error is a *smooth* field — a still
frame looks like a slightly wrong field of view, and only when the camera moves does the field move
with it.

**The fix is not more bits.** Sending the matrix as `f64` would buy an order of magnitude and leave
the same subtraction in place. The subtraction is what should not exist: **the eye is removed from
the view before the inverse is taken**, so the unprojection lands in a space centred on the camera
and hands back an offset directly.

```rust
let mut rotation = view;
rotation.w_axis = glam::Vec4::W;   // a look-at view is rotation * translate(-eye)
(projection * rotation).inverse()
```

Nothing then cancels, the aim is **0.00013 pixels wrong wherever the camera stands**, and the matrix
is far better conditioned since no entry is the size of the world any more.

**Why nothing caught it.** The unprojection had a test, and it was a round trip — correct, and blind
to this, because it ran with the camera at `(1, 2, 3)`. **A precision fault that vanishes at the
origin needs a test that leaves it**, so the assertion is now made at four camera positions out to
the far corner of the map, and the same scene rendered at the origin and at 200,000 units has to
produce the same picture. The old unprojection fails the second on pixel coverage alone.

**What is still `f32`, and why that is fine.** Hit positions quantise to 0.024 units out there, but
the shadow ray bias is 1.5 units, a terrain blend weight moves by 5e-5 of a tile, and a wave phase
by 0.005 radians. Terrain vertices are baked in world space and carry the same 0.024, but that is a
static property of the geometry rather than something that moves when the camera does.

### 8.8 Terrain blends four tile centres, and the quadrant is what carries them

A cell's `VTEX` names one texture per 512-unit tile, 16×16 of them and nothing in between, so ground
met its neighbours along a straight edge and Seyda Neen's shore was a staircase running diagonally
across the slope. The fix is bilinear blending between the four nearest tile *centres*; the design
question is where the four ids live.

**Not a splat map.** The obvious shape is a per-cell weight texture and a second binding, and it is
not worth it: the weights are a fixed function of world position and need no storage. What needs
storing is *which four* textures a point blends, and that is constant over a region.

**The quadrant.** Split each tile into 2×2 quads — 32×32 quadrants per cell — and give each the 2×2
block of tile centres it falls inside. The four ids pack as 4×u16 into the two words `GpuMaterial`
already had spare, so **no new binding, no new buffer, and 48 bytes stays 48 bytes.** They intern
like any other material: a cell has 1,024 quadrants and about 78 distinct tuples (worst measured
121), because most quadrants sit inside a run of tiles naming the same texture.

A cell origin is a whole number of tiles, so it cancels and the weights are a function of world
position alone, 0 at the lower-left centre and 1 at the upper-right, wrapping exactly where the
quadrant's own tuple shifts by one tile so the ramp stays continuous.

**The first version of that ramp was wrong in two ways, and §8.14 is what it should have been.** It
ran linearly from one tile centre to the next, and it stopped at the cell boundary.

**Cost:** three extra texture taps on ground pixels — **6.60 ms against 5.95 ms** of trace at
1920×1080 on a view that is almost entirely terrain, and less than that with a horizon in frame. An
early-out for the case where all four ids agree, most of a cell, measured at **6.60 ms, no change at
all**: the taps are not serialised on anything a branch can skip, and the branch is not coherent
across a warp. Reverted.

**Proof.** `tests/terrain.rs` renders a plane through the real `SceneRenderer` with the four layers
set to red, green, blue and white, so green *is* the x weight and blue *is* the y weight, and
predicts every pixel to within two levels of 8-bit quantisation.

### 8.9 A horizon costs 1.5 ms, and the lights would have more than tripled it

Before this the world ended at the streaming window, about 410 m, and from anywhere with a view the
land stopped mid-slope against the sky. Vvardenfell is only some 36 cells across, so the fix is not
a level-of-detail hierarchy but a second tier: `CellDetail::Distant` out to twelve rings, terrain
and objects, at a sixteenth of the triangles.

**Decimating is enough, and stitching is not needed.** `Mesh::from_land` takes a stride; at 4 a cell
is 17×17 vertices and 512 triangles rather than 65×65 and 8,192. Because the stride divides the 64
quads a cell spans and the shared last row is kept whatever it is, **two cells at the same stride
still meet vertex for vertex** — asserted exactly, not to a tolerance. Only the one ring where the
detailed window meets the distant tier can crack, and over five real cells the coarse chord parts
from the fine surface by at most **64 units, half a vertex spacing** — under three pixels at 1080p,
at a boundary never closer than three cells. Skirts and stitching were designed and not built.

**Cost at 1920×1080 from a hilltop over the whole island**, the worst case there is since every
pixel is horizon:

| | trace | window load |
|---|---|---|
| the 7×7 window alone | 4.77 ms | 208 ms |
| + distant terrain, 12 rings | 5.40 ms | 511 ms |
| + the objects on it | **6.29 ms** | 1,155 ms |
| + the lights those objects carry | 9.85 ms | — |

**That last row is the finding.** Distant objects — 21,772 instances over 428 cells — cost 0.89 ms.
The 229 `LIGH` references among them cost **3.8 ms more**. A lamp a kilometre away with a radius of
a few hundred units reaches nothing on screen, so a distant cell places its lamps and drops their
lights, and the image is unchanged. Peak GPU memory with the whole visible world resident is 620 MB.

**Residency and detail are separate questions, and keeping them separate is what stops a hole.** The
first shape evicted a cell whose tier no longer matched where it was — and a camera crossing one
boundary demotes a whole column at ring 3, so every crossing deleted seven cells of good coarse
terrain and showed sky until the detailed copies loaded. Now `still_resident` is blind to the tier
and only asks whether a cell has left the world's edge, while `rebuild_as` asks whether to *request*
the other tier, so the swap happens in the frame the replacement lands rather than the frame the
camera moved. The hysteresis lives in `rebuild_as`: a cell earns a rebuild only a whole ring past
the boundary.

**Cells arrive sixteen a frame, and "arrive" has to include the misses.** Of the 625 squares the
horizon asks for only 428 exist — the rest are open sea with no record — and a loop that stops as
soon as a cell fails to place still drains one sea square a frame:

| | frame 50 | frame 100 | full |
|---|---|---|---|
| stopping on a miss | 163 cells | 268 | ~250 frames |
| taking sixteen either way | 234 cells | 429 | ~100 frames |

What made one-at-a-time the rule for the near window was the top-level rebuild, and that happens
once per frame however many cells landed in it.

### 8.10 Lights are binned into a world grid

The shader walked every light in the scene for every shading point, primary and bounce alike, with
no bound on count or distance: **0.031 ms per light per frame at 1920×1080** whether or not it
contributes. Three filters were tried on the *geometry* before the lights were suspected, and their
failure is what pointed at them — removing 99% of the instances saved 0.9 ms while removing the
lights saved 3.8. **A cost that does not move when the geometry does is not the geometry's.**

**A uniform grid over the lights, in world space** rather than screen space, because a bounce hit is
not on screen and a screen-space cluster list could not answer for it. Cell `i` owns
`indices[offsets[i]..offsets[i + 1]]` — a prefix sum with a trailing sentinel, so the whole
structure is two buffers however many cells it has, built by a counting sort at the same commit that
rebuilds the top level. The cell size starts at one terrain tile and **doubles until the grid fits
two budgets**: 65,536 cells and 262,144 index entries. The second is not redundant — a wide world
overruns the first, and one light with an enormous reach overruns the second while the grid is
small.

Vivec is the worst light density in the game: 173 lights in one 7×7 window, against 53 in Balmora
and 20 in a typical interior.

| lights | with the grid | walking them all |
|---|---|---|
| 173, as shipped | **4.78 ms** | 6.91 ms |
| +500 | 5.06 ms | 19.03 ms |
| +2,000 | 5.05 ms | 66.00 ms |

Flat where the old loop was linear, and 1.8 ms — a quarter of the trace — on the real scene. Below
about fifty lights the two are within noise, which is the honest bound: **this buys a town, not a
room.**

**The output is bit-identical.** Forcing the grid to a single cell reproduces the old walk through
the same code path and the two renders differ in 0 of 2,073,600 pixels. The grid may offer a light
that turns out not to reach — it bins by bounding box and the shader's distance test settles it —
but it must never withhold one that does, and that one-sidedness is asserted against the brute-force
answer over a sweep of probes. Binning by a light's centre instead of its reach fails it.

### 8.11 Cell frustum culling buys nothing, and the reason generalises

Built, measured, reverted; kept because someone will otherwise propose it again. Six planes from the
view-projection, each resident cell's world bounds tested against them, the top level rebuilt over
what survived, plus the ring around the camera kept whatever the frustum said because the sun does
not care which way the camera faces.

Out of 6,455 instances across a 48-cell window it removed 4,124 looking along the ground and 1,709
looking at the sky — a 74% cut, as selective as this can ever get:

| | median | min | max |
|---|---|---|---|
| culled | 1.91 ms | 1.91 | 1.94 |
| everything resident | 1.88 ms | 1.85 | 2.10 |

**Nothing. Not a small win inside the noise — zero, at three quarters of the scene removed.**

The premise is what is wrong, and it applies to any culling scheme proposed above the acceleration
structure: **a bounding volume hierarchy already culls, spatially, and does it per ray rather than
per frame.** A ray that never travels toward a cell never descends the subtree holding it, so
removing that subtree removes work nobody was doing. Frustum culling is a rasteriser's idea — it
exists because a rasteriser must *submit* geometry before it can reject it, and a ray tracer never
submits anything. Against that, the costs are real: 1.1 ms and a device-idle stall each time a turn
brings a cell across a frustum edge, and the image changes by 3,066 pixels of two million because
bounce and shadow rays that reached a culled cell now escape to the sky.

If the top level ever does become the cost, the thing to reach for is fewer *instances* rather than
fewer cells — merging a cell's static clutter into one structure shrinks what the hierarchy has to
describe rather than hiding part of it.

### 8.12 The visibility shader is six files

`primary_visibility.comp` had reached 1,243 lines and was more than half water. It is now 78 — a
header, six includes and `main` — with the rest in `.glsl` modules beside it. The split is not by
topic but by **dependency order**, which is the only order GLSL allows:

| | |
|---|---|
| `bindings.glsl` | the descriptor set and the structs in it — the whole of what the host must agree with |
| `sampling.glsl` | hashing and direction sampling, no bindings touched |
| `surface.glsl` | attribute fetch, cone LOD, the cutout test, and both traversals |
| `lighting.glsl` | next-event estimation, the sky, one bounce |
| `waves.glsl` | the height field and its gradient |
| `water.glsl` | Fresnel, absorption, caustics |

**One forward declaration in the whole file set.** Lighting needs the sun dimmed on its way down
through water and water needs a surface shaded, which is a cycle; declaring `sun_through_water` and
`daylight_reaching` at the top of `lighting.glsl` breaks it. Putting water first instead would have
cost three. The refactor is pixel-identical, which is the only claim worth making about it.

### 8.13 Motion vectors, and the second place the world's size would have shown

DLSS Ray Reconstruction needs them, and so does the thing worth spending down after it: the sun costs
sixteen shadow rays a pixel, and reusing that estimate across frames is what makes one a frame
viable. Both want the same buffer, so it is built once, ahead of either.

**What a pixel stores.** The displacement, *in pixels*, from where its surface is now to where it was
on the previous frame's screen — the offset to add to a pixel coordinate to find its own history. A
miss stores zero, which a temporal filter reads as "this pixel did not move"; for the sky that is
true.

**Reprojected as an offset, never as a world point.** The obvious formulation takes the hit position,
projects it with the previous frame's view-projection, and subtracts. That is §8.7's mistake with the
operands swapped: the hit position is world-scale, and so is the previous view's translation. Instead
the shader keeps everything camera-relative —

```glsl
vec3 was = direction * surface.t + frame.camera_motion;
vec4 before = frame.previous_clip_from_offset * vec4(was, 1.0);
```

`direction * t` is the offset from *this* eye, already computed without forming a world position;
`camera_motion` is `now - before`, differenced on the host where both are known; and
`previous_clip_from_offset` is the previous frame's `projection * rotation`, with its translation
dropped for the same reason the forward matrix has none. Nothing world-scale is ever subtracted on
the device.

**The camera delta is exact.** Subtracting two `f32`s within a factor of two of each other is exact,
and a camera does not cross half the world in a frame — so sending the delta costs nothing that
sending the previous position would have saved, and spares the shader a subtraction it could not
afford.

**Full floats, unlike the rest of the G-buffer.** A motion vector spans the frame — a couple of
thousand pixels when the camera turns — and a half float's eleven-bit mantissa lands only on whole
pixels above 1024. That is exactly the range temporal reuse most needs right.

**Behind the previous eye there is no answer**, and the perspective divide would fold such a point
back into the frame as a plausible-looking coordinate. `w > 0` is checked and the vector left at
zero.

**Cost: not measurable.** One eight-byte store and a matrix multiply per pixel, against run-to-run
variance of ±2 ms at 1920×1080 on a machine that had been busy for hours — the arm *without* the
write measured higher. Recorded as unmeasured rather than as a number.

**What is asserted**, because a reprojection can be plausible and wrong in three different ways:

- A camera that did not move leaves every pixel where it is. Not exactly, and it cannot be — a still
  frame is an unproject followed by a project and `f32` rounds in between. A hundred-thousandth of a
  pixel is what is left, or sixty thousand frames before a history has crept one pixel sideways.
- A camera that only *turns* moves every surface by the same amount whatever its distance, because
  rotation has no parallax.
- A camera that *steps* moves near surfaces further than far ones — checked against the naive
  world-space projection carried out in **double precision**. That is the calculation the shader must
  not make, done exactly, which is what makes it an independent answer rather than the same
  arithmetic agreeing with itself. The first attempt at this test failed because its reference was
  the `f32` world-space projection: the reference was wrong, not the code, which is §8.7 turning up
  a second time from the other side.

Over a real trace, two walls at 200 and 400 units move by 0.64 and 0.32 pixels for a four-unit step,
hand-computed from the field of view. Ignoring the hit distance fails it; dropping the camera delta
fails it. The sign is asserted too: the camera stepped left, so a surface's *previous* screen position
is left of where it is now, and a vector pointing the other way smears history backwards.

**One camera, one type.** `view`, `projection` and the eye now travel together as `Viewpoint`, for
this frame and the previous one alike. They have to agree — the position must be the point the view
looks from — and passed as three arguments nothing said so; a frame whose rays start somewhere its
matrices do not would render a plausible picture of the wrong place.

### 8.14 The ground was blended everywhere and nowhere

Reported from a screenshot: terrain reading as overlapping translucent squares, and elsewhere a
razor-straight seam with no blend at all. Two faults in §8.8, and they pull in opposite directions.

**A tile has to read as itself somewhere.** The ramp ran the whole 512 units from one tile centre to
the next, so *no point on the map drew a single texture* — every one was a mix of four, and a tile
came out as a translucent square laid over its neighbours rather than as ground. The original engine
does not do that. It blends through a map of **two texels per tile**, each tile's own pair at full
weight, and lets bilinear filtering do the rest — `components/esmterrain/storage.cpp:497`, where the
comment is *"We need to upscale the blendmap 2x with nearest neighbor sampling to look like
Vanilla"*. That confines the transition to the 256 units straddling the boundary and leaves the
middle half of every tile pure. Written directly, it is the same one line with a clamp:

```glsl
vec2 weight = clamp(fract(world.xy / 512.0 - 0.5) * 2.0 - 0.5, 0.0, 1.0);
```

**And the blend stopped at the cell boundary.** `terrain_materials` read one cell's `VTEX` and
clamped past its edge, so a cell's outermost quadrant blended a tile *with itself*. Two cells
therefore met in a 512-unit band of flat ground — 256 from each — with a hard seam down the middle,
every 8,192 units. The reported coordinates landed on tile row 15.67 of 16, which is exactly that
band; the arithmetic said so before the render did.

The fix is to read the eight neighbours' `VTEX` as well, as one grid three cells on a side. A
neighbour that is open sea has no `LAND` record at all and keeps the old clamped value — which is
the right answer where there is genuinely nothing to blend with, rather than everywhere.

**Cost: 230 ms across a 428-cell window**, 1,387 against 1,155, or about half a millisecond a cell.
That is why `LandRecord::textures_of` exists: eight full `LandRecord::parse` calls per cell would
have undone eight delta-coded heightmaps and eight normal fields to read one subrecord. Interning
only the tiles a quadrant can reach — the cell and a border of one, rather than the whole
neighbourhood — is a further 40 ms; it was worth having and it is *not* where the time goes, which
the first guess at it had backwards.

**What is asserted.** The render test now predicts every pixel of the new profile exactly, *and*
that a point well inside a tile draws that tile alone — the property the first version had no way to
fail. On real data, Seyda Neen's ground must draw at least one texture its own `VTEX` never names,
and every such texture must belong to a cell beside it: reverting to the clamped version fails it
with "the ground draws only this cell's own 8 textures", while still producing a perfectly plausible
list of Bitter Coast art. That was the trap in the original test, which pinned the *names* and so
could not see a missing blend at all.

### 8.15 A promoted cell kept the terrain it had when it was far away

Reported as hard-edged rectangles of the wrong ground, a few hundred units across, on terrain right
under the camera — and only after flying. A fresh screenshot of the same coordinates never showed it,
which is the whole clue: the cell had to have been somewhere else first.

**Mesh slots are grow-only and keyed by source path.** That is what makes a neighbouring cell nearly
free (§8.9) — the second cell to name a rock gets the one already uploaded. Terrain is keyed by its
cell, `land:x,y`, because a heightmap belongs to exactly one and no other cell can share it.

The distant tier (§8.9) broke that premise without changing the key. **A cell has one heightmap per
level of detail**, and both were called `land:x,y`. So a cell that arrived in the distant ring at
stride 4 and was later promoted into the detailed window found its name already taken and was handed
back the coarse mesh — geometry *and* the material indices baked into its submeshes at upload. Near
terrain then drew 512-unit quads each carrying a single quadrant's four textures, which under the
clamped ramp of §8.14 saturate into exactly the reported rectangles. Before that clamp the same fault
was there and merely looked like a smear.

The key is now `land:x,y@stride`. Two meshes per cell instead of one, in a table the design already
keeps for the life of the renderer.

**Why the tests missed it.** Every one loaded a cell once. `cell_residency.rs` covered the sharing
this depends on — two cells naming one mesh upload it once — and nothing covered a cell coming back
*different*, which is the case the second tier introduced. There are now two: the scene names its
terrain by detail, and the renderer, handed the same cell at another detail, uploads a mesh rather
than reusing the one under the old name.

**Out of scope, noted:** anything else a cell can name that changes with its detail would have the
same problem. Nothing does today — objects are keyed by file path and are the same mesh at any tier —
but the key is the invariant, not the terrain.

### 8.16 The guides an upscaler reads, written before there is one

§5.2 lists what DLSS Ray Reconstruction wants and warns that the specular guide is *"easy to forget
and awkward to add late"*. The awkwardness is real and specific: the guide is a quantity the **trace**
has to record at the hit, so retrofitting it means going back into the shader rather than adding a
pass. It is written now, with nothing reading it.

**Specular albedo, roughness, and the specular hit distance.** A reflection does not move across the
screen the way the surface carrying it does — it moves with whatever is reflected — so a temporal
filter given only depth reprojects every mirror wrongly. The distance to the reflected hit is what
fixes that, and `water_ray` was already returning it and throwing it away: *"the reflection's distance
is discarded"*.

Vanilla Morrowind is matte. `NiSpecularProperty` is force-disabled at this NIF version, so **water is
the only thing in the world with a specular response** — its Fresnel term is the albedo and the lobe
left by waves too small to resolve is the roughness, both already computed. Everything else reports
`Guides(vec3(0), 1, 0)`.

**One new image, not three.** Specular albedo and roughness share an `rgba16f`; the distance rides in
the **albedo target's alpha**, which the composite does not read and which was a constant 1.0 — the
same idiom the normal target already uses for depth.

**Jitter, off by default.** Sub-pixel offsets let successive frames resolve detail no single frame
holds; on their own they are shimmer, so nothing turns them on until something accumulates. Halton in
bases 2 and 3 rather than a random offset, and the test asserts the property that distinguishes them:
16 frames must touch all four quadrants of the pixel and 64 must touch all sixteen sixteenths, which
a random sequence fails by clumping.

**The convention I got wrong, and how.** The obvious reading of "motion vectors must exclude jitter"
is to measure against the pixel centre, and that is what I wrote. The fault injection did not fire,
which was the tell: working out why showed the change was backwards. The jitter is applied to the
*coordinate*, not the matrix, so a hit produced by a jittered ray projects back through that same
matrix to exactly the jittered coordinate — measuring against it cancels the jitter, and measuring
against the centre *introduces* it. With the centre version a still camera reports **0.30 pixels** of
motion that is purely its own jitter, which is history fetched from the wrong place every frame. The
original line was right; the test that now pins it did not exist, and both existing motion tests pass
either way with jitter off.

**Cost:** one more `imageStore` per pixel and an image at 1080p. The visible output is unchanged —
0 of 921,600 pixels differ, jitter being off.

### 8.17 NGX links by hand, and `wchar_t` is 32 bits here

The first slice of M7's second half: prove the SDK links, and ask it what it needs.

**Hand-written FFI, no generator.** NGX's parameter map is a C++ class with a vtable, which looks
like a reason to reach for `bindgen` or a C++ shim — and is not. Every call this needs is exported
with **C linkage**, checked with `nm` on `libnvsdk_ngx.a` before a line was written: the SDK's own
helper headers reach the map through `NVSDK_NGX_Parameter_SetI` and friends rather than through the
vtable, and so does this. Twenty mangled symbols exist in that archive and none of them is named
here.

**The extension query comes first because nothing else can.** `NVSDK_NGX_VULKAN_RequiredExtensions`
takes no Vulkan objects, and what it returns has to be enabled *when the instance and device are
created* — so it is the one call that must work before anything else exists. On this driver:

| | |
|---|---|
| instance | `VK_KHR_get_physical_device_properties2` |
| device | `VK_NVX_binary_import`, `VK_NVX_image_view_handle`, `VK_EXT_buffer_device_address`, `VK_KHR_push_descriptor` |

Queried rather than hardcoded: the list has changed between SDK versions, and these are what *this*
one wants.

**`wchar_t` is 32 bits on Linux**, and declaring `GetNGXResultAsString` as `*const u16` reads UTF-32
at half the stride — every error name came back one character long, its second byte being the
terminator. `NVSDK_NGX_Result_FAIL_OutOfDate` rendered as `"N"`. The test caught it because it asks
for the SDK's actual name rather than for "some letters", which the truncated version would have
passed. This is the failure mode hand-written FFI has, and the reason each declaration carries the
header line it was checked against.

### 8.18 NGX comes up, and four wrong parameters on the way

**DLSS Ray Reconstruction reports available on the RTX 4090 Laptop GPU.** That is the answer M7 hangs
on — it depends on the driver, the SDK and the GPU together, so nothing but NGX can give it — and
reaching it meant getting four things wrong first. Each returned a code that named itself, which is
the whole reason `Display` asks the SDK rather than printing a number.

**The device could not be created with what NGX asked for.** It names
`VK_EXT_buffer_device_address`, because it supports drivers older than this one; the same capability
is core in Vulkan 1.2 and this device enables it there, and the spec forbids both — `vkCreateDevice`
rejects the pair rather than ignoring one. Superseded names are now dropped, and the test asserts
they are gone rather than trusting it.

**`FAIL_UnableToWriteToAppDataPath`.** The header gives `InApplicationDataPath` a default of null and
NGX rejects null: it wants somewhere to put its logs and any feature library it downloads. A C++
default argument is not the same as an optional parameter, which is the sort of thing a hand-written
binding has to learn by being told.

**`FAIL_InvalidParameter`.** The project id is a **UUID and is parsed as one**. Mine read
`…-rtxmw000001`, which is memorable and not hexadecimal. NGX says only that some parameter was
invalid, not which.

**Available, but reporting itself unavailable.** `libnvidia-ngx-dlssd.so` is neither on the loader
path nor beside the binary, and NGX's default search is the application folder alone — so it has to
be *told*, through `NVSDK_NGX_FeatureCommonInfo`. That struct carries its logging block by value
rather than behind a pointer, so the whole thing has to be declared even though only the path list is
set; a short one would leave NGX reading past the end.

**What this bought beyond DLSS.** `Device::new` now takes extensions from its caller, enabled where
present — the rule the optional set already followed — and `PhysicalDevice::supports` answers for
names this crate has never heard of. Neither belongs to NGX: the point is that the list is queried
from whoever knows, rather than becoming another constant in the Vulkan layer.

### 8.19 DLSS agrees with the frame budget

The optimal-settings query, and the answer §5.3 was written against:

| preset | render resolution for 3840×2160 |
|---|---|
| Performance | **1920×1080** |
| Balanced | 2227×1253 |
| Quality | 2560×1440 |

That is the number the entire budget is measured at, now asked of DLSS rather than assumed. Dynamic
resolution is offered too — the reported range runs from 1920×1080 up to native — which is a lever
worth remembering if the trace ever overruns.

**The query is not an exported symbol.** It is a function pointer the driver's feature library puts
*into the capability map*, fetched by name and called with that same map, reading its inputs and
writing its answers back through it. That is why the SDK's own helper returns `FAIL_OutOfDate` when
the pointer is absent: it means the feature library was never found, which is a different problem
than the name suggests and the one §8.18 spent a while on.

**The three presets are asserted against each other, not just against a number.** A query that
ignored the quality value would return 1920×1080 for all three and pass a test that only checked
Performance. Each is also required to sit inside the dynamic range it reports, which is what a
renderer varying resolution would be handed.

### 8.20 The G-buffer moves to the layout DLSS reads, and depth stops being a half float

DLSS Ray Reconstruction wants diffuse albedo, specular albedo, normals, roughness, depth, colour and
motion vectors. §8.16 produced all of them — but not in the arrangement it reads them from, and the
repack turned up a bug that had nothing to do with DLSS.

**Roughness moves into the normal target's `w`.** That is `Roughness_Mode_Packed`, which is one fewer
resource to bind and one fewer image to write. It was in the material target's alpha, which is now
spare, and the material target carries specular albedo alone.

**Depth becomes its own target, at full precision.** It used to ride in the normal target's `w`, in
an `rgba16f` — where the largest representable value is **65,504**. That was fine when the world
ended at the streaming window and stopped being fine when §8.9 pushed the horizon past 100,000 units:
every pixel beyond that stored infinity, and the à-trous filter's edge test divides one distance by
another. Under the ceiling it was merely coarse — eleven bits of mantissa puts the step at about
eight units by ten thousand, and sixty-four by sixty thousand.

The new target is `rg32f`: **clip depth in `r`** for the upscaler, **distance from the eye in `g`**
for the filter. Two different questions that were being answered by one number — the upscaler
reprojects with clip depth and the filter stops edges on world distance, and a reverse-Z clip value
would have made the filter's tolerance mean something different at every distance.

**Measured effect: 7 pixels of 921,600 move by 2 levels**, scattered rather than banded — the filter
seeing exact distances where it had quantised ones. Small, and in the direction of correct.

**What the tests had to learn.** `water.rs` read specular albedo and roughness from one image, which
would have kept passing had roughness been written to the wrong target. It now reads each from where
DLSS reads it: the albedo from the material target's `rgb`, the roughness from the *normal* target's
`w`.

### 8.21 The feature builds, once NGX is asked what it dislikes

Ray Reconstruction is created at 1920×1080 → 3840×2160 and released cleanly. Two faults on the way,
and the second is the more useful lesson.

**`Use.HW.Depth` describes the depth's shape, not where it came from.** The enum is `Linear = 0`,
`HW = 1`, and §8.20 writes clip depth — projected and reverse-Z — whoever computed it. I set `Linear`
on the reasoning that a compute shader wrote it rather than the depth test, which is true and
irrelevant.

**`MVLowRes` reads as a description, not a request.** It says the motion vectors *are* at the low —
render — resolution, which §8.13 writes them at. I reasoned it the other way round and left it out.

Both came back as `FAIL_InvalidParameter`, which names no parameter.

**What found it was NGX's own log**, which is off by default and was off here because the logging
level in `NVSDK_NGX_FeatureCommonInfo` was left at zero. Turned up, it says:

```
Error: Low resolution Motion Vectors required
NVSDK_NGX_Result_FAIL_InvalidParameter
```

That message exists nowhere in the API surface — no status code carries it, and no parameter can be
queried for it. It is the difference between reading the answer and bisecting a parameter map.

**And it cannot be had quietly**, which is why it is `RTXMW_NGX_LOG` rather than simply on: the
feature libraries write **1,018 lines to the console** on one successful run, enough to bury the
assertion message of whatever failure sent someone looking. Off, they write nothing at all — not even
the files. So it is a switch, in the shape `DLSS_SDK_DIR` and `MORROWIND_DATA_DIR` already use.

**Ownership, since NGX has two things to release and an order.** The parameter map a feature is
built from is not the capability map — that one is NGX's, for asking questions — and it has to
outlive the feature. Both belong to one `Feature`, released together: the handle first, then the map.
The first attempt handed the map to a closure that `map_err` dropped whether or not the error path
ran, which would have destroyed it on *success*.

### 8.22 Ray Reconstruction runs, and what that test does not prove

1920×1080 in, 3840×2160 out, on real Vulkan images, with the validation layer silent — which is a
separate claim from NGX returning success. A status of success says only that NGX liked the parameter
map; DLSS records its own commands into the buffer afterwards, and the layer is what has an opinion
about the resources those commands touch.

**Two names that are not the obvious ones.** Ray Reconstruction reads its albedos from
`DLSS.Input.DiffuseAlbedo` and `DLSS.Input.SpecularAlbedo`, not the generic `GBuffer.Albedo` and
`GBuffer.Specular` sitting beside them in the same header. And the entry point is
`NVSDK_NGX_VULKAN_EvaluateFeature_C`, not the unsuffixed symbol next to it, which takes a C++
callback type — the SDK's own helper calls the `_C` one.

**What the test proves and what it does not.** Swapping the output for a 1080p image fails it, so the
plumbing has teeth. Swapping *depth* for *motion vectors* does **not** — both are `rg32f` at render
resolution, and neither NGX nor the validation layer can tell one from the other. Every input here is
an empty allocation, so there is no picture to check either. What this establishes is the resource
wrapper, the parameter names and the call sequence; whether the right image reaches the right name is
a question only a real frame can answer, and that is the next step rather than a gap in this one.

### 8.23 Ray Reconstruction is wired into the frame, and produces black

The frame path now reorders around it — trace, composite, **DLSS**, exposure, tone curve — with the
à-trous filter set to zero, the tone curve moved to the upscaled size and jitter turned on. On a real
cell it reports `1920x1080 to 3840x2160` and takes about 4 ms. **And the picture is black.**

**What is established.** Read back directly, the upscaled image has peak 1.0 with exactly 8,294,400
non-zero channels — one per pixel of a 3840×2160 frame, which is the alpha. So DLSS runs, writes its
output, and writes nothing but alpha. Its colour input is a real frame: 7.6 of 8.3 million channels
above zero. Nothing errors, and the validation layer is silent.

**Two real bugs found on the way there**, both mine and both fixed: no barrier between the composite
writing the colour and DLSS reading it, and the tone curve dispatched over the *render* extent while
writing a 4K image — which left three quarters of the frame as the allocator had it.

**What was ruled out.** Reverse-Z clip depth with `DepthInverted` and `Use.HW.Depth`, against linear
world distance with neither: black both ways. The parameter names are the RR-specific ones, checked
against the header — `DLSS.Input.DiffuseAlbedo`, `DLSS.Input.SpecularAlbedo`, `GBuffer.Normals`,
which the helper sets twice and the second one wins.

**Still to try**, in the order I would: the normal encoding, since nothing has confirmed DLSS reads
world-space `[-1, 1]` rather than packed `[0, 1]`; the subrect base parameters, which the SDK's helper
sets for every input and this does not; and feeding a constant colour to separate "DLSS ignores its
input" from "DLSS rejects its guides".

**Superseded by §8.24**, which found the cause. Everything ruled out above stayed ruled out.

**It is opt-in twice over** until that is understood: `--features dlss` to compile it, and
`RTXMW_DLSS=1` to attach it. Building with the feature and not setting the variable is **pixel-for-
pixel identical to the default build** — checked, 0 of 921,600 — so nothing already working is
standing on this.

*Both sentences above are history.* §8.31 makes `dlss` a default feature and DLSS on by default, and
retires `1` as a spelling; `RTXMW_DLSS=off` is what turns it back off.

### 8.24 The black frame was a missing usage flag

**DLSS samples its inputs.** Every image handed to Ray Reconstruction has to be created with
`VK_IMAGE_USAGE_SAMPLED_BIT`, and ours were `STORAGE | TRANSFER_SRC` — which is everything the
renderer's own passes need and one bit short of what NGX needs. An image it cannot sample reads as
zero. NGX returns success, the validation layer says nothing, and the network resolves a black field
to a uniform 2⁻²³ — the second-smallest half-float subnormal, its floor rather than a real value.

Adding the bit to the G-buffer, the trace target and the upscaled output turned 640×360 into a clean
1280×720 in one change.

**What found it was varying an input and watching nothing happen.** Constant colours of 0.25, 1.0 and
100.0 produced bit-identical output, which says DLSS never read that image — and the output image was
being written through the same wrapper, so the wrapper, the struct layout, the `SetVoidPointer` path
and the parameter name were all fine by construction. That narrowed it to a property of the image
rather than of the code around it.

**Two things ruled out on the way, both now reverted.** The `AutoExposure` creation flag and the
`DLSS.Pre.Exposure` / `DLSS.Exposure.Scale` scalars the SDK's helper substitutes 1.0 into: measured
bit-identical with and without, and §3.7 of the RR integration guide says exposure is not supported by
Ray Reconstruction at all. The SDK helper setting them is DLSS-SR heritage.

**The test now asserts a picture.** A constant frame is the one input whose correct output is
arithmetic rather than a reimplementation of the network — a flat field can only upscale to itself —
so `[0.25, 0.5, 0.75]` in has to come back as itself, per channel, and does to within 0.15%. Three
different values rather than one, because a grey would pass a single-channel check while proving
nothing about which channel was read. Removing the usage bit fails it with the 2⁻²³ floor in the
message.

**The upscale got its own timing stage** on the way out. It had been recorded inside the composite's
window, which reported a 0.65 ms upscale as 0.65 ms of compositing — the composite's real cost is
0.01 ms. A stage that is the most expensive thing in the frame cannot be measured as part of the
cheapest.

### 8.25 The sampler redrew the same noise every frame

A still camera produced **bit-identical frames**: 0 of 921,600 pixels differed between frame 1 and
frame 8. The hash streams were seeded `hash(pixel, stream, sample)` with no frame term, so the
estimator's error was a fixed pattern rather than something that averages away.

**A spatial filter hides this and a temporal one cannot.** À-trous filters each frame on its own and
never asks whether the noise moved, so the fault survived M6 unnoticed. Ray Reconstruction
accumulates across frames, reads a pattern that never changes as scene detail, and preserves it —
the frame came out covered in salt-and-pepper speckle that 64 frames of convergence did nothing to.
`DLSS-RR Integration Guide` §3.5 states the requirement it violated: samples must have minimal
correlation *temporally* as well as spatially.

The fix is `sample_stream` in `sampling.glsl`, which exclusive-ors the pixel with a word derived
from a new `sequence` field in the frame constants. Exclusive-or with a fixed word is a bijection, so
pixels still never collide within a frame — the rotation changes which stream each pixel draws from,
not how many there are.

**It costs the path that does not need it.** `sampling.glsl` had argued the other way, and was right
for the renderer it was written for: with only an à-trous pass, a fixed seed dithers and holds still
while reseeding leaves crawling static nothing averages away. Consecutive filtered frames of a still
camera went from bit-identical to 0.7% RMSE apart. The trade inverts under a temporal filter, which
is why it flipped — but the old rationale was sound and is recorded here rather than deleted.

### 8.26 The reference was rewarding aliasing

Measured against a native-4K 1024-sample reference, the à-trous path scored 32.1 dB and Ray
Reconstruction 24.3 — a 7.6 dB deficit for the far more sophisticated denoiser, which was reason to
distrust the *measurement* rather than the denoiser.

**The reference was aliased.** It renders one sample per pixel spatially with jitter off, as does the
à-trous path; Ray Reconstruction resolves sub-pixel detail across jittered frames and antialiases.
On a shore full of fences and railings, PSNR penalised RR for edges that are *more* correct than the
reference's.

Re-rendering the reference at 7680×4320 and box-filtering to 4K — four spatial samples per output
pixel, 1024 indirect ones, 18.7 s of device time — moves both figures:

| 1 spp, 3840×2160 out | vs aliased reference | vs supersampled reference | device |
|---|---|---|---|
| à-trous, 4 passes, native | 32.1 dB | **26.8 dB** | 23.5 ms |
| RR DLAA, native | 24.3 dB | **24.9 dB** | 59.0 ms |
| RR Performance, 1920×1080 in | 21.9 dB | **22.2 dB** | 10.4 ms |

The gap that mattered was under 2 dB, not 7.6, and it moved in opposite directions for the two
methods — which is the signature of a metric measuring the reference's own defect.

**A reference has to be at least as correct as the thing it judges, on every axis at once.** This one
was more converged and less antialiased, and only the first was being thought about.

### 8.27 M7's performance gate is met

3840×2160 output on the Seyda Neen shore, release build, RTX 4090 Laptop:

| | device | trace | upscale |
|---|---|---|---|
| native 4K | 34.50 ms | 29.80 | — |
| 1920×1080 → 4K, RR Performance | **10.76 ms** | 6.39 | 4.18 |

29 fps against 93 fps, with about 6 ms of the 16.7 ms budget unspent — and this is an exterior, where
§5.3's table was measured on an interior. Take the figures as a ratio rather than as absolutes: the
laptop's clocks move with thermals, and repeated runs of the same frame during this session spanned
10.4 to 21.7 ms.

**The quality gate is not met.** 22.2 dB against a supersampled reference is not "comparable to a
1024-sample reference" by any reading, and RR trails the à-trous filter it replaces by 1.9 dB at
matched resolution while costing five times as much there. What is established is that it is not the
frozen noise, not the sky guides, not exposure, and not the jitter sequence length; §3.5's other
requirements — hash quality and sample correlation — and the hit-distance guides RR is never given
are what remain.

### 8.28 The upscaled frame was exposed by the noisy one

Ray Reconstruction trailed the à-trous filter it replaces by 1.9 dB, and §3.5's sampling requirements
turned out not to be why. Two things it names were tested and neither moved: replacing the sampler's
hash with `pcg4d` from the paper it cites gained **+0.01 dB**, and taking two full 32-bit words per
sample instead of two 16-bit halves gained **+0.005 dB**. Both were reverted — the old hash is a XOR
of four products with a short finaliser, which is the family that paper measures as weak, and on this
content it measures as sufficient.

**What it was: a 2.7% brightness bias, uniform across the frame.** Mean luminance came out 0.7493
against the reference's 0.7298, and splitting the frame into sky, middle and ground gave +2.5%, +2.8%
and +2.9% — a global gain, not a region getting it wrong.

The auto-exposure histogram bins `log2(luminance)` **per pixel**, and the mean of a log sits below
the log of the mean by roughly the variance. A noisy frame therefore measures darker than it is and
the tone curve opens to compensate. Attaching an upscaler sets the à-trous passes to zero — Ray
Reconstruction is the denoiser — so what exposure read was a single sample per pixel.

The fix is one binding: exposure measures **the frame the tone curve is about to map**, which is the
upscaled one where there is an upscaler. It had read the render-resolution frame either way, to avoid
averaging four times the pixels for what looked like the same answer; it is not the same answer. The
ordering already allowed it, since DLSS runs before exposure in the frame, so no lag is introduced.

| 1 spp, 3840×2160 out, vs the §8.26 supersampled reference | before | after |
|---|---|---|
| à-trous, 4 passes, native | 26.80 dB | 26.80 dB (bit-identical) |
| RR DLAA, native | 24.89 dB | **27.49 dB** |
| RR Performance, 1920×1080 in | 22.23 dB | **23.54 dB** |

Ray Reconstruction now **beats** the filter it replaces at matched resolution, by 0.7 dB, while also
antialiasing. Exposure at 4K costs 0.10 ms against 0.05 at render resolution, and the shipped frame
is 11.27 ms.

**A residual, deliberately left.** Rendering natively with `--denoise 0` still exposes 0.7% bright,
because there is then no denoised image for exposure to read. That is a diagnostic setting rather
than a shipping one, and the real cure is a histogram that averages luminance before taking its log.
The bias is smaller there only because the tone curve's own compression of noisy highlights happens
to pull the other way.

**No test guards this**, and that is a choice rather than an omission: without an upscaler the change
is provably inert — the native figure is bit-identical — so only a DLSS-gated test on this hardware
could see it, and it would re-measure what this section records. The invariant is instead structural:
`bind_targets` binds both passes from one `source`, and `record` dispatches both over one extent
expression.

### 8.29 Ray Reconstruction reaches the window

It had been reachable only from `--screenshot`, which is to say only from a stationary camera —
and a stationary camera cannot show what a temporal upscaler gets wrong. The windowed renderer now
builds the same upscaler: NGX's device extensions at device creation, the **window's own size as the
output** so the blit to the swapchain is a copy rather than the upscale it is without one, and DLSS's
answer as the size to trace at.

The wiring is shared rather than copied. `upscaler.rs` holds what both front ends need, because the
two bring up Vulkan separately and a second copy would be a second place for them to disagree about
what DLSS was told.

**Two bugs, both only reachable from a window.**

A compositor sends a resize on first map that changes nothing, so `recreate` ran on frame one — and
it built the replacement upscaler *before* releasing the old one. Each `Upscaler` owns its own `Ngx`,
and dropping one calls `Shutdown1` for the whole device, so the feature just built was orphaned the
moment the old one went. NGX reports that as `FAIL_NotInitialized` at every evaluation and says
nothing at build time; the log showed the shape of it — one context initialised, **two** features
created, **three** shutdowns. The order is now release, then build.

The same first-map resize also meant a full rebuild for a size that had not changed, which costs a
weight upload. The swapchain still has to be recreated — it may be out of date for its own reasons —
but everything sized by the window is now left alone when the window's size is what it already was.
That matters more during a drag, which sends one of these a frame.

**What this does not yet cover.** There is no explicit history reset for a camera that jumps: `reset`
is derived from whether a previous frame exists, which is true from the second frame onward. Nothing
in the engine teleports yet, so there is nothing to test it against — but a fast-travel or a
coc-style jump will need one, and the symptom will be a smear rather than a crash.

Note for a wide display: the output follows the window, so a 7680×2160 one traces 3840×1080 — twice
the pixels §5.3 budgets for. That is the right answer for that window and the wrong one for the
budget, and the two only agree at 3840×2160.

### 8.30 The jitter handed to DLSS had the wrong sign, on both axes

The image shook — about a pixel, every frame, plainly visible in motion and invisible in every
measurement taken until then. A still camera turns it into a number: consecutive frames differed by
**0.0335 RMSE** under Ray Reconstruction against **0.0073** unupscaled. A temporal accumulator that
is *less* stable than the raw path with a stationary camera is doing the opposite of its job.

Sweeping the four sign combinations settles it in four renders:

| `Jitter.Offset` | frame-to-frame RMSE, still camera |
|---|---|
| `+x, +y` (what it was) | 0.0335 |
| `+x, -y` | 0.0196 |
| `-x, +y` | 0.0289 |
| **`-x, -y`** | **0.0016** |

**The trace adds the offset to the sample coordinate**, moving where inside its pixel a ray is fired.
NGX wants the offset as applied to the *projection*, which moves the frustum the other way for the
same picture. Handing over the coordinate's sign leaves Ray Reconstruction un-jittering in the
direction that doubles the offset rather than cancelling it.

Nothing reports this. The feature builds, evaluates and returns success, and the wrong sign still
resolves an image — just an unstable one, and one whose stillness nobody had measured because every
number so far came from a single frame.

Quality against the §8.26 supersampled reference, 1 spp:

| | inverted jitter | corrected |
|---|---|---|
| RR DLAA, native | 27.49 dB | **30.23 dB** |
| RR Quality, 2560×1440 in | — | **27.74 dB** |
| à-trous, 4 passes, native | 26.80 dB | 26.80 dB |

### 8.31 The default build is the one worth looking at

`dlss` is a default feature and DLSS runs at **Quality** unless told otherwise, so a plain
`cargo run` is the engine with everything it has switched on. `RTXMW_DLSS` now exists to turn it
*off* — `off` or `0` — or to name another mode, which is what an A/B against the unupscaled path
needs and the only reason it is still a variable. An unrecognised value is refused rather than
silently rendering at a mode nobody asked for.

The cost: NGX's SDK is not in the repository and `.refs/` is gitignored, so a fresh clone builds only
once the SDK is fetched there or `DLSS_SDK_DIR` points at it. `--no-default-features` is the way out.

### 8.32 One place reads settings

`RTXMW_DLSS` had been read inside `upscaler.rs`, which is a module with no business knowing that an
environment exists. Every setting is now declared once, in `cli`, as an ordinary argument:

```
--dlss <MODE>    off, performance, balanced, quality or dlaa
```

clap covers the flag **and** the variable from that one declaration — `env = "RTXMW_DLSS"` — and the
`.env` layer sits underneath as the argument's default, read through the same `from_dotenv` the
game's own directory is found by. So the order is flag, then variable, then `.env`, then the built-in
default, and there is one reader rather than two. `upscaler::build` takes the resolved mode as a
parameter; nothing outside `cli` calls `env::var` at all, which is a property a grep can check.

**A setting that reads the environment cannot be pinned in a test.** The parser tests flatten `dlss`
away in the two helpers every one of them already goes through, because a machine with `RTXMW_DLSS`
set would otherwise fail all of them, and none of them is about that setting — it has its own test,
which exercises the value parser directly.

The type is `Upscaling(Option<Preset>)` rather than `Option<Preset>`: clap reads an `Option` field as
"this argument may be absent", and absent is exactly what this one must not mean.

### 8.33 The exposure residual is a property, not a defect

§8.28 left a 0.85% brightness gap between a filtered frame and the same frame unfiltered, and it is
closed here by measurement rather than by a change. Output luminance at 1920×1080, DLSS off:

| à-trous passes | 0 | 2 | 4 | 8 |
|---|---|---|---|---|
| 4 spp | 0.7303 | 0.7281 | 0.7276 | 0.7272 |
| 1 spp | 0.7344 | — | 0.7277 | — |

**It scales with the noise and converges with the filtering**, which is what the log's concavity
predicts: the histogram bins `log2(luminance)` per pixel, the mean of a log sits below the log of the
mean by roughly the variance, so a noisier frame measures darker and the curve opens. Four passes
against eight differ by 0.05%, and every configuration that ships is on that end — the default reads
DLSS's denoised output, and `RTXMW_DLSS=off` reads a filtered frame.

So the gap appears only when an unfiltered frame is asked for, and then it is the honest answer:
§8.28's rule is that exposure measures the frame the tone curve maps, and a noisy frame measures as
what it is. Chasing it further would be optimising a diagnostic.

**One attempt is recorded as failed** so it is not retried blind. Averaging luminance over a 2×2
block before binning does not fix it — the gap flips to −0.77% and the filtered frame's own exposure
moves with it, because a block average changes which samples fall under the histogram's black cutoff
as well as how noisy they are. The concavity is the dominant term but not the only one.

### 8.34 A test for the thing no test could see

§8.30's sign error survived a feature that built, evaluated, returned success and passed the
validation layer, and every quality number taken from it — because all of them came from a *single*
frame. `tests/upscaler_stability.rs` renders ten and compares the last two with the camera held
still, which is the one thing an inverted jitter cannot fake.

**Its own test binary, and so its own process.** NGX is global per device and the SDK does not
promise to survive concurrent initialisation, which the unit test in `dlss/mod.rs` already depends
on; two NGX users in one binary would be racing. DLAA rather than an upscaling preset, so what is
measured is the temporal resolve alone rather than reconstruction error folded in beside it.

**The fixture had to be real content, and finding that out took two tries.** A synthetic wall with
nine bars each way passed with *every* sign combination — a bar was fifty pixels wide, so the frame
had thirty-six edges in it and a misaligned history had almost nothing to disagree about. Bars two
pixels wide separated the populations by 1.5×, still too thin to assert on. Only a real cell, whose
every surface carries texture detail at pixel scale, gives a misaligned history something to show:

| `Jitter.Offset` | frame-to-frame RMS, colour |
|---|---|
| **`-x, -y`** (correct) | **0.00090** |
| `+x, -y` | 0.00371 |
| `-x, +y` | 0.00485 |
| `+x, +y` | 0.00731 |

The bound is 0.0018 — the geometric mean of the correct value and the nearest failure, so each side
has a factor of two. It fails on **either** axis inverted, not only on both.

**A machine without the game skips**, rather than falling back to the synthetic grid. The grid was
kept as a fallback at first and that was a mistake twice over: it measured 0.0054 with the signs
*correct*, so it failed a bound calibrated on real content, and it could not have caught the fault
anyway. A check that cannot fail on the thing it exists for is worse than an honest skip.

A stable black frame would also pass, so the test asserts the frame is lit as well.
