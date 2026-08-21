# rtxmw — renderer design

Immediate goal: **render static Morrowind locations with a free noclip camera**, with hardware ray
tracing as the primary rendering mode rather than an effect over a rasterizer. §4 records what is
built and what each milestone measured; §8 records mistakes worth not repeating.

## Decisions

| Question | Decision | Consequence |
|---|---|---|
| Graphics API | **`ash` (raw Vulkan) + GLSL via `glslc`** | wgpu removed. Refit, opacity micromaps, SER and RT pipelines all stay reachable |
| License | **MIT OR Apache-2.0** | DLSS Ray Reconstruction is available; M7 is an integration rather than writing a denoiser |
| Material data | **De-light vanilla textures offline** | §5.1 — de-lighting recovers albedo only |
| Perf target | **1920×1080 internal → 3840×2160 output @ 60 fps** | ~8–9 ms denoiser, ~7 ms for everything else. 16:9 output, not native ultrawide |
| First scene | **Interiors through M8; exteriors at M9** | Fastest path to a correct image |
| Windowing | **`winit`** | Gamepad and audio need separate crates when they arrive |

---

## 1. Graphics API: `ash` and GLSL

wgpu 30 was audited against its vendored sources and removed. It has a complete ray-query-from-compute
surface, bindless with non-uniform indexing, working BLAS/TLAS build and compaction, and `as_hal` down
to `ash::Device`. What it cannot do, none of it an edge case here:

| Gap | Consequence for a Morrowind engine |
|---|---|
| **No BLAS refit** — `ALLOW_UPDATE`/`PreferUpdate` accepted and silently ignored | Every skinned actor is a full rebuild every frame |
| **No opacity micromaps**, though the hardware exposes `VK_EXT_opacity_micromap` | Alpha-tested foliage, grates and banners are the biggest RT cost multiplier in the game |
| **No RT pipelines** at the safe API level; `as_hal` does not reach `BindGroup`/`PipelineLayout`/`ShaderModule` | No SBT-driven material dispatch, no intersection or callable shaders |
| **No shader execution reordering**, though the hardware reports real `REORDER` | Substantial performance unclaimed on divergent hit shading |
| **TLAS instances are a CPU `Vec`**, re-serialized every build | ~1 ms per 1,000 instances, 98.7% of encode time. Exteriors are thousands of small statics |

bevy_solari, wgpu's flagship RT consumer, still lists transparent and alpha-masked materials, skinned
meshes, point lights, environment lighting, LODs and mipmaps as unsupported — approximately the
Morrowind feature set — and has already dropped to `wgpu-hal` to escape TLAS-build overhead.

**The counter-case, to re-read if this is revisited.** For static geometry wgpu is sufficient: BLASes
and the TLAS are built once, refit never fires, and ray-queries-in-compute is mainstream. It would
have saved one to two weeks of allocator, descriptor, barrier and SBT plumbing. Two things defeat it:
the gaps above are certainties rather than risks, and the DLSS advantage is illusory — `dlss_wgpu`
exists to inject Vulkan extensions before device creation, which under `ash` is calling NGX with a
device we own.

**GLSL**, compiled by `glslc` from `build.rs` and validated with `spirv-val`. `GLSL_EXT_ray_tracing`
and `GLSL_EXT_ray_query` are complete, and nearly every readable open-source RT reference is GLSL.
Slang is the better language and both compile to SPIR-V, so shaders can migrate individually; revisit
once the renderer's shape settles. Rejected: WGSL (dead end off wgpu), HLSL/DXC (no position fetch,
micromaps or SER), rust-gpu (no RT examples, incomplete buffer device address).

---

## 2. Module structure

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

Flat storage throughout: a cell is thousands of small objects and `Vec<Vec<_>>` would allocate per
instance. `StaticScene` is parallel vectors of meshes, materials, textures, instances and lights plus
the cell ambient; a mesh is one flat vertex and index buffer with a `submeshes` table of index ranges
tagged by material.

- **A hit resolves through one indexed read.** The TLAS instance's 24-bit custom index carries the
  mesh's `first_submesh`; adding the hit's `geometry_index` lands on a `GpuGeometry` entry naming the
  material and the vertex/index offsets. That is why the submesh table is flat rather than per-mesh.
- **Positions get their own tightly packed `Float32x3` stream**, shading attributes in a parallel
  buffer on the same vertex id. This is what the AS build API wants, not an optimization.
- **Frame constants** were lifted field-for-field from OpenMW's `components/fx/stateupdater.hpp`. They
  outgrew Vulkan's guaranteed 128 push-constant bytes and live in a storage buffer read with `scalar`
  layout, so a `vec3` packs at four-byte alignment and matches the `repr(C)` struct field for field.

---

## 4. Implementation plan

### M0 — Foundations — **done**
winit window, Vulkan device with the RT extension set, swapchain, noclip camera. Position fetch, ray
tracing maintenance1 and opacity micromap all report available on the target hardware.

- **Vulkan 1.3, not 1.4** — `ash` 0.38 ships 1.3.281 headers.
- **The swapchain cannot be a storage image**: sRGB formats expose no
  `VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT`, so RT output goes to an offscreen HDR image and reaches the
  swapchain through a blit — which makes the offscreen target an M3 requirement, not an M8 one.
- Limits: shader group handle size 32, base alignment 64, max ray recursion 31, max BLAS geometries
  and TLAS instances both 16,777,215.

### M1 — Data: VFS + BSA + ESM enumeration — **done**
`esmtool` is not installed, so the cross-check is the file's **own header record count**: walking
Morrowind.esm yields exactly the 48,295 records the header declares, which catches a mis-sized record
shifting every subsequent offset.

Measured: 20,952 VFS paths across the three BSAs plus loose files (7,319 meshes, 6,256 textures);
1,134 interiors and 1,404 exteriors holding 316,116 references.

### M2 — Geometry: NIF — **done**
All 7,319 shipped meshes parse: 4,579,361 triangles, 4,631,142 vertices, 41,702 geometry blocks. The
cross-check is self-consistency: every triangle index inside its vertex buffer, every UV set and
normal array matching the vertex count, the block walk landing exactly on the root list.

**Blocks carry no size at version 4.0.0.2**, so a parser off by one byte shifts every subsequent block
and the failure surfaces far from its cause. Four bugs, three the same mistake: `bool` is four bytes
at this version, but several fields that look like booleans are declared `char` and are one. What made
them findable was wrapping every block failure in an error carrying the block's index and type —
"6,601 × read past the end" became "6,601 × NiSourceTexture", which names the bug.

### M3 — First light: RT primary visibility — **done**
One BLAS per mesh, compacted; one TLAS over the cell's instances; a ray-query compute pass into an
offscreen HDR target. Seyda Neen's office: 104 meshes, 21,113 vertices, 17,835 triangles, 0.85 MiB.

- `GeometryBuffers::POSITION_STRIDE` is asserted to be 12: the stride is a number the build is *told*
  rather than one it derives, so padding the vertex would misplace every triangle rather than fail.
- **Indices stay mesh-local**, with `MeshRange::first_vertex` passed as the build's `firstVertex`.
  Rebasing ties a mesh's index data to where it landed, and cells relocate meshes at M9.
- **A mesh that flattened to nothing keeps its slot** as a zero-length range, so `MeshId` stays a
  direct index.
- **`gpu-allocator` pads an allocation to the memory requirement's alignment**, so an 8-byte buffer
  maps 16. `Buffer::mapped` trims to the requested size, or every readback returns padding.
- **One `cmd_build_acceleration_structures` call for all 104**, each with its own scratch region cut
  from a single buffer and the compaction-size query in the same submission behind a barrier.
  Compaction takes the bottom level 1.31 MiB → 0.58 MiB, 56%, which is why `ALLOW_COMPACTION` is on
  by default.
- **Scratch alignment is `minAccelerationStructureScratchOffsetAlignment`, 128 here**, and is *not*
  satisfied by the buffer's natural requirement. `Buffer::with_alignment` raises it and asserts the
  resulting address, because a heap that satisfies it by luck stops under fragmentation.
- **Instances use `TRIANGLE_FACING_CULL_DISABLE`.** Morrowind authors single-sided planes and winding
  is inconsistent across the mesh library.

**Verified headlessly, pixel by pixel:** a wall 100 units ahead covers the fraction of the frame 75°
of FOV predicts; geometry to the **north appears on the left** and **above appears at the top**, which
pins the whole handedness chain — instance transform, projection convention, Vulkan's Y-down NDC, the
shader's unprojection — where a mirror anywhere in it still draws a wall in the wrong place; two
meshes in one buffer render at their own offsets, the direct test of the build range triple.

The camera's projection uses the *offscreen* aspect ratio, not the window's.

### M4 — Textures and bindless materials — **done**
All 6,256 shipped textures decode: 190.8 MiB across 4,181 BC1, 1,971 BC2, 93 uncompressed BGRA8 and
11 TGA. A survey cut the scope before any of it was written:

- **There is no DXT5.** The library is DXT1 and DXT3 only, asserted, because a replacer pack
  introducing BC3 would render as noise.
- **No transcode is needed.** DXT1 and DXT3 *are* BC1 and BC2, and the files carry mip chains up to
  eleven levels deep.
- **TGA is 11 files**, all uncompressed 24-bit. RLE and colour maps are rejected rather than
  implemented.
- **`TextureFormat` says nothing about colour space** — the same BC1 bytes are sRGB as albedo and
  UNORM as a normal map, so the consumer chooses.
- **DXT1 maps to BC1 *with* alpha** — that one bit is the foliage and grates the cutout needs.
- **Mip levels share one buffer with a range table**, which is also what `vkCmdCopyBufferToImage`
  wants. The corpus test asserts header plus data accounts for the whole file; checking only that the
  level table tiles the buffer passes for a decoder that *drops* a level.

**Materials.** 7,319 meshes flatten into 26,869 submeshes drawing 4,593 materials over 4,311 textures.

- **A model is one mesh but rarely one surface**, so flattening keeps runs of indices tagged by
  material. Adjacent runs sharing a material merge; non-adjacent ones do not, because collapsing those
  needs reordering and the index ranges are what a build reads.
- **NIF properties are inherited**, so resolution carries a property stack down the node graph beside
  the transform.
- **The material table is scene-wide** — one bindless array and one material buffer per cell.
- **Two path fixups the original data needs**: a texture name is relative to `textures/`, and it
  routinely claims an extension the shipped file does not have.
- **45 of 4,311 texture references resolve to nothing**, dangling in the shipped data. The corpus test
  asserts a *rate* under 2% rather than perfection.

**Device side.** A flat `GpuGeometry` table, a `GpuMaterial` table, UV interpolation at the hit.

- **Geometry opacity follows the material.** `OPAQUE` lets traversal commit without invoking a shader:
  right for a wall, catastrophic for a grate.
- **The shader declares `scalar` block layout.** Under default std430 a `vec3` pads to sixteen bytes
  and every table entry after the first is misread. `spirv-val` needs `--scalar-block-layout`.
- **The bindless array is the set's last binding** — Vulkan permits a variable descriptor count only
  on the final element.
- **Slot zero is a magenta fallback** and a material addresses `id + 1`, so missing textures are
  absorbed rather than special-cased.
- **Every format maps to an sRGB view.** Sampling gamma-encoded albedo as UNORM darkens midtones by
  about half and cannot be tuned out; pinned by a test, because it looks merely "a bit dark".
- **`textureLod`, never `texture`** — implicit LOD needs derivatives a compute shader lacks.
  Anisotropy stays off: it is a rasterizer's answer to a footprint problem a ray tracer solves with
  ray differentials.

The test that matters is `two_materials_in_one_mesh_shade_differently`: both halves share an instance
and a custom index, so only `geometry_index` separates them. Dropping that term fills exactly the same
pixels and fails only this test.

**The alpha cutout runs in the candidate loop**, with `gl_RayFlagsOpaqueEXT` gone from the query so
the per-geometry bits decide whether traversal asks at all. The survey reads the opposite of the
obvious way: **Opaque 3,982, Blend 539, Mask 72**. Only 72 materials are *explicitly* alpha-tested; it
is the 539 blended ones that carry the foliage, grates and banners, drawn with `NiAlphaProperty` over
a texture whose alpha is very nearly binary. Blended materials get a stand-in cutoff of 0.5 and run
the same cutout path until ordered transparency replaces it. Marking only the 72 would have looked
correct and been wrong.

### M5 — Direct lighting — **done**
`LIGH` point lights with shadow rays, cell ambient, and the sun.

- **Light colours and cell ambient are sRGB-encoded** in the file, decoded on the way in, pinned by
  tests at mid grey where the two spaces diverge most.
- **Morrowind stores no intensity** — a `LIGH` record carries a colour and a radius, because the
  original renderer's fixed attenuation curve supplied brightness. Radiant intensity is derived as
  `radius² × INTENSITY`, so a lamp differs from a candle by its size.
- **Attenuation is inverse square, windowed to reach exactly zero at the radius** — Morrowind's radius
  is a hard cutoff and a clipped inverse square leaves a visible edge.
- **Carried, negative and off-by-default lights are not placed.** A negative light *subtracts*
  illumination, a trick for a renderer accumulating into a framebuffer.
- **Shadow rays run the cutout too**, so a grate throws bars, and use `TerminateOnFirstHit`.

**Soft shadows.** Each light is sampled as a sphere over eight shadow rays. The emitter size is
invented, because Morrowind records none: 8% of reach with a 10-unit floor. The sample pattern is a
**stable** per-pixel hash rather than a per-frame one — without temporal accumulation, reseeding turns
the penumbra into crawling noise (reversed by §8.25).

The test took two attempts. Counting distinct brightness levels across the shadow boundary *passed
with a point light*, because the lit wall already varies through attenuation and the cosine term. The
measure that isolates visibility is the ratio against the same scene with the occluder removed.

**The sun.** Direction is Morrowind's own `(-400 · orbit, 75, -100)` with `orbit` 1 at sunrise to −1
at dusk — the direction light *travels*, so the shader negates it. The **angular diameter is ours**:
OpenMW's sun is a pure direction with no size anywhere. Half a degree is the real sun and the only
reason a shadow has a penumbra — 38 rows of penumbra against 0 from a point. Shadow rays sample the
cone uniformly over solid angle rather than over the disc. A real sun's penumbra is about **one pixel
wide** when camera and blocker are equidistant from the surface, so the fixture is deliberately odd to
make the softness measurable.

**The sky comes from the ambient** rather than constants of its own, brighter overhead than at the
horizon, so indoors a missed ray yields the room's own dark fill. The disc is drawn but is **not**
energy-consistent with the directional term: a real sun's radiance is some sixty thousand times the
value used, and reconciling them means making the directional light an area light.

**Cost:** the exterior went 1.75 → 3.5 ms at 1920×1080, nearly all in the trace — sixteen shadow rays
a pixel where most reach the sky unobstructed. First thing to spend down if the budget tightens.

**The interior comes out dark, and that is the honest result.** Ambient 0.038 and lights reaching
64–128 units in a room spanning 1,757 × 2,559. §5.1 from the other direction: the original engine
leaned on pre-lit albedo *and* a flat ambient, so lighting that albedo correctly leaves it underlit.

### M6 — Indirect lighting — **done**
One diffuse bounce per pixel, cosine-weighted, with next-event estimation at the bounce hit. `trace`
owns a ray query and returns a resolved `Surface`, so primary and bounce rays share one traversal;
`occluded` keeps its own copy because `glslc` rejects `rayQueryEXT` as a parameter.

**Ambient became the environment's radiance rather than an unconditional fill.** A bounce ray that
escapes returns the cell's ambient, so at zero samples the estimator's mean *is* the ambient and the
term collapses to the `albedo * ambient` applied before. With rays, geometry occludes that fill —
ambient occlusion is not a separate effect, it is the same integral, sampled. A sealed interior loses
9% mean brightness, concentrated in corners and under furniture, which is §5.1 again: the albedo
already has AO painted into it.

- **The Lambertian `1/pi` moved into the shader.** It had been folded into `INTENSITY`, where a single
  scale on the only lighting term was unobservable; with a second term integrating over the hemisphere
  the ratio became real. The direct-lit image is unchanged.
- **Cosine-weighted by Malley's method** — a uniform sphere point added to the normal, exact rather
  than approximate, reusing the sphere sampler the soft shadows already had. The pdf cancels both the
  cosine and the `1/pi`.
- **Shadow rays at a bounce hit are cut from eight to one** — a bounce is already averaged over four
  directions; resolving *its* penumbra costs thirteen rays to change nothing.

**The variance baseline**, RMSE against a 256-sample reference: 1 spp 0.0713, 4 spp 0.0355, 16 spp
0.0174. Ratios of 2.01 and 2.04 per quadrupling — textbook `1/sqrt(N)`, so the error is Monte Carlo
noise with no bias underneath it. A stuck sample index fails that flat (0.0000227 at every count).

The synthetic scenes are hand-checkable: a white wall with a coloured floor, lit by one white light
overhead, so the red-minus-blue gap at a pixel *is* the indirect term. Predictions computed before the
trace came back within 2%. Both tests read a 17×17 patch, because at four samples a pixel holds one of
five levels.

**Not done, deliberately:** stratifying the bounce directions, and sampling one light rather than
looping all of them — thirteen is cheap and lower-variance but `O(lights)` per bounce. §8.10 is what
that became.

### Spawning: where a cell puts the player

**Morrowind stores the arrival point on the door you leave through, not in the cell you enter.** A
door reference carries `DODT`, a position *in its destination cell*, and `DNAM`, that cell's name. So
travelling through a door is a local question answered with no search, while arriving without walking
through anything means one pass over all 316,116 references, about 25 ms. There is no cheaper route: a
cell does not record how it is entered.

`DNAM` is absent for a door to an exterior, where `CellId::containing` floors the world coordinates by
the 8,192-unit grid. **Flooring rather than truncating matters**: truncation puts everything between
−8192 and 8192 in cell zero and mirrors half the map onto the other half.

**Editor markers place no geometry, and are excluded from the door search too.** Morrowind ships six
`meshes/Marker_*.nif` placed 1,145 times, including a solid 160-unit `NorthMarker` in the census
office. `PrisonMarker` is filed as a `DOOR`, carries a destination into that office, and is the
*first* such door in file order — so the obvious rule picked it and the camera started inside the
furniture.

- **Yaw is a compass bearing.** The stored rotation turns about the **negated** Z axis, so zero faces
  +Y (north) and a quarter turn faces +X (east) — the opposite handedness to a maths library's default.
- **Only the arrival's horizontal position is authored data; its height is a hint.** Across sixteen
  arrivals in twelve interiors it sits a median of 89 units above the floor, ranging 22 to 144. The
  original engine throws it away, raising the actor 20 units and tracing down, so this does too through
  `StaticScene::ground_below` — 0.3 ms over 46,251 triangles, fine once per cell and hopeless per frame.
- **Standing eye height is 160 units**: twice the median arrival height, times nine tenths, landing 83%
  up a 194-unit door. `MWRender::Camera::mHeight`, 124, is the *third-person* orbit pivot applied only
  `if (mMode != Mode::FirstPerson)` — adding it to a marker that already contained a body height put
  the camera at 394 in a room whose ceiling is at 420. **A constant lifted from a reference
  implementation without reading the branch it sits in is not a citation.**

### Test harness: the renderer is the thing under test

`rtxmw_render::SceneRenderer` owns the pass, the target and the loaded cell; `rtxmw::Renderer` adds
surface, swapchain, frame ring and present. Before the split, `Renderer` owned both and lived in the
binary crate where no test could reach it, so `primary_visibility` assembled its own copy of the
load-and-trace sequence — a parallel abstraction whose every assertion was about a reconstruction of
the engine.

- **The uploader is borrowed, never owned.** Giving each `SceneRenderer` its own made twelve tests
  submit to one queue concurrently — every parallel test failed and every serial one passed. An
  uploader wraps one command pool on one queue, so it is a device-wide resource.
- **Image readback moved off `RenderTarget`** onto `readback::image_to_rgba8`, because the image worth
  reading back is the renderer's own output, which no test owns.

`cargo run -- --screenshot <path> [WIDTHxHEIGHT] [CELL]` renders one frame on a device with no surface
extensions, ~0.6 s warm against tens of seconds for the windowed binary.

### M8 — Exposure, tone curve and sRGB — **core done**
Everything before this wrote linear radiance and clamped to `0..1`, and an interior traces at about
three hundredths of mid grey.

**Auto-exposure is measured from a log histogram, not an average.** Thirteen candle flames above
luminance 1 in a room at 0.02: a linear mean is dominated by whichever population has more pixels —
for a frame five sixths dark the mean is 1.7, and exposing for it puts the room at 0.002, zero after
encoding. The mean of the *logs* is 2⁻⁴·¹.

Two dispatches, because a reduction cannot see every pixel's contribution until every workgroup has
written it and a dispatch boundary is the only barrier that wide:

- `luminance_histogram.comp` bins log luminance into 256 bins spanning 2⁻¹⁰ to 2⁶, tallying into shared
  memory so the global buffer sees one atomic per bin per workgroup. The buffer is cleared with
  `vkCmdFillBuffer` — anything zeroing it inside the same dispatch would race.
- **Bin zero is reserved for pixels with no light on them**, excluded from the divisor as well as the
  sum. Counting them halves the mean bin, two stops darker than the truth: 103 correct against 230 on
  a half-black frame.
- `exposure.comp` reduces the bins in one workgroup of exactly 256 threads.

**The tone curve is Khronos PBR Neutral**, chosen against the milestone's done-when: an original
Morrowind screenshot and a render of the same viewpoint compare without a gamma mismatch, and the
original applied no tone curve. It rolls off only above 0.76 where ACES would twist every midtone, and
its desaturation term keeps a torch flame going white instead of clipping channel by channel.

**It is not the identity below the shoulder.** The operator subtracts a flat 0.04 above the toe, so
middle grey enters at 0.18 and leaves at 0.14. §8.52 fixed what that offset did to the *darks*; the
midtone shift remains, and whether to remove it is a decision about what the vanilla comparison is
worth rather than a defect to fix quietly.

**sRGB is encoded in the shader and the swapchain is `UNORM`** to stop it happening twice. sRGB formats
expose no storage capability, and the alternative — a linear 8-bit intermediate for the presentation
engine to encode — bands badly in exactly the near-black range an interior lives in. The payoff is that
**the screenshot is byte-identical to what the window shows**.

The chain is pinned by one hand-computable number: a flat frame of *any* radiance reaches the file at
103 (0.18 key → 0.14 after the shadow lift → sRGB 0.404 → 103). Unchanged across a hundredfold change
in scene radiance, and the same assertion catches a missing encode (36), a double one (172) and a
missing tone curve (118). §8.48 later replaced the flat-key property with a partial adaptation, and the
test's name changed with it.

**Not done:** bloom, colour grading, sharpening; exposure adaptation over time; HDR output.

### M7 groundwork — a G-buffer and a denoiser over it

**The trace no longer writes a picture. It writes what a surface is, separately from what light reaches
it.** All the noise is in the lighting, while the albedo a ray reads from a texture is exact, so
filtering the lighting alone smooths noise without touching a texel of surface detail and recombining
is one multiply. DLSS-RR consumes the same G-buffer, so only the filter is replaced.

- `albedo` — **half float, not the eight bits a reflectance in `0..1` appears to need.** The composite
  multiplies albedo by unbounded illumination, so a quantisation step scales with the light: eight bits
  moved the mean pixel by 0.32 of 255 and the worst by 37, half float moves the worst by 1.
- `normal_depth` — world normal and distance from the eye, together because they are the pair an
  edge-stopping filter tests.
- `illumination` — light arriving, albedo divided out, double-buffered for the ping-pong.
- Emissive surfaces and the sky stay in the existing target, added by the composite: neither noisy nor
  demodulated.
- A miss writes **zeroes** across the G-buffer rather than being skipped — the filter reads a
  neighbourhood, and an untouched pixel holds whatever the allocator left there.

**The filter is an edge-stopping à-trous wavelet**, four passes with tap spacing doubling each time —
5×5 taps reaching sixteen pixels either way, where a direct blur of that radius needs nearly a thousand.
Weights are a normal term and a *relative* depth term; relative because Morrowind's units put a room's
walls tens apart and a hillside thousands. Neighbour-to-neighbour roughness on a flat lit surface falls
8.8 → 0.07 with the albedo step still one pixel wide.

The fixture had to grow twice to earn that: with only a wall and a floor, whose normals are
perpendicular, deleting the *depth* term changed nothing, so a panel parallel to the wall and
identically surfaced was added. Composing the lighting before filtering — the regression this design
exists to prevent — spreads the albedo step over fifteen columns.

### M7 — Denoise and upscale — **done for accuracy, timing deferred to §5.3**
DLSS Ray Reconstruction via NGX: denoise, antialias and upscale in one pass, with no separate TAA —
running TAA over a temporally-accumulated denoiser double-blurs and compounds ghosting. **Done when:**
a still frame at 1 spp is comparable to a 1024-sample reference by a numeric metric, and the frame holds
60 fps at the §5.3 target.

`crates/rtxmw/src/upscaler.rs` brings NGX up for both front ends, `--dlss off|performance|balanced|
quality|dlaa` selects it, and it replaces the à-trous filter rather than sitting beside it.

**The accuracy half is met** — §8.88, and `tests/reconstruction.rs` is what holds it: against a
4,096-sample reference, one sample a pixel comes out at 0.1918 relative RMSE as traced, 0.0300 under
the four à-trous passes Ray Reconstruction replaced, and **0.0085 under Ray Reconstruction itself**.

**The timing half is §5.3's now rather than this milestone's.** It was measured here at 15.6 ms by
night and 14.1 by day against a 16.7 ms budget — inside it, but only on a settled clock, with the same
five runs spanning to 37 ms and the first of every batch the slow one. Since then the weather set has
gone in and the frame is 44 ms at the §5.3 target, which is recorded there along with why it is not
being worked on yet.

Practical notes: NGX ships real Linux `.so` files and needs specific Vulkan instance and device
extensions **at creation time**; OTA model updates are broken on Linux, so the baked-in model is what
gets used; Streamline is Windows-only, so call NGX directly.

### Frame timing — the first measurement against §5.3

A timestamp query pool brackets each stage. **Device time, not wall clock** — meaningful only because
the stages are separated by full barriers. Shipped interior, RTX 4090 Laptop, four bounce samples, four
à-trous passes, medians of five:

| internal resolution | total | range | trace | denoise | composite | exposure | tonemap |
|---|---|---|---|---|---|---|---|
| 1280×720 | 1.40 ms | 1.39–1.41 | 0.96 | 0.39 | 0.02 | 0.02 | 0.01 |
| **1920×1080** | **3.43 ms** | 3.31–4.69 | 2.41 | 0.83 | 0.04 | 0.03 | 0.02 |
| 2560×1440 | 5.96 ms | 4.85–9.35 | 4.35 | 1.62 | 0.09 | 0.04 | 0.04 |

**Take the spread seriously.** At 720p five runs agree to 0.02 ms; at 1440p they span nearly two to one,
and a single run straight after the test suite read 5.77 ms at 1080p. Scaling is close to linear in
pixels but not exactly — 2.25× the pixels from 720p to 1080p cost 2.45× the time — so there is no hidden
fixed cost, and no figure can be extrapolated.

### M9 — Exteriors: terrain and streaming — **done**
This is where TLAS instance counts get large, the point at which wgpu's CPU marshalling would have
become the bottleneck. It did not become one here: the 477 cells resident from a hilltop rebuild the top
level once a frame without showing up in the budget.

**`LAND` is two encodings that both look plausible when decoded wrongly**, which is what the tests are
shaped around:

- **`VHGT` stores gradients, not heights.** Each row's first column steps from the row above and every
  other from its left-hand neighbour, over a float offset for the whole cell. Read as absolute values it
  still produces a surface — just not this one.
- **`VTEX` is sixteen 4×4 blocks, not a 16×16 grid.** Read flat it scrambles the texturing into the
  right textures in the wrong places.
- **A `VTEX` index is one past its `LTEX` index**, because zero means the region's default.

Verified against every `LAND` record: 1,292 cells from −2,152 to 18,952 units, neighbouring vertices
never more than 1,016 apart, 548 centred below sea level — Vvardenfell's shape, which is what a corpus
catches that a round trip cannot. A further **98 `LAND` records carry no terrain at all**, only
coordinates and a flag word; treating them as parse failures would discard 7% of the world silently.

**Terrain is placed rather than instanced.** A heightmap belongs to exactly one cell, so `Mesh::from_land`
writes world coordinates and the instance carries no transform. Its 65×65 grid shares its last row and
column with the neighbouring cell, which is what makes adjacent terrain meet without a seam and why 65
vertices span 64 quads. Texture tiles become submeshes, so Seyda Neen is eight submeshes over 84 meshes
and 289 instances, loading in 19 ms.

The test names the eight textures exactly. Asserting they belong to the Bitter Coast was not enough,
because **the palette is grouped by region**, so dropping the index offset still lands on Bitter Coast
art: `Tx_BC_rock_01` where `Tx_BC_rock_03` belongs.

**A fault-injection harness reported four false negatives** before that was noticed: it treated a compile
failure as "the test passed". It now distinguishes *did not build* from *not caught* — the second time in
this project that a verification step reported success for work it never ran.

### M9 — Streaming cells one at a time

**The window is nearly free past 5×5.** At 1920×1080, four bounce samples, four à-trous passes, sixteen
sun rays:

| radius | cells | instances | frame | trace |
|---|---|---|---|---|
| 0 | 1 | 289 | 3.12 ms | 2.52 |
| 1 | 9 | 1,700 | 6.93 ms | 6.13 |
| 2 | 25 | 3,888 | 8.03 ms | 7.15 |
| 3 | 49 | 6,406 | 8.16 ms | 7.30 |
| 4 | 81 | 10,545 | 8.15 ms | 7.33 |

Tripling the cells from 5×5 to 9×9 costs **0.12 ms** — the extra cells are beyond anything a ray reaches
and the top level dismisses them in log time. The expensive step is the *first* neighbour, and not because
of instance count: rays that used to escape to the sky now hit terrain and spawn shadow and bounce rays.
**Draw distance is not what this budget is spent on. Ray termination is.**

**The first attempt was a block reload** — cross a boundary, load the 7×7 around the new centre, replace
everything. 195 ms dev / 133 ms release, of which the shape is what matters: **three passes over the same
79 MB file** to find records whose location never changes (read 21 ms, model index 7, scene 35, door search
26), then a re-upload of a scene 90% identical to the resident one — 49 neighbouring cells share 378 meshes.

**Identity, and two lifetimes.** Every mesh carries the path it was loaded from (terrain is keyed by its
cell), and on that identity the renderer keeps two tiers split by lifetime rather than by kind:

- **Assets are grow-only** — mesh data, bottom-level structures, textures, materials — keyed by source,
  uploaded by the first cell to name them, kept for the life of the renderer. The ceiling is the shipped
  library: 4,311 textures against 8,192 slots, some three thousand meshes, around half a gigabyte for a
  session that visited every cell. With eviction the arena would fragment and slots would be reused, which
  means renumbering, which means rewriting the tables that name them. A mod with 2K textures would blow the
  ceiling; until a refcount exists, the ceiling is asserted rather than assumed.
- **Cells own placements and nothing else** — instances and lights. Evicting one is dropping two lists.

A commit therefore rebuilds only what the resident *set* determines: the geometry and material tables
(~100 KB), the lights, the instance buffer and the top level.

**One walk, then two reads.** The file is walked once into a `CellIndex`: the offset of every cell's `CELL`
and `LAND` record, plus the `LTEX` palette and (§8.62) the regions. The palette has to live there for the
same reason the offsets do — `LTEX` records are scattered with no relation to the cells that use them, so a
cell loaded on its own would resolve its ground against however much of the palette happened to precede it.
The door search is not run for a streamed cell: a third of the cost, answering a question a cell the camera
walks into has already answered.

**Hysteresis is two radii.** Cells load within Chebyshev radius 3 and are evicted only past 4, so crossing a
boundary pushes the far column out of the load window but not out of the kept one. One crossing settles it;
no timers, no margin, one mechanism.

**Cost:** per cell with neighbours resident, **0.7–3.7 ms to add, 0.6–2.5 ms to commit**. Filling the whole
7×7 window from cold is 48 cells in **136 ms** against 195 for the block reload. Windowed at 1920×1080:
**81 fps while a cell arrives every frame**, 108–112 once full. The frame that takes a cell is a slower
frame, not a dropped one.

### M10 — Water — **done**
Per-cell water plane, RT reflection and refraction, absorption and scattering, analytic caustics. Design
and findings: §7. **Not done:** foam at the shoreline, sun shafts underwater.

### M11 — Sky, weather and time of day — **done**
A clock, the sun on Morrowind's own twelve-hour arc, both moons as lit spheres carrying their vanilla
portraits, the star field, the painted cloud deck, and the ten weathers out of `Morrowind.ini`:
volumetric fog, rain, snow, blown ash, lightning, and the film rain leaves behind. Design and findings:
§8.38–§8.87. Almost none of it is invented — the ini holds ten blocks of 49 fields, `Morrowind.esm` a fog
density for every one of the 1,134 interiors and the region lists, and where the file says nothing the
game still ships a mesh that does. What this renderer supplies is the levels and the light transport.

**Not done:** weather never changes — `;` cycles the region's list and the choice holds, which is why
`Thunder Threshold` is parsed and unread; the ambient and `Sun *` schedules stay parsed and unused
deliberately (§8.60, §8.61); shafts are cast for the sun and one moon only, since shadowing the fog costs
a ray per light per step (§8.43, §8.80); nothing accumulates — no snow on the ground, no ash on ledges;
and no Purkinje shift or absolute units, which §5.1 blocks rather than this milestone.

### M12 — Animation: actors, controllers and particles — **step 1 done**

§5.4's scope boundary, lifted. Written before any of it was built, because the sizing decides the
architecture and the sizing is not what the plan assumed — and the first step has since been built
and measured, which corrected the plan again (§8.91).

**What the content actually holds**, counted across the three masters and every shipped mesh:

| | count |
|---|---|
| Animated references placed in cells | **4,066** — 3,043 `NPC_`, 863 `CREA`, 160 `ACTI` with a `.kf` |
| Busiest single cell | **22** (Rotheran Arena: 18 NPCs, 4 creatures); mean where any are present, 3.2 |
| Skinned meshes | **556 files**, mean 1,401 vertices and 1,678 triangles; heaviest 22,267 vertices |
| `.kf` clips | 162, none of which parse today |
| `NiKeyframeController` | 10,480 blocks in 557 files |
| `NiParticleSystemController` | 637 in 258 files |
| `NiGeomMorpherController` | 324 in 273 files |
| `NiUVController` / `NiVisController` / `NiAlphaController` | 116 / 309 / 188 blocks |

**A worst-case frame therefore deforms about 35,000 vertices and rebuilds about 42,000 triangles of
bottom level.** Small enough that a full rebuild was expected to be the right answer, and
`ALLOW_UPDATE` — which a structure traced by every ray and built once should not have to carry —
expected to cost more than it saved. **Both halves of that were wrong**, and step 1 below is where
it was measured rather than argued. §1's row still stands as a statement about wgpu; what would
actually have hurt there is the row below it, TLAS instances marshalled through a CPU `Vec` on every
build, which for a per-frame top level over a hilltop's worth of statics is the fatal cost.

**One block type stands between the parser and every animation in the game.** All 162 `.kf` files
fail at block zero on `NiSequenceStreamHelper`, which has no arm; Morrowind blocks carry no size, so
there is no skipping past it. Everything else is already there — 59 block types are sized exactly,
including `NiKeyframeData`'s four interpolation modes with their tangents, `NiSkinInstance`'s bone
links and `NiSkinData`'s per-bone inverse binds and sparse weight lists. The layouts are written
down; the work is turning `cursor.skip` into a read. `base_anim.nif` is 2.7 MB of which 41 KB is the
skeleton and skin — the rest is keyframes, which is why the `x`-prefixed sibling exists and why
OpenMW's rule is to prefer `xfoo.nif` where `xfoo.kf` is present.

**The vertex streams need no new architecture, only a different allocation unit.** A skinned
instance takes its own slice of the *existing* shared position and attribute buffers — per instance,
not per mesh — and its own mesh slot pointing at it. `Geometry.first_vertex` already rebases every
index, so `surface.glsl` is untouched by the whole feature. At 35,000 vertices that is about 1.1 MB,
and the buffers already carry `STORAGE_BUFFER` usage, so a compute pass can write them. Index data
can be shared between instances of the same model, since a mesh slot names an index range it does
not own.

**Skinning runs on the GPU into that slice, double-buffered, because motion vectors need last
frame's positions.** §8.7's motion vector is built from the camera alone, which is right for a world
where nothing moves and wrong for the first thing that does. Ray Reconstruction's accumulation is
what pays for it — §8.90 measured how sharply it responds to a guide that lies — so the previous
deformed position is part of this from the start, not after.

**Per-frame acceleration structures, in the frame's own command buffer.** Everything today builds
through `Uploader::submit_and_wait`, allocates its scratch inline, and hands back a *new*
`AccelerationStructure`; none of those three survive contact with a per-frame path. The top level
must be built in place so the descriptor written in `VisibilityPass::bind` stays valid, its instance
buffer persistently mapped with only the animated rows rewritten, and its scratch preallocated at
cell load. `tests/frame_allocations.rs` already covers `record`, so the zero-allocation budget
polices this from the first commit. `VK_KHR_ray_tracing_maintenance1` is detected and unused today;
its sync2 stages are what the skinning-to-build and build-to-trace barriers want.

**The order is smallest-slice-first, with the risk taken before the content.**

1. **A measured spike — done.** Twenty-two placements of 1,682 triangles apiece, the busiest cell
   in the game, deformed by a compute pass and rebuilt through the whole per-frame path.
   `tests/deforming.rs` is it, and §8.91 is what it found: **refitting builds in 0.108 ms against
   rebuilding's 0.242 and traces in the same 0.085 either way**, so it is the default. The
   architecture the measurement forced is the useful half — per-instance vertex regions in the
   shared streams, a persistent scratch, a top level rebuilt in place so the descriptor survives,
   and scoped barriers rather than the load path's full one.
2. **Banners — done.** §8.92. `furn_de_banner_pawn_01.nif` is six nodes, twenty vertices, one skin
   and three keyframe controllers, and the whole path behind it is now real: `.kf` and inline
   animation parse, a rig is built from the node graph, a compute pass skins into the placement's
   own vertex region, and the structures over it are built inside the frame. What it does not have
   yet is a clip chosen from anything — a model plays the one animation it carries.
3. **Creatures — done.** §8.93. They were already placed by step 2, because 87 of the 89 models
   animate from their own NIF; what they needed was the text keys that say which part of that
   animation is standing still. `XSCL` and a clip loaded from a `.kf` are still ahead.
4. **NPCs — the body, done.** §8.94. A skeleton chosen by race and sex, thirteen parts found by the
   naming convention `BODY` records use instead of a race field, and the face and hair the record
   names itself. Worn `NPCO` inventory, which overrides the skin it covers, is what is left — and
   with it the 273 `CREA` records that are humanoid and assembled the same way.
5. **Particles and the rest of the controller family.** Particles belong in the transparency target
   beside precipitation rather than in the acceleration structure — §8.87 already transcribed
   `ashcloud.nif` by hand and the upscaler already composites that layer. `NiUVController` is a
   material offset, `NiVisController` and `NiAlphaController` are host-side scalars,
   `NiGeomMorpherController` reuses step 1's deformed slice, and `NiBillboardNode` reuses its
   per-frame top level.

**Deliberately not in it**: AI, pathing, locomotion and animation blending. An actor stands where
the cell put it and plays an idle; what this milestone owns is that the world stops being still.

### What is left, across the milestones

Collected rather than decided here — each is recorded where the work was done:

- **M8's post chain.** Bloom, grading, sharpening, exposure adaptation over time, HDR output.
- **M10's water.** Shoreline foam — the Jacobian's *sign* is computed and thrown away — and underwater
  shafts (§7.8).
- **§5.1's other half.** Settled — roughness dropped, replacer packs refused, and the normal map
  synthesised from painted relief built in §8.90. What waits on it is the specular coat §8.89 tried
  early, and the absolute units §8.51–§8.53 each recorded a debt for.
- **§5.4's scope boundary.** Bind pose only: no animation, no particles, no creatures or NPCs. Planned
  as M12, which is the last feature block before §5.3 opens.
- **§5.3's budget.** 44 ms at 3840×2160 against 16.6, knowingly, and not to be opened until the feature
  set is closed. Opacity micromaps, SER and cluster acceleration structures wait there.

---

## 5. Decisions still open

### 5.1 Vanilla Morrowind has no material data, and its albedo is pre-lit

OpenMW states it plainly: *"Morrowind format NIF files do not support normal maps or specular maps."*
Vanilla assets are 256²-era diffuse textures with **lighting, shading and ambient occlusion painted into
the albedo**, authored for a fixed-function renderer with no per-pixel lighting.

Physically-based ray tracing on top of pre-lit albedo double-lights everything: surfaces read flat and
muddy, ambient occlusion appears twice, and no denoiser tuning fixes it. This is the single largest threat
to "looks great", and it is an art-pipeline problem no renderer architecture solves.

1. **Accept it and tune.** Cheapest. Will beat vanilla, will not look like a modern RT title.
2. **Support replacer texture packs.** Mature ecosystem; OpenMW's `_n` / `_nh` / `_spec` conventions are
   the de-facto standard. Cheap, and shifts the quality ceiling onto the pack.
3. **Synthesize normal and roughness from diffuse.** Height-from-luminance is unreliable; a small image
   model would do better. Medium effort, uncertain payoff.
4. **De-light the vanilla textures offline.** Research-grade, and the only path that makes *vanilla*
   assets physically correct.

**Decided: option 4** (built — §8.35), and **option 2 is refused**: a replacer pack is mod compatibility,
which is the first thing this project says does not rank, and it moves the quality ceiling onto someone
else's art in a renderer whose premise is vanilla content. The failure mode to watch is over-correction —
flat, washed-out output where the algorithm removed genuine painted detail — so vanilla stays available as
an A/B (§8.37).

**De-lighting recovers base colour only, and the two halves of what is left were surveyed apart.** Option
3 covers a normal map and a roughness map, and the evidence for them runs opposite ways.

**Roughness: dropped.** There is no signal to recover and no data to temper an invention with. Of the
19,415 `NiMaterialProperty` records across the 7,319 shipped meshes, **19,125 carry a glossiness of zero**;
12,416 carry a black specular colour and the rest exporter defaults, none of it ever rendered because
`NiSpecularProperty` is force-disabled at this version. Texture names do not stand in: only **1,020 of
4,456** carry a material word at all, and the naming is by object and set — `a_bear_pauldron`,
`amulet_heartfire` — rather than by substance. Nor does the literature offer a way in: every method that
recovers roughness reliably reads a specular highlight, from Deschaintre's flash-lit capture onward, and a
painted 256² texture contains none. What is left is invention, and §8.89 is what invention looked like —
a uniform dielectric floor exceeds the diffuse return of anything below about 2.5% albedo, which is most
of an interior, and arrives as exactly the wash this section exists to avoid.

**Normals: taken, and built — §8.90.** The signal is really there, because the relief in these textures
is *painted*: an artist drew the mortar dark because it is deep and the stone light because it stands
proud. Read as a height field, the log of luminance is that account and its gradient is the normal, taken
per hit off the colour texture already resident rather than baked into a second map. The failure mode is
mild by construction — a normal map redistributes light rather than adding a floor, so getting it partly
wrong lights a wall oddly instead of washing it out.

**The Retinex discriminator this section planned on does not exist in the content.** The premise was that
painted relief is achromatic where a pigment change is not; measured across fifty shipped textures, the
chromaticity step *rises* with the luminance step beside it, 0.0027 to 0.059, so the strongest edges are
the most chromatic and a gate keyed on colour deletes exactly the shape. §8.90 has the numbers and the two
causes. Straight log-luminance, with no gate and no per-texture normalisation, is what shipped.

**A coat waits on that map rather than the other way round.** §8.89 tried the specular layer first, over a
world with a single roughness in it, and a sheen with no variation in it reads as a film over the lens.
The map is now there for it.

**And it was the blocker for absolute units.** `DAYLIGHT`, `SKY_STRENGTH` and the night floor are numbers
tuned to reproduce a screenshot, so nothing keyed to cd/m² — the exposure literature, the moons' real
irradiance, a Purkinje shift — can be applied as published until that is redone (§8.51, §8.52, §8.53).

### 5.2 Licensing — settled: MIT OR Apache-2.0

Permissive, so the NVIDIA RTX SDK licence's source-disclosure prohibition does not apply and **DLSS Ray
Reconstruction is available**. Under GPL-3 the denoiser would have been hand-rolled SVGF from the BSD-3
references — roughly 700 lines of shader to work, 2,000 to be good.

The cost is a fat G-buffer: DLSS-RR requires diffuse albedo, specular albedo, normals, roughness, colour,
depth, motion vectors and a specular guide, plus jitter offset and a reset flag. The specular guide in
particular is easy to forget and awkward to add late (§8.16).

### 5.3 Output resolution is the hardest constraint in the project

**Confirmed against DLSS itself (§8.19):** for a 3840×2160 output under Performance it answers
**1920×1080** — the number the whole frame budget is measured at.

Denoising scales with *output* pixels, not internal ones. DLSS-RR alone costs ~6.1 ms at 3200×1800 on a
3080; anything above 4K output spends 10–14 ms on the denoiser before a single ray is cast, so driving a
high-resolution ultrawide natively is off the table.

**Decided: 1920×1080 internal → 3840×2160 at 60 fps.** 8.3 M output pixels, roughly **8–9 ms of
denoiser**, leaving about **7 ms of a 16.6 ms frame** for acceleration structures, rays, GI and post. If
the renderer proves efficient enough the internal resolution moves up to 2560×1440 at no architectural
cost. Morrowind's shape helps — ~1.1 GB of data, 256² textures, low-poly meshes — so BLAS memory and build
time are a non-issue in 16 GB. The pressure is **TLAS instance count** and **alpha-tested foliage**.

**Features first, and the bar is not chased until they are done.** Measured against this at 3840×2160 the
frame costs 44 ms, roughly two and a half times the budget, and that is knowingly where it stays for now.
Optimising a renderer that is still growing light paths means tuning work that the next feature invalidates,
and the techniques that close a gap this size — opacity micromaps for the foliage above, SER, cluster
acceleration structures — are structural rather than incremental, so they are cheaper to fit once against a
finished set of passes than repeatedly against a moving one. The number is recorded so it cannot be
forgotten, not so it can be worked on.

### 5.4 Smaller decisions — settled by default

- **Light units.** Physical units throughout, 69.99125109 units per metre. `LIGH` radius maps to inverse
  square with a radius-derived cutoff, not Morrowind's original curve.
- **Emissive vs analytic.** A torch is *both* a `LIGH` record and a glowing mesh: use the record as an
  analytic light and mark the mesh emissive **but excluded from light sampling**.
- **Colour management.** sRGB textures, linear working space, tonemap at the end.
- **Asset cache.** Measured and declined — §8.54.
- **Debug affordances.** A debug-view selector, shader hot reload, a headless golden-image mode.
- **Scope boundary for "static".** M0–M11 render the **bind pose only, no particles, no animation**. NIF
  controllers are sized exactly but not evaluated; creatures and NPCs are not placed. That is what kept
  the per-frame acceleration structure out of the renderer until it was proven — M12 lifts it.

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
| build-time: NGX SDK | DLSS Ray Reconstruction | **not a crate and not in this repository** |

**The NGX SDK is fetched, not vendored.** It is NVIDIA's, under the RTX SDK licence, so it lives in
`.refs/dlss` beside the OpenMW checkout and is gitignored like it:

```
git clone --depth 1 https://github.com/NVIDIA/DLSS.git .refs/dlss
```

`DLSS_SDK_DIR` overrides the location. The binding is hand-written FFI against the C header — no generator
and no crate.

**The feature requires the SDK**, and `build.rs` fails with the command to fetch it if it is absent. The
first version warned and compiled the feature out so `--all-features` would build without it — which put
"is it really here" in a `cfg` only `rtxmw-render` can see, which the binary then had to gate on and could
not. `--all-features` therefore needs the SDK fetched, the way the tests need the game installed.

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

1. **Sea level is z = 0 everywhere outdoors.** No per-cell lookup, no interpolation across a boundary.
2. **The flag is the gate, not the value.** Every interior carries a water height whether or not it has
   water, so reading `WHGT` and testing for presence would flood 941 dry rooms. `has_water()` is
   `(flags & 0x02) || is_exterior()`.
3. **Water is shallow**, so the seabed is visible almost everywhere, which is what makes caustics worth
   having.
4. **The shoreline is the most-seen feature** — 41% of land cells contain one.

### 7.2 A sum of trochoidal waves, not an FFT ocean

Tessendorf's spectrum is right for deep water at kilometre scale; we have coastal shallows and interior
pools, and it would cost a compute pass and three textures a frame to simulate a spectrum whose defining
feature — long deep-water swell — does not belong in a swamp. A direct sum of sinusoids is
**differentiable in closed form**, which is what makes the analytic caustic possible at all.

### 7.3 The surface and its shading

One unit quad, instanced per water cell with the level in its transform — water is the *ideal* shared
asset, unlike terrain. It goes in the acceleration structure rather than being intersected analytically so
reflections, shadows and refraction rays all see it without a second code path.

On a water hit, with `n` the wave normal and `η = 1/1.333`: **Fresnel** (Schlick, `F0 = 0.02`); **a
reflection ray** and **a refraction ray**, each traced and shaded; **Beer–Lambert absorption** with σ from
a Jerlov coastal water type rather than a hand-picked blue, which is *why* shallow water reads green and
deep water blue; **single scattering** added back; and **sun glint**, GGX against the wave normal.

**This dodges the denoiser by construction.** The à-trous filter is demodulated by albedo and water has
none — but a mirror reflection and a refraction are *deterministic*, one ray each, no sampling, so water
shades into the emissive/sky channel that already bypasses the denoiser. Perfectly specular water is not a
simplification to be undone later, it is what makes water compose with the filter.

### 7.4 Caustics from the Jacobian

Caustics are ray-density change. With an analytic height field the refracted direction is known in closed
form, so the convergence of the refracted bundle at depth `d` is the determinant of the Jacobian of that
map and intensity is `1/|det J|` — a few ALU per underwater hit, evaluated where the seabed is already
being shaded. No photons, no buffer, no filtering. Held in reserve if it disappoints: photon-splatted
caustic maps, *Ray Tracing Gems II* ch. 30, reported at 0.5–2 ms on RTX hardware.

### 7.5 Underwater

The same model inverted: Beer–Lambert on every primary ray, total internal reflection past the critical
angle of 48.6° — which is why the surface from below is a mirror ringed by a bright disc of sky — and the
sun's colour filtered by depth before it lights anything.

### 7.6 What was built

**The flat plane made the frame faster** — 134 fps with water against 108–116 without. A water pixel
*replaces* a diffuse one, and a diffuse pixel is the expensive kind: sixteen shadow rays and four bounce
rays against water's two deterministic ones.

**Water must not cast a shadow, and how it is told matters.** Building it non-opaque so the any-hit loop can
wave shadow rays past *works* and **costs half the frame rate** — 68 fps against 134 — because every shadow
ray crossing the sea then invokes a shader where traversal alone had been enough. Water carries a mask bit
instead and `occluded` asks only for solid geometry: free.

**Waves.** Trochoidal components with real dispersion, `sqrt(gk)` with Morrowind's own gravity, so long waves
outrun short ones and the pattern never sets into one rigid shape. The quad stays flat and only the normal
moves — displacing two triangles buys nothing, and the silhouette against a shore comes from the terrain
behind it. **Waves cost nothing measurable.**

- **A wave shorter than the pixel looking at it is averaged away**, using the ray cone footprint already
  carried for texture LOD. Picking one of a crest and a trough instead is what makes distant water a field of
  crawling white sparks.
- **What a ray cone cannot resolve is not gone, it is rough.** The variance of the discarded octaves comes
  back as a widened specular lobe — LEAN mapping in one dimension. Its most visible consequence is the sun: a
  mirror shows one hard dot, a mile of ruffled water a shimmering road, because the glitter path **is** the
  wave-slope distribution made visible. The disc is widened by the lobe and dimmed by the same factor, so a
  rougher sea spreads the sun without adding light.
- **Which side of the surface a ray is on is a question about the plane, not about a wave.** Taking it from
  the wave normal reads a facet tilted away at a glancing angle as "the camera is underwater", sends the
  reflection into the seabed, and turns the far water white.

**Caustics.** `J = I - bend·depth·H` with `H` the Hessian of the same sinusoids the normals come from. The
finding that made it work was not about caustics: `water_ray` traced reflection and refraction at the
*bounce* cone spread, where a coarse mip is correct for a diffuse bounce. A reflection and a refraction are
specular and carry the pixel's own cone; at the bounce rate a seabed a hundred units down was sampled with a
hundred-unit footprint, so every texture read its top mip and every wave was averaged out of the caustics.
The term was varying by 25% on its own and arriving at the frame as 4%. Fixing the cone turned the pattern on
and sharpened every reflection in the game.

**Where the model stops.** `q = p - bend·grad(h)` holds while the refracted bundle has not crossed itself.
Past the first focus the rays have folded, and because the term is evaluated at the seabed rather than at the
surface it came from it starts *making* light — three quarters more at four hundred units. The depth fed to
the lens is capped at 140 units, which holds the error under 6% and says something true anyway: caustics are
sharp in a shallow pool and washed out in deep water.

**Chromatic dispersion is kept and worth almost nothing.** Cauchy's fit gives 1.3326 / 1.3342 / 1.3392 at
600 / 550 / 450 nm — three determinants over a Hessian that does not depend on the channel. **Twelve pixels in
ninety thousand differ by more than one level.** Kept because it is right and free; if the sea ever gets steep
enough for the determinant to approach zero, this is what puts prism edges on cusps.

**Shore and underwater.** The waterline fades over the last thirty-five units of depth, and a camera below the
surface fogs every primary ray. Total internal reflection came free out of `refract` returning zero past the
critical angle.

- **The seam is a grazing-angle artefact.** From above, three units of water is almost invisible whatever the
  shader does — the first test compared views straight down, passed, and went on passing with the fade deleted
  entirely. Edge-on, Fresnel turns that same water into a mirror.
- **Underwater the *albedo* is dimmed, not the lighting.** The filter divides lighting by albedo, so dimming
  both would put the water straight back, and what the depth took is a property of the path.
- **A ribbon of flat colour along every waterline.** The refraction ray was offset to the *far* side of the
  plane, so wherever the bed sat nearer the surface than the 1.5-unit offset the ray began under the ground,
  travelled down through open air and reported water of unbounded depth. Both water rays now leave from the
  viewer's side and trace against solid geometry only; culling water from its own reflection and refraction
  removes the self-intersection the offset existed to avoid. Every test passed throughout, because the
  waterline test used three units of water — twice the offset, and the artefact lives below it.
- **Deep water was a milky sheet, and the cause was the scattering albedo rather than the colour.** A channel
  whose scattering albedo approaches one settles at a bright colour however deep it gets. Halving the albedo
  makes the deep go dark while lowering the extinction keeps the shallows transparent — the two complaints
  pull on different terms.
- **The in-scattering integral was wrong in the direction that made it worse.** Light scattered toward the eye
  has to reach the point it scattered from, and only the return leg was attenuated. Integrating both replaces
  `1 - T` with `(1 - T²) / 2`: identical in the shallows, half as bright where it settles, and markedly less
  red, because squaring the transmittance costs red twice.
- **The sun was attenuated on its way down and the sky was not.** Found by measuring an invariant rather than
  by looking: the same column of water seen from ten units above and ten below has to agree, and does, to 3%.
  At a *slant* they legitimately differ by 11% — entering at 53° a ray bends to 37° and reaches a floor 200
  units down in 250 units of water, where the same look from below costs 317. **Water really is clearer from a
  boat than from under it.**

The extinction and scattering coefficients are art direction resting on physics. The tests derive every
expectation from a single `EXTINCTION` constant so a tuning pass is one line rather than five pieces of
arithmetic that quietly stop describing the shader.

### 7.7 The spectrum is empirical, and its short end is a limit in time

**Caustics tiled when the octaves were spaced too widely**, and the reason is in the derivative: **curvature
weights an octave by `A k²`** where slope weights it by `A k`. A geometric series with gain 0.55 and lacunarity
0.618 gives a curvature ratio of **1.44 — above one** — so the finest one or two octaves own the Hessian
entirely, and two plane waves crossing is a grid. Slope's ratio was 0.89, so every octave contributed there.
Two fixes, and the second alone still leaves a visible grain:

- **Space the octaves closely** so five or six components land at comparable short scales pointing in different
  directions — broad in direction where the swell is narrow.
- **Carry the ripples on the swell.** A low-frequency displacement applied to the sample position before the
  waves are evaluated: physically, short waves riding the orbital motion of long ones. Thirteen units of drift
  is most of a wavelength to the shortest waves and a rounding error to the longest. The Hessian is taken with
  respect to the drifted position, dropping the chain rule's contribution from the drift itself — that field
  turns over six hundred units against a curvature set by ten, so its Jacobian is within a fifth of the
  identity and the omission is a slow variation indistinguishable from what real water does.

**The series is the TMA spectrum** — JONSWAP under Kitaigorodskii's shallow-water attenuation — spread by
**Donelan-Banner**, which is Horvath's pairing. Thirty-two components: eight wavenumber bands, four directions
each, sampled by *quantile* of the directional spread so every component carries the same energy and the
spread's shape is exact however few are taken.

- **The depth term is the coastal correction this game needs.** A six-metre swell over half a metre of water
  travels at two thirds of its open-sea speed; over three metres at full speed. That is why this is TMA.
- **The spread is frequency-dependent by construction** — the lowest band fans across 68°, the highest past 120°.
- `alpha` never appears: it is a constant multiplier on the whole spectrum, so it cancels, and the table is
  scaled instead to a **significant wave height**.

**The short cutoff is a limit in time, not in space.** TMA's tail is the Phillips saturation range, where
steepness is *constant* with wavenumber, so `A k²` climbs without bound. Cutting at eighteen units produced the
best caustics this renderer has drawn — and made them **tear**. **A wave's period falls with its length, so the
waves that focus hardest are the ones that move fastest. They are the same waves.**

| shortest wave | caustic contrast | change per twelfth of a second |
|---|---|---|
| the old hand-tuned series | 17.7 | 49% |
| 18 units | 24.6 | 73% |
| **32 units** | **18.5** | **51%** |
| 50 units | 16.5 | 33% |

**Choppiness is in and changes almost nothing to look at** — 788 pixels of a shore by at most a twentieth. The
displacement's Jacobian contributes the steepness `A k`, summing to 0.28 across the spectrum, while the
refraction term contributes `bend·depth·A k²`, ten times that at a few metres. One thing it buys outright: with
displacement the map from surface to seabed is a *ratio* of determinants — a patch covers `det(I + dD)` of
surface and lands on `det(I + dD − bend·H)` of bottom — which is 1 at zero depth by construction, where a single
determinant would brighten the bottom of a depthless puddle.

**This surface does not focus light; it modulates it by tens of percent.** Focus needs `bend·depth·A k² ≈ 1`;
summed over the whole spectrum with every octave aligned, which never happens, that reaches 1.21 at the deepest
water the term is allowed. Three plausible culprits were cleared: the brightness ceiling raised 3 → 8 was
pixel-identical, filtering off was pixel-identical, and 14 octaves → 18 (shortest wave 8.4 → 2.5 units) came out
finer and noisier rather than bolder. That last is the useful negative: curvature grows with wavenumber, but the
cells it focuses are the size of those waves, so the pattern gets *finer* rather than sharper, and below a pixel
it is noise.

**A wind-chop band was built, measured and reverted.** Bold caustics need roughly nine times the energy a
swell-shaped spectrum puts in the metre band, and a second log-normal peak supplies it — a bimodal sea of swell
plus wind waves is ordinary oceanography. **It does what it was supposed to**, contrast over a shore patch rising
20.0 → 24.2, and costs more than that is worth:

| cost | measured |
|---|---|
| the whole sea roughens | distant water's pixel-to-pixel variation 14.0 → 23.1, the exact failure the ray-cone filtering exists to avoid |
| caustics alias where they sharpen | stipple 9.1 → 21.9 |
| **the water stops obeying Beer–Lambert** | looking up from below, transmission fell to 0.649 of the near view where the analytic answer is 0.807 |

The third settled it: a surface rough enough to focus light that hard refracts the view far enough that
straight-line attenuation no longer describes what comes out — and absorption, scattering and the sun's path to
the seabed are all built on that attenuation. A band-limited widening of the curvature cone works against the
aliasing (stipple 3.5× below baseline) but is too blunt: the widening that cleans a close view erases the pattern
at a middling one. Something widening with the *sharpness* rather than the distance would be the right shape.
**Vvardenfell's water is a sheltered coastal shelf, and a sheltered coastal shelf does not throw pool caustics.**

### 7.8 Not built

Foam at the shoreline and sun shafts underwater. Foam has a natural source already computed: the Jacobian
determinant's **sign** detects surface self-intersection, and where a surface folds is where whitecaps
belong — the shader currently takes its absolute value.

**No absolute frame rate from this machine is worth quoting.** Measurements ranged 116 to 382 fps for the
same scene, because the GPU idles at 315 MHz and only ramps to 2,280 under sustained load. Back-to-back A/B
pairs are the only comparisons that survive.

---

## 8. Findings

Things that were wrong, or were measured and turned out not to be worth it. Each is here because the
mistake was not visible from the code and something else would have made it again.

### 8.1 Ray offsets follow the triangle's plane, not the shading normal

Rugs sparkled along their edges and smooth-shaded rocks came out under salt-and-pepper, and **the noise
changed strength from triangle to triangle**, which is the diagnosis. Every ray left offset along the
*interpolated* normal, which on geometry this coarse can point tens of degrees away from its triangle — so
the origin lands under the surface on some triangles, and a shadow ray that starts underneath is stopped by
the surface it started from.

Rays now leave along the triangle's own plane, from the cross product of its edges. **Which way that plane
faces is the subtlety, and getting it wrong blacks out half an interior:** turned toward the *ray*, a shadow
ray leaving the back of a tapestry sets off for the sun, meets the tapestry, and reports shadow. The plane
has to face the side the surface is *lit* from — the side its shading normal points at.

### 8.2 The shading normal faces the ray

Dark dust over every tree, speckle on tapestries, sparkle along a rug's edge: one defect. **The shading
normal was whichever way the vertices were authored, so a surface hit from its far side reported the light
landing on its near one** — lit through its own body. Morrowind's foliage is thousands of single cards packed
below a pixel apiece, wound every which way, so neighbouring pixels came back at opposite brightnesses.

The normal now faces the ray, which is what `gl_FrontFacing` does for every rasteriser. It could not be done
before §8.1: while offsets followed the shading normal, turning it toward the viewer sent shadow rays out
from under the surface.

**Which side the ray met is decided by the triangle's plane, not by the normal being turned.** An interpolated
normal near a silhouette can point away from a face the camera looks straight at, so the obvious reading flips
part of a surface along a seam that slides across the floor as the camera moves. A plane cannot disagree with
itself that way. That rests on the winding agreeing with the authored normals, which nothing in the format
enforces, so it was measured: **77 of 60,215 triangles** are wound against their own normals, a fifth of a
percent. Three of this repo's own *fixtures* were, and had never been wrong before, because nothing consulted
a winding until now.

Five suspects were ruled out first, each by one render: alpha cutoff swept 0.15/0.5/0.85 (17.8/19.0/17.1 —
real but a tenth of it), +3 levels of mip bias (19.0 → 17.3, visually unchanged), alpha coverage held at 44%
across mips (19.0 → 19.8, *worse*), 4 → 64 bounce samples (unchanged, so not stochastic at all), sun forced
fully visible (unchanged). Rendering **albedo alone** came back clean, which put it in the lighting.

Two things found and deliberately not acted on: Morrowind's cutout art is black wherever it is transparent —
1,449 of the canopy texture's 1,635 fully transparent blocks — so a filtering sampler mixes black into every
leaf edge, and dividing it back out by alpha changed nothing measurable; and `NiStencilProperty` would name
two-sided surfaces outright, except the three shipped archives contain **no** stencil property at all.

### 8.3 Sheets are lit from both sides

**Morrowind hangs single layers of triangles and the renderer treated every one as the skin of a solid.** A
layer has no inside: lit from the far side it should glow, and shaded as a solid it goes black. A run carries
a `thin` bit into `GpuGeometry`, and the shading term is `max(N·L, 0) + T·max(−N·L, 0)` with `T = 0.5` — a
Lambertian sheet, view-independent. The indirect gather takes the hemisphere the ray came from rather than the
one the normal names, so an inside-out triangle no longer gathers the inside of the hull it is nailed to.

**Deciding which runs are sheets is the whole difficulty, and two plausible tests each fail on real data.**
Asking what a run *encloses* works for a flat rug and fails for a curved sail. Asking whether it has a *border*
is exact for closedness but cannot be asked of a run: a run is a material boundary, so a wine bottle with three
textures arrives as three open patches — measured that way, 268 of 308 runs classified as cloth.

The test that holds asks both at the level each is meaningful: the border of the whole *mesh*, the enclosed
volume of the *run*. **Edges are keyed by quantised position, not by vertex index** — Morrowind splits vertices
at every texture seam, so two triangles sharing an edge routinely name four different vertices for it, and
counted by index every seam reads as a border and every solid as cloth.

**The shape test finds a rug and a sail and cannot find a tree.** A canopy is hundreds of leaf cards joined at
the branches, and the cupped cluster wraps as much air as a room's shell — `Flora_BC_Tree_02` scores 0.031
against a cube's 0.068. **The material knows.** A run whose alpha is anything but opaque is a cutout, and
Morrowind has no solid cutouts: the mode is set on foliage, thatch, banners, grates and glass, every one a
single layer. Together the two signals mark 47 of a shore's 419 runs and 50 of an office's 308. Backlit foliage
is 27% brighter with no measurable change in noise.

### 8.4 A model's outermost node transform is discarded

Seyda Neen's fireplace presented its back to the room — which is what made it hard, since a wrong *convention*
would have turned the whole room. `in_nord_fireplace_01.nif` carries a half turn about Z on its outermost node,
and **the original engine ignores that transform** — for block zero only, only when that block is a node, and
never for one named `bip01`. 455 of 7,317 models carry a turn there and 423 a translation or scale; 259 are
rooted at `bip01`, 255 with a real transform, so the exception is as load-bearing as the rule: discarding a
rig's animation root would flatten every piece of armour in the game.

The anchor that needed no screenshot: **a hearth and the fire inside it are placed as separate references**, so
the fire says which way the stack faces — −0.99 against the fire before, +0.99 after.

### 8.5 A reference's Euler angles apply Z first

OpenMW composes the rotation as `Quat(z, -Z) * Quat(y, -Y) * Quat(x, -X)`, which reads as X-first and was
transcribed that way. It is not: **OSG writes its quaternion product in the opposite order to everyone else**,
so that expression means Z first and X last. The tell is a page away — `Misc::toEulerAnglesZYX` recovers the
angles `makeOsgQuat` was given, and only inverts it under the reversed reading. Over four thousand random
angles, reversed round-trips to zero and the ordinary reading is out by up to two units of a unit vector.

The argument that delayed this: OpenMW writes the *same* order for Bullet a few lines below, Bullet's product is
the ordinary one, and the two were assumed to agree because physics ought to match graphics. They do not agree —
that is a latent inconsistency in OpenMW, not evidence about OSG.

It moves only references that turn about more than one axis, 22 of one interior's 268, and the obvious
measurements are blind to it: a plate's tilt out of horizontal is *identical* under both orders whenever the Y
angle is zero. What separates them is that things stop resting where they were put — the book's base sits 8
units below the board its cups stand on, which is the test.

### 8.6 `NiStencilProperty` was five bytes short, and nothing could have noticed

It read flags, a versioned `bool` and five words. The format is flags, a **one-byte** enabled flag — a `char`,
not the four-byte `bool` this version uses elsewhere — and seven words. Twenty-six bytes read against thirty-one.

**Blocks in this version carry no size**, so a property that reads one field too few leaves everything after it
decoding as garbage, silently. No shipped mesh could catch it — **0 of 7,319 name the block** — which is why the
corpus test passed. Every fixed-size property now has its byte count asserted directly: a synthetic block of
exactly the documented length must be consumed whole, and one byte shorter must fail rather than quietly stop.
The same mistake in the other direction is annotated three blocks away at `NiSourceTexture`, where a `char` read
as a `bool` over-consumed by three.

### 8.7 The unprojection lost the world, and it looked like jitter

Reported as the camera shaking when it moved, everywhere except near the world origin. That last detail is the
whole diagnosis:

```glsl
vec4 target = frame.inverse_view_projection * vec4(ndc, 1.0, 1.0);
vec3 direction = normalize(target.xyz / target.w - frame.camera_position);
```

The unprojection lands a **world-space point on the near plane**, 0.05 units from the eye, and the subtraction
recovers the direction. Both operands are the size of the world; their difference is 0.05. At Seyda Neen's
75,000 units the gap between representable `f32` values is 0.024, so the answer had two or three bits left in
it — 0.00 px of aim error at the origin, 2.1 at 1,000 units, **127 at Seyda Neen**, **377 at the far corner**.
It read as jitter rather than as a broken projection because the error is a *smooth* field.

**The fix is not more bits.** The subtraction is what should not exist: **the eye is removed from the view
before the inverse is taken**, so the unprojection lands in a space centred on the camera and hands back an
offset directly.

```rust
let mut rotation = view;
rotation.w_axis = glam::Vec4::W;   // a look-at view is rotation * translate(-eye)
(projection * rotation).inverse()
```

Nothing then cancels, the aim is **0.00013 pixels wrong wherever the camera stands**, and the matrix is far
better conditioned.

**Why nothing caught it.** The unprojection had a test, and it was a round trip — correct, and blind to this,
because it ran with the camera at `(1, 2, 3)`. **A precision fault that vanishes at the origin needs a test that
leaves it**, so the assertion is now made at four camera positions out to the far corner, and the same scene
rendered at the origin and at 200,000 units has to produce the same picture.

**What is still `f32`, and why that is fine.** Hit positions quantise to 0.024 units out there, but the shadow
ray bias is 1.5 units, a terrain blend weight moves by 5e-5 of a tile, and a wave phase by 0.005 radians.

### 8.8 Terrain blends four tile centres, and the quadrant is what carries them

A cell's `VTEX` names one texture per 512-unit tile, 16×16 of them and nothing in between, so ground met its
neighbours along a straight edge. The fix is bilinear blending between the four nearest tile *centres*; the
design question is where the four ids live.

**Not a splat map.** The weights are a fixed function of world position and need no storage. What needs storing
is *which four* textures a point blends, and that is constant over a region.

**The quadrant.** Split each tile into 2×2 quads — 32×32 quadrants per cell — and give each the 2×2 block of
tile centres it falls inside. The four ids pack as 4×u16 into two words `GpuMaterial` already had spare, so **no
new binding, no new buffer, and 48 bytes stays 48 bytes.** A cell has 1,024 quadrants and about 78 distinct
tuples (worst 121), because most quadrants sit inside a run of tiles naming the same texture. A cell origin is a
whole number of tiles, so it cancels and the weights are a function of world position alone.

**Cost:** three extra texture taps on ground pixels — **6.60 ms against 5.95 ms** of trace at 1920×1080 on a
view that is almost entirely terrain. An early-out where all four ids agree measured **6.60 ms, no change at
all**: the taps are not serialised on anything a branch can skip, and the branch is not coherent across a warp.
Reverted.

**Proof.** `tests/terrain.rs` renders a plane with the four layers set to red, green, blue and white, so green
*is* the x weight and blue *is* the y weight, and predicts every pixel to within two levels. The first version
of the ramp was wrong in two ways — §8.14.

### 8.9 A horizon costs 1.5 ms, and the lights would have more than tripled it

Vvardenfell is only some 36 cells across, so the fix for a world that ended at the streaming window is not an
LOD hierarchy but a second tier: `CellDetail::Distant` out to twelve rings, terrain and objects, at a sixteenth
of the triangles.

**Decimating is enough, and stitching is not needed.** `Mesh::from_land` takes a stride; at 4 a cell is 17×17
vertices and 512 triangles. Because the stride divides the 64 quads a cell spans and the shared last row is kept,
**two cells at the same stride still meet vertex for vertex** — asserted exactly, not to a tolerance. Only the
ring where the detailed window meets the distant tier can crack, and the coarse chord parts from the fine surface
by at most **64 units, half a vertex spacing** — under three pixels at 1080p, at a boundary never closer than
three cells.

**Cost at 1920×1080 from a hilltop over the whole island**, every pixel horizon:

| | trace | window load |
|---|---|---|
| the 7×7 window alone | 4.77 ms | 208 ms |
| + distant terrain, 12 rings | 5.40 ms | 511 ms |
| + the objects on it | **6.29 ms** | 1,155 ms |
| + the lights those objects carry | 9.85 ms | — |

**That last row is the finding.** 21,772 distant instances cost 0.89 ms; the 229 `LIGH` references among them
cost **3.8 ms more**. A lamp a kilometre away with a radius of a few hundred units reaches nothing on screen, so
a distant cell places its lamps and drops their lights, and the image is unchanged. Peak GPU memory with the
whole visible world resident is 620 MB.

**Residency and detail are separate questions, and keeping them separate is what stops a hole.** The first shape
evicted a cell whose tier no longer matched where it was — and a camera crossing one boundary demotes a whole
column at ring 3, so every crossing deleted seven cells of good coarse terrain and showed sky until the detailed
copies loaded. `still_resident` is now blind to the tier and only asks whether a cell has left the world's edge,
while `rebuild_as` asks whether to *request* the other tier, so the swap happens in the frame the replacement
lands. The hysteresis lives in `rebuild_as`: a cell earns a rebuild only a whole ring past the boundary.

**Cells arrive sixteen a frame, and "arrive" has to include the misses.** Of the 625 squares the horizon asks for
only 428 exist, and a loop that stops as soon as a cell fails to place still drains one sea square a frame — 163
cells by frame 50 and ~250 frames to fill, against 234 and ~100 taking sixteen either way. What made
one-at-a-time the rule for the near window was the top-level rebuild, and that happens once a frame however many
cells landed.

### 8.10 Lights are binned into a world grid

The shader walked every light for every shading point, primary and bounce alike: **0.031 ms per light per frame
at 1920×1080** whether or not it contributes. Three filters were tried on the *geometry* before the lights were
suspected, and their failure is what pointed at them — removing 99% of the instances saved 0.9 ms while removing
the lights saved 3.8. **A cost that does not move when the geometry does is not the geometry's.**

**A uniform grid over the lights, in world space** rather than screen space, because a bounce hit is not on screen.
Cell `i` owns `indices[offsets[i]..offsets[i + 1]]` — a prefix sum with a trailing sentinel, so the structure is
two buffers however many cells it has, built by a counting sort at the same commit that rebuilds the top level.
The cell size starts at one terrain tile and **doubles until the grid fits two budgets**: 65,536 cells and 262,144
index entries. The second is not redundant — a wide world overruns the first, and one light with an enormous reach
overruns the second while the grid is small.

Vivec is the worst light density in the game: 173 lights in one 7×7 window, against 53 in Balmora and 20 in a
typical interior.

| lights | with the grid | walking them all |
|---|---|---|
| 173, as shipped | **4.78 ms** | 6.91 ms |
| +500 | 5.06 ms | 19.03 ms |
| +2,000 | 5.05 ms | 66.00 ms |

Below about fifty lights the two are within noise, which is the honest bound: **this buys a town, not a room.**

**The output is bit-identical.** Forcing the grid to a single cell reproduces the old walk through the same code
path, differing in 0 of 2,073,600 pixels. The grid may offer a light that turns out not to reach — it bins by
bounding box and the shader's distance test settles it — but it must never withhold one that does, and that
one-sidedness is asserted against the brute-force answer over a sweep of probes. Binning by a light's centre
instead of its reach fails it.

### 8.11 Cell frustum culling buys nothing, and the reason generalises

Built, measured, reverted; kept because someone will otherwise propose it again. Out of 6,455 instances across a
48-cell window it removed 4,124 looking along the ground and 1,709 looking at the sky — a 74% cut, as selective as
this can ever get. Median frame: **1.91 ms culled against 1.88 ms with everything resident.** Not a small win
inside the noise — zero, at three quarters of the scene removed.

The premise is what is wrong, and it applies to any culling scheme proposed above the acceleration structure: **a
bounding volume hierarchy already culls, spatially, and does it per ray rather than per frame.** A ray that never
travels toward a cell never descends the subtree holding it. Frustum culling is a rasteriser's idea — it exists
because a rasteriser must *submit* geometry before it can reject it. Against that, the costs are real: 1.1 ms and
a device-idle stall each time a turn brings a cell across a frustum edge, and the image changes by 3,066 pixels
because bounce and shadow rays that reached a culled cell now escape to the sky.

If the top level ever does become the cost, the thing to reach for is fewer *instances* rather than fewer cells.

### 8.12 The visibility shader is six files

`primary_visibility.comp` had reached 1,243 lines and was more than half water. It is now 78 — a header, six
includes and `main`. The split is by **dependency order**, which is the only order GLSL allows:

| | |
|---|---|
| `bindings.glsl` | the descriptor set and the structs in it — the whole of what the host must agree with |
| `sampling.glsl` | hashing and direction sampling, no bindings touched |
| `surface.glsl` | attribute fetch, cone LOD, the cutout test, and both traversals |
| `lighting.glsl` | next-event estimation, the sky, one bounce |
| `waves.glsl` | the height field and its gradient |
| `water.glsl` | Fresnel, absorption, caustics |

**One forward declaration in the whole file set.** Lighting needs the sun dimmed through water and water needs a
surface shaded, which is a cycle; declaring `sun_through_water` and `daylight_reaching` at the top of
`lighting.glsl` breaks it. Putting water first would have cost three. The refactor is pixel-identical, which is
the only claim worth making about it.

### 8.13 Motion vectors, and the second place the world's size would have shown

**What a pixel stores.** The displacement, *in pixels*, from where its surface is now to where it was on the
previous frame's screen. A miss stores zero, which a temporal filter reads as "this pixel did not move"; for the
sky that is true.

**Reprojected as an offset, never as a world point.** The obvious formulation projects the hit position with the
previous view-projection and subtracts — §8.7's mistake with the operands swapped. Instead:

```glsl
vec3 was = direction * surface.t + frame.camera_motion;
vec4 before = frame.previous_clip_from_offset * vec4(was, 1.0);
```

`direction * t` is the offset from *this* eye; `camera_motion` is `now - before`, differenced on the host;
`previous_clip_from_offset` is the previous frame's `projection * rotation` with its translation dropped. Nothing
world-scale is ever subtracted on the device. **The camera delta is exact** — subtracting two `f32`s within a
factor of two of each other is exact, and a camera does not cross half the world in a frame.

**Full floats, unlike the rest of the G-buffer.** A motion vector spans the frame — a couple of thousand pixels
when the camera turns — and a half float's eleven-bit mantissa lands only on whole pixels above 1024.

**Behind the previous eye there is no answer**, and the perspective divide would fold such a point back into the
frame as a plausible coordinate. `w > 0` is checked and the vector left at zero.

**Cost: not measurable** against run-to-run variance of ±2 ms. Recorded as unmeasured rather than as a number.

**What is asserted**, because a reprojection can be plausible and wrong in three ways: a still camera leaves every
pixel where it is (to a hundred-thousandth of a pixel — a still frame is an unproject followed by a project and
`f32` rounds in between); a camera that only *turns* moves every surface by the same amount whatever its distance;
a camera that *steps* moves near surfaces further than far ones — checked against the naive world-space projection
carried out in **double precision**, which is the calculation the shader must not make, done exactly. The first
attempt at that test failed because its reference was the `f32` world-space projection: the reference was wrong,
which is §8.7 turning up a second time from the other side. Two walls at 200 and 400 units move by 0.64 and 0.32
pixels for a four-unit step, hand-computed from the field of view. The sign is asserted too.

**One camera, one type.** `view`, `projection` and the eye travel together as `Viewpoint`. They have to agree, and
passed as three arguments nothing said so; a frame whose rays start somewhere its matrices do not would render a
plausible picture of the wrong place.

### 8.14 The ground was blended everywhere and nowhere

Two faults in §8.8, pulling in opposite directions.

**A tile has to read as itself somewhere.** The ramp ran the whole 512 units from one tile centre to the next, so
*no point on the map drew a single texture*. The original engine blends through a map of **two texels per tile**,
each tile's own pair at full weight, and lets bilinear filtering do the rest
(`components/esmterrain/storage.cpp:497`, *"We need to upscale the blendmap 2x with nearest neighbor sampling to
look like Vanilla"*). That confines the transition to the 256 units straddling the boundary and leaves the middle
half of every tile pure. Written directly it is one line with a clamp:

```glsl
vec2 weight = clamp(fract(world.xy / 512.0 - 0.5) * 2.0 - 0.5, 0.0, 1.0);
```

**And the blend stopped at the cell boundary.** `terrain_materials` read one cell's `VTEX` and clamped past its
edge, so a cell's outermost quadrant blended a tile *with itself* and two cells met in a 512-unit band of flat
ground with a hard seam down the middle, every 8,192 units. The fix is to read the eight neighbours' `VTEX` as one
grid three cells on a side. A neighbour that is open sea has no `LAND` record and keeps the old clamped value —
the right answer where there is genuinely nothing to blend with, rather than everywhere.

**Cost: 230 ms across a 428-cell window**, about half a millisecond a cell. That is why `LandRecord::textures_of`
exists: eight full `LandRecord::parse` calls per cell would undo eight delta-coded heightmaps to read one
subrecord. Interning only the tiles a quadrant can reach is a further 40 ms — worth having, and *not* where the
time goes, which the first guess had backwards.

**What is asserted.** The render test predicts every pixel of the new profile exactly, *and* that a point well
inside a tile draws that tile alone. On real data, Seyda Neen's ground must draw at least one texture its own
`VTEX` never names, and every such texture must belong to a cell beside it: reverting to the clamped version fails
that while still producing a perfectly plausible list of Bitter Coast art. That was the trap in the original test,
which pinned the *names* and so could not see a missing blend at all.

### 8.15 A promoted cell kept the terrain it had when it was far away

Hard-edged rectangles of the wrong ground under the camera, and only after flying. A fresh screenshot of the same
coordinates never showed it, which is the clue: the cell had to have been somewhere else first.

**Mesh slots are grow-only and keyed by source path**, and terrain is keyed by its cell, `land:x,y`. The distant
tier (§8.9) broke that premise without changing the key. **A cell has one heightmap per level of detail**, and both
were called `land:x,y`. A cell that arrived in the distant ring at stride 4 and was later promoted found its name
taken and was handed back the coarse mesh — geometry *and* the material indices baked into its submeshes at upload.
Near terrain then drew 512-unit quads each carrying a single quadrant's four textures, which under §8.14's clamped
ramp saturate into exactly the reported rectangles.

The key is now `land:x,y@stride`.

**Why the tests missed it.** Every one loaded a cell once. `cell_residency.rs` covered two cells naming one mesh;
nothing covered a cell coming back *different*, which is the case the second tier introduced.

**Out of scope, noted:** anything else a cell can name that changes with its detail would have the same problem.
Nothing does today — objects are keyed by file path and are the same mesh at any tier — but the key is the
invariant, not the terrain.

### 8.16 The guides an upscaler reads, written before there is one

§5.2 warns that the specular guide is easy to forget and awkward to add late. The awkwardness is specific: the
guide is a quantity the **trace** records at the hit, so retrofitting it means going back into the shader rather
than adding a pass.

**Specular albedo, roughness, and the specular hit distance.** A reflection does not move across the screen the
way the surface carrying it does, so a temporal filter given only depth reprojects every mirror wrongly. The
distance to the reflected hit is what fixes that, and `water_ray` was already computing and discarding it.

Vanilla Morrowind is matte — `NiSpecularProperty` is force-disabled at this NIF version, so **water is the only
thing in the world with a specular response**: its Fresnel term is the albedo and the lobe left by unresolved
waves is the roughness. Everything else reports `Guides(vec3(0), 1, 0)`.

**One new image, not three.** Specular albedo and roughness share an `rgba16f`; the distance rides in the
**albedo target's alpha**, which the composite does not read.

**Jitter, off by default** until something accumulates. Halton in bases 2 and 3 rather than a random offset, and
the test asserts the property that distinguishes them: 16 frames must touch all four quadrants of the pixel and
64 all sixteen sixteenths, which a random sequence fails by clumping.

**The convention I got wrong.** The obvious reading of "motion vectors must exclude jitter" is to measure against
the pixel centre. The jitter is applied to the *coordinate*, not the matrix, so a hit produced by a jittered ray
projects back through that same matrix to exactly the jittered coordinate — measuring against it cancels the
jitter, and measuring against the centre *introduces* it: a still camera reports **0.30 pixels** of motion that is
purely its own jitter. The original line was right; the fault injection not firing was the tell.

### 8.17 NGX links by hand, and `wchar_t` is 32 bits here

**Hand-written FFI, no generator.** NGX's parameter map is a C++ class with a vtable, which looks like a reason to
reach for `bindgen` and is not. Every call this needs is exported with **C linkage**, checked with `nm` on
`libnvsdk_ngx.a`: the SDK's own helper headers reach the map through `NVSDK_NGX_Parameter_SetI` and friends rather
than through the vtable.

**The extension query comes first because nothing else can.** `NVSDK_NGX_VULKAN_RequiredExtensions` takes no
Vulkan objects, and what it returns has to be enabled *when the instance and device are created*. On this driver:
instance `VK_KHR_get_physical_device_properties2`; device `VK_NVX_binary_import`, `VK_NVX_image_view_handle`,
`VK_EXT_buffer_device_address`, `VK_KHR_push_descriptor`. Queried rather than hardcoded, since the list has changed
between SDK versions.

**`wchar_t` is 32 bits on Linux**, and declaring `GetNGXResultAsString` as `*const u16` reads UTF-32 at half the
stride — every error name came back one character long. `NVSDK_NGX_Result_FAIL_OutOfDate` rendered as `"N"`. The
test caught it because it asks for the SDK's actual name rather than for "some letters". Each declaration carries
the header line it was checked against.

### 8.18 NGX comes up, and four wrong parameters on the way

**DLSS Ray Reconstruction reports available on the RTX 4090 Laptop GPU.** Four things wrong first, each returning a
code that named itself — the reason `Display` asks the SDK rather than printing a number:

- **The device could not be created with what NGX asked for.** It names `VK_EXT_buffer_device_address` because it
  supports older drivers; the same capability is core in Vulkan 1.2 and this device enables it there, and the spec
  forbids both. Superseded names are dropped, and the test asserts they are gone.
- **`FAIL_UnableToWriteToAppDataPath`.** The header gives `InApplicationDataPath` a default of null and NGX rejects
  null. A C++ default argument is not the same as an optional parameter.
- **`FAIL_InvalidParameter`.** The project id is a **UUID and is parsed as one**; mine read `…-rtxmw000001`, which
  is memorable and not hexadecimal.
- **Available, but reporting itself unavailable.** `libnvidia-ngx-dlssd.so` is neither on the loader path nor beside
  the binary, and NGX's default search is the application folder alone — so it has to be told, through
  `NVSDK_NGX_FeatureCommonInfo`. That struct carries its logging block by value, so the whole thing must be declared
  even though only the path list is set.

**What this bought beyond DLSS.** `Device::new` takes extensions from its caller, enabled where present, and
`PhysicalDevice::supports` answers for names this crate has never heard of — the list is queried from whoever knows
rather than becoming another constant in the Vulkan layer.

### 8.19 DLSS agrees with the frame budget

| preset | render resolution for 3840×2160 |
|---|---|
| Performance | **1920×1080** |
| Balanced | 2227×1253 |
| Quality | 2560×1440 |

Dynamic resolution is offered too, from 1920×1080 up to native — a lever if the trace ever overruns.

**The query is not an exported symbol.** It is a function pointer the driver's feature library puts *into the
capability map*, fetched by name and called with that same map. That is why the SDK's helper returns
`FAIL_OutOfDate` when the pointer is absent: it means the feature library was never found, which is §8.18's problem
under a misleading name.

**The three presets are asserted against each other**, not just against a number: a query that ignored the quality
value would return 1920×1080 for all three and pass a test that only checked Performance. Each is also required to
sit inside the dynamic range it reports.

### 8.20 The G-buffer moves to the layout DLSS reads, and depth stops being a half float

**Roughness moves into the normal target's `w`** — `Roughness_Mode_Packed`, one fewer resource to bind and one
fewer image to write. The material target carries specular albedo alone.

**Depth becomes its own target, at full precision.** It used to ride in an `rgba16f`, whose largest representable
value is **65,504** — fine when the world ended at the streaming window, and not once §8.9 pushed the horizon past
100,000 units: every pixel beyond stored infinity, and the à-trous edge test divides one distance by another. Under
the ceiling it was merely coarse: eleven bits of mantissa puts the step at about eight units by ten thousand.

The new target is `rg32f`: **clip depth in `r`** for the upscaler, **distance from the eye in `g`** for the filter.
Two different questions were being answered by one number, and a reverse-Z clip value would have made the filter's
tolerance mean something different at every distance. Measured effect: 7 pixels of 921,600 move by 2 levels.

**What the tests had to learn.** `water.rs` read specular albedo and roughness from one image, which would have kept
passing had roughness been written to the wrong target. Each is now read from where DLSS reads it.

### 8.21 The feature builds, once NGX is asked what it dislikes

- **`Use.HW.Depth` describes the depth's shape, not where it came from.** The enum is `Linear = 0`, `HW = 1`, and
  §8.20 writes clip depth — projected and reverse-Z — whoever computed it. Setting `Linear` because a compute shader
  wrote it is true and irrelevant.
- **`MVLowRes` reads as a description, not a request.** It says the motion vectors *are* at render resolution, which
  §8.13 writes them at.

Both came back as `FAIL_InvalidParameter`, which names no parameter. **What found it was NGX's own log**, off by
default because the logging level in `NVSDK_NGX_FeatureCommonInfo` was left at zero. Turned up: `Error: Low
resolution Motion Vectors required`. That message exists nowhere in the API surface — no status code carries it and
no parameter can be queried for it.

**And it cannot be had quietly**, which is why it is `RTXMW_NGX_LOG` rather than simply on: the feature libraries
write **1,018 lines to the console** on one successful run, enough to bury the assertion message of whatever failure
sent someone looking.

**Ownership, since NGX has two things to release and an order.** The parameter map a feature is built from is not the
capability map, and it has to outlive the feature. Both belong to one `Feature`, released together: the handle first,
then the map. The first attempt handed the map to a closure that `map_err` dropped whether or not the error path ran,
destroying it on *success*.

### 8.22 Ray Reconstruction runs, and what that test does not prove

1920×1080 in, 3840×2160 out, on real Vulkan images, with the validation layer silent — a separate claim from NGX
returning success, which says only that NGX liked the parameter map.

**Two names that are not the obvious ones.** RR reads its albedos from `DLSS.Input.DiffuseAlbedo` and
`DLSS.Input.SpecularAlbedo`, not the generic `GBuffer.Albedo`/`GBuffer.Specular` beside them in the same header. And
the entry point is `NVSDK_NGX_VULKAN_EvaluateFeature_C`, not the unsuffixed symbol next to it.

**What it does not prove.** Swapping the output for a 1080p image fails it, so the plumbing has teeth. Swapping
*depth* for *motion vectors* does **not** — both are `rg32f` at render resolution, and neither NGX nor the validation
layer can tell one from the other. Every input is an empty allocation, so there is no picture to check.

### 8.23 Ray Reconstruction is wired into the frame, and produces black

Frame order is trace, composite, **DLSS**, exposure, tone curve, with the à-trous filter at zero and jitter on. It
reports `1920x1080 to 3840x2160`, takes about 4 ms, **and the picture is black.** Read back directly the upscaled
image has exactly 8,294,400 non-zero channels — one per pixel, which is the alpha — while its colour input is a real
frame. Nothing errors and the validation layer is silent.

**Two real bugs found on the way**, both fixed: no barrier between the composite writing the colour and DLSS reading
it, and the tone curve dispatched over the *render* extent while writing a 4K image.

**Ruled out** (and still ruled out after §8.24 found the cause): reverse-Z clip depth with `DepthInverted` and
`Use.HW.Depth` against linear world distance with neither — black both ways; the RR-specific parameter names, checked
against the header.

### 8.24 The black frame was a missing usage flag

**DLSS samples its inputs.** Every image handed to Ray Reconstruction has to be created with
`VK_IMAGE_USAGE_SAMPLED_BIT`, and ours were `STORAGE | TRANSFER_SRC` — everything the renderer's own passes need and
one bit short of what NGX needs. An image it cannot sample reads as zero; NGX returns success, the validation layer
says nothing, and the network resolves a black field to a uniform 2⁻²³, the second-smallest half-float subnormal.

**What found it was varying an input and watching nothing happen.** Constant colours of 0.25, 1.0 and 100.0 produced
bit-identical output, which says DLSS never read that image — and the output was written through the same wrapper, so
the wrapper, struct layout, `SetVoidPointer` path and parameter name were fine by construction.

**Two things ruled out on the way, both reverted.** The `AutoExposure` creation flag and the `DLSS.Pre.Exposure` /
`DLSS.Exposure.Scale` scalars measured bit-identical; §3.7 of the RR integration guide says exposure is not supported
by Ray Reconstruction at all, and the SDK helper setting them is DLSS-SR heritage.

**The test now asserts a picture.** A constant frame is the one input whose correct output is arithmetic rather than a
reimplementation of the network, so `[0.25, 0.5, 0.75]` in has to come back as itself per channel, and does to within
0.15%. Three different values rather than one, because a grey would pass a single-channel check while proving nothing
about which channel was read.

**The upscale got its own timing stage.** It had been recorded inside the composite's window, reporting a 0.65 ms
upscale as 0.65 ms of compositing where the composite's real cost is 0.01 ms.

### 8.25 The sampler redrew the same noise every frame

A still camera produced **bit-identical frames**: the hash streams were seeded `hash(pixel, stream, sample)` with no
frame term, so the estimator's error was a fixed pattern rather than something that averages away.

**A spatial filter hides this and a temporal one cannot.** À-trous filters each frame on its own; Ray Reconstruction
accumulates across frames, reads a pattern that never changes as scene detail, and preserves it — the frame came out
covered in salt-and-pepper that 64 frames of convergence did nothing to. `DLSS-RR Integration Guide` §3.5 states the
requirement it violated: samples must have minimal correlation *temporally* as well as spatially.

`sample_stream` exclusive-ors the pixel with a word derived from a new `sequence` frame constant. XOR with a fixed
word is a bijection, so pixels still never collide within a frame.

**It costs the path that does not need it.** With only an à-trous pass, a fixed seed dithers and holds still while
reseeding leaves crawling static: consecutive filtered frames of a still camera went from bit-identical to 0.7% RMSE
apart. The trade inverts under a temporal filter, which is why it flipped — but the old rationale was sound.

### 8.26 The reference was rewarding aliasing

Measured against a native-4K 1024-sample reference, à-trous scored 32.1 dB and Ray Reconstruction 24.3 — reason to
distrust the *measurement*. **The reference was aliased**: it renders one sample per pixel spatially with jitter off,
as does the à-trous path, while RR resolves sub-pixel detail across jittered frames. On a shore full of fences, PSNR
penalised RR for edges that are *more* correct than the reference's.

Re-rendering at 7680×4320 and box-filtering to 4K — four spatial samples per output pixel, 18.7 s of device time:

| 1 spp, 3840×2160 out | vs aliased reference | vs supersampled reference | device |
|---|---|---|---|
| à-trous, 4 passes, native | 32.1 dB | **26.8 dB** | 23.5 ms |
| RR DLAA, native | 24.3 dB | **24.9 dB** | 59.0 ms |
| RR Performance, 1920×1080 in | 21.9 dB | **22.2 dB** | 10.4 ms |

The gap that mattered was under 2 dB, not 7.6, and it moved in opposite directions for the two methods — the
signature of a metric measuring the reference's own defect. **A reference has to be at least as correct as the thing
it judges, on every axis at once.**

### 8.27 M7's performance gate is met

3840×2160 output on the Seyda Neen shore, release, RTX 4090 Laptop: native 4K is 34.50 ms (29.80 trace);
1920×1080 → 4K under RR Performance is **10.76 ms** (6.39 trace, 4.18 upscale). 29 fps against 93, with about 6 ms
of the budget unspent — and this is an exterior, where §5.3's table was measured on an interior. Take it as a ratio:
repeated runs of the same frame spanned 10.4 to 21.7 ms.

**The quality gate is not met.** 22.2 dB against a supersampled reference is not "comparable to a 1024-sample
reference", and RR trails the à-trous filter by 1.9 dB at matched resolution while costing five times as much there.

### 8.28 The upscaled frame was exposed by the noisy one

§3.5's sampling requirements were not the reason. Two things it names were tested and neither moved: replacing the
sampler's hash with `pcg4d` gained **+0.01 dB**, and taking two full 32-bit words per sample instead of two 16-bit
halves gained **+0.005 dB**. Both reverted — on this content the old hash measures as sufficient.

**What it was: a 2.7% brightness bias, uniform across the frame** (sky +2.5%, middle +2.8%, ground +2.9% — a global
gain, not a region getting it wrong). The auto-exposure histogram bins `log2(luminance)` **per pixel**, and the mean
of a log sits below the log of the mean by roughly the variance, so a noisy frame measures darker and the tone curve
opens to compensate. Attaching an upscaler sets the à-trous passes to zero, so what exposure read was a single sample
per pixel.

The fix is one binding: exposure measures **the frame the tone curve is about to map**, which is the upscaled one
where there is an upscaler. DLSS already runs before exposure, so no lag is introduced.

| 1 spp, 3840×2160 out, vs §8.26's reference | before | after |
|---|---|---|
| à-trous, 4 passes, native | 26.80 dB | 26.80 dB (bit-identical) |
| RR DLAA, native | 24.89 dB | **27.49 dB** |
| RR Performance, 1920×1080 in | 22.23 dB | **23.54 dB** |

Exposure at 4K costs 0.10 ms against 0.05 at render resolution.

**No test guards this**, and that is a choice: without an upscaler the change is provably inert, so only a DLSS-gated
test on this hardware could see it. The invariant is structural — `bind_targets` binds both passes from one `source`,
and `record` dispatches both over one extent expression.

### 8.29 Ray Reconstruction reaches the window

It had been reachable only from `--screenshot`, which is to say only from a stationary camera — and a stationary
camera cannot show what a temporal upscaler gets wrong. The windowed renderer now builds the same upscaler, with the
**window's own size as the output** so the blit to the swapchain is a copy. `upscaler.rs` holds what both front ends
need, because the two bring up Vulkan separately and a second copy would be a second place for them to disagree.

**Two bugs, both only reachable from a window.** A compositor sends a resize on first map that changes nothing, so
`recreate` ran on frame one — and it built the replacement upscaler *before* releasing the old one. Each `Upscaler`
owns its own `Ngx`, and dropping one calls `Shutdown1` for the whole device, so the feature just built was orphaned.
NGX reports that as `FAIL_NotInitialized` at every evaluation and says nothing at build time; the log showed one
context initialised, **two** features created, **three** shutdowns. The order is now release, then build. The same
resize also meant a full rebuild for an unchanged size, which costs a weight upload — everything sized by the window
is now left alone when the size is what it already was, which matters during a drag.

**Not covered:** there is no explicit history reset for a camera that jumps. `reset` is derived from whether a
previous frame exists. Nothing teleports yet; a fast-travel will need one, and the symptom will be a smear.

Note for a wide display: the output follows the window, so a 7680×2160 one traces 3840×1080 — twice the pixels §5.3
budgets for.

### 8.30 The jitter handed to DLSS had the wrong sign, on both axes

The image shook about a pixel every frame, plainly visible in motion and invisible in every measurement taken until
then. A still camera turns it into a number: consecutive frames differed by **0.0335 RMSE** under RR against
**0.0073** unupscaled. A temporal accumulator less stable than the raw path with a stationary camera is doing the
opposite of its job. Sweeping the four sign combinations settles it in four renders: `+x,+y` 0.0335; `+x,-y` 0.0196;
`-x,+y` 0.0289; **`-x,-y` 0.0016**.

**The trace adds the offset to the sample coordinate**, moving where inside its pixel a ray is fired. NGX wants the
offset as applied to the *projection*, which moves the frustum the other way for the same picture. Handing over the
coordinate's sign leaves RR un-jittering in the direction that doubles the offset rather than cancelling it. Nothing
reports this: the feature builds, evaluates and returns success, and the wrong sign still resolves an image.

Quality against §8.26's reference, 1 spp: RR DLAA native 27.49 → **30.23 dB**; RR Quality (2560×1440 in) **27.74 dB**;
à-trous unchanged at 26.80.

### 8.31 The default build is the one worth looking at

`dlss` is a default feature and DLSS runs at **Quality** unless told otherwise, so a plain `cargo run` is the engine
with everything switched on. `--dlss off` turns it back off, which is what an A/B against the unupscaled path needs.
An unrecognised value is refused rather than silently rendering at a mode nobody asked for. The cost: a fresh clone
builds only once the SDK is fetched into `.refs/` or `DLSS_SDK_DIR` points at it; `--no-default-features` is the way
out.

### 8.32 One place reads settings

`RTXMW_DLSS` had been read inside `upscaler.rs`, a module with no business knowing that an environment exists. Every
setting is declared once, in `cli`, as an ordinary argument; clap covers the flag **and** the variable from that one
declaration (`env = "RTXMW_DLSS"`), and the `.env` layer sits underneath as the argument's default. Order is flag,
variable, `.env`, built-in default. Nothing outside `cli` calls `env::var`, which is a property a grep can check.

**A setting that reads the environment cannot be pinned in a test.** The parser tests flatten `dlss` away in the two
helpers they all go through, because a machine with `RTXMW_DLSS` set would otherwise fail all of them; the value
parser has its own test.

The type is `Upscaling(Option<Preset>)` rather than `Option<Preset>`: clap reads an `Option` field as "this argument
may be absent", and absent is exactly what this one must not mean.

### 8.33 The exposure residual is a property, not a defect

Output luminance at 1920×1080, DLSS off:

| à-trous passes | 0 | 2 | 4 | 8 |
|---|---|---|---|---|
| 4 spp | 0.7303 | 0.7281 | 0.7276 | 0.7272 |
| 1 spp | 0.7344 | — | 0.7277 | — |

**It scales with the noise and converges with the filtering**, which is what the log's concavity predicts. Four passes
against eight differ by 0.05%, and every shipping configuration is on that end. The gap appears only when an
unfiltered frame is asked for, and then it is the honest answer.

**One attempt is recorded as failed** so it is not retried blind: averaging luminance over a 2×2 block before binning
flips the gap to −0.77% and moves the filtered frame's exposure with it, because a block average changes which samples
fall under the histogram's black cutoff as well as how noisy they are.

### 8.34 A test for the thing no test could see

§8.30's sign error survived a feature that built, evaluated, returned success and passed the validation layer, and
every quality number taken from it — because all of them came from a *single* frame. `tests/upscaler_stability.rs`
renders ten and compares the last two with the camera held still.

**Its own test binary, and so its own process.** NGX is global per device and the SDK does not promise to survive
concurrent initialisation. DLAA rather than an upscaling preset, so what is measured is the temporal resolve alone.

**The fixture had to be real content.** A synthetic wall with nine bars each way passed with *every* sign combination
— a bar was fifty pixels wide, so a misaligned history had almost nothing to disagree about; bars two pixels wide
separated the populations by 1.5×, still too thin to assert on. Only a real cell, textured at pixel scale, works:

| `Jitter.Offset` | frame-to-frame RMS |
|---|---|
| **`-x, -y`** (correct) | **0.00090** |
| `+x, -y` | 0.00371 |
| `-x, +y` | 0.00485 |
| `+x, +y` | 0.00731 |

The bound is 0.0018 — the geometric mean of the correct value and the nearest failure — and it fails on **either**
axis inverted, not only on both.

**A machine without the game skips**, rather than falling back to the synthetic grid. The grid measured 0.0054 with
the signs *correct*, so it failed a bound calibrated on real content, and it could not have caught the fault anyway.
A check that cannot fail on the thing it exists for is worse than an honest skip. A stable black frame would also
pass, so the test asserts the frame is lit as well.

### 8.35 De-lighting, as an estimate divided out at sample time

§5.1 chose option 4. Three findings changed the shape of it, and each is worth more than the code.

**A Morrowind plugin cannot override a texture.** Only `LTEX` names a texture path in the record stream, and that
covers the ~100 land textures; every other reference is `NiSourceTexture::mFilename` inside a NIF, which no ESM
record reaches. What replacer packs actually use is the VFS — later archives win and loose files beat BSAs — so the
override belongs there, and this engine has its own VFS.

**A cache is not needed.** Estimating a shading map for the *entire* library — 4,783 textures, 148.7 MB — takes
**162 ms**, against 50 ms to decode them. A cell's 118 textures is well under a millisecond, so the maps are computed
on load and thrown away with the cell.

**Rewriting the texels was the wrong mechanism.** `rtxmw-texture` deliberately never decompresses. De-lighting the
pixels would need a decoder *and* either an encoder — trading Bethesda's compression quality for a hand-rolled one —
or storing them uncompressed at four times the VRAM. What is being removed is low-frequency by nature, so a 32×32 map
per texture holds it, and BC1 gives block averages from its endpoint pairs without decompressing anything.

**The estimate**, in `shading_map.rs`: mean luminance over a 32-square grid, three box blurs, normalised to a mean of
one and clamped to `0.5..=2.0`. Normalising is what makes it a redistribution rather than a brightness change. It is
written through the sRGB transfer, because the bindless array binds these as an sRGB format.

**It rides in the existing array, interleaved.** Vulkan allows a variable descriptor count only on a set's final
binding, so there cannot be a second bindless array; a texture id addresses a pair, `1 + 2n` for the colour and
`2 + 2n` for its map. Terrain gets the correction for free, because it blends four texture *ids*.

**The default is on**, and `--delight 0` is the A/B. At full strength on the census office the frame moves by 0.020
RMS, which is the scale a normalised, clamped, heavily blurred estimate should move things.

**Three bugs, all found by review or by an existing test rather than by looking at the picture.**

- `cone_lod` measures texel density with `textureSize` on an array *slot*, and all three call sites still passed the
  pre-interleave index — so every mip selection in the renderer read a 32×32 shading map's dimensions. It costs
  nothing visible on a still frame and everything on a grazing one.
- A texture smaller than the grid lands in a handful of cells and leaves the rest empty; reading those as black made
  the estimate a spike on a floor and drove the correction to the clamps. Empty cells now take the average of the
  sampled ones, so too few texels to resolve shading is the same as having none. The terrain test caught this,
  because its fixture is one texel per tile deliberately.
- A material whose texture never loaded got no map, leaving the array's magenta stand-in — whose red channel decodes
  to the top of the range, so every untextured surface was divided by two. Missing textures now get a neutral map.

Four tests carry the algorithm, and the third is the one that matters: a flat texture must come back neutral to
within 0.01, a painted 3:1 ramp must be recovered (0.575 to 1.431, mean 1.000), **a checkerboard must be left alone**
— detail alternating every texel is not lighting, and following it is the over-correction that flattens a texture —
and a BC1 block's mean must match the hand-computed average of its palette.

### 8.36 Emissive was being added past the albedo

Mushroom caps rendered as flat white discs. `flora_bc_mushroom_01.nif`'s cap material carries
`emissive: [0.5, 0.5, 0.5]` where the stalk's carries zero, and this renderer wrote emissive to the frame *beside*
the albedo rather than through it. A term added past the texture cannot show the texture.

**The original engine treats emissive as light, not as colour** — `objects.frag:232` sums it with the diffuse and
ambient terms and `:236` multiplies the whole by the diffuse texture. `emitted` is now zero for a hit and emissive
joins the lighting the surface receives, **scaled** because the two renderers do not agree on what one means: there a
fully lit surface reached one, here direct sun is `DAYLIGHT = 8`. Carried across at the sky's scale rather than the
sun's — what these materials are for is being visible in shade.

**De-lighting was the obvious suspect and was not the cause**: across the mushroom textures the brightest cell goes
*down* under correction (0.37 → 0.33, 0.69 → 0.53, 0.65 → 0.58) and no cell is pushed past white.

**A wrong first attempt.** Deleting `emitted = surface.emissive` and adding the term to the lighting washed the frame
out to a blue haze: `emitted` is initialised to `sky(direction)` and that line was what *overwrote* it for a hit, so
removing it left the sky added on top of every surface. The tell was the scale — with six of 181 materials emissive,
nothing about emissive could touch every pixel, and the difference map was every pixel. Zeroing `emitted` explicitly
is the other half; with it the frame moves by 0.007 RMS with mean luminance unchanged to six figures.

### 8.37 A contact sheet, so de-lighting can be judged rather than argued about

An estimate that removes painted detail and one that removes painted light move the numbers the same way, so this is
not something a metric sees. `--textures <PATH> [CELL]` writes every texture a cell uses, vanilla beside de-lit.

**It needed a decompressor, which this crate had deliberately never had.** `Texture::to_rgba8` is that, and the
module note now says which of the two it is rather than claiming the crate never decompresses at all. Its tests are
hand-computed against BC1's palettes, both the four-colour one and the three-colour mode whose fourth entry is
transparent.

**The correction on the sheet is the shader's arithmetic**, at the same strength and against the same map.
Thumbnails are nearest-neighbour on purpose: a smoothed one would hide exactly the loss of detail this exists to find.

### 8.38 Fog that lies low, drifts, and is lit by what stands in it

Twenty-four steps along the primary ray, density falling off exponentially with height and modulated by drifting
value noise, with every lamp the light grid offers scattering into each step.

**Marched rather than integrated, deliberately.** Exponential height fog has a closed form, but only while density is
uniform across the horizontal plane, and fog that cannot move was the one thing this was not to be.

**It needed no new binding.** Fog attenuates the whole frame, and the trace has only the two halves the composite adds
— but

    (emitted + albedo · lighting) · T + inscatter  ==  (emitted · T + inscatter) + albedo · (lighting · T)

so folding it into both halves in the trace is the same as fogging their sum, and the trace is where the lights
already are. Binding the light grid, the light buffer and a matrix to the composite would have duplicated the trace's
whole view of the world.

**Three things the data said that guesswork would not have.** Exteriors carry *no fog at all* — only interiors have an
`AMBI` record, and an exterior's fog belongs to the weather system (§8.59). The same recorded density means different
things indoors and out: the original set fog by a start and end distance *scaled to the view range*, so 0.75 filled a
room and 0.75 filled a valley; this renderer's extinction is absolute, so `INDOOR_FOG_SCALE` makes the conversion
explicit. And height fog alone only fills hollows — `FOG_FLOOR` is the fraction that does not fall off, which turns
"pools in the valleys" into "the far hillside is paler". (§8.39 removed it again once the layer was measured from the
water.)

**Cost**, minimum of six traces at 960×540: 3.09 ms without, 3.14 with. At 1920×1080 the difference is inside the
run-to-run spread.

Three test files turn fog off, each for a reason written beside them: `output.rs` because an unlit surface with fog on
it is a lit one, `primary_visibility.rs` and `terrain.rs` because their assertions are hand-computed radiance and blend
weights — fog also varies with world position, which is what caught the scene-far-from-the-origin test.

Not done: shafts. Shadowing the fog needs a ray per light per step (§8.43 did it for the sun alone).

### 8.39 Fog that gathers over water, and stops looking like a texture

**It pools where water is.** Density falls off from the cell's own water level and `FOG_FLOOR` is gone, so a hill
stands clear of the layer. A dry cell has no water level and the shader is handed negative infinity, so it falls back
to the origin rather than putting the fog infinitely far down.

**Three octaves, each on its own heading at its own speed.** The differing speeds are the point: a single field
scrolling is a texture sliding past, where octaves shearing against each other make the shapes themselves form and
pull apart. The third carries a little vertical drift. The frequency step is 2.27 rather than 2 so the lattices never
line up.

**And a coverage band, which is what makes it patchy rather than merely uneven.** Scaling density by a noise gives fog
that is everywhere and varies; mapping a band of the noise onto clear-to-thick gives banks with gaps between them.

**The band has to sit inside the noise's own range, and getting that wrong made the fog vanish.** Averaging octaves
narrows the distribution — three land mostly within a quarter either side of a half — so `smoothstep(0.42, 1.0, fbm)`
squared left average coverage near a third of a percent. At `0.30..0.70` unsquared it is fog again. **Do the arithmetic
on a threshold rather than picking one that looks reasonable for a single octave's spread.**

The larger structures needed a thicker layer and more extinction to read at all, because a band that is clear half the
time needs the other half to count for twice as much.

### 8.40 Higher, and warped so it stops looking like a gradient

The layer's scale height goes 520 → 2600, so fog fills the air rather than hugging the shore. **And the domain is
warped** — Quilez's `fbm(p + w·fbm(p))`, at one level with a single-octave warp rather than the full construction's
fbm-per-component-per-level: two extra noise samples instead of six. Horizontal only, because the vertical shape of
this fog is the height falloff.

**The thing that made the difference was making the noise *coarser*, which is the opposite of the instinct.**
Twenty-four steps over a ray that can run thirty thousand units puts more than a thousand between samples, so
structure finer than that aliases into noise the jittered start and the temporal filter take back out. What reads as
a bank of fog is the *coarsest* octave, so its cells went 1,400 → 3,000 units and the warp to 1,500, and the coverage
band narrowed to `0.40..0.56`.

### 8.41 Structure you can see from the ground, which took two wrong diagnoses

§8.40's coarser noise gave fog whose shape was visible only from a ridge, because from the ground everything is near.

**Wrong diagnosis one: uniform steps.** A ray running thirty thousand units gives the first hundred a twentieth of one
sample. `fog_depth` now squares the parameter so the first step spans about fifty units and the last a couple of
thousand — the same reasoning that makes a froxel grid slice its frustum exponentially. That let the grain come back
down from 3,000 to 900 units.

**Wrong diagnosis two, and the one that mattered: it was never a sampling problem.** Sampling finely where the fog is
*thin* buys nothing. What the eye reads over a distant hillside is thousands of units of integration, and structure at
any scale averages out of it. **Fog has visible shape when it is optically thick over a short distance**, so a bank
reads white while the air beside it is clear.

So the fix was **sparse and dense rather than uniform and thin**: the coverage band moved to `0.44..0.66`, which clears
more of the volume, and extinction doubled to 5e-4 to pay for it. Overshooting is instructive too: at `0.55..0.65` with
2e-3 the distance becomes a white wall with hard-edged dark blobs punched through it, which is a narrow band's edges
meeting a step count that cannot resolve them.

**Still not taken**: Perlin-Worley for the base shape with high-frequency Worley eroding the silhouette, as Horizon
Zero Dawn's clouds do. Worth revisiting now that the near field is sampled finely, since the reason for skipping it was
a sampling rate that no longer applies close to the camera.

A fourth test file turns fog off: the sun's penumbra width, because fog scatters light into a shadow.

### 8.42 A room is not a valley, and the census office was a steam room

§8.41 doubled the outdoor extinction to 5e-4. `INDOOR_FOG_SCALE` had been chosen against 1.2e-4 and was not touched, so
every interior silently became four times as thick as the value anyone had looked at.

**Two things were wrong, and only one was the number.** The other was that interiors ran the outdoor coverage field —
banks are a landscape feature, and inside a room they read as a rendering fault. The field is now selected rather than
assumed, on `fog_density > 0` in the cell's own record, which only interiors carry:

```glsl
float banks = smoothstep(FOG_CLEARING, FOG_SOLID, fog_fbm(position));
float coverage = mix(banks, FOG_EVEN, frame.fog_uniform);
```

`FOG_EVEN` is 0.5, the middle of what the banked field spans, so switching between them changes the shape without
changing how much air there is. `INDOOR_FOG_SCALE` went 0.035 → 0.006 — more than the factor of four the extinction
accounts for, the rest by eye, because a room reads better holding a veil than holding air.

The fog test caught the retune: its fixture is an interior, so both changes landed on it at once. Raising the fixture's
recorded density to 140 — far above anything the game ships — is the honest fix, because what it tests is the integral
over distance and an interior no longer buys enough of one at any density a cell actually has.

### 8.43 The sun in the fog

Fog was lit by the sky and by lamps and not at all by the sun, which is five times the sky's radiance. Facing the sun
rendered *identically* to facing away.

**The phase function is free, and that decides which one to use.** The sun is directional, so the angle between the
view ray and it is the same at every point along a ray: one evaluation for a whole march. That takes the usual
argument for a single Henyey-Greenstein lobe off the table, so this uses Jendersie and d'Eon's HG-Draine blend fitted
to tabulated Mie, at a droplet diameter of eight micrometres. Two lobes and four `exp`s, and the difference between a
lobe peaking at 45 times isotropic and real fog, which peaks at 4,337 and still sends a sixth of isotropic straight
backwards. Both halves are what fog looks like: the blaze around a low sun, and fog not going black when you turn
around.

**Wrong once, by exactly `4*pi`.** The first version normalised the phase so isotropic came out at 1.0, matching the
lamps' convention. That is right for a source integrated over the sphere and wrong for a delta: the sky arrives from
everywhere and a phase function integrates to one, while the sun arrives from one direction as *irradiance* and what
comes back is that irradiance times the phase function **per steradian**.

**Wrong twice, by leaving out the light's own path.** Single scattering that lights every point in a bank as though it
were the first the sun touched is several times too bright. The density falls off exponentially with height, so the
column from a point to the sky along the line to the sun is closed form — `sigma * H / cos(zenith)` — and costs one
`exp` on an extinction the march already computed. That is what puts a gradient down through a bank and what makes a
sun on the horizon correctly light nothing.

**Shadows: eight rays, and the count came from measuring.** A shadow ray costs about four march steps, so one per step
would cost more than four times what the whole fog costs. The march is cut into eight stretches with one ray apiece,
drawn from a jittered point along its stretch and aimed at a point on the sun's disc — softening the edge is what
keeps a binary visibility test from aliasing against a march that samples it far more coarsely. Against a 24-ray
reference, RMSE falls 0.0155 / 0.0134 / 0.0087 / 0.0048 for 1 / 2 / 4 / 8.

Cost at 1920×1080, whole frame, best of five: no fog 5.33 ms; fog at 2 rays 6.69; at 4 7.35; **at 8 7.53**; at 24
9.69. Eight rays cost 0.18 ms over four — they are perfectly coherent, and a gate skips them wherever the sun's phase
falls below a fiftieth of the sky's term, which is everything but the sunward part of the frame and all of every
interior.

**What the industry does, and why this does not.** Nothing shipped ray-marches volumetric transmittance per pixel:
Wronski's froxel grid is universal, at about 240×135×64 at 1080p, and where ray tracing enters it replaces the shadow
*map* at roughly one ray per froxel (NVIDIA's default in RTX Remix works out at 0.75 rays per output pixel), always to
decouple cost from light count rather than for integration quality. Eight rays per pixel at full resolution is over ten
times that density; it is affordable because there is one sun rather than nine hundred lights, and worth it because a
froxel grid at an eighth of screen resolution is where a shaft through a tree loses its edges. If this ever needs to
survive a real light count, the froxel grid is the fallback.

Ray Reconstruction sees the fog composited into the colour, which is also what NVIDIA ships —
`rtx.rayreconstruction.compositeVolumetricLight` defaults to true in Remix and the separate-layer path is documented
there as flicker-prone.

The test is a ratio rather than a value, because it is the only thing the fixture computes exactly: two frames differ
in nothing but which side of the camera the sun is on, so what is left is `p(16.7°) / p(163.3°)`. The readback patch
spans nine to twenty-four degrees off the sun, so the answer must land between 21.3 and 59.5; it measures 31.97. An
isotropic fog gives exactly 1.0. The second test drops a lid over the march that the camera's own ray never reaches and
asserts the air beneath it scatters *exactly* what air with no sun scatters — not merely less, which would pass while
leaking.

### 8.44 Lamps: the factor they owed the air, and the reach the records never gave them

**The `4*pi` the lamps owed.** Like the sun, a lamp arrives at a point as *irradiance*, so even an isotropic fog owes
it a factor of `1/4*pi`. Without it every lamp lit the air twelve and a half times more strongly than a sky of the same
radiance, which is why interiors read as though the air itself were glowing.

They stay isotropic, unlike the sun, for two reasons: a lamp's angle to the view ray changes at every step and for
every lamp, where the sun's is fixed for a whole march; and the real phase function's forward peak is 4,300 times
isotropic, which for a source at a *finite* distance is a firefly waiting for a march step to land on the line from the
eye through it.

**Morrowind's radii are tiny, and reach is now separate from brightness.** Seyda Neen's street lanterns record 256
units and the census office runs 64 to 256 — 0.9 to 3.7 metres, so a lantern lit its own post. One number was doing
three jobs. Intensity still comes from the recorded radius, because a lamp's brightness is what the lamp *is*, as does
the emitter size that gives its shadows a penumbra. Only the falloff's reach is stretched:

    reach = radius * 2.0 + 128.0

so a lantern is exactly as bright at arm's length as before and its light runs out to nine metres instead of three and
a half. The flat term is what saves the small lights: doubling 64 units is still nothing, and it is the candles that
most need to leave their own table. The light grid bins on the same number.

At 1920×1080, best of five: interior 5.01 → 7.35 ms, exterior 7.26 → 7.75. The interior pays 2.34 ms mostly in shadow
rays — more lights reach a point, each gets eight. It turns a pool-lit room into a lit one, the largest single change
to how an interior reads since de-lighting.

**And the fog is thinner.** `OUTDOOR_FOG_DENSITY` 0.75 → 0.30. Half the light off a surface survived fifty-three metres
at the old figure; it now survives a hundred and thirty. The bank structure is untouched, since the coverage field says
where the fog *is* and the density only how thick it is there.

**A test rotted quietly**, which is the failure mode that does not announce itself. `a_lamp_in_the_fog_lights_it`
proved a lamp lights *air* by standing it far enough from a grey wall that its recorded radius could not touch it.
Stretching reach moved the lamp past the wall and the test went on passing — on light bouncing off the wall, the one
thing it was written to exclude. Its wall is now black.

### 8.45 The sun moves, and Morrowind's own constant would not let it

Everything from §8.38 to §8.44 had only ever been seen at one hour.

**Morrowind has no sunrise.** The game's direction is `(-400 · orbit, 75, -100)`, and the third component is a
constant: the sun swings east to west without ever descending, standing fourteen degrees above the horizon at the
moment the clock says it rises. The vertical is now scaled — by `sqrt(1 - orbit²)` here, replaced by a cosine in §8.58
— which leaves noon exactly where the constant put it (100 of 125, 53 degrees) and lets both ends of the day reach the
horizon they are named for. **A deliberate departure from the game data, and the only one here.**

**The atmosphere does the colour, so nothing is a tint chosen by hand.** A beam to a sun `h` above the horizon crosses
Kasten and Young's air mass — 1.25 atmospheres at noon, 38 at the horizon, and finite there, which `1/sin h` is not —
and loses `exp(-tau * mass)` per channel against the Rayleigh depths `(0.046, 0.108, 0.265)`. That expression is the
whole of why a low sun is orange. The sky was the same arithmetic read the other way, `(1 - T) * T` — see §8.48, which
had to replace it. Two constants are tuned rather than derived: a greying term standing in for the multiple scattering
a single-scattering model has none of, and a scale set so the default hour lands on the blue this replaced.

**The sky belongs to the renderer, not to the cell.** It used to be two literals in `StaticScene` written at load
time, which is wrong by construction: an hour later the same geometry is lit by a different sun. An exterior now
records *no* lighting at all, and `SceneResidency` keeps whatever record the resident cell had so `relight()` can
re-derive whenever the cell or the sky moves. `SceneRenderer::set_sky` is the whole of the per-frame path.

Three things are pinned by rendering rather than by argument: an interior is **bit-identical** at 03:00 and 13:00 and
bit-identical to before any of this existed, and `--time 9:30` is bit-identical to no argument at all.

### 8.46 A clock to look at the world with

**One clock, because both are the same clock.** The fog's drift is a distance the wind has carried it and the sun's
height is an hour of the day; a `WorldClock` holds the seconds and the hour and advances both by the same real delta
times one speed. At speed 1 it runs at Morrowind's own `timescale` of 30 — right to play at, useless to look at. The
step is ×4 and the ceiling 256: at 64 the fog reads, banks forming and pulling apart at about one a second, and at 256
a whole day is under seven seconds.

**The timescale is a factor of thirty before the speed touches it**, and forgetting that is how the ceiling first came
out at 4096, at which the entire day passes in four tenths of a second.

**And nudges, which are the useful half.** `,` and `.` move the hour half an hour without moving the clock, so they
work while paused. Reaching dusk by speeding time up drags the fog through an hour of wind on the way, and two frames
that differ in the fog as well as the light compare nothing.

**A clock is not a velocity.** A camera that missed a second simply did not move; a clock that missed one has to decide
how much time passed, and a cell load stalls the better part of a second — unclamped at top speed that is two and a
half game hours for a frame in which nothing happened. `advance` carries at most a twentieth of a second.

The keys act on a key's *repeat* as well as its press, so holding one scrubs — except the pause, which a repeat would
flicker thirty times a second. `WorldClock` writes itself into the caller's formatter rather than handing back strings,
because the title is built twice a second on the frame path.

### 8.47 The darkest frame of the day was the one the sun rose in

At 05:59 everything went black — **eighty-two times darker than midnight** — while 05:48 and 06:01 looked right, and
06:00 exactly was mathematically *zero*. Three faults, one root.

**The sun never went below the horizon.** `arc = sqrt(1 - orbit²)` was clamped with `max(0.0)`, so past sunrise the sun
parked *on* the horizon and stayed there all night, and every statement the renderer made about night came from an
invented `dusk` parameter counting how far the orbit had run past ±1. A circle's continuation is a hyperbola — so the
radicand's sign flips and the sun goes under, matching in value and slope, both running to infinity there, which is
exactly why a sun sets quickly and then slows.

**The sky's daylight term was scaled by the sine of the sun's elevation, which is zero at the horizon.** A sunset sky
is the brightest sky there is. The sine asks where the sun is relative to the *observer*, and what lights a sky is
sunlit air — the air above an observer is still in sunlight after the sun has left their horizon, which is what
twilight is. It is now a horizon value of 0.2 rising as the square root of the climb by day, times `exp(climb / sin 6°)`
below the horizon: six degrees is civil twilight, so the glow is a quarter gone by nautical at twelve and into the
rounding by astronomical at eighteen.

**And the night floor was crossfaded with that term rather than added to it.** `lerp(daylight, NIGHT_SKY, dusk)`
between a daylight of *zero* and the floor produces everything below the floor. Starlight is another source, not a
replacement for the sun, and a sum of two positive quantities cannot come out under either. It is `+ NIGHT_SKY` now and
`dusk` is gone entirely.

| | 05:48 | 05:59 | 06:01 | midnight |
|---|---|---|---|---|
| before | 0.0020 | **0.00017** | 0.046 | 0.014 |
| after | 0.036 | 0.047 | 0.120 | 0.019 |

**Two tests, because nothing would have caught this.** One walks every minute of the day and asserts the sky is never
darker than its own night floor. The other asserts it never *dims* from midnight to mid-morning — a curve that merely
avoids zero could still have had the three-hundredfold dip. That second immediately found something real and not a
fault: past about 08:40 the luminance eases off toward noon, because `(1 - T) · T` is largest at middling optical depth
and a noon sky is deep blue where blue carries little luminance. The test stops at 08:00.

### 8.48 A sky that went green, a night that went pink, and a renderer with no night in it

**Green evenings: `(1 - T) * T` is the wrong shape for a sky.** Each channel peaks where its *own* transmittance is a
half, and green's half lands at 6.4 air masses — an elevation of nine degrees — so every evening travelled blue, then
**green**, then red.

The fix is to stop computing a magnitude per channel and compute a *direction* between two spectra. One end is what the
air scatters, which is Rayleigh and always blue; the other is what survived the air, which reddens as the sun descends.
How far along is how much the air took, `1 - mean(T)`. **Two endpoints and a mix cannot produce green, because the line
from blue to red runs through grey.** A test walks every minute of the day and asserts the green channel never leads.

**Pink midnight: the hyperbola was too shallow.** §8.47's `-sqrt(orbit² - 1)` is the analytic continuation, matching in
value and slope, and it descends infinitely fast at the boundary and then flattens, reaching only twelve degrees under
at midnight — a seventh of the sunset glow, deep red, burning all night. A real sun's hour angle turns at a constant
rate, so it descends at a steady 70° per unit of orbit, about ten an hour. The greying also rises with air mass now,
which is the part that is not a fudge: more air is more bouncing, so less colour survives.

**And night was bright because the renderer normalised every frame onto the key.** Measured means at 1280×720: midnight
0.716, an interior 0.631, noon 0.727 — within two percent of each other, so the engine had no night in it at any hour.
Clamping the exposure ceiling is not the answer: at a ceiling of 6 night only fell to 0.567 while the interior fell just
as far.

Adaptation is compressive, not complete — a room at dusk goes on looking dimmer than the same room at noon however long
you sit in it. So `exposure = (KEY / mean)^0.75`, which makes rendered luminance `KEY^a · mean^(1-a)`: a frame already
on the key is untouched, and a scene fifty times darker comes out two and a half times darker rather than identical.
Night measured 0.626 against noon's 0.739.

Two tests changed rather than broke, and their names were the tell.
`a_flat_frame_lands_on_middle_grey_whatever_its_radiance` asserted the exact property being removed; it is now
`exposure_carries_a_frame_toward_the_key_without_flattening_it`, and because this fixture's mean luminance *is* its
albedo every byte in it is arithmetic — 0.02 → 72, 0.18 → 103, 2.0 → 148.

### 8.49 A sky with a direction, and the bounce that finally sees it

**The shape.** `Sky` already derived the two spectra a sky interpolates between and then averaged them into a single
ambient. `Sky::shape` now gives a direction's radiance from three things a view direction knows:

- **Paleness**, `1 - exp(-(m_view - 1) / 6)`. The horizon is thirty-eight atmospheres, and light that has scattered
  that many times has forgotten what wavelength it started as — so the tint runs from Rayleigh blue overhead to white
  at the horizon, and the zenith is the only part of the dome that keeps its colour at every hour.
- **Warmth**, the same mix toward the transmitted spectrum, weighted by how sunward the direction is.
- **Brightness**, the Rayleigh phase function normalised to average one, times the paleness again (thicker air is more
  lit air), times a term that dims the anti-solar side once the sun is low — the Earth's own shadow said as a fraction
  rather than modelled.

**Two mixes, never a product.** That is the third time this file has had to learn it: multiplying a rising spectrum by
a falling one peaks in the middle and the middle is green.

**The ambient is derived from the shape rather than beside it.** `Sky::ambient` is the dome's average over 256
Fibonacci-sphere directions — the cheapest quadrature that is even everywhere and has no seam at a pole. The light a
surface is given and the sky behind it are therefore the same sky by construction, where before they were two
expressions that happened to agree.

**And the bounce finally sees it, which is the half that makes it matter.** `gather_indirect` terminated escaped rays
with the flat ambient, so a directional sky would have been *drawn* and invisible to lighting. Escaped rays now take
`sky(towards)`, one evaluation of a function the frame already runs on every pixel that sees sky.

The equation is written twice — Rust so the host can average it, GLSL so a pixel can have its own direction — and
nothing but a test stops those drifting. `tests/sky_dome.rs` renders a frame of nothing but sky at three hours and
checks **every unclipped pixel** against `Sky::shape`; they agree to under one part in 255. A second test asserts what
the cross-check cannot: that the sunward horizon is warm, the opposite one cool and dimmer, and the zenith blue — a
pair of implementations that were both flat would pass the first and fail this.

**Still not modelled:** the anti-solar dark band and the Belt of Venus are a geometric shadow rather than a path
length, and `AWAY_SHARE` stands in for both with one number.

### 8.50 Morrowind's day is twelve hours long, not fourteen

`TimeOfDay`'s sunset was 20:00, documented as coming from the original engine's ini fallbacks. `Morrowind.ini` ships
`Sunrise Time=6` and **`Sunset Time=18`**, and OpenMW's fallback list agrees — the two-hour-longer day was invented to
go with a citation that was half right, **which is the worst kind of wrong: sourced, and not.**

The default hour moved 9:30 → 9:00, which is the hour whose `orbit()` is 0.5 on a twelve-hour day, so every image made
against it still holds. Everything downstream that had counted the old day was carrying its arithmetic: `NIGHT_DESCENT`
is 70° per unit of orbit, described as "about ten an hour over the seven hours an orbit unit is worth" — an orbit unit
is six hours now, so it is twelve an hour and the sun reaches seventy degrees under at midnight; `WorldClock`'s speed
table is a table of how long a day takes and every row was for a fourteen-hour one; and eight test hours meant "noon"
or "just before sunset" by number rather than by name.

**And one of those tests turned out to be weaker than it read.** `sky_dome.rs` asserted the horizon opposite a low sun
is *cool*; it passed on a difference of two parts in ten thousand. The model says something else and says it clearly:
at the horizon the paleness has saturated, so that direction is **neutral**, and the blue lives thirty degrees up where
it is three times the red.

### 8.51 Night as a thing the world knows, not a thing the picture measures

**Measured first.** The scene really is dark: mean linear luminance 0.189 at noon against 0.00352 at 23:00, a ratio of
53. So the lighting is fine and something downstream is giving it back. Sweeping the adaptation exponent 0.75 → 0.42 —
a 5.5-fold change in exposure — moved the night sky by 1.25. Clamping the ceiling to 1.5, nearly no lift at all, moved
it to 0.52 and took the interior down with it. **Every lever that scales the whole frame is blunt**, because
auto-exposure has no absolute anchor.

Lowering the night floor is worse than blunt. The sky and the ground are the same number here by construction (§8.49),
so it takes the ground with it, and the ground has an albedo besides. It lands where Khronos PBR Neutral's shadow offset
crushes — §8.52.

**What the literature says.** Narkowicz states the principle: *"we want to have a darker image in low light conditions
and a brighter image in high light conditions."* Krawczyk, Myszkowski and Seidel fitted a key value to the scene's own
luminance, `1.03 - 2 / (2 + log10(L + 1))`, which runs about **four and a half stops** from sunlight to starlight.
Unreal exposes it as an `AutoExposureBiasCurve` keyed to average scene EV100, Unity HDRP as a `curveMap` with per-EV min
and max curves, CryEngine as EV Min / EV Max / EV Auto Compensation **animated over the 24-hour Time-of-Day curve**, and
Infamous: Second Son shipped a manual exposure offset per time of day. The clamp windows are worth noting: HDRP ships
−1…14 EV100, Source a two-stop window, Godot three stops of ISO — Unreal's default −10…20 is the outlier, and is why an
unconfigured Unreal night looks washed.

**So the bias is keyed to the sky, not to the frame.** `Sky` knows the hour absolutely, which the histogram can only
guess at, so it emits the multiplier directly. The shape is Krawczyk's — a soft S in log luminance, saturating at both
ends rather than clamping — fitted between this dome's own darkest and brightest and normalised so noon is untouched.

**1.75 stops rather than the literature's 4.5** (2 after §8.52), and the reason is worth writing down: this renderer has
no absolute luminance scale to hang the published curve on. Noon to midnight here is fifty to one where the world's is a
hundred million to one. At 2.5 stops the night sky read beautifully and the near ground went black.

**And that trade is exactly the one the literature says cannot be won with exposure.** Ghost of Tsushima names both
halves: *"Make night feel like night, and not just darker day. Increase visibility in dark areas."* Their answer is a
Purkinje shift — rods taking over below about 3 cd/m², which desaturates, shifts blue, and **raises** the apparent
brightness of dark areas — in four instructions and two precomputed matrices. It is a different lever from exposure and
moves both things at once. The matrices come from spectral integration the talk does not print, and real units (§5.1)
are what would let the published curve be used as published.

### 8.52 The tone curve was squaring its own shadows

**Khronos PBR Neutral subtracts `x - 6.25x²` from the darkest channel below 0.08.** For that channel the output is
therefore exactly `6.25x²` — a log-log slope of **two**, contrast doubled through the whole bottom of the range. A linear
0.01 keeps 6% of itself. And because the same amount is taken from all three channels while only the smallest is
squared, a night colour comes out with its blue-to-red ratio inflated about fivefold on the ground.

**What the offset is for does not apply here.** Khronos put it in so a glTF `baseColor` reproduces exactly under even
white lighting: a dielectric of IOR 1.5 adds a 4% Fresnel floor which desaturates the render against the authored albedo,
and shifting the curve down by 0.04 cancels it. It is a colour-management guarantee for an asset viewer; this renderer is
judged against Morrowind screenshots. It also quietly cost the property the curve was chosen for — M8 said it was the
identity below `START`, and middle grey went in at 0.18 and came out at 0.14.

**Removing it outright was worse, and that is the useful half.** The night's ground came back and the *day* went flat —
the offset had been doing real shadow-contrast work. What both wants is the offset ramped to zero rather than squared
into it:

    colour -= SHADOW_OFFSET * clamp(darkest / (3 * SHADOW_OFFSET), 0, 1)

The amount taken falls away with the colour itself, so black stays black and nothing is multiplied by its own smallness.
Above `3 × offset` it saturates and the curve is bit-for-bit the reference one, which is why every pinned midtone held —
`FLAT_GREY` is still 103.

| darkest channel | reference keeps | ramped keeps |
|---|---|---|
| 0.005 | 3.1% | 66.7% |
| 0.01 | 6.3% | 66.7% |
| 0.05 | 31.2% | 66.7% |
| 0.18 | 77.8% | 77.8% |

With the crush gone the hour's bias reaches **2 stops**. Two and a half was tried and judged too dark by eye — before
the fix it was not reachable at all.

**What the research says is still missing**, in the order it ranked them: absolute units, without which nothing keyed to
cd/m² can be applied (§5.1); then the bias as a retarget curve keyed on true EV100 rather than fitted to this dome's
range, which is the mechanism Assassin's Creed Unity → Watch Dogs 2 → Far Cry 5 arrived at; then metering illuminance
rather than post-albedo luminance, so a snow-covered corridor and a dark one do not meter alike; then the moons, where a
full moon raises the sky five to eight times but the ground sixty to four hundred, a four-stop swing that is the only
thing which inverts sky-brighter-than-ground; then the Purkinje shift, per pixel and after the curve — Far Cry 5 declined
it precisely because they had pushed the moon off physical, and Ghost of Tsushima ran it because they had not, which is a
fork to take deliberately.

### 8.53 Two moons, lit by the same sun as everything else

**Everything about them is read out of the game.** `[Moons]` gives `Masser Size=94` and `Secunda Size=40`, and
each moon's sky mesh says how far away that radius sits: `sky_moon_large.nif` at 606.84, `sky_moon_small.nif` at
503.58. So `atan(94/606.84)` is a disc **17.6 degrees across** — thirty-five times the real moon, and the sky
Morrowind is remembered for — against Secunda's 9.08.

**This shipped at a third of that and was wrong**, and how the mistake was made is the point: `sky_night_01.nif`
puts the *star dome* at 2000, that looked like the one radius the sky is drawn on, and both `Size`s were read
against it. Two things say otherwise and both were there to be checked. Secunda's mesh is authored at radius
**39.0625** and its `Size` is **40** — the ini number *is* the mesh's own radius. And the ratio falls out right:
sizes 2.35 apart, angles at their own distances **1.94**, and the two portraits authored 512 pixels against 256.
**A ratio that matches the art is worth more than a radius that matched a different mesh.**

The faces are `tx_masser_full.dds` and `tx_secunda_full.dds`, mean opaque texel (0.0332, 0.0099, 0.0123) and
(0.0440, 0.0373, 0.0295) linear — one red, one grey, the red one two and a half times darker, kept rather than
normalised away.

**The phase is geometry.** The game ships eight painted phases per moon; this draws the `full` face only and
carves the terminator by reconstructing the sphere's own normal at each pixel and asking whether the sun reaches
it. One `sqrt` and a dot product, less code than a phase selector, and it cannot disagree with the sky it is in:
a crescent points at the sun because there is no other direction available to it. It also moves continuously.

**Where the moons are follows from the same commitment.** A full moon is opposite the sun, so a moon's place in
the sky and its phase are one fact: the moon rides a great circle whose pole is read off `Sun::at(0.0)`
(Morrowind's noon `(0, 75, -100)` is a 3-4-5 triangle, so its 53.13° gives a pole at 36.87), delayed round that
circle by however far through its cycle it is. `Daily Increment` sets the cycle: 1 for Masser is an eighth a day,
an eight-day month; 1.2 for Secunda six and two thirds. **A full moon therefore rises at sunset without anything
being told to do that**, and a new one is up all day and invisible; `sky_dome.rs` asserts that over forty days.

`Axis Offset` is the one number whose meaning had to be chosen. The ini gives 35 and 50 and does not say around
which axis, and the obvious reading — tilt the orbital pole away from the celestial one — does not survive the
arithmetic: a pole 35 degrees higher culminates 35 degrees *lower*, leaving Masser crawling at eighteen degrees.
Swinging the pole around the zenith keeps both moons as high as the sun gets and moves where they rise; taking
the two in opposite directions is what makes their arcs cross.

**McEwen's lunar-Lambert for the disc, and Allen's measured law for the light.** A Lambertian sphere is brightest
in the middle and falls to its limb; the real moon reads as a flat disc, because a rough dusty surface scatters
back the way the light came. Lommel-Seeliger's `mu0 / (mu0 + mu)` is that in one divide — but alone it puts the
sunward limb at exactly **twice** the disc's middle at every phase but full, because its emission cosine goes to
zero at the limb while the incidence cosine does not. Blending toward a Lambertian term, whose cosine *does*
vanish there, is the standard correction: `1 - 0.019a + 0.000242a² - 1.46e-6 a³`. The *total* light is a different
question: Lommel-Seeliger integrated says a half moon is 0.38 of a full one and the lit fraction says 0.5, where
photometry says **0.09**. Allen's fit `dm = 0.026|a| + 4e-9 a⁴` gives that, and its quartic term is the opposition
surge — why the nights either side of full are so much darker than full itself.

| | radiance of the lit face | irradiance delivered |
|---|---|---|
| real full moon against the sun | 1 / 640,000 | 1 / 400,000 |
| here | 1 / 44 | 1 / 16 |

Neither is physical and there is no scale on which they could be (§5.1). What *is* pinned is the radiance, by
something other than taste: a moon bright enough to blow all three channels is a white disc whatever colour it was
given, which throws away the only reason to draw Masser rather than a bright dot. 0.18 lands its red channel at the
top of the range with its blue a fifth of that. The irradiance is not free either — the two moons' share is
`(size / Masser's size)²`, so Secunda delivers a fifth of Masser's area's worth rather than a second number to keep
in step.

**Cost.** At 1920×1080, best of four: a night trace of **5.88 ms** became 10.47 at sixteen shadow rays a moon and
**9.12 at eight**, which is what shipped. Eight is enough where sixteen is not for the sun because the questions
are different sizes — Masser subtends thirty-five times the sun's angle, so its penumbra is spread that much wider
— and because the light is a fraction of the sun's. Day frames are unchanged at 7.97 ms.

**Those figures were wrong by a factor of 2.25 when first written down.** The timing line reported durations and no
resolution, `--screenshot 1920x1080` names the *output*, and the default `--dlss quality` traces at 1280×720.
`FrameTimings` now prints what it traced at and what it displayed to.

**And a moon is a rock.** The disc was *added* to the sky the ray already had, which is right for the dome's own
glow — that is air in front of the moon — and wrong for the stars, which are behind it: the constellations showed
through Masser's face. `moon_covers` is the cone test on its own, and where it holds the star field is skipped; the
same flag suppresses the sun's disc, so an eclipse now works, and with Masser at eighteen degrees against the sun's
half a degree it is total when it happens.

**And they are drawn but not gathered**, like the stars and for a sharper version of the same reason. Masser's disc
is a thousand times the sun's solid angle, so a bounce ray finds one about once in a thousand at three hundred
times the night sky's floor — a firefly every few hundred pixels. The light they contribute arrives as a resolved
directional term instead.

The clock had to learn the date: `WorldTime` counts hours since the world began rather than hours since midnight,
because a phase advances between one midnight and the next. `--time 25` is one in the morning of the second day.

**The bright outline round every moon was none of that.** It survived lunar-Lambert and turning the upscaler off,
and the shading law measured smooth. Rendering with the portrait disabled found it: **the vanilla portraits are not
premultiplied.** Past the edge of the painted disc the file's colour climbs back — for Secunda to 0.39 of its mean,
where the disc just inside has fallen to 0.14 — and the alpha that exists to mask it was being sampled and thrown
away. Multiplying by alpha removes it and hands over the silhouette's antialiasing for nothing.

**The face does not rotate with the phase, and the game agrees.** The eight painted phases are one face under eight
terminators: their maria correlate at 0.29 to 0.77 as painted against −0.14 to +0.30 mirrored. That is also what a
tidally locked moon does.

**What was missing is the other rotation.** A locked moon keeps its face toward *us* and its orientation toward its
*orbit*, so as it crosses the sky the face turns against the horizon — **106 degrees across one of Masser's
transits**, measured. The face's up was being built from the world's, which pins it to the horizon all night; that
is what a billboard does, and it is what vanilla does. It now stands upright against the moon's own orbital pole.

**What this unblocks and has not spent.** `NIGHT_SKY`'s note has said since §8.49 that nothing scaling the whole
scene can separate a dark sky from a legible ground, and that the fix is a light falling on surfaces and not on the
sky. That light now exists; the floor is left where it is all the same, because the hours a moon is down are still
lit by it alone.

### 8.54 The game is opened once, not three times

Opening the installed game costs **46 ms warm**: 25 to read `Morrowind.esm`'s 79 megabytes, 10 to index the archives, 2.4
for the cell index and 8.7 for the model table. A single run was paying it **twice over, plus a third pass over the
archives** — the startup cell opened everything and dropped it, the streaming thread opened everything again and kept it,
and the moons' portraits opened the archives for two files. 158 MB read to use 79.

**The reason it was that way turned out not to be true.** `GameFiles`' own note said the cell index and the model table
borrow the reader, so whatever owns the bytes must outlive all three. They do not borrow it: `CellIndex` and `ModelIndex`
take a reader to `build` and own every byte they keep, and the single borrower is `EsmReader` over the bytes alone, whose
construction is a header parse measuring 0.0 ms. So `GameData` owns the bytes, the archives and both indices, and hands
out a reader on demand.

It is a `OnceLock<Option<GameData>>` behind `GameData::shared() -> Result<Option<&'static Self>>`. `&'static` rather than
an `Arc` because the data lives as long as the process either way; sharing it across threads was already safe, because
`BsaArchive::read` reads at an offset rather than seeking — a decision made for exactly this. **Failures are not cached**,
so a caller that reports one and carries on does not poison every later attempt.

**Measured, best of six, on a `--screenshot` run: 2259 ms → 2150 ms.** About 110 ms — the 46 ms open plus two bare archive
opens at 10 apiece plus 11 ms of indexing the streamer no longer repeats, plus the page faults on 79 MB of fresh anonymous
memory that is now never allocated.

**What was measured and deliberately not built.** Across a 7×7 window there are 1,837 NIF parses of 379 distinct meshes
(79% repeats) and 2,084 texture decodes of 358 distinct textures (83% repeats) — which looks damning until the repeats are
costed. Parsing every NIF the window names takes **27 ms**; parsing each distinct one once takes **7 ms**. The repeated
meshes are the cheap ones — clutter, furniture and rocks — while the expensive parses are one-offs. Texture "decode" is a
header parse and a copy rather than a transcode, so all 2,084 come to 12 ms. A full asset cache saves about 30 ms of a
173 ms window — **17%** — and costs `Arc<Mesh>` or a shared arena plumbed across the `rtxmw-scene`/`rtxmw-render` seam
plus a lifetime policy. Cell loading already runs off the frame path. Not worth it.

And it is worth recording where the time in that 173 ms actually goes: under 40 ms of the 1,436 ms initial window fill is
reading or decoding anything. The rest is ESM record walking, `Mesh::from_nif` and terrain building, which is where a
profile should start.

### 8.55 The sun jumped forty degrees at midnight, and nothing had noticed

`Sun::at` takes an `orbit` that runs 1 at sunrise to −1 at sunset and off both ends into the night — and **wraps**,
reaching −2 a minute before midnight and +2 a minute after. Past the horizon the old night branch built its heading from
`-400 * orbit` the way the daylit half does, so at the wrap the heading flipped from west of north to east of it:
**40 degrees in 1.2 game minutes.**

**It had been there since the night branch was written and nothing had ever read it.** The sun is seventy degrees under at
midnight, its colour is zero, and the sky's twilight term is into the rounding — so every consumer of the sun's *direction*
at that hour was multiplying it by nothing. The first thing that wasn't was a moon's terminator, which is carved from where
the sun actually is: the crescent snapped across the disc at 00:00, and the bug surfaced three sections after the code that
caused it.

The night now runs on one continuous parameter — nought at sunset, one at sunrise. The descent reproduces `|orbit| - 1`
exactly, and the bearing sweeps from where the sun set through **due north at midnight** to where it rises. Sunset and
sunrise sit symmetrically either side of north, so both ends meet the daylit formula to the bit. A test walks the whole
night at a tenth of a game minute and asks for no step larger than a twentieth of a degree.

**Worth keeping as a shape of bug rather than a bug.** A quantity that is continuous where it is read and discontinuous
where it is not stays correct until something starts reading it there. The three constants the two branches share are named
now, because the property the test asserts is only true while both read the same numbers.

### 8.56 Clouds, which are a vanilla asset lit rather than shown

Morrowind's sky is nine painted sheets, one per weather — `tx_sky_clear.dds` through `tx_sky_blight.dds`, 512 square, BGRA
— scrolled over a flat cap of radius 1000 by `Cloud Speed`.

**Each sheet is a photograph of a sky with 2002's lighting in it**, which is the whole of how to use one. Compositing one
over the dome puts the sky in twice and lights every cloud twice — §5.1 exactly. So the sheet supplies *shape* and the
light supplies colour: the alpha its artist drew the clouds with, and — where that alpha carries nothing, which is every
overcast weather, all at 255 — the texel's own luminance against the sheet's mean. The same split as the moons' portraits,
and it pays off at dusk, where a cirrus deck goes gold because the light reaching it has crossed thirty atmospheres.

**A shell over a curved world, not a dome round the eye.** `sky_clouds_01.nif` is a cap of radius 1000 rising 100 to 307 —
flat enough that its own artist was drawing this picture — but a mesh centred on the viewer meets every ray at one
distance, so the last few degrees above the horizon smear the sheet into radial streaks. Five hundred metres up over the
Earth's radius gives a ray `h` overhead and `sqrt(2Rh)` — **80 km** — along the horizon, which is what a sky's depth is
made of. One tile spans two kilometres, which puts a cloud feature at about 200 metres.

**And that geometry is a precision trap.** The obvious root of the ray-shell quadratic is `-b + sqrt(b*b + c)`, where `b`
is the world's radius times the ray's climb: 4.5e8 in game units, so `b*b` is 2e17 and `f32` has four digits left of it.
Subtracting `b` from its own square root throws away the answer — straight up it came out **17 units** from a distance that
is exactly the altitude. The conjugate form `c / (b + sqrt(b*b + c))` is the same root, adds two positive numbers instead
of cancelling two huge ones, and is exact there. The second time this project has been bitten by differencing world-scale
numbers — §8.7 was the first.

**A cloud is darker than the sky it covers.** The sky-lit term was set at 0.9 of the dome's radiance on the argument that a
cloud is lit from the whole hemisphere — true of the irradiance arriving and silent about what leaves. A thick cloud
reflects most of that *upward*; plane-parallel theory puts a deck's transmission at 0.2 to 0.3. At 0.9 a night deck was 90%
of the sky it hid. At 0.3 a solid deck at night is a dark shape blotting out stars, and thin cloud is not dragged down with
it because how much sky a wisp replaces at all is its own alpha.

**And the clock is where the wind had to stop being physical.** `timescale` is 30, so a game hour passes in two minutes of
watching and a real 19 km/h breeze carries a cloud across ten degrees of sky every two seconds — a conveyor belt. The drift
is set against the clock the sky is watched on instead: about ten degrees a minute. It is the honest place to break from the
physical figure, because the sun's hour is chronology and has to run at the game's rate, while a cloud's drift is ambience.

Two things a review caught. The sun's disc was composited *after* the layer and **replaced** the colour, so a sun under
solid overcast came through at full strength with the deck drawn around it; it is blended by the same coverage now. And the
sheet was sampled with plain `texture`, which in a compute shader takes mip zero — while `reach` at the horizon is a hundred
times what it is overhead. It picks a level off the ray cone now, the same argument `cone_lod` makes for a surface.

### 8.57 The clouds get their own horizon, and cast a shadow on the world

**A cloud is still in sunlight after the ground has lost it, and that is the whole of a sunset.** The layer was handed
`Sky`'s own sun — already faded over the disc's half-degree at the *ground's* horizon — so it lost the sun at the same
instant the ground did: 70% of the clouds' colour in 1.2 game minutes, against 1.5 to 3.4% either side.

Three things were wrong. The layer's **horizon is lower**: a deck at height `h` over a world of radius R keeps the sun until
it is `sqrt(2h/R)` under the ground's horizon — 0.72 degrees at five hundred metres — and the layer reaches its own horizon
80 km away whose clouds keep it that much longer again, so the deck goes out over roughly twice the dip. It crosses **less
air**: five hundred metres up is above a twentieth of the atmosphere's mass, so at sunset its beam has crossed twenty-seven
air masses against the ground's thirty-eight, which is why a sunset cloud is gold rather than black. And the fade is keyed
to **how far the sun is under**, not to the sine of where it is.

That last was a symptom of the singularity §8.58 removed; the three fixes took the clouds from about 25% of their daylit
level per 2.4 seconds of watching down to **12.3%**.

**And the layer casts.** The same sheet, sampled along the ray to each light above it, at a coarse mip because a shadow is a
soft thing and every shading point pays for it. A deck lets a quarter through — the same fifth-to-a-quarter that `SKYLIT` is
the other side of — so a cloud shadow is not black, which would read as an eclipse. Fault injection confirms the path:
forcing full shadow moves 40.8% of the frame's pixels by more than 16 of 255.

**And the shadows were invisible until the layer shrank, which is a scale argument rather than a lighting one.** A cloud's
shadow is the size of the cloud. At eight-kilometre tiles a feature was 780 metres across against a visible landscape of
about three hundred, so the whole view sat under one cloud and dimmed as a body. Shrinking the tile alone would have shrunk
the clouds in the sky too, so the base came down with it: the angle a cloud subtends is `feature / altitude`, and holding
that fixed keeps the sky while the shadow halves. Two kilometres of tile at five hundred metres of base is the pair that
gives both — at two kilometres of base the same tile turned the sky into a mackerel plaid. Toggling the shadow moves the
frame by max 49 of 255 where the first scale managed 12.

**Three things were hiding them, and only one was a bug.** The scale was. Clear weather's sheet is **cirrus** — its alpha
averages a quarter, and cirrus in life casts almost nothing, so drawing exactly what the sheet says is faithful and
invisible (2.0% of pixels moved). And the exterior fog was still a placeholder thick enough to wash out most of what does
get cast (the same toggle with `--fog 0` moves 6.9%). So the shadow is deepened past what the sheet asks for, by four,
taking an average cirrus sky from a fifth of the sun blocked to three quarters and the toggle to 6.1% of pixels at max 83.
**It is the one number in the layer chosen rather than derived**: a shadow that cannot be seen is not worth tracing. The
weather system should retire it — `tx_sky_cloudy`'s alpha averages 0.74 against clear's 0.25.

It also forced the layer to be **anchored to the world** rather than to the viewer. The sheet had been addressed by the
ray's direction alone, so the whole deck travelled with the camera — invisible on its own at 140,000 units up, and fatal the
moment a shadow had to lie under the cloud casting it.

### 8.58 The sun's elevation was a circle where it should have been a cosine

`sqrt(1 - orbit²)` has an infinite derivative at the horizon, so the sun dropped from 0.57 degrees to nothing in the last
eighteen seconds of the day and everything keyed to its elevation stepped there. **Measured on the ground at Seyda Neen: the
light fell from 1.33 to 0.46 between 17:58 and 18:00**, a 19.9% step in one 2.4 seconds of watching.

**The circle was this project's own, not Bethesda's**, which is what made it takeable. `cos(orbit * pi/2)` gives the same
ends — one at noon, nought at both horizons — is smooth there, and is more physical besides: a real sun's elevation against
an hour angle linear in time is a cosine, and feeding a linear quantity to a function that expects one already cosine-shaped
is precisely what put the vertical tangent at the horizon.

The step is **19.9% down to 7.2%**. What remains is a kink rather than a cliff: the day reaches the horizon at 22 degrees per
unit of orbit while the night leaves it at `NIGHT_DESCENT`'s 70. Matching them means reshaping the night's descent, and 70
degrees at midnight is what §8.47 bought to stop the twilight glow burning all night — at 22 the glow would be 2.8% of its
horizon value at midnight rather than 0.012%. Not obviously worth taking, and not taken.

**Two things had to be re-fitted, and both were doing their job.** `SKY_STRENGTH` exists to pin the default hour at a
luminance of 0.518 so every image made against it stays comparable; the sun is lower at nine now (18.3° against 22.1), so it
went 1.5786 → 1.7000. `DAY_LUMINANCE`, the noon figure the exposure curve is fitted between, follows 0.702 → 0.755.

And one test changed rather than one behaviour: the zenith is no longer blue at 17:30, because at 2 degrees of elevation
`warmth` is near one and the dome's warm tint reaches the top. That is a real weakness in spreading warmth by `sunward` alone
— the zenith looks through one air mass and should stay blue at any hour — but the new curve exposed it rather than caused it.

### 8.59 Weather, read out of the ini rather than written down here

Morrowind keeps its weather in `Morrowind.ini`: ten `[Weather X]` blocks of 49 fields apiece, and a general `[Weather]`
section saying when each changes. Three of its figures had already been copied into this crate by hand — the day's length,
the star schedule, the moons' sizes — because there was nothing to parse the file with. `GameData` now owns a parsed `Ini`
beside the master file and the archives.

**Each family of colours changes over on its own schedule, which is why a Morrowind dusk does not happen all at once.**
`Sky Pre-Sunset Time=1.5` and `Sky Post-Sunset Time=.5` against the ambient's 1 and 1.25: the sky begins turning ninety
minutes before the sun goes down and has finished thirty after, while the ground starts sixty before and takes until
seventy-five after. Twelve such figures, four families, and nothing had to be invented.

**What the game supplies and what stays this renderer's.** The hues are Bethesda's and the levels are not: `Land Fog Day
Depth` is in the original engine's units, so `FOG_SCALE` ties clear weather's 0.69 to the 0.30 settled by eye and every other
weather follows the ini's *ratio* to it.

**Normalised by its brightest channel, not by its luminance**, and the difference is the whole of whether blight works. Its
`Fog Day Color` is (128, 19, 19), whose luminance is a twentieth of its red — so dividing by that gave a multiplier of 4.0 in
red and the fog came out *brighter* than the light that lit it. What this quantity is is a scattering albedo, and an albedo
cannot exceed one. Against the maximum, blight is a deep red darker than a clear day's.

Three placeholders are retired: `OUTDOOR_FOG_DENSITY`, `CLEAR_SPEED` (the layer drifts at the weather's `Cloud Speed`), and
the cloud layer's hard-coded full cover (now `Clouds Maximum Percent`, 0.66 for rain). `--weather` names one of the ten; an
unknown name is clear rather than a refusal to start, since the list is the game's.

**Two of the ten are Bloodmoon's and carry its own prefix** — `Tx_BM_Sky_Snow` and `Tx_BM_Sky_Blizzard` against the eight
`Tx_Sky_*`. The test that every weather names a sheet asks the archives rather than assuming the shape, which is how that was
found.

**What is not done.** The ambient schedule is parsed and unused deliberately, as the check the derivation is held against
(§8.60). `Sun * Color` is unused because what it would add is the extinction a weather's medium puts on the beam (§8.61).

### 8.60 A deck is a lid, and the ini agrees

**Sky and ambient move in opposite directions.** Overcast's `Sky Day Color` is 1.19 times clear's in luminance while its
`Ambient Day Color` is 0.44; foggy's are 2.59 and 0.55. A cloud deck is a bright sheet that leaves the ground dim, and
Bethesda wrote both halves down.

This renderer had only the first. `dome_average` excluded the clouds deliberately — a dome that counted their own light would
light the ground by its own clouds — but leaving out their *blocking* with it made an overcast noon as bright underfoot as a
clear one. The fix is one line and no authored figure: the covered fraction of the dome is worth `SKYLIT` of the open one,
the same 0.3 a cloud sends down of the sky that lit it. What a weather hides is its sheet's own mean alpha times its
`Clouds Maximum Percent`.

**The ini is then the check rather than the source:**

| | derived | ini | | | derived | ini |
|---|---|---|---|---|---|---|
| clear | 1.00 | 1.00 | | rain | 0.65 | 0.58 |
| foggy | **0.55** | **0.55** | | cloudy | 0.59 | 1.06 |
| overcast | 0.36 | 0.44 | | thunderstorm | 0.65 | 0.38 |
| snow | 0.36 | 0.44 | | ashstorm | 0.36 | 0.14 |
| blizzard | 0.36 | 0.44 | | blight | 0.36 | 0.17 |

Where the two part is informative rather than random. A **dust storm** is dark because the air is full of ash, not because a
deck is over it — the ini asserts 0.14 and this renderer has no airborne dust to derive it from. **Thunderstorm** is the same
shape. And **cloudy** is the one weather whose authored answer this does not believe — its ambient is clear's to within 6%
despite three-quarters cloud cover, which reads as a copy.

**The other half did not land, and both attempts are worth keeping.** The plan was to let each weather's `Sky * Color` move
the dome as a departure from clear's. As a tint on the *dome* it double-counts: foggy's sky is 2.59 times clear's **because**
of its deck, so dimming by that deck as well made overcast brighter than clear. Moved to the *deck* instead it turns overcast
**orange** — the departure is (2.31, 1.13, 0.46), because dividing a grey by clear's blue is a warm ratio however it is
normalised. **The mistake behind both is treating one number as two things.** The ini's sky colour is *the whole sky's
average* — the blue air under clear, the deck under overcast — and nothing in the file splits those apart. §8.61 is what
settled it.

### 8.61 The sky colour is a medium, and the file says so by writing it twice

Blight was red fog under a pale white sky: the ground haze took `Fog Day Color` and the dome stayed the physical model's.

**The file does separate something, and it is not what was being looked for.** Six of the ten weathers write their sky colour
and their fog colour as *literally the same number*, in all four keys: overcast `143,146,149` twice over, ashstorm
`124,073,058`, and the same for rain, thunderstorm, snow and blizzard. Only clear, cloudy, foggy and blight write two, and the
first two of those are the weathers whose sky is actually blue. Twelve bytes agreeing exactly is an authoring decision:
**when the medium fills the air, the sky is the medium.**

So the question stops being "what colour is the deck" and becomes "how far up does this weather's own medium reach", which has
one answer per weather and both numbers to derive it from already here.

**A `Veil`, which is the fog seen looking up.** The same medium in the column above the ground haze, in front of *everything*
the sky has: the dome, the cloud deck, both moons, the stars and the sun's disc. That placement is the fix for the original
defect — under blight the deck covers the whole dome, so a veil applied only to the air would have left a white sheet across a
red sky.

**Chromatic and nothing else**, which is what keeps it from landing on work already done. How much light a weather takes away
is derived and checked by §8.60's table, so the veil is a hue on each pixel at the luminance that pixel already had. Every
figure in that table is byte-for-byte what it was, and a clear-weather frame is *pixel-identical*. It is also what keeps the
deck legible: at full strength a veil that replaced the sky with one flat colour would erase the sheet it is drawn over.

**How much of it there is, as a constrained least squares that is allowed to refuse.** The sky the renderer would draw, the
weather's fog, and the weather's asserted sky are compared as colours at one luminance, and the amount is the projection of
the third onto the segment between the first two. Six weathers land on 1.000 at every hour by identity; blight and foggy land
between, 0.58 and 0.76 at noon, because their two colours differ and the dome still shows through.

Clear and cloudy are the interesting case: their skies are *bluer* than either the renderer's dome or their own fog, so the
unclamped projection runs past one, and clamping alone would assert a full veil for exactly the two weathers that should have
none. So the fit answers for itself: what the medium could not account for is the **sine** of the angle between them — the sine
of an angle, so a medium whose blue is a fortieth of its red is judged on the same footing as one that is grey. Across the day
the eight it explains never exceed **0.280** and clear and cloudy never come in under **0.52**, so the cut sits at 0.4 and any
value in that gap gives the same ten answers.

**That refusal is the right way round.** The ini's clear sky is one flat swatch; this renderer computes a Rayleigh dome with an
air mass and a twilight in it. Where the game has something the renderer cannot derive — dust — it wins; where the renderer has
something the game never had, it keeps it.

**What it comes to.** Against the ini's own `Ambient` schedule, which nothing here is fitted to, the summed hue error over the
ten at noon falls from **2.64 to 1.46**. Ashstorm alone goes 0.825 → 0.149. Overcast stays grey.

**One trap had to be closed first.** `Weather::clear()` is the fallback for a machine with no game installed, and its colour
schedules came out *white*, documented as safe only while nothing read them. The veil reads them: white sky against white fog
fits perfectly and asserts an opaque white medium filling the sky. The schedules now fall back to `[Weather Clear]`'s own
forty-eight bytes.

**Still not done.** `Sun * Color` remains parsed and unused. Blight's is `224,084,084` against clear's `255,252,238`, and the
ratio is the extinction the weather's medium puts on the beam — real, and unambiguous in a way the sky colour never was. Its
*level* cannot be taken with it: under overcast the same ratio is 0.69, and that is the deck blocking the sun, which the layer
already does.

### 8.62 A region is the only thing that says where a weather can happen

`Morrowind.ini` has nothing to say about which weathers belong where. What does is `REGN`: every exterior cell names a region by
id, and every region gives each of the ten a percentage summing to a hundred. **A zero is not "rare" — it is the file saying
that weather does not happen here.** Seyda Neen's shore comes out clear, cloudy, foggy, rain, thunderstorm, which is a swamp;
nothing on Vvardenfell snows because snow arrived with Bloodmoon's Solstheim.

**Two orders, and they are not the same one.** `WEAT`'s bytes run clear, cloudy, foggy, overcast, rain, thunderstorm, ashstorm,
blight, snow, blizzard — the game's own order, which reads like a forecast. The ini's sections come out **sorted by name**,
because that is what indexing sections does, so `Weather::table` opens on ashstorm and puts clear fourth. Matching a chance to a
weather by position would silently pair blight's percentage with blizzard's sky. `RegionRecord::ORDER` is written down for that
reason.

**Where nothing narrows them, the answer is all ten.** An interior names no region, a handful of exteriors name none either, a
region the file does not describe rules nothing out, and a region whose every chance is zero is bad data rather than a place with
no weather.

**It cost no extra pass over the file.** `CellIndex` already carries the `LTEX` palette for the same reason, so regions ride the
same walk.

**What changes when the camera crosses a boundary is the list, not the sky.** A blight storm that blew in over the Ashlands does
not stop at the coast — the region says what can *begin* here. So a weather standing outside the local list is normal, and the
step enters the list at whichever end it came from rather than refusing to move.

On the device, a weather change is one bindless slot written twice. Filling it drops the image that was in it, so it waits for the
device to go idle first — a key press is not a frame path, and the alternative is a deferred-destroy queue for something that
changes when a person asks. The failure that matters is invisible from the host: a slot whose memory changed but whose descriptor
did not would keep drawing the first sheet while every number the host derives came from the second. `tests/sky_textures.rs`
renders clear, then overcast, then clear again, and asserts the middle frame moved and the last came back.

### 8.63 One wind, and it decides three things about the fog

**The measurement that set the direction.** Across the ten, far-field contrast said clear (depth 0.69) and foggy (1.0) were
**indistinguishable** — 42.0 against 43.3, and the difference has the wrong sign because the veil moves it more than the density
does. Thinning clear's own fog raises contrast steadily, 42.0 to 50.4 as the depth goes 0.69 to 0.17, so the curve responds fine;
the ini's ratios are simply undramatic. **Bethesda's fog depth was a view-distance dial rather than a density.**

So the density ratios stay the game's and the character comes from the one number nothing was reading: **`Wind Speed`** — 0 for
foggy and snow, .1 clear, .2 cloudy and overcast, .3 rain, .5 thunderstorm, .8 ashstorm, .9 blight and blizzard. Bethesda putting
a fog bank at dead still is right: a radiation fog forms in still air and sits in it.

**One number, three effects, because all three are the same physics** — turbulent mixing driven by wind shear is what carries air
past, what lifts what is in it off the ground, and what stirs it until the banks are gone:

- **Advection**, which is not what the existing drift was. The three `FOG_CHURN` headings disagree with each other, and that
  shearing is what makes shapes form and pull apart — air doing that in a dead calm is the whole reason a still fog is not a
  frozen texture. The wind adds the separate thing: the entire field carried downwind together, on the heading the cloud layer
  already drifts along, because there is one wind over a landscape.
- **Lift.** `FOG_HEIGHT` is clear weather's figure in still air, and the layer stands deeper by the weather's own `Land Fog Depth`
  — the field is *named* depth — times what its wind adds. This is also what makes the fog agree with §8.61 rather than contradict
  it: the veil says eight of the ten fill the sky, and a medium that filled the sky while pooling in a 37-metre bank was two
  answers to one question.
- **Mixing.** `fog_uniform` was one bit for indoors; outdoors it is now the wind.

**Each of the three is separately measurable, which took two wrong assertions to find.**

- Over a minute, advection and churn have *both* saturated — the field is uncorrelated with where it was, and every measure levels
  off at the same number. At ten seconds advection reads 3.29 against churn's 1.09; at sixty it reads 4.19 against 4.68. **A
  difference metric between two noise fields has a ceiling, and a test that runs past it compares two ceilings.**
- A gale's frame varies *more*, not less, because lift dominates: standing the layer up puts fog where there was none. The stirring
  only shows where lift cannot act — at the base of the layer, where the height falloff is one whatever it has been scaled to.
  Measured in the bottom quarter of the frame, spread falls monotonically: 4.50, 4.02, 3.26, 2.76, 2.38.

The tests now isolate each. Advection is two winds of *equal strength blowing opposite ways*, which agree on lift and mixing by
construction and draw the same frame before the clock has run at all. Lift is the top of the frame against the bottom, three to
seven times as much. Mixing is the ground band alone.

**The wind alone could not set the height, and the picture found it before the arithmetic did.** Driving the layer's depth from the
wind and nothing else gave the weather *named* foggy the shallowest layer of the ten, because Bethesda puts its wind at nought — so
a foggy day let you see **further** than a clear one, confirmed at 43.3 against 39.4. Depth is what the ini's `Land Fog Depth` is
called, and it belongs in the height as much as in the density; the wind multiplies it rather than replacing it. Once the two
multiply they have to be ordered against each other — foggy is 1.45 times clear's depth and blows at nought against clear's 0.1, so
the wind's coefficient has to leave `1 + 0.1 x` under 1.45, and **anything over 4.5 inverts them**. Four is what is there, and it is
a bound rather than a taste.

**And the ratios had to give, which is the one place in the weather system the game's own numbers are not obeyed.** With all three
axes in, foggy was still only about a sixth more fog across the frame than clear, because the only thing separating them in density
is the authored 1.45×. In the original engine that number set where fog reached full against a fixed far plane, so a small change
moved a hard cutoff doing most of the work; here it is a density integrated along a ray. **No rescaling that keeps a clear day clear
can pull 1.45 apart** — halving the absolute and quartering it left foggy the same sixth clearer. So the order stays the game's and
the spacing does not: a fourth power leaves clear exactly where it was by construction (its render is pixel-identical), moves cloudy
and overcast by a tenth, doubles rain, and puts foggy at four and a half times a clear day. Far-field contrast: clear 40.3, rain
34.1, foggy 24.1, blight 8.1.

**The top needed a cap, and the number that says so is 0.03.** Blizzard's `Land Fog Depth` is 2.8 against clear's 0.69, and a fourth
power of four-times is **271** — which rendered at 0.03 of a clear day's contrast, meaning not one thing in the frame was visible.
Sixteen leaves it a whiteout you can still see the boat from and binds nothing else: ashstorm and blight sit at six and a half.

**The wind was picked and should have been read.** `FOG_GALE` began at 120, chosen against `FOG_GRAIN` so the strongest weather
crossed one cell of the coarsest noise in about nine seconds — and nine seconds to cross thirteen metres is 1.4 metres a second,
which is a still afternoon rather than an ash storm. Twenty metres a second is a Beaufort 8 gale and there are seventy units to the
metre, so the figure is 1,400 and every weather's `Wind Speed` becomes a real one: clear's 0.1 is a two-metre breeze, rain's 0.3 six,
thunderstorm's 0.5 ten, ashstorm's 0.8 sixteen, blight and blizzard eighteen. **A constant chosen to make a noise field look busy was
hiding the fact that the units were already there to derive it.** That moved the tests' own window with it: at 1,400 advection
saturates within a couple of seconds, so the wind is measured over one second and the churn — eleven to nineteen units a second —
over ten.

**And the layer was in the water as well as over it.** §8.39 measured the fog's height from the water. What that leaves below the
surface is `max(z - water, 0)` clamped to zero — the layer's *full* thickness, everywhere, all the way down — so every submerged ray
carried a second medium laid over the one `water.glsl` already attenuates and colours. Removing it moves a wholly submerged frame by
**36 of 255**; from the ship's deck it is a quarter of the pixels at a mean of 0.15, and under a blizzard it is nothing at all,
because at sixteen times clear's density the ray is opaque before it reaches the water. The guard is `water_level - z > 0`, the idiom
`primary_visibility.comp` already uses, which needs no flag: a dry cell carries negative infinity for its level.

### 8.64 Rain is signal, and Ray Reconstruction cannot tell

The first rain was composited inside the trace, into `emitted`, and came out of DLSS smeared. Two mechanisms, both unavoidable at
that position. Ray Reconstruction **denoises what it is given**, and a rain streak is not noise in a ray-traced estimate; it is the
answer. And it **accumulates temporally against motion vectors**, which describe the surface *behind* each streak, because one vector
per pixel cannot describe both a drop falling at fifty-seven metres a second and the hillside it crosses. NVIDIA's own guidance says
RR "does struggle with fine particles, rain droplets ... causing them to either ghost, blur, or a bit of both."

**The SDK already had the way out.** `nvsdk_ngx_helpers_dlssd_vk.h` carries `pInTransparencyLayer`, commented *"optional input res
particle layer"*, with `pInTransparencyLayerOpacity` beside it and the parameter strings `DLSS.TransparencyLayer` and
`DLSS.TransparencyLayerOpacity` in `nvsdk_ngx_defs.h`. It is a layer at render resolution that RR **composites rather than filters**.

So the trace writes what falls into two images of its own, premultiplied colour and coverage, and never into the frame. What that cost
elsewhere:

- **The bindless texture array had to move.** Vulkan allows a variable descriptor count only on a set's *final* binding, so the two
  new storage images could not go after it; the array is binding 19 now and they are 17 and 18.
- **Something has to composite the layer when there is no upscaler.** That belongs in the composite, and getting it there took two
  goes. The composite returned early on a ray that hit nothing — every pixel of sky, which is where rain shows most — so the tone
  curve looked like the place instead. It is not: **the exposure pass meters what the composite leaves behind**, so putting the rain
  in after the curve left auto-exposure reading a frame with rain in it under an upscaler and without under none, and the two paths
  chose different exposures for the same weather. The early return is gone and the overlay is a push constant on the composite; with
  an upscaler it is zero and the layer is never read.
- **The rain tests had to read the finished image rather than the traced target**, which is a better test than it was: it now covers
  the compositing as well as the drawing.

**And with the blur gone, two defects it had been hiding were plain**: a horizontal band of long streaks across the horizon and,
looking down, a ring of disconnected blobs around the eye. Four attempts were made at the first before the second arrived and gave
both away at once.

**What found it was painting the suspects.** Rendering `abs(dot(direction, fall))` puts a black line exactly through the middle of the
band: it sits where the ray runs perpendicular to the fall, which for near-vertical rain is the horizon. Painting `reach` came out
flat grey looking out to sea, which ruled out per-pixel depth *for that view* — and that was the near miss, because looking **down** it
is not flat at all.

**The root cause is that the lattice was derived from the ray.** The cell across the fall was the volume divided by the number of
samples, and the volume is `min(surface distance, Rain Diameter)` — a per-pixel quantity. So the lattice changed size per pixel:
neighbouring pixels disagreed about where the drops *were*. That is incoherent on its face — a drop is a thing in the air, and how big
it is cannot depend on which ray looks at it. It drew the ring, whose edge sat exactly where the floor's distance crossed the volume's;
and it drew the band, which is the same incoherence read along a different contour, because the screen-space shape of an aliased lattice
is set by how fast its coordinates move per pixel, a rate that peaks where the ray is perpendicular to the fall.

So the lattice is a **world constant** now, and the march walks it **one cell at a time** — which keeps consecutive samples in adjacent
cells rather than skipping across dozens, and is the difference between walking a lattice and aliasing it. **What does not move is how
much rain there is**: the physical answer is what a ray crossing `n · 4r² · L` of real drop cross-section blocks, and what the shader
solves for is the opacity a *drawn* streak needs so that however many a coarse lattice holds come to the same total. Appearance and
quantity are separated, and only the first is a choice.

The four that failed are worth naming, because each was a plausible reading of the symptom: more layers (over-counts — ten consecutive
samples in one 122-unit cell charging for the same streak); closing the ray on the streak's axis (unbounded, so every cell any sample
touches reports a hit); the same clamped to a sample's own span (the clamp bounds the approach but the *cell* is still picked at the
midpoint); and holding the drops back from the lens (moves where the artefact is, not what it is).

**A streak's length is not the game's fall speed.** `Precip Gravity` times rain's entrance speed is 4,025 units a second — *fifty-seven
metres*, six times what a raindrop reaches, because the original draws long sprites and they have to move to read. Taking a streak's
length from that made it a full metre, which at half a metre from the eye subtends fifty-six degrees. Shrinking the lattice did nothing
for it and could not — a streak's size on screen is its radius over its distance, the radius is a fraction of the cell and the nearest
sample sits half a cell out, so the two cancel and a finer lattice only ever made *more* drops the same size. The ini's speed still
carries the field past, which is the game's look; how far one drop smears while the shutter is open is nine metres a second, which is
physics.

**And the counts are spread like the fog depths.** How much of a ray the rain covers goes as the count, and the file puts a thunderstorm
at 650 drops against rain's 450 — a ratio of 1.44. §8.63's argument applies unaltered: a number tuned against a fixed sprite budget is
not a rate. Rain stays exactly where the file puts it and a thunderstorm comes out two and a half times it.

### 8.65 Rain that lands

**Which weather rings is the file's own claim, not a choice here.** The general `[Weather]` section carries `Rain Ripples=1` beside
`Snow Ripples=0` — a raindrop breaks the surface and a flake settles onto it. The shader returns before it reads anything when the
weather is snow.

**A lattice of impacts, the same trick the drops themselves use.** Each cell of the water plane holds one impact at a hashed place on
its own phase, leaving a ring expanding at half a metre a second — a little above the minimum phase speed of capillary-gravity waves —
and dying within `RIPPLE_LIFE`. The nine neighbouring cells are read as well as the one underfoot, because a ring outlives its own cell.
`water_normal` already sums slopes from the wave spectrum, so a ring is one more slope term. **Not part of the spectrum, and it must not
be**: the caustics stage differentiates the swell a second time, and a ring eleven centimetres across carries none.

**How many rings is not how many drops.** A real rain deposits thousands of drops a second on a square metre and a surface cannot show
them as separate rings; what an eye picks out is a few tens. The cell is twenty units, which puts a dozen impacts on a square metre with
a handful ringing at any moment, and measured against the same frame with the term off they move **a quarter of the open water**.

**Testing it took three goes, and the two failures are the interesting part.** Comparing a rainy frame against a dry one measures the
*drops in the air*, not the rings — a snowy frame differs from a dry one on the flakes alone, and 137 pixels of a 4,096-pixel fixture
said so. Reading the normal target instead looks like the clean isolation and is not: water writes `-direction` there rather than its own
normal, deliberately, so a ring never reaches it — §8.20's reason, still holding.

What works is a weather whose drop *volume* is too small to draw. A drop is drawn only within the weather's own `Diameter` of the eye, so
a couple of units of that puts nothing in the transparency layer while the water still knows it is raining — and then rain against
**snow** differs by the rings and nothing else. With the term stubbed out the test reports zero, which is what says it bites.

**And the fixture had to come down to the water.** A ring is eleven centimetres across and the cone fade averages it away exactly as it
does the swell; a camera four hundred units up at sixty-four pixels square read that correctly as no ripples at all.

**The other half** — rain darkening and glossing whatever the sky can see, which is what sells a shower on stone and timber rather
than on water — is §8.70.

### 8.66 The exposure was answering for the rain

Every assertion in `tests/precipitation.rs` passed on the auto-exposure. Metering follows the frame's overall brightness, so
putting rain in the air drops every pixel three levels and snow drops them twenty: a difference count came back saturated at
16,384 of 16,384 while 572 pixels actually had a drop on them. One threshold cleared its bar by seven pixels of rounding; "snow
drifts where rain falls" compared 16,329 against 16,377 and **reverses** on the honest measure.

`lit_by` is the fix, and one constant suffices because the bias has a sign: precipitation only adds light, so the exposure it
provokes can only subtract it. Four levels is above a twenty-level shift in the wrong direction and far below a streak.

Fixing the measurement is not enough to fix the snow test. Snow lights ten times as many pixels as rain, so it turns more of
them over while moving a tenth as far — the comparable quantity is turnover as a *fraction* of coverage. Read that way, over a
hundredth of a second rain scores 1.99 of its own coverage (a complete turnover) against snow's 0.74.

**And no rain under the water.** Two clips: nothing when the eye is submerged, and a downward ray cut where it crosses the
level. An opaque hit already ends the march; the second covers a ray that finds nothing and would carry rain down through open
water.

### 8.67 A streak is the weather's own speed

`PRECIP_TERMINAL` was a speed — 630 units a second — so every weather's smear was rain's. A flake at 345 units a second came out
smeared over the same fifteen centimetres: a needle ten times longer than it was drawn wide.

It is a ratio now, `630 / 4025`, and rain is bit-identical. Snow drops to 2.1 units, set by the floor rather than by its 0.90
physical smear: never shorter than twice the drawn radius, because below that a shape is not short but pointed the wrong way.

Coverage is untouched by construction — `column = streak / PRECIP_DUTY`, so the share of a column a streak fills is constant
whatever its length. Shortening the smear stacks more of them in the same air rather than thinning the shower.

Shape is what tells the two apart: lit pixels with a lit neighbour below against one to the right give rain 6.16 and snow **1.42
→ 1.06**. Bounded on the excess over round, because snow covers a third of the frame and fusing adds neighbours in both
directions equally, pulling any ratio toward one whatever the flakes are shaped like.

### 8.68 The clamp that made a storm and a shower the same picture

Rain and thunderstorm rendered **byte-identical**. `alpha` came out at 1.8 and 4.6, both clamped to 1, so `Max Raindrops`,
`RAINFALL` and `COUNT_CURVE` decided nothing for the two weathers they were written for.

The cause: the drawn radius was a fraction of the sampling cell, so in `steps · pi r^2 / cell^2` the two cancelled and `alpha`
had nowhere to go. Same error as the ray-derived lattice behind §8.64's horizon band — how wide a drop is drawn cannot depend on
how finely the ray is sampled. With the radius in world units, `walked` cancels and the cell solves in closed form:

    cell = spacing * cbrt(PRECIP_OPACITY * r^2 * duty / (PRECIP_SPREAD * PRECIP_RADIUS^2))

The cell tracks each weather's drop spacing and the drawn radius to the two-thirds power, which holds a flake drawn three times
a drop's width to the same opacity instead of a ninth. Every weather lands at `PRECIP_OPACITY` and the clamp cannot bind. Per
weather is not per pixel, so the band stays fixed.

Two things rendering it made obvious and reasoning did not:

- **`RAINFALL` at 5,000 is a wall of static.** It is 2,900, and derived: rain's cylinder holds 450 drops in 412 cubic metres —
  1.09 per cubic metre, not the 0.6 the doc claimed, which predates the `height = high - low` fix — against Marshall and
  Palmer's `N0 / Λ` of 3,165. The 2,750 tuned against screenshots first landed within five percent of it, which is why the
  derivation replaced the tuning.
- **`COUNT_CURVE` was a workaround for the clamp**, added because 650-against-450 looked invisible. That was the clamp, not the
  ratio. Deleted, with `LIGHTEST`.

**The cross-section was wrong twice.** The target was `4 * PRECIP_RADIUS^2`, called a cross-section beside itself; a sphere
presents `pi r^2`, and `PRECIP_RADIUS` was a large drop's radius applied to every drop. For `N(D) = N0 exp(-ΛD)` the mean square
diameter is `2 / Λ^2`, so the mean cross-section is `π / (2Λ^2)` and the equivalent radius is

    r = 1 / (Λ sqrt 2)

0.280 mm at moderate rain — a fifth of the 1.5 mm that stood there, a twenty-ninth of the area.

Run that out and moderate rain blocks two parts in a thousand of a three-metre ray. Real rain is nearly transparent; streaks
show because a drop is a lens far brighter than what is behind it. This renderer cannot draw that — sub-pixel specks at fifty
times the background alias into static and Ray Reconstruction eats them (§8.64) — so the signal is spread: more coverage, dimmer
streaks, `coverage * radiance` held. `PRECIP_SPREAD` is that factor, and the only number here physics does not give. Its check:
`PRECIP_SPREAD * PRECIP_LIT` is a lens gain near fifty, the order Garg and Nayar and Wang both put a drop's at.

### 8.69 The rain fell upward

`precip_fall.z` is negative and the shader sampled at `position + drift`, which solves for the drop at *minus* the drift. The
whole field climbed.

Rain hid it: a streak seven pixels tall and one wide, slid along its own axis, lies on top of itself. Only a round flake has
anywhere to go. The correlation must fit inside one lattice column — every cell is jittered on its own hash, so a drift of one
column maps each cell onto a *different* neighbour and the profile past that is flat at chance. A fiftieth of a second measured
a peak six percent above chance, meaning nothing; a two-hundredth gives **952 pixels one row down against 517 one row up**, and
the old sign inverts it.

### 8.70 What the rain leaves behind

A wet surface is a substrate under a film: darker because the film keeps handing light back for another pass, glossed because
the film has a top.

**The darkening is a closed form.** Light leaving the substrate meets the film's top from inside, and everything past the
critical angle — 48.6 degrees, most of the hemisphere by solid angle — is turned round for the stone to absorb again. Egan and
Hilgeman's fit gives 0.475 at water's index, and the whole path is a geometric series in `albedo * F`. That is Lekner and Dorf,
and the series is why it cannot be a multiplier: a bright substrate gets more back each pass, so **a dark floor loses
proportionally more than a bright one**. A flat 0.6 makes the test report 0.5998 against 0.5998 where the closed form separates
them.

**Only where the rain reaches**, one ray back along `precip_fall` — the wind's slant is already in that vector, so a dry patch
sits offset from the eave that casts it. A single direction drew the boundary as the building's own silhouette, polygon by
polygon; rain gusts and splashes, so it samples a ten-degree spread.

**Slope runs opposite to the rate.** The square root of the cosine was wrong: a surface saturates once covered, but what it
*holds* is deposition against runoff, and a tilted face sheds what a level one keeps. Squared.

Four looks were rejected, each a different fault:

- **Plastic** — the reflection was added on top of the full diffuse response, and Schlick runs to one at grazing. It has to take
  from the diffuse what it returns.
- **One giant puddle** — widening `cone_spread` by the lobe picks a coarser mip and leaves the ray a perfect mirror, so
  roughness blurred textures and never geometry. Roughness must move the ray: GGX puts the lobe near `alpha = roughness^2` and a
  reflection turns by twice its normal, drawn per frame for Ray Reconstruction to accumulate.
- **Washed out** — there was no BRDF at all, nothing to say the facets facing the eye at a glancing angle are shadowed by those
  in front. At roughness 0.55 that gave **0.45** at grazing where the answer is **0.107**. The reflection is a cone sample of
  the environment, which is what the split-sum approximation is for; Lazarov's fit agrees with Schlick to a thousandth head-on,
  and the whole difference is the shadowing.
- **White spots** — Lagarde: a *thin* film reflects the disturbed normal of what is under it, a thick one a flat normal. With no
  normal map the film disturbed an interpolated vertex normal across a flat plank — the puddle case everywhere. The rain is the
  disturbance; `waves.glsl`'s lattice rings a wet deck because it is the same rain. But a ring only shows if the reflection is
  sharp enough to bend, and a thirty-degree cone average is uniform enough that tilting the normal changes nothing. **Standing
  water is smooth and soaked-in water is not**: roughness runs `FILM_ROUGHNESS` → `FILM_PUDDLE` with how much stands, so rings
  bend what a pool reflects while the damp ground stays matte. `FILM_HELD` is Lagarde's porosity with no map to read it from.

**The film reaches every hit.** `shade` is the only place a bounce learns what it landed on; left out, ground under shallow
water was drawn dry against a wet bank and the two met at the waterline. 0.9 ms.

**Still owed:** returning one for drowned ground put the bed darker than the shore beside it in weather with no rain, which
§8.20's no-seam rule caught at 0.205 against 0.269. The film fades to nothing over `SHORE_FADE` instead — hiding the step rather
than being right about it.

**One measurement worth keeping:** tracing the reflection cost 16.7 ms against 7.9 and was nearly abandoned, but the cost was
entirely *shading* the hit — a shadow ray per light — not the trace. Flat-lighting it, which a lobe this wide deserves, brought
it back to 7.9 with no visible difference.

### 8.71 A flash is a place, not an ambient

Lightning was folded into `frame.ambient` — light from everywhere at once, so it lit every face equally, cast nothing, and put a
white bay beside a dark shore, water returning a lit dome one for one. Split in two: `FLASH_LIT` is what a discharge throws on a
surface at `FLASH_REFERENCE`, `FLASH_SEEN` how far the dome brightens toward it. The bay is the constraint on the second.

The schedule is the ini's — `Thunder Frequency`, `Threshold` and `Sound Decrement` are already a Poisson process with a fixed
decay. One roll picks four kinds: 22% to ground, 11% crawler, 12% in-cloud, the rest sheet, against 20–25% observed
cloud-to-ground. `STROKES` restrikes `RESTRIKE` apart are the flicker; `CHANNEL` is 30,000 K.

Two bugs the structure hid:

- **`hash(0) == 0`** — splitmix64's finalizer fixes zero, so every storm's first flash drew the same shape. Golden-ratio
  increment. The tell was that it was always *that* flash.
- **The speed key reached the weather.** It multiplies the clock by up to 256, so a quarter-second flash lasted one frame at
  16×. `weather_seconds()` is a second clock it never touches.

### 8.72 Four times, saturation was mistaken for a shape

A channel is 2.4 px across and its glow runs to a quarter of the frame. No one falloff spans that, so there are three tiers —
`BOLT_CORE`, `BOLT_HALO`, `BOLT_CORONA` — the core deliberately blown out. Each failure was read as a bounding-volume artefact
first:

- **A Lorentzian never reaches zero**, so any cut leaves a step. Compact support is what makes the bound sound: `BOLT_REACH` is
  the widest tier and the profile is exactly zero at its radius, so cut and profile are one number and cannot disagree.
- **A hard tube**, from the halo saturating while `(1-x)^2` is flat at the centre. Glare's shape times the window, not plus.
- **A white capsule.** The amplitude was written as a ratio to `BOLT_ARC`, a number in the tens of thousands, so whether it came
  out above white was invisible where it was written: it peaked at **58 times white** and stayed saturated to seven tenths of
  its radius. Peaks are now in display terms, bounded at compile time.
- **A ring at 45 px, in fog only.** The corona's weather term was read off the depth of whichever *narrow* tier won the pixel,
  which is absent past the halo's reach — a **5.3-fold drop in the width of a pixel**, invisible in still air because there the
  term is nought either side.

The last was found by measuring a screenshot rather than the code: `107, 106, 107, 111, 119, 123` across the edge, 16 levels in
three pixels. The rule it left: nothing whose support is narrower than the corona may scale the corona, asserted `HALO < CORONA`.

### 8.73 A discharge is a line, and the point at its middle drew a bulb

`flash_reaching` modelled the arc as a point at its midpoint floored at `FLASH_NEAREST`, which is a ball that wide. A crawler
runs 74,000 to 140,000 units, so the fog painted a glowing sphere at its halfway mark with no bolt inside it — while
`flash_light` already sampled along the channel, so the two halves of one discharge had disagreed from the start.

The inverse square along a segment is closed form:

    integral ds / (r^2 + s^2) = atan(s / r) / r

divided by the run, so the arc carries one discharge however long it is. Far off the arctangents collapse to `1 / d^2`, which
keeps the calibration and lets a sheet — source and ground one point — take the same expression with no branch.

It fixes shape rather than brightness: **7.604 at a crawler's middle against 7.522 a quarter of the way down**, where a point at
its centre gave 100 and 0.99. `FLASH_NEAREST` came down 5,000 → 2,500 with it; 71 m is wider than the piece of deck being looked
at, 36 m is a channel's luminous envelope.

### 8.74 The halo is the air, so it follows how much air there is

Attenuating the corona by haze charges the air for hiding what it makes — the narrow tiers are an object seen through weather,
the wash is that weather scattering the channel's light back. Drawn that way it was deleted exactly when it should have been
strongest, cut to a fifteenth in the only weather that has lightning.

The amplitude follows the same argument:

    peak = BOLT_CORONA_PEAK + BOLT_CORONA_AIR * (1 - haze)

A fixed figure cannot serve both ends: what reads against a night sky vanishes inside a storm, and what reads inside a storm
flattens a clear night into a pale wash. `fog_strength` was also missing from the flash's haze, so a scene asked for no fog
still charged the cell's density — collapsing the two conditions this exists to tell apart into one.

### 8.75 The deck lit from inside, and the fog that had been deleting it

Over half of all lightning never leaves the cloud; what shows is a region of weather going bright. `flash_on_deck` asks the line
source where each ray crosses the shell, so the glow sits over the part of the deck the channel is in and follows a crawler
along its length.

It could not be seen. A storm stands at **7.5 nepers** to the shell — `fog_density` 2.31 over `FOG_HEIGHT` lifted five times,
half covered — so the deck arrives multiplied by 5e-4. Raising the term fourteenfold moved the frame by four hundredths of a
level, which is what identified the cause: a constant that does nothing is not one that is too small. So it takes `FLASH_HAZE`'s
treatment, composited *in front of* the fog under a capped haze. The deck's own colour stays behind it — you cannot see the
cloud — and only what the lightning lit survives.

Read two mip levels coarser, which is what multiple scattering means (Dobashi and Nishita): at its own level the glow inherited
the painting's hard alpha edge, which is a cloud's silhouette and not a glow's. Both reflections carry it along with the
channel.

### 8.76 Measuring against exposure measures exposure

`ADAPTATION` is 0.75, so a sixfold flash survives as `6^0.25` — 1.6. Three measurements here were built wrong first:

- Differencing a lit frame against a dark one measures the exposure shift. `moved()` passed every precipitation assertion on
  that alone.
- Two shots compare two tone curves: the side that gained most measured least, and the far side came back *negative*. Both
  halves in one wide shot gives 70.3 against -12.9, where a flat wash — the failure being guarded — gives 0.88 of the near side
  against -0.18.
- On water the absolute lift *falls* when the reflection gains the lit deck, 29 against 37. The ratio survives the curve:
  **0.419 with it against 0.183 without**.

A statistic that mixes the effect with the frame's mean brightness is measuring the tonemapper.

### 8.77 A beam and the medium it crosses cannot be reading different layers

`fog_density_at` thins the fog over `FOG_HEIGHT * fog_lift`; the sun's slant depth measured across a bare `FOG_HEIGHT`.
The two disagreed about how deep the air is by whatever the weather's lift was — **eighteenfold in a blizzard**, where
the layer stands highest. Generalising the term to any direction is what surfaced it: while it was `fog_sun_depth` there
was nothing to compare it against.

### 8.78 The moons light the air, and the disc is what lights it

At night the only thing lighting the haze was `frame.fog`, the dome's own colour — and since the disc itself is
extinguished by the weather in front of it, a rainy night drew none of Masser's red: **R/G 0.945 against 2.34 in clear
air**. Not a washed-out moon, a different colour.

- **From the disc's `colour`, not from `light`.** `FULL_RADIANCE` is set by where the tone curve stops keeping colour;
  `light` is what a surface should receive, thirty-two times larger at Masser. Mixing them put a halo brighter than the
  moon inside it and turned the night sky pink.
- **Capped at the source's radiance**, which the geometry gives free: a disc of radiance `L` over `solid` delivers
  `L * solid` head-on, and `moon.colour` is that `L`, so skirt and face agree by construction.
- **Measured to the limb, not the centre.** Draine's peak is a fraction of a degree and Masser is eighteen; read as a
  point it spent itself on the centre texel and drew a red spark in a grey moon.

### 8.79 A moon's face is an image, not an average

Taken whole, single scattering draws a lobe at nearly half the moon's brightness — right for a bright source in water
haze, and on a night sky it read as a lamp behind frosted glass. It could not simply be scaled: the same term is why a
red moon stays red in rain, and dimming everywhere took the hue to 0.98 against 1.33 as it stands.

So face and skirt split. **Small-angle scattering carries a source's image rather than averaging it** — light turned by
less than the moon's own width still arrives from the part of the moon that emitted it — so the face returns the disc as
drawn, and `disc * (1 - T) + disc * T` is a moon behind rain at its own radiance with its own structure. The average had
pasted a flat circle over the portrait, which is the washed-out coin. Off the face, `FOG_MOONLIGHT` takes the skirt down
to 0.15: radial rings out from Masser against a sky of 39 run 51/43 and rejoin at a hundred and ten, against 42 and gone
by a hundred.

### 8.80 A headland covers a moon, and two different things say so

The image term exists only where the ray points *at* the moon, so the ray's own hit is the occlusion test, already paid
for; without it the march painted a whole moon onto a cliff, hard-edged where the disc crossed the skyline. The skirt is
air lit from a direction the ray is not pointing along and needs a shadow ray — one for the pair, aimed at whichever
delivers more, spending at night what the day already spends.

**Read as a hue, because a leak is red and the sky is not:** under a lid the dome is 0.0039 red to 0.0045 green, a leak
0.024 to 0.008. On brightness it hid an order of magnitude under the threshold, the open frame being the disc's blown
core.

**And the two meet at the limb.** The face darkens to nothing there while a step outside it the whole disc contributes
at once — left to disagree, a dark ring between a lit face and its own halo. The skirt is the floor on the face, which
is also the physics: the air over the limb sees all of the moon.

### 8.81 The painted alpha is the silhouette, and everything behind has to agree

`moon_covers` hid the sky geometrically; `moon_disc` draws through the portrait's alpha, which ramps to nothing over the
outermost texels and is what antialiases the limb. The stars were cut off across the band where the moon had already
faded — a **dark ring inside every setting moon**. `covering` now comes out of the call that draws the face, so one
number does both, and the sun's eclipse is partial at the limb rather than binary.

### 8.82 A cloud is a depth, not a coverage

An alpha composites the sky behind it. Run a moon through the same mix and half a cloud leaves half a moon, which is
still the brightest thing in a night sky — so the deck read as ignoring the moons while swallowing the stars. A cloud is
a depth of droplets: `exp(-CLOUD_MOON_DEPTH * hidden)`, eight e-foldings at full thickness, while dome and stars keep
the mix. The layer is therefore read before the moons are drawn and composited after.

### 8.83 Every interior the game ships is fogged, and none of them showed

§8.42 cut `INDOOR_FOG_SCALE` to 0.006, which is not a veil but **nothing**: the median interior came out at a
sixty-seventh of a clear day's air, and a fixture there rendered byte-identical fog on and off. **1,134 interiors carry
a density in `Morrowind.esm`, none zero, range 0.25 to 1.5, mean 0.87** — showing none of that discards the record
rather than reading it. Settled by eye between the Guild of Mages and Abaelun Mine.

The test reads a hall's depth, because across a closet the veil resolves to one 8-bit level, and reads the fog's red
against grey rather than a brightness. **The fixture's density is half a product and moves with it:** 140 left standing
when the scale rose made two tests so opaque that forward and back came out identical, a ratio of NaN.

### 8.84 A coverage that was carrying density, and a dead calm that was a photograph

`FOG_EVEN` is the coverage when fog is even rather than banked, and a half is the midpoint of `FOG_CLEARING` and
`FOG_SOLID` — but those are thresholds on the *noise*, and what the smoothstep between them produces averages **a
third**. The constant exists so indoors and out differ in character rather than in how much air there is, and at a half
every step toward evenness was also a step toward more air.

That mattered once `FOG_STIRRED` was added beside it: `Weather` gives foggy and snow a wind of exactly nought, so their
coverage was the raw threshold — banks with clear gaps, which reads as blotches rather than weather. It lifts the whole
set a quarter toward even, so a rain at 0.3 comes out near a half and an ashstorm at 0.8 barely moves. Indoors is never
banked, so a room's coverage simply *is* `FOG_EVEN`, and `INDOOR_FOG_SCALE` rose by the same ratio to 0.045.

### 8.85 A curve that respaces the weathers must not respace the hour

`DEPTH_CURVE` spreads the ten apart with clear fixed. Applied to the *interpolated* depth it amplified the gap between a
weather's day and night as hard as the gap between weathers: foggy's night is 1.9 times its day in the ini, and a fourth
power made that four times the air, pinned at `DEEPEST` — a blizzard's whiteout for a fog meant to be a little thicker
than noon's. So the curve reads the day depth, the figure the ten are ordered by, and the hour rides on top as the ini's
plain ratio; `stands * hour` is the same product either way and only decides where the power lands.

The exponent came down 4 → 3.5 with it: cloudy and overcast move a fiftieth, rain a fourteenth, foggy, snow and
thunderstorm a sixth, ashstorm and blight a fifth, and a blizzard not at all — 134 either way, with `DEEPEST` always
deciding. Foggy goes from four and a half times a clear day to three and two thirds.

### 8.86 Three seconds of every start was one shader

`primary_visibility.comp` is **172,043 words of SPIR-V** and the driver takes three seconds on it; the other five
modules take one millisecond together. A `VkPipelineCache` on disk takes a fresh process to **72 ms against 3,169**
and a second renderer in-process to 20 against 3,100 — which the suite paid fourteen times over.

- **Keyed by nothing here, because Vulkan keys it already** — entries are indexed by the whole pipeline, so a changed
  shader misses and recompiles. Naming the file by a source hash would key it again, coarser, and throw away every other
  pipeline. A blob from another driver is *safe*: the header carries vendor, device and driver.
- **Written once the pipelines exist, not at teardown** — the tests share a device out of a `static`, which is never
  dropped. The once-per-run latch is set on the write, or a call arriving before anything was compiled would burn it.
- **Through a temporary and a rename**, because a dozen test processes finish into the same half-megabyte file. Last
  writer wins, with a superset of what it started from.

### 8.87 Ash is carried, not dropped, and the ini says nothing about it

Ashstorm and blight carry colours, a fog depth, a wind speed and a `Storm Threshold` — nothing about what is blowing.
What the game ships instead is `meshes/ashcloud.nif`: seven `NiParticleSystemController` emitters on the camera, zero
triangles, 172 vertices, spanning ±1,224 and standing 116 to 1,394. A sprite-era budget in exactly the way `Max
Raindrops`'s 450 is, so the count, spread and height are read off it. **Told from a blizzard by its sheet** — both carry
a threshold, only Bloodmoon's is `tx_bm_` — which is §8.59's argument reused.

**It goes where the air goes.** A mote fine enough to stay up has a terminal velocity of centimetres a second against a
wind of tens of metres, so `Wind Speed` through `GALE` is its speed and gravity gets a twentieth. That is why `velocity`
belongs to `Precipitation`: it is the one thing the three kinds differ in by more than a constant.

**The density is derived and then knowingly missed by three orders.** Ten milligrams of PM10 to the cubic metre, a
twenty-micron grain at 2,500 kg/m³ massing 1.05e-11 kg — **9.6e5 motes/m³**. One to a cube of `spacing` at that density
is 0.71 units on a side, far under what the march resolves, and specks finer than a pixel are what §8.64 and §8.68 are
about: at 2.5 units it was static across the frame, at 10 scattered specks, at **7** a field of motes with the ship
legible through it — 982/m³, a thousandth of real. What stands in for the rest is the weather's own fog, thickest of the
ten after a blizzard. The dust that is not drawn is the dust you are looking through.

**Rock, not crystal.** Snow's albedo is nine tenths, volcanic ash's a seventh — the whole reason a dust storm is a brown
gloom where a blizzard is a white one, and at a flake's brightness the motes were pale grains against their own storm.
Drawn 1.6 times a drop's width, between a flake's 3 and a drop's 1, and moving sixteen metres a second across the view,
so it comes out short and slanted with no case of its own. `precip_snow` became `precip_kind`, **ordered so one
comparison answers the commonest question**: anything above rain does not wet a surface or ring the water.

### 8.88 One sample a pixel, against four thousand

M7's done-when, and the half that had never been measured: stability is not accuracy, and a filter
returning last frame unchanged would pass every other check here. One reference, four candidates.

| | relative RMSE |
|---|---|
| 1 spp as traced | 0.1918 |
| 4 à-trous passes | 0.0300 |
| Ray Reconstruction, first frame | 0.0295 |
| **Ray Reconstruction, settled (DLAA)** | **0.0085** |
| Ray Reconstruction at Performance | 0.0255 |

**Relative, and in units of the frame's own mean.** The frame is scene-referred and unbounded, so a
plain RMSE over linear radiance is decided by whichever handful of pixels holds the ceiling lamp.
Rousselle's ε of 1e-2 taken literally would have been the same mistake wearing the other hat: this
interior's mean radiance is **0.0168**, so the floor would swamp every denominator in the frame and
quietly turn a relative error back into an absolute one. Dividing both frames by the reference's mean
first is what makes the number mean the same thing in a cave and on a beach — §8.76 again.

**The reference is 32 frames of 128 samples, averaged in linear and jittered.** Averaged in linear
because the tone curve is concave where a firefly lands, so the mean of tone-mapped frames is not the
tone-mapped mean. Jittered because Ray Reconstruction resolves sub-pixel detail out of the jitter it
is given, and a reference sampled at pixel centres would charge it for every edge it got *right*.
**Bought in samples rather than in frames**, because every estimator's stream rotates with the frame
index — so frames converge all of them and samples only the bounce, but a frame is read back over the
bus and a sample is not: over 64 frames, 16, 32 and 64 samples give 0.0062, 0.0045 and 0.0034 of
residual for 0.52, 0.57 and 0.63 seconds. The same four thousand at half the frames costs 0.46.

**Splitting the reference by parity measured a half-pixel shift, not noise.** The two halves exist to
say what the reference's own residual is; taken as evens against odds it came back at 0.0248, four
times the truth, because Halton in base 2 is a bit reversal — every odd index lands in the right half
of the pixel and every even one in the left. Consecutive runs each cover the pixel evenly: 0.0062.

**That residual is then taken back out in quadrature**, since two independent errors add that way and
these draw from different streams. The evidence it is sound is that the corrected candidate figures
are *identical* across references of three different lengths — 0.0085 and 0.0256 at 16, 32 and 64
samples a frame, where the uncorrected readings move. It is guarded rather than trusted: the residual must stay under
the best candidate's own error, or the correction is what is doing the measuring.

Two things the table says that a single number would not. **Ray Reconstruction's first frame, with no
history, measures 0.0295 against the à-trous filter's 0.0300** — the two are the same picture, and
everything Ray Reconstruction gains over the filter it replaced is temporal. And **Performance, at a
quarter of the traced pixels, lands level with that filter at full resolution**, which is what §5.3 is
buying with.

### 8.89 A rough lobe is not the diffuse hemisphere, and the floor is where that shows

§5.1's other half starts by noticing that the specular layer is already written. `filmed` — Lazarov's
split-sum environment BRDF, a cone-sampled reflection, the `Film{kept, reflected}` energy split and
the guides Ray Reconstruction reads — is a complete dielectric coat, gated entirely on rain. Making
every surface wear one is a change of gate, not a new light path. `FILM_ROUGHNESS` even turns out to
be the substrate's own roughness under another name: *"a millimetre of water over stone is a
millimetre of water shaped like stone"*, guessed because there was nothing to read it from.

**The gate was load-bearing in a way nothing said.** The reflection is flat-lit — `hit.albedo *
frame.ambient`, no shadow ray, which is what makes it affordable — and that is sound only because
`wetness` has already established a clear line to the sky before it ever runs. Run under a lid it
painted an unshadowed sky onto the underside of stone and doubled what a shaded surface returned.

The repair looked better than the thing it replaced: above a roughness of 0.6 the lobe spans most of
the hemisphere, so read `gather_indirect` — which traced that hemisphere *with occlusion in it* — and
the coat costs no ray at all. **It is the wrong hemisphere.** `gather_indirect` is cosine-weighted
about the *normal*; a specular lobe is centred on the *mirror direction*. Those coincide head on and
diverge as the view goes grazing — which is exactly where the split-sum's Fresnel is largest, 0.051
against 0.025. A floor is seen at a grazing angle, faces the brightest part of the room, and is dark
stone, so it took the maximum Fresnel times the maximum radiance, added flat and uncoloured by its
own albedo: a pale sheet over the one surface that fills the frame.

Two measurements are worth keeping from it. On M8's own fixtures a black albedo went from 0 to **80**
and a 2% albedo from 72 to **99** — a dielectric floor of 2.5–5% exceeds the diffuse return of
anything darker than about 2.5% albedo, which is most of an interior. That is correct physics and it
is also most of §5.1's washed-out failure mode arriving by the front door. And **the game records no
roughness to temper it with**: of 19,415 `NiMaterialProperty` records across the shipped meshes,
19,125 carry a glossiness of **zero**, 12,416 a black specular colour, and the rest exporter
defaults; only 1,020 of 4,456 texture names carry a material word at all, and those name objects
rather than substances.

So the coat waits for the derived map. A uniform 4% sheen over a world with one roughness in it has
no variation to read as material — it reads as a film over the lens, which is what it looked like.

### 8.90 Relief out of the albedo, read as a gradient and given to no upscaler

§5.1's remaining half, built — and built differently than it was planned, because the discriminator
it named is not in the content.

**The height is the log of luminance and the answer is its gradient.** Log because a painted shadow
multiplies the pigment under it, so the ratio carries the shape and a difference of logs is
invariant to how brightly the texture was painted. A gradient rather than an integrated height,
because a normal map need not be integrable and solving for one throws the answer away again. Four
bilinear taps half a texel out on each diagonal, which is a 3×3 Sobel *exactly* — a corner tap
returns the mean of the four texels around it, and differencing those means gives Sobel's own 1-2-1
weights with the averaging done by the sampler. A plain central difference is two taps and visibly
grainier on 256² art. The offsets are a texel of the level being read, held to the levels the
texture has, so relief coarsens with distance the way a normal map's mip chain would.

**The Retinex gate is refused, by measurement.** §5.1 planned to separate painted relief from
pigment on the premise that relief is achromatic. Across fifty shipped textures, between neighbours
both above 5% grey, the mean chromaticity step *rises* monotonically with the luminance step beside
it — **0.0027** where luminance barely moves, **0.059** across the strongest edges. The strongest
edges are the most chromatic, so the gate suppresses exactly the mortar lines and carved mouldings
that carry the shape; rendered, the stonework flattened and noise stood where the relief had been.
Two causes compound: BC1 quantises a block to a segment between two 5-6-5 endpoints, which cannot
darken without leaving grey unless that segment happens to lie along it, and the art shadows with a
cooler pigment rather than with less of the same one.

**No per-texture normalisation either.** Log space already removes exposure, so what is left in the
spread of gradient magnitudes across textures — p90 from 0.6 on carved wood to 2.1 on cave rock — is
genuine: the rock really is rougher. Equalising them would have been the one thing worth
precomputing, and there is nothing to precompute. The slope is one constant, compressed toward its
bound by `tanh` rather than clamped, because a clamp is flat past its knee and renders a strongly
painted edge as one facet with a crease down it.

**The upscaler is guided by the untilted normal, and that is worth more than it sounds.** A guide
normal answers *which surface is this pixel on*, which is what history is reprojected and rejected
against. Relief is detail inside one surface and it is already in the albedo the upscaler has.
Handing over the tilted normal costs Ray Reconstruction most of its temporal accumulation: against
the §8.88 reference, settled DLAA error goes **0.0085 → 0.0126** with the tilted guide and
**0.0085 → 0.0093** with the mesh's own, for the same picture. Nearly the whole reconstruction cost
of the feature was a guide describing texture as geometry. The à-trous filter is indifferent —
0.0300 → 0.0303 — so nothing is smeared by telling it the surface is flat.

**Terrain blends gradients, not heights.** A height is defined only up to the constant each tile was
painted around, so a mix of four steps wherever those constants differ and its derivative is a wall
along every tile boundary; a mix of four derivatives has no such term. Sixteen taps on a ground hit,
which is the cost, and it stays until §5.3 opens.

The measured effect is a 3–6% mean change in radiance on lit surfaces and much more at an edge, off
a base that was flat everywhere. `--relief 0` is the A/B.

### 8.91 Refit wins, and an unbuilt structure is a lost device

M12's first step, measured. Twenty-two placements of a 900-vertex, 1,682-triangle lattice — the
count the busiest cell in the game places and the size the mean skinned mesh is — deformed by a
compute pass into vertex regions of their own and rebuilt inside the frame's own command buffer.
`tests/deforming.rs`, against the real `SceneRenderer`.

| | build | trace |
|---|---|---|
| Rebuild, `PREFER_FAST_BUILD` | **0.242 ms** | 0.085 ms |
| Refit, `PREFER_FAST_TRACE \| ALLOW_UPDATE` | **0.108 ms** | 0.085 ms |
| Nothing moving | 0.000 ms | 0.080 ms |

**Both halves of the prediction were wrong.** Refitting is not a marginal saving over a rebuild at
this scale, it is less than half the cost; and `ALLOW_UPDATE` — which was expected to cost the
traversal, since these are traced by every ray and built once — costs it nothing measurable. Nor
does the tree degrade as the pose leaves the one it was built for: at an amplitude of a third of the
mesh's own size the two trace in 0.087 and 0.088. So refitting is the default, and the switch stays
because the answer is a property of the content rather than of the hardware.

**A top level over an unbuilt bottom level is a lost device, not a wrong picture.** The deforming
structures are created unbuilt — their vertex regions hold whatever the allocator left there until
the first deform writes them — and the commit that follows built a top level referencing them. The
frames after it were all correct, because each rebuilds both levels in order; what faulted was the
*next* load, several renderers later, which made it look like exhaustion. It reproduced only with
`ALLOW_UPDATE` set, which changes the structure size and so the garbage, and that sent an hour into
the refit path before the bisection landed on a build mode that was never the difference. The load
now primes them through one submission of the same two passes the frame uses.

**What the shape of the per-frame path had to be.** Nothing here could go through `Uploader`: every
existing build blocks on a fence, allocates its scratch inline and hands back a *new* structure, and
all three are wrong sixty times a second. So the top level is built **in place** — the descriptor a
pass binds is written per cell, and a new handle every frame would mean rewriting it every frame —
its scratch is allocated once at load, and the barriers are scoped to `ACCELERATION_STRUCTURE_BUILD`
rather than the load path's `ALL_COMMANDS`. `tests/frame_allocations.rs` covers `record`, so the
whole path is inside the zero-allocation budget from the first commit, and it is.

**And a deforming placement needed no new architecture.** It takes a slice of the same position and
attribute buffers a cell uploads into, per instance rather than per mesh, and a mesh slot of its own
pointing at it; every index is already rebased by `first_vertex`, so `surface.glsl` is untouched by
the feature. Indices and submesh descriptions are shared with the pose it was cloned from — a
`MeshRange` names an index range it does not own.

**One fixture bug worth recording, because it looked exactly like a renderer bug.** The lattice was
wound against the normals it carried, so every hit was shaded as the surface's own back and the
scene rendered black under a sun that was working perfectly. It cost the same hour again, chasing a
deformation that was reaching the picture the whole time. `primary_visibility.rs`'s `quad` helper
has carried the winding fix since M3; a fixture that builds its own geometry has to do the same.

### 8.92 The animation the format already carried

M12's second step. One block type stood between the parser and every animation in the game, and
everything behind it was already being read for its width and thrown away — so the work was turning
`cursor.skip` into a read, and then working out what the numbers meant.

**All 7,481 shipped files now parse**, the 7,319 meshes and the 162 `.kf` clips together. `.kf` is a
NIF with `NiSequenceStreamHelper` at its root and nothing else new in it, which is why one arm
unlocked all of them. Retained beside it: keyframe channels, skin instances and their bind
transforms and weights, the controller header, and named string extra data.

**Three measurements decided the rest of the design.**

- **Linear, and the file may say what it likes.** Of 3.44 million rotation keys, **97.9% are linear**
  already; of 692,327 translations, 95.1%. What is left is quadratic or TCB, whose tangents this
  drops. Scale is the exception — 4,224 of its 4,391 keys are quadratic — and 4,391 keys is the whole
  game. **No channel anywhere is `XYZ`**, so the per-axis rotation path is parsed and never taken.
- **Four influences per vertex.** 390,381 vertices follow one bone, 72,629 two, 20,834 three, 3,602
  four and **107 follow five** — two hundredths of one percent, which lose their smallest share to a
  renormalisation. A fifth slot for them would cost every vertex in the world a fifth of its weight
  data. A skin names at most 64 bones, so an index is a byte and four of them are one word.
- **The bind pose is only in `NiSkinData`.** Across all 556 skinned files there is *no* composition
  of the node rest transforms with the inverse binds that comes out the identity — not `world·inv`,
  not `inv·world`, with or without the skin transform or the skeleton root. A NIF's node transforms
  are whatever pose the file was saved in. So a rig is posed as `worldPose(bone) · inverse_bind` and
  the rest transforms are only what a joint with no channel falls back to.

**A skinned mesh does not look like itself until this runs.** `furn_de_banner_pawn_01.nif` stores its
twenty vertices flat in the xy plane at z of zero; the bind transforms stand it upright in xz,
hanging from a root bone at z=63. Six joints, three of them bones, a four-second clip. Every banner
and every sail in the game has been drawn lying flat since M2 and nobody noticed, because a flat
thing seen from the side is a line.

**Rigid pieces are bound as one-bone skins.** A model with a skeleton can hold geometry with no skin
on it, and binding that to the node it hangs from — with that node's own rest transform undone — is
the same arithmetic as a skinned vertex with a single full-weight bone. One path, no special case at
the shader.

**The top level is *updated* per frame, where the bottom levels are rebuilt.** §8.91 measured refit
winning below and expected it to lose above, because a top level is the structure every ray enters.
It does not: over an exterior's worth of statics an update is **0.34 ms against 0.60**, and the
traversal cannot tell — **2.42 ms against 2.44**, inside the run-to-run spread. The instance records
never change between frames; what makes a new build necessary is the bounds of the bottom levels
underneath them, which is what an update is for.

**And the counts are what the survey said.** Seyda Neen's shore cell places two animated references
and Balmora's nine, against 0.34 ms of animation in a frame that traces in 2.4.

### 8.93 A creature's NIF holds everything it can do, in one line

M12's third step, and it turned out to be a smaller thing than the plan expected — with one large
consequence.

**87 of the 89 creature models animate from their own NIF.** Of them, 80 carry a skin, inline
keyframes *and* text keys; 7 are node-animated with no skin, like the banners; 2 carry a skin and no
keys and need their `.kf`. So the `.kf` plumbing the plan put here is not what stood between the
renderer and a moving cast — step 2's path already placed them. What stood in the way was that a NIF
holds **every animation a creature has, laid end to end in one channel**, and it was playing all of
it: an ash ghoul idled, walked, ran, turned, attacked, was knocked down, died, and did it again.

**The boundaries are written as text against the moments they fall on.** 505 files carry
`NiTextKeyExtraData`, and a key's text is several lines of `group: marker` — `idle: start`,
`walkforward: loop stop`. The vocabulary across the shipped content: 790 `idle3`, 780
`weapononehand`, 758 `idle2`, 720 `idle`, 698 `walkforward`, 690 `knockout`, 664 each of
`runforward`, `turnright` and `turnleft`, down through `attack1..3`, `death1`, `hit1`, `blink`,
`talk` and `spellcast`. **`soundgen` is not one of them** — at 7,290 lines it is by far the most
common name in the game, and every one of them is a footfall or a scream rather than a span of
anything; `sound` is another 494.

**An idle is not spelled one way.** `idle3` outnumbers plain `idle`, and a model carrying only the
numbered ones is ordinary, so the choice runs `idle`, `idle1`, `idle2`, `idle3` and falls back to the
model's *first* group rather than to its whole reel. A group with no `loop start` repeats the whole
of itself.

The measure of it: an ash ghoul declares **22 groups**, and its idle is **2.6 seconds of a 44.7
second reel**. `rig.rs`'s test is that assertion plus the one that says which span was picked —
posing the mesh across the chosen span and across `walkforward` and comparing how far it travels,
which separates standing still from walking away without knowing anything about ash ghouls.

**Not done here**: a clip loaded from a `.kf`. The controllers in one hang off a sequence helper
rather than off the nodes they drive, and the targets are named by the string chain beside them —
which is the shape NPCs need, because `base_anim.nif` is a skeleton with no animation in it and
`xbase_anim.kf` is where the animation went.

### 8.94 A person is thirteen files and a skeleton

M12's fourth step. **2,772 of the 3,049 `NPC_` records carry no model at all**, and the 277 that do
name a `base_anim` variant rather than a body — so until now the game's whole human cast was placed
nowhere and drawn as nothing.

**The skeleton already knows where the parts go.** `base_anim.nif` carries 61 nodes, and beside the
`Bip01` chain that drives them sit attachment nodes named for the parts themselves — `Head`, `Neck`,
`Chest`, `Groin`, `Left Hand`, `Left Wrist`, `Left Forearm`, `Left Upper Arm`, `Left Clavicle`,
`Left Foot`, `Left Knee`, `Left Ankle`, `Left Upper Leg`, and the right of each. Every one carries a
placeholder geometry — `Tri Head`, `Tri Chest` — so the mapping from a `BODY` record's part index to
a bone is a lookup rather than a table anyone had to invent.

**And the placeholders are why the shortcut does not exist.** `base_anim.nif`'s geometry carries
flag 5 — bit zero, hidden — where `base_anim_female.nif`'s carries 4 and simply forgot to. So the
male base flattens to *nothing*, the female to a body, and "just place the skeleton" was never an
option. It also means the 273 `CREA` records that name a `base_anim` variant have been placing an
empty mesh since M2: those are humanoid creatures assembled the same way, and they are still
invisible until the same treatment reaches them.

**A `BODY` record has no race field.** Its four-byte `BYDT` gives the part (0–14), a flags byte
whose bit zero is female, and a kind — 533 skin, 331 clothing, 247 armour. Which race a part belongs
to is in its *id*: `b_n_dark elf_f_forearm`. OpenMW resolves it the same way, because there is
nothing else to resolve it by. Coverage across the ten races is complete but for two deliberate gaps
and one accident: **no race has a clavicle skin** (the chest covers it, so a clavicle is only ever
armour), **only Khajiit and Argonians have a tail**, and Argonian women have no forearm.

**The face and the hair are named by the record.** `BNAM` and `KNAM` are `BODY` ids, and across all
2,675 `NPC_` records in the master **every one of them resolves** — no head or hair names a record
that is not there.

**Race decides three things.** Which skeleton — `base_animkna.nif` for Argonians and Khajiit, whose
women are animated as their men because there is no female variant of it — and a height multiplier
per sex, which is what makes a Bosmer at 0.90 shorter than a Nord at 1.06 out of one skeleton.

A male Wood Elf comes out as **3,630 vertices over 125 bones, measuring 36 × 26 × 133 units**, which
is a person. `assembled_actor.rs`'s test is that measurement plus the one underneath it: every
vertex's weights must sum to one, because a vertex that falls short is dragged toward the origin and
one that overshoots is thrown away from it, and either reads as a body coming apart.

**Two allocations a frame, both inside `ash`.** Placing actors put the animation path into the
census office, which is what `tests/frame_allocations.rs` measures — and it went from zero to four
allocations per frame. Two were descriptions rebuilt every frame, now built once per commit. The
other two were `ash`'s own: its safe `cmd_build_acceleration_structures` collects the `&[&[..]]` of
range infos into a `Vec` on every call, so both builds now go through the function pointer with the
pointer array held as scratch. Zero again — and 31% faster with it, the refit path going from 0.108
ms to **0.074**, because describing the build once per commit was worth more than the allocation.

**Not done here**: worn clothing and armour. An NPC's `NPCO` inventory overrides the skin parts it
covers, which is why the woman above is in her underwear.

### 8.95 One arm is the other one reflected

Reported from the window: an NPC's limbs on one side turn the wrong way. Two bugs behind it, and
one of them was not the one being reported.

**Morrowind's body parts are authored for the right side.** One file is both arms — `ActorPart`
hands the same mesh to `Left Forearm` and to `Right Forearm` — and what makes one of them a left arm
is a reflection along the bone's own length, applied between the bone and the mesh. OpenMW does
exactly this and by exactly this test: `components/sceneutil/attach.cpp:166` looks for `Left` in the
attachment's name and scales that subtree by `(-1, 1, 1)`.

**Measured rather than taken on trust**, because a limb segment is a tube and a wrongly reflected
one still looks like a tube. The skeleton carries its own placeholder for every part — `Tri Left
Upper Arm` sits under the node `Left Upper Arm` — so the game's own data says where a part belongs.
Against it, an upper arm fits its **left** socket at 3.87 units mean nearest-neighbour mirrored
against 4.49 as-is, and its **right** socket at 3.87 as-is against 4.49 mirrored. The foot and the
forearm cannot tell the difference at all — 3.95 either way — which is why the arm is what settled
it.

**A reflection turns a triangle inside out**, and the shading normal is chosen by the triangle's own
plane. The winding of a reflected part is reversed where it is stored, so the plane comes back out
the right way once the reflection is applied; the normal needs nothing, because for a matrix that is
its own inverse and symmetric the plain transform *is* the inverse transpose. The consequence is
that an assembled body is deliberately wound against its own normals wherever a part went left,
which is why `the_shipped_meshes_wind_their_triangles_to_agree_with_their_normals` now measures only
meshes that came out of a file.

**And the second bug, found on the way.** `b_n_breton_f_hand` and `b_n_breton_f_hand.1st` both match
the naming convention a body is chosen by, and the first-person one is often first in the file — so
a Breton woman was wearing the pair of arms the player sees down their own sleeve. `.1st` is now
excluded.

**A skinned file is the whole of what it covers.** `B_N_Dark Elf_F_Skins.nif` holds both hands *and*
the torso, and the `hand` record and the `chest` record both name it; so a file that binds by its own
bone names is added once however many parts point at it and however many sides they ask for. That
had been adding it twice, which is what made the first assembled bodies look padded — 3,630 vertices
over 125 bones against the 2,194 over 55 they actually are.
