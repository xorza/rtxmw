//! The block types a Morrowind NIF can contain.
//!
//! Every type here must consume exactly its own bytes. Blocks in this file version carry no size,
//! so a parser that reads one field too few or too many shifts every block after it — the failure
//! surfaces far from its cause, usually as a nonsensical block type name.

use crate::cursor::{Cursor, Link};
use crate::error::{NifError, Result};
use crate::keyframe_data::{ColourKey, KeyframeData, Track};
use crate::nif_file::version;
use crate::particle_modifier::ParticleModifier;
use crate::particle_system::ParticleSystem;
use crate::skin_data::SkinData;
use crate::skin_instance::SkinInstance;
use crate::text_keys::TextKeys;
use crate::time_controller::{ControllerKind, TimeController};
use crate::uv_controller::{UvController, UvData};

/// Bytes a bounding sphere occupies: a centre and a radius.
const BOUNDING_SPHERE_BYTES: usize = 16;

/// Fields every named object carries.
#[derive(Debug, Clone, Default)]
pub struct ObjectNet {
    pub name: String,
    pub extra_data: Link,
    pub controller: Link,
}

impl ObjectNet {
    fn read(cursor: &mut Cursor<'_>, _version: u32) -> Result<Self> {
        Ok(Self {
            name: cursor.string()?,
            extra_data: cursor.link()?,
            controller: cursor.link()?,
        })
    }
}

/// A node's local transform, as stored.
///
/// The rotation is row-major and must be transposed for a column-vector convention — the single
/// most common way to get a NIF importer subtly wrong.
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub translation: [f32; 3],
    pub rotation: [[f32; 3]; 3],
    pub scale: f32,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            scale: 1.0,
        }
    }
}

/// Fields every scene-graph object carries.
#[derive(Debug, Clone, Default)]
pub struct AvObject {
    pub net: ObjectNet,
    pub flags: u16,
    pub transform: Transform,
    pub properties: Vec<Link>,
}

impl AvObject {
    /// Set on a node that should not be drawn.
    pub const FLAG_HIDDEN: u16 = 0x1;

    fn read(cursor: &mut Cursor<'_>, ver: u32) -> Result<Self> {
        let net = ObjectNet::read(cursor, ver)?;
        let flags = cursor.u16()?;
        let transform = Transform {
            translation: cursor.vec3()?,
            rotation: cursor.matrix3()?,
            scale: cursor.f32()?,
        };
        // Velocity, present only through 4.2.2.0 and unused.
        if ver <= version(4, 2, 2, 0) {
            cursor.skip(12)?;
        }
        let count = cursor.u32()? as usize;
        let properties = cursor.links(count)?;
        if ver <= version(4, 2, 2, 0) && cursor.bool(ver)? {
            read_bounding_volume(cursor)?;
        }
        Ok(Self {
            net,
            flags,
            transform,
            properties,
        })
    }

    /// Whether the node is marked invisible.
    pub fn is_hidden(&self) -> bool {
        self.flags & Self::FLAG_HIDDEN != 0
    }
}

/// Consumes a bounding volume without keeping it; the renderer computes its own bounds.
fn read_bounding_volume(cursor: &mut Cursor<'_>) -> Result<()> {
    let kind = cursor.u32()?;
    match kind {
        // Base: no payload.
        0xFFFF_FFFF => {}
        // Sphere: centre and radius.
        0 => cursor.skip(BOUNDING_SPHERE_BYTES)?,
        // Box: centre, three axes, extents.
        1 => cursor.skip(60)?,
        // Capsule: centre, axis, extent, radius.
        2 => cursor.skip(32)?,
        // Lozenge: radius, extents, centre, two axes.
        3 => cursor.skip(56)?,
        // Union: a count followed by that many nested volumes.
        4 => {
            let count = cursor.u32()? as usize;
            for _ in 0..count {
                read_bounding_volume(cursor)?;
            }
        }
        // Half space: plane and centre.
        5 => cursor.skip(28)?,
        other => {
            return Err(NifError::BadValue {
                what: "bounding volume type",
                value: other,
            });
        }
    }
    Ok(())
}

/// Fields shared by every controller: the chain link, timing, and the node it drives.
fn read_time_controller(cursor: &mut Cursor<'_>) -> Result<()> {
    cursor.skip(4)?; // Next controller in the chain.
    cursor.skip(2)?; // Flags.
    cursor.skip(4 * 4)?; // Frequency, phase, start and stop times.
    cursor.skip(4)?; // Target node.
    Ok(())
}

/// Fields shared by every extra-data block at this version.
fn read_extra(cursor: &mut Cursor<'_>) -> Result<()> {
    cursor.skip(4)?; // Next extra-data block.
    cursor.skip(4)?; // Byte count, recomputed from the payload itself.
    Ok(())
}

/// How the values between two animation keys are interpolated.
mod interpolation {
    pub(super) const LINEAR: u32 = 1;
    pub(super) const QUADRATIC: u32 = 2;
    pub(super) const TCB: u32 = 3;
    pub(super) const XYZ: u32 = 4;
    pub(super) const CONSTANT: u32 = 5;
}

/// Consumes a list of animation keys, returning the interpolation type it declared.
///
/// `value_bytes` is the width of one keyed value. Quaternion channels never store tangents even
/// under quadratic interpolation, which is why `has_tangents` is separate from the type.
fn read_key_list(
    cursor: &mut Cursor<'_>,
    value_bytes: usize,
    has_tangents: bool,
    is_morph: bool,
) -> Result<u32> {
    let count = cursor.u32()? as usize;
    // An empty list omits the interpolation type entirely — except in a morph, where it is always
    // written.
    if count == 0 && !is_morph {
        return Ok(0);
    }
    let interpolation = cursor.u32()?;
    // A morph always writes the type even with no keys, and then it may be `Unknown`. With nothing
    // to interpolate that is legal, so the type is only meaningful once there are keys.
    if count == 0 {
        return Ok(interpolation);
    }
    let per_key = match interpolation {
        interpolation::LINEAR | interpolation::CONSTANT => 4 + value_bytes,
        interpolation::QUADRATIC => {
            4 + value_bytes + if has_tangents { value_bytes * 2 } else { 0 }
        }
        // Tension, bias and continuity follow the value.
        interpolation::TCB => 4 + value_bytes + 12,
        // The caller reads the per-axis lists itself.
        interpolation::XYZ => 0,
        other => {
            return Err(NifError::BadValue {
                what: "interpolation type",
                value: other,
            });
        }
    };
    cursor.skip(count * per_key)?;
    Ok(interpolation)
}

/// A grouping node.
#[derive(Debug, Clone, Default)]
pub struct Node {
    pub av: AvObject,
    pub children: Vec<Link>,
    pub effects: Vec<Link>,
}

/// Vertex data shared by every geometry type.
#[derive(Debug, Clone, Default)]
pub struct GeometryData {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    /// One entry per UV set; Morrowind meshes use zero or one.
    pub uv_sets: Vec<Vec<[f32; 2]>>,
    pub triangles: Vec<[u16; 3]>,
    /// Declared vertex count, which particle data needs even when the positions were omitted.
    pub(crate) vertex_hint: usize,
}

impl GeometryData {
    /// Reads the fields common to all geometry, then the triangle count.
    fn read_base(cursor: &mut Cursor<'_>, ver: u32) -> Result<Self> {
        let vertex_count = cursor.u16()? as usize;

        let mut data = Self {
            vertex_hint: vertex_count,
            ..Default::default()
        };
        if cursor.bool(ver)? {
            data.vertices.reserve_exact(vertex_count);
            for _ in 0..vertex_count {
                data.vertices.push(cursor.vec3()?);
            }
        }
        if cursor.bool(ver)? {
            data.normals.reserve_exact(vertex_count);
            for _ in 0..vertex_count {
                data.normals.push(cursor.vec3()?);
            }
        }
        cursor.skip(BOUNDING_SPHERE_BYTES)?;
        if cursor.bool(ver)? {
            data.colors.reserve_exact(vertex_count);
            for _ in 0..vertex_count {
                data.colors.push(cursor.color4()?);
            }
        }

        // At 4.0.0.2 this whole field is the UV set count; later versions narrow it to a mask.
        let mut uv_count = cursor.u16()? as usize;
        if !cursor.bool(ver)? {
            uv_count = 0;
        }
        data.uv_sets.reserve_exact(uv_count);
        for _ in 0..uv_count {
            let mut set = Vec::with_capacity(vertex_count);
            for _ in 0..vertex_count {
                set.push(cursor.vec2()?);
            }
            data.uv_sets.push(set);
        }

        Ok(data)
    }

    /// Triangle-based geometry adds a triangle count; particle data does not.
    fn read_tri_based(cursor: &mut Cursor<'_>, ver: u32) -> Result<Self> {
        let data = Self::read_base(cursor, ver)?;
        cursor.skip(2)?; // Triangle count, re-derived from the index list below.
        Ok(data)
    }

    fn read_tri_shape(cursor: &mut Cursor<'_>, ver: u32) -> Result<Self> {
        let mut data = Self::read_tri_based(cursor, ver)?;
        let index_count = cursor.u32()? as usize;
        let indices = cursor.u16s(index_count)?;
        data.triangles = indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| [c[0], c[1], c[2]])
            .collect();

        // Match groups: vertices that share a position, used by the original tools only.
        let groups = cursor.u16()? as usize;
        for _ in 0..groups {
            let count = cursor.u16()? as usize;
            cursor.skip(count * 2)?;
        }
        Ok(data)
    }

    fn read_tri_strips(cursor: &mut Cursor<'_>, ver: u32) -> Result<Self> {
        let mut data = Self::read_tri_based(cursor, ver)?;
        let strip_count = cursor.u16()? as usize;
        let lengths = cursor.u16s(strip_count)?;
        for length in lengths {
            let strip = cursor.u16s(length as usize)?;
            data.triangles.extend(triangles_from_strip(&strip));
        }
        Ok(data)
    }

    /// Total triangles, after strips are expanded.
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }
}

/// Expands a triangle strip into independent triangles.
///
/// Winding alternates along the strip, and degenerate triangles — which strips use to stitch
/// separate pieces together — are dropped rather than emitted as zero-area geometry.
fn triangles_from_strip(strip: &[u16]) -> Vec<[u16; 3]> {
    let mut out = Vec::new();
    for (offset, window) in strip.windows(3).enumerate() {
        let (a, b, c) = (window[0], window[1], window[2]);
        if a == b || b == c || a == c {
            continue;
        }
        out.push(if offset % 2 == 0 {
            [a, b, c]
        } else {
            [a, c, b]
        });
    }
    out
}

/// A drawable piece of geometry.
#[derive(Debug, Clone, Default)]
pub struct Geometry {
    pub av: AvObject,
    pub data: Link,
    pub skin: Link,
}

/// One texture slot of a texturing property.
#[derive(Debug, Clone, Default)]
pub struct TextureSlot {
    pub source: Link,
    pub uv_set: u32,
}

/// Which textures a geometry uses.
#[derive(Debug, Clone, Default)]
pub struct TexturingProperty {
    pub base: Option<TextureSlot>,
}

/// Surface colours and shininess.
#[derive(Debug, Clone)]
pub struct MaterialProperty {
    pub ambient: [f32; 3],
    pub diffuse: [f32; 3],
    pub specular: [f32; 3],
    pub emissive: [f32; 3],
    pub glossiness: f32,
    pub alpha: f32,
}

/// How a surface blends.
#[derive(Debug, Clone, Copy)]
pub struct AlphaProperty {
    pub flags: u16,
    pub threshold: u8,
}

impl AlphaProperty {
    /// Alpha blending is enabled.
    pub fn blends(&self) -> bool {
        self.flags & 0x1 != 0
    }

    /// Alpha testing is enabled.
    pub fn tests(&self) -> bool {
        self.flags & 0x200 != 0
    }

    /// Whether what is drawn *adds* to the frame rather than covering it.
    ///
    /// Bits five to eight are the destination factor, and zero there is `ONE` — the frame kept
    /// whole and the surface piled on top. That is what makes a flame a flame rather than a hole:
    /// nothing behind it is taken away, so it needs no sorting and no occlusion of its own.
    pub fn adds(&self) -> bool {
        self.blends() && (self.flags >> 5) & 0xF == 0
    }
}

/// Where a texture's pixels come from.
#[derive(Debug, Clone, Default)]
pub struct SourceTexture {
    /// Path relative to `textures/`, as stored. Empty when the pixels are embedded.
    pub file_name: String,
    pub external: bool,
}

/// A parsed block.
///
/// `Other` covers types that parse correctly but carry nothing the renderer needs — keeping them
/// distinct from an unknown type, which is fatal.
#[derive(Debug, Clone)]
pub enum Block {
    Node(Node),
    Geometry(Geometry),
    GeometryData(GeometryData),
    Texturing(TexturingProperty),
    Material(MaterialProperty),
    Alpha(AlphaProperty),
    SourceTexture(SourceTexture),
    Controller(TimeController),
    Keyframe(KeyframeData),
    Skin(SkinInstance),
    SkinData(SkinData),
    /// The node a particle system draws through: its transform, and the texture and blend mode
    /// every particle takes. Carries no triangles — the sprites are the drawing.
    Particles(Geometry),
    /// An emitter: a flame, a steam vent, a plume of smoke.
    ParticleSystem(ParticleSystem),
    /// One link of an emitter's modifier chain.
    ParticleModifier(ParticleModifier),
    /// A colour keyed over a particle's life — see [`ColourKey`].
    Colour(Track<ColourKey>),
    /// What slides a texture across the surface it is painted on.
    UvController(UvController),
    /// The channels it slides it along.
    UvData(UvData),
    /// A named string hung off a node — in a `.kf` file, the name of the node a controller drives.
    StringExtra(String),
    /// What an animation is called at each moment of it — see [`TextKeys`].
    TextKeys(TextKeys),
    /// A `.kf` file's root, whose two chains are the whole of the animation it carries.
    Sequence(ObjectNet),
    /// Parsed correctly, but carries nothing a renderer needs.
    ///
    /// The name is a `&'static str` rather than nothing: it allocates nothing and it is what makes
    /// a `Debug`-printed block list legible, which is how a desync gets diagnosed.
    Other {
        kind: &'static str,
    },
}

impl Block {
    /// Parses the block named `kind`, consuming exactly its bytes.
    pub fn read(kind: &str, cursor: &mut Cursor<'_>, ver: u32) -> Result<Option<Self>> {
        let block = match kind {
            "NiNode" | "NiBSAnimationNode" | "NiBSParticleNode" | "AvoidNode"
            | "RootCollisionNode" | "NiCollisionSwitch" | "NiBillboardNode" => {
                let av = AvObject::read(cursor, ver)?;
                let child_count = cursor.u32()? as usize;
                let children = cursor.links(child_count)?;
                let effect_count = cursor.u32()? as usize;
                let effects = cursor.links(effect_count)?;
                Self::Node(Node {
                    av,
                    children,
                    effects,
                })
            }
            "NiTriShape" | "NiTriStrips" => {
                let av = AvObject::read(cursor, ver)?;
                let data = cursor.link()?;
                let skin = cursor.link()?;
                Self::Geometry(Geometry { av, data, skin })
            }
            "NiTriShapeData" => Self::GeometryData(GeometryData::read_tri_shape(cursor, ver)?),
            "NiTriStripsData" => Self::GeometryData(GeometryData::read_tri_strips(cursor, ver)?),
            "NiTexturingProperty" => {
                let _net = ObjectNet::read(cursor, ver)?;
                cursor.skip(2)?; // Property flags.
                cursor.skip(4)?; // Apply mode.
                let slot_count = cursor.u32()?;
                let mut property = TexturingProperty::default();
                for slot in 0..slot_count {
                    if !cursor.bool(ver)? {
                        continue;
                    }
                    let source = cursor.link()?;
                    cursor.skip(4)?; // Clamp mode.
                    cursor.skip(4)?; // Filter mode.
                    let uv_set = cursor.u32()?;
                    cursor.skip(4)?; // PS2 filtering settings, present through 10.4.0.1.
                    cursor.skip(2)?; // Unknown short, present through 4.1.0.12.
                    if slot == 0 {
                        property.base = Some(TextureSlot { source, uv_set });
                    }
                    // The bump-map slot adds a luma bias and a 2x2 matrix.
                    if slot == 5 {
                        cursor.skip(4 + 16)?;
                    }
                }
                Self::Texturing(property)
            }
            "NiMaterialProperty" => {
                let _net = ObjectNet::read(cursor, ver)?;
                cursor.skip(2)?; // Property flags.
                Self::Material(MaterialProperty {
                    ambient: cursor.color3()?,
                    diffuse: cursor.color3()?,
                    specular: cursor.color3()?,
                    emissive: cursor.color3()?,
                    glossiness: cursor.f32()?,
                    alpha: cursor.f32()?,
                })
            }
            "NiAlphaProperty" => {
                let _net = ObjectNet::read(cursor, ver)?;
                Self::Alpha(AlphaProperty {
                    flags: cursor.u16()?,
                    threshold: cursor.u8()?,
                })
            }
            "NiSourceTexture" => {
                let _net = ObjectNet::read(cursor, ver)?;
                // A `char` in the format, so one byte — not the four-byte `bool` used elsewhere
                // in this version. Reading it as a bool over-consumes by three and desyncs.
                let external = cursor.u8()? != 0;
                let mut texture = SourceTexture {
                    external,
                    ..Default::default()
                };
                // Internal textures precede their pixel data with a "has data" byte.
                let has_data = if external { false } else { cursor.u8()? != 0 };
                if external {
                    texture.file_name = cursor.string()?;
                }
                if has_data {
                    cursor.skip(4)?; // Link to the embedded NiPixelData.
                }
                // Pixel layout, mipmap format, alpha format, then the one-byte "is static" flag.
                cursor.skip(4 * 3)?;
                cursor.skip(1)?;
                Self::SourceTexture(texture)
            }
            // Properties with a fixed tail after the shared header.
            "NiZBufferProperty" => {
                let _net = ObjectNet::read(cursor, ver)?;
                cursor.skip(2)?;
                Self::Other {
                    kind: "NiZBufferProperty",
                }
            }
            "NiVertexColorProperty" => {
                let _net = ObjectNet::read(cursor, ver)?;
                cursor.skip(2 + 4 + 4)?;
                Self::Other {
                    kind: "NiVertexColorProperty",
                }
            }
            "NiStencilProperty" => {
                let _net = ObjectNet::read(cursor, ver)?;
                // Flags, then a **one-byte** enabled flag — a `char` in the format, not the
                // four-byte `bool` this version uses elsewhere — and then seven words: the test
                // function, the reference and mask, the fail, z-fail and pass actions, and the
                // draw mode. `components/nif/property.cpp:539`.
                cursor.skip(2 + 1 + 4 * 7)?;
                Self::Other {
                    kind: "NiStencilProperty",
                }
            }
            "NiShadeProperty"
            | "NiSpecularProperty"
            | "NiWireframeProperty"
            | "NiDitherProperty" => {
                let _net = ObjectNet::read(cursor, ver)?;
                cursor.skip(2)?;
                Self::Other {
                    kind: "flag property",
                }
            }
            // Controllers. None of them drive anything yet, but each has to be sized exactly.
            "NiKeyframeController"
            | "NiVisController"
            | "NiAlphaController"
            | "NiFlipController"
            | "NiMaterialColorController"
            | "NiRollController" => {
                let controller = match kind {
                    "NiKeyframeController" => ControllerKind::Keyframe,
                    "NiVisController" => ControllerKind::Visibility,
                    "NiAlphaController" => ControllerKind::Alpha,
                    "NiFlipController" => ControllerKind::Flip,
                    "NiMaterialColorController" => ControllerKind::MaterialColour,
                    _ => ControllerKind::Roll,
                };
                Self::Controller(TimeController::read(cursor, controller)?)
            }
            "NiUVController" => Self::UvController(UvController::read(cursor)?),
            "NiGeomMorpherController" => {
                read_time_controller(cursor)?;
                cursor.skip(4)?; // Data block.
                cursor.skip(1)?; // Always-active flag.
                Self::Other {
                    kind: "NiGeomMorpherController",
                }
            }
            "NiSkinInstance" => Self::Skin(SkinInstance::read(cursor)?),
            "NiTextureEffect" => {
                let _av = AvObject::read(cursor, ver)?;
                // NiDynamicEffect's affected-node list.
                let affected = cursor.u32()? as usize;
                cursor.skip(affected * 4)?;
                cursor.skip(36 + 12)?; // Projection rotation and position.
                cursor.skip(4 * 4)?; // Filter, clamp, texture type, coord-gen type.
                cursor.skip(4)?; // Source texture.
                // A `uint8_t` in the format, not the four-byte `bool` used for other flags.
                cursor.skip(1)?; // Clip plane enabled.
                cursor.skip(16)?; // Clip plane.
                cursor.skip(4)?; // PS2 shorts.
                cursor.skip(2)?; // Unknown short, present through 4.1.0.12.
                Self::Other {
                    kind: "NiTextureEffect",
                }
            }
            "NiCamera" => {
                let _av = AvObject::read(cursor, ver)?;
                cursor.skip(4 * 6)?; // Frustum: left, right, top, bottom, near, far.
                cursor.skip(4 * 5)?; // Viewport plus the LOD adjust.
                cursor.skip(4)?; // Scene node.
                cursor.skip(4)?; // Unused.
                Self::Other { kind: "NiCamera" }
            }
            "NiVertWeightsExtraData" => {
                read_extra(cursor)?;
                let count = cursor.u16()? as usize;
                cursor.skip(count * 4)?;
                Self::Other {
                    kind: "NiVertWeightsExtraData",
                }
            }
            // Particle systems are geometry with extra per-particle arrays.
            "NiParticles" | "NiAutoNormalParticles" | "NiRotatingParticles" => {
                let av = AvObject::read(cursor, ver)?;
                let data = cursor.link()?;
                let skin = cursor.link()?;
                Self::Particles(Geometry { av, data, skin })
            }
            "NiParticlesData" | "NiAutoNormalParticlesData" | "NiRotatingParticlesData" => {
                let data = GeometryData::read_base(cursor, ver)?;
                cursor.skip(2)?; // Particle count.
                cursor.skip(4)?; // A single radius at this version.
                cursor.skip(2)?; // Active count.
                if cursor.bool(ver)? {
                    cursor.skip(data.vertex_hint * 4)?; // Per-particle sizes.
                }
                if kind == "NiRotatingParticlesData" && cursor.bool(ver)? {
                    cursor.skip(data.vertex_hint * 16)?; // Per-particle rotations.
                }
                Self::GeometryData(data)
            }
            "NiKeyframeData" => Self::Keyframe(KeyframeData::read(cursor)?),
            "NiFloatData" => {
                read_key_list(cursor, 4, true, false)?;
                Self::Other {
                    kind: "NiFloatData",
                }
            }
            // Four channels: U and V offset, U and V tiling.
            "NiUVData" => Self::UvData(UvData::read(cursor)?),
            "NiVisData" => {
                // Not a key list: each entry is a time and a one-byte visibility flag.
                let count = cursor.u32()? as usize;
                cursor.skip(count * 5)?;
                Self::Other { kind: "NiVisData" }
            }
            "NiPosData" => {
                read_key_list(cursor, 12, true, false)?;
                Self::Other { kind: "NiPosData" }
            }
            // Kept rather than sized, because it is what fades a puff of smoke: every one of the
            // game's alpha-blended emitters carries one and none of its additive ones do.
            "NiColorData" => Self::Colour(Track::<ColourKey>::read(cursor, false)?),
            "NiMorphData" => {
                let morphs = cursor.u32()? as usize;
                let vertices = cursor.u32()? as usize;
                cursor.skip(1)?; // Relative-targets flag.
                for _ in 0..morphs {
                    read_key_list(cursor, 4, true, true)?;
                    cursor.skip(vertices * 12)?;
                }
                Self::Other {
                    kind: "NiMorphData",
                }
            }
            "NiSkinData" => Self::SkinData(SkinData::read(cursor)?),
            // `NiBSPArrayController` is the same record under a different name.
            "NiParticleSystemController" | "NiBSPArrayController" => {
                Self::ParticleSystem(ParticleSystem::read(cursor)?)
            }
            "NiPathController" => {
                read_time_controller(cursor)?;
                cursor.skip(4 * 3)?; // Bank direction, max bank angle, smoothing.
                cursor.skip(2)?; // Follow axis.
                cursor.skip(4 + 4)?; // Path and percentage data.
                Self::Other {
                    kind: "NiPathController",
                }
            }
            "NiPixelData" => {
                cursor.skip(4)?; // Pixel format.
                cursor.skip(16)?; // Colour masks.
                cursor.skip(4)?; // Bits per pixel.
                cursor.skip(8)?; // Compare bits.
                cursor.skip(4)?; // Palette.
                let mipmaps = cursor.u32()? as usize;
                cursor.skip(4)?; // Bytes per pixel.
                cursor.skip(mipmaps * 12)?; // Width, height and offset per level.
                let pixels = cursor.u32()? as usize;
                cursor.skip(pixels)?;
                Self::Other {
                    kind: "NiPixelData",
                }
            }
            "NiGravity" => Self::ParticleModifier(ParticleModifier::read_gravity(cursor)?),
            "NiParticleColorModifier" => {
                Self::ParticleModifier(ParticleModifier::read_colour(cursor)?)
            }
            "NiParticleGrowFade" => {
                Self::ParticleModifier(ParticleModifier::read_grow_fade(cursor)?)
            }
            "NiParticleRotation" => {
                Self::ParticleModifier(ParticleModifier::read_rotation(cursor)?)
            }
            "NiPlanarCollider" => {
                cursor.skip(4 + 4)?; // The chain link and controller every modifier begins with.
                cursor.skip(4)?; // Bounce factor.
                cursor.skip(8)?; // Extents.
                cursor.skip(12 * 4)?; // Position, X and Y vectors, plane normal.
                cursor.skip(4)?; // Plane distance.
                Self::Other {
                    kind: "NiPlanarCollider",
                }
            }
            "NiStringExtraData" => {
                // The next block in the chain, which nothing follows: a `.kf` file's strings are
                // walked from the sequence helper's own extra-data link in order.
                let _next = cursor.link()?;
                cursor.skip(4)?; // Byte count, recomputed from the string itself.
                Self::StringExtra(cursor.string()?)
            }
            // A `.kf` file's root. Everything it holds hangs off the `ObjectNet` links it carries:
            // the strings naming targets on one, the controllers driving them on the other.
            "NiSequenceStreamHelper" => Self::Sequence(ObjectNet::read(cursor, ver)?),
            "NiTextKeyExtraData" => Self::TextKeys(TextKeys::read(cursor)?),
            _ => return Ok(None),
        };
        Ok(Some(block))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_strip_expands_with_alternating_winding() {
        // 0-1-2-3 is two triangles; the second must be wound the other way.
        let triangles = triangles_from_strip(&[0, 1, 2, 3]);
        assert_eq!(triangles, vec![[0, 1, 2], [1, 3, 2]]);
    }

    #[test]
    fn degenerate_strip_triangles_are_dropped() {
        // Repeated indices are how strips stitch separate pieces together; they enclose no area.
        let triangles = triangles_from_strip(&[0, 1, 1, 2, 3]);
        for triangle in &triangles {
            assert!(
                triangle[0] != triangle[1]
                    && triangle[1] != triangle[2]
                    && triangle[0] != triangle[2],
                "degenerate triangle {triangle:?} survived"
            );
        }
        assert_eq!(triangles, vec![[1, 2, 3]]);
    }

    #[test]
    fn a_strip_shorter_than_three_indices_yields_nothing() {
        assert!(triangles_from_strip(&[]).is_empty());
        assert!(triangles_from_strip(&[0, 1]).is_empty());
    }

    /// One property block's bytes: a `NiObjectNET` header naming nothing, then `tail` zero bytes.
    ///
    /// The header is what every property starts with — a length-prefixed name and two links — so a
    /// block that reads its own tail correctly consumes exactly this and stops.
    fn property_bytes(tail: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; 4];
        bytes.extend([0u8; 8]);
        bytes.extend(std::iter::repeat_n(0u8, tail));
        bytes
    }

    #[test]
    fn every_fixed_size_property_consumes_exactly_its_own_bytes() {
        // **Blocks in this version carry no size**, so a property that reads one field too few
        // leaves the file pointing into its own tail and every block after it decodes as garbage.
        // Nothing downstream notices — the next block type is read from a table, not from the
        // stream — which is why this is asserted per block rather than left to the corpus.
        //
        // Each tail is spelled out from `components/nif/property.cpp`:
        let properties = [
            // Flags alone. The test function these gained is 4.1.0.12 and later.
            ("NiZBufferProperty", 2),
            ("NiShadeProperty", 2),
            ("NiSpecularProperty", 2),
            ("NiWireframeProperty", 2),
            ("NiDitherProperty", 2),
            // Flags, then the vertex and lighting modes.
            ("NiVertexColorProperty", 2 + 4 + 4),
            // Flags, a byte for enabled, then seven words.
            ("NiStencilProperty", 2 + 1 + 4 * 7),
            // Flags and a one-byte threshold.
            ("NiAlphaProperty", 2 + 1),
        ];
        let version = crate::nif_file::version(4, 0, 0, 2);
        for (kind, tail) in properties {
            let bytes = property_bytes(tail);
            let mut cursor = Cursor::new(&bytes);
            Block::read(kind, &mut cursor, version)
                .unwrap_or_else(|e| panic!("{kind} should parse: {e}"))
                .unwrap_or_else(|| panic!("{kind} should be a known block"));
            assert_eq!(
                cursor.remaining(),
                0,
                "{kind} left {} of its {tail} tail bytes unread",
                cursor.remaining()
            );

            // And it must not be satisfied by fewer: one byte short has to fail rather than
            // quietly stop, which is the half of this a length check alone would miss.
            let short = property_bytes(tail - 1);
            let mut cursor = Cursor::new(&short);
            assert!(
                Block::read(kind, &mut cursor, version).is_err(),
                "{kind} read a block one byte shorter than the format allows"
            );
        }
    }

    #[test]
    fn alpha_flags_decode_blending_and_testing_independently() {
        let blending = AlphaProperty {
            flags: 0x1,
            threshold: 0,
        };
        assert!(blending.blends() && !blending.tests());

        let testing = AlphaProperty {
            flags: 0x200,
            threshold: 128,
        };
        assert!(!testing.blends() && testing.tests());

        let both = AlphaProperty {
            flags: 0x201,
            threshold: 128,
        };
        assert!(both.blends() && both.tests());
    }
}
