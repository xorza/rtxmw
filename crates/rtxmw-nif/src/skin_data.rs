//! How one skinned mesh's vertices follow the bones that move it.

use crate::block::Transform;
use crate::cursor::Cursor;
use crate::error::Result;

/// Bytes a stored bounding sphere occupies: a centre and a radius.
const BOUNDING_SPHERE_BYTES: usize = 16;

/// Reads a transform in the order a skin block writes one.
///
/// **Rotation first, where a scene-graph object writes its translation first.** The two orders are
/// the same forty-four bytes and a transform read in the wrong one still looks like a transform,
/// which is why this is a function of its own rather than a method on [`Transform`] that either
/// caller could reach for.
fn bind_transform(cursor: &mut Cursor<'_>) -> Result<Transform> {
    Ok(Transform {
        rotation: cursor.matrix3()?,
        translation: cursor.vec3()?,
        scale: cursor.f32()?,
    })
}

/// One vertex's share of one bone.
#[derive(Debug, Clone, Copy)]
pub struct VertexWeight {
    pub vertex: u16,
    pub weight: f32,
}

/// One bone's bind pose, and the run of weights it owns.
///
/// **Morrowind stores skinning per bone, not per vertex**: each bone names the vertices it moves
/// and by how much, and a vertex appears once in every bone's list that touches it. Turning that
/// round is the loader's job, not the format's.
#[derive(Debug, Clone)]
pub struct BoneSkin {
    /// Skin space to this bone's space — what a posed bone is measured against.
    pub inverse_bind: Transform,
    /// Range into [`SkinData::weights`].
    pub weights: std::ops::Range<u32>,
}

/// The whole of what a `NiSkinData` block says.
///
/// One weight list with a range table beside it rather than a list per bone: a bone owns a run of
/// it and nothing is pushed to after the parse.
#[derive(Debug, Clone, Default)]
pub struct SkinData {
    /// Skin space to the skeleton root's space.
    pub transform: Transform,
    pub bones: Vec<BoneSkin>,
    pub weights: Vec<VertexWeight>,
}

impl SkinData {
    pub(crate) fn read(cursor: &mut Cursor<'_>) -> Result<Self> {
        let transform = bind_transform(cursor)?;
        let count = cursor.u32()? as usize;
        // The partition a later NIF version splits the weights into, which this one does not write.
        cursor.skip(4)?;
        let mut bones = Vec::with_capacity(count);
        let mut weights: Vec<VertexWeight> = Vec::new();
        for _ in 0..count {
            let inverse_bind = bind_transform(cursor)?;
            cursor.skip(BOUNDING_SPHERE_BYTES)?;
            let owned = cursor.u16()? as usize;
            let first = weights.len() as u32;
            weights.reserve(owned);
            for _ in 0..owned {
                weights.push(VertexWeight {
                    vertex: cursor.u16()?,
                    weight: cursor.f32()?,
                });
            }
            bones.push(BoneSkin {
                inverse_bind,
                weights: first..weights.len() as u32,
            });
        }
        Ok(Self {
            transform,
            bones,
            weights,
        })
    }
}
