//! What a keyframe controller reads: one node's movement, as three channels of keys.

use crate::cursor::Cursor;
use crate::error::{NifError, Result};

/// How the values between two keys are read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Interpolation {
    /// No keys at all, which is what an empty channel declares.
    #[default]
    None,
    Linear,
    /// Cubic, with an in and an out tangent stored beside every value — except on a quaternion
    /// channel, which stores none.
    Quadratic,
    /// Tension, continuity and bias stored beside every value.
    Tcb,
    /// A rotation channel written as three float channels instead — see [`KeyframeData::axes`].
    Xyz,
    /// Hold the previous value until the next key.
    Constant,
}

impl Interpolation {
    fn read(cursor: &mut Cursor<'_>) -> Result<Self> {
        let value = cursor.u32()?;
        Ok(match value {
            1 => Self::Linear,
            2 => Self::Quadratic,
            3 => Self::Tcb,
            4 => Self::Xyz,
            5 => Self::Constant,
            other => {
                return Err(NifError::BadValue {
                    what: "interpolation type",
                    value: other,
                });
            }
        })
    }
}

/// One keyed rotation. Stored `w` first, which is not the order `glam` takes.
#[derive(Debug, Clone, Copy)]
pub struct QuaternionKey {
    pub time: f32,
    pub value: [f32; 4],
}

/// One keyed translation.
#[derive(Debug, Clone, Copy)]
pub struct VectorKey {
    pub time: f32,
    pub value: [f32; 3],
}

/// One keyed scalar: a scale, or one axis of an [`Interpolation::Xyz`] rotation.
#[derive(Debug, Clone, Copy)]
pub struct FloatKey {
    pub time: f32,
    pub value: f32,
}

/// One channel of an animation: its keys, and how to read between them.
#[derive(Debug, Clone)]
pub struct Track<K> {
    pub interpolation: Interpolation,
    pub keys: Vec<K>,
}

// Written out rather than derived, which would demand a `Default` of the key type for no reason:
// an empty channel is empty whatever it would have carried.
impl<K> Default for Track<K> {
    fn default() -> Self {
        Self {
            interpolation: Interpolation::None,
            keys: Vec::new(),
        }
    }
}

impl<K: Keyed> Track<K> {
    /// Reads a channel, tangents and all.
    ///
    /// **An empty channel omits its interpolation type entirely** — except inside a morph, where it
    /// is always written and may name a type that means nothing with no keys to read between.
    fn read(cursor: &mut Cursor<'_>, is_morph: bool) -> Result<Self> {
        let count = cursor.u32()? as usize;
        if count == 0 && !is_morph {
            return Ok(Self::default());
        }
        let interpolation = Interpolation::read(cursor)?;
        if count == 0 {
            return Ok(Self {
                interpolation,
                keys: Vec::new(),
            });
        }
        // The caller reads the per-axis channels itself; nothing follows here.
        if interpolation == Interpolation::Xyz {
            return Ok(Self {
                interpolation,
                keys: Vec::new(),
            });
        }
        let mut keys = Vec::with_capacity(count);
        for _ in 0..count {
            let time = cursor.f32()?;
            let value = K::read_value(cursor)?;
            match interpolation {
                // **Tangents are consumed and dropped.** Reading between keys is done linearly —
                // see `docs/design.md` §8.92 for the measurement that says what that costs.
                Interpolation::Quadratic if K::HAS_TANGENTS => {
                    K::read_value(cursor)?;
                    K::read_value(cursor)?;
                }
                Interpolation::Tcb => {
                    cursor.skip(12)?;
                }
                _ => {}
            }
            keys.push(K::key(time, value));
        }
        Ok(Self {
            interpolation,
            keys,
        })
    }

    /// Whether this channel says anything about the node it belongs to.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// A value a channel can carry.
///
/// A trait rather than three copies of the reader: the three channels differ only in how wide a
/// value is and whether quadratic interpolation writes tangents beside it.
pub trait Keyed: Sized {
    /// Whether tangents are stored under quadratic interpolation. Quaternions never carry them.
    const HAS_TANGENTS: bool;
    /// The stored value, without its time.
    type Value: Copy;
    fn read_value(cursor: &mut Cursor<'_>) -> Result<Self::Value>;
    fn key(time: f32, value: Self::Value) -> Self;
}

impl Keyed for QuaternionKey {
    const HAS_TANGENTS: bool = false;
    type Value = [f32; 4];
    fn read_value(cursor: &mut Cursor<'_>) -> Result<[f32; 4]> {
        Ok([cursor.f32()?, cursor.f32()?, cursor.f32()?, cursor.f32()?])
    }
    fn key(time: f32, value: [f32; 4]) -> Self {
        Self { time, value }
    }
}

impl Keyed for VectorKey {
    const HAS_TANGENTS: bool = true;
    type Value = [f32; 3];
    fn read_value(cursor: &mut Cursor<'_>) -> Result<[f32; 3]> {
        cursor.vec3()
    }
    fn key(time: f32, value: [f32; 3]) -> Self {
        Self { time, value }
    }
}

impl Keyed for FloatKey {
    const HAS_TANGENTS: bool = true;
    type Value = f32;
    fn read_value(cursor: &mut Cursor<'_>) -> Result<f32> {
        cursor.f32()
    }
    fn key(time: f32, value: f32) -> Self {
        Self { time, value }
    }
}

/// One node's animation: where it points, where it stands and how large it is, over time.
#[derive(Debug, Clone, Default)]
pub struct KeyframeData {
    pub rotations: Track<QuaternionKey>,
    /// Three float channels standing in for the rotation, where it was written as
    /// [`Interpolation::Xyz`]. Empty otherwise; never any other length.
    pub axes: Vec<Track<FloatKey>>,
    pub translations: Track<VectorKey>,
    pub scales: Track<FloatKey>,
}

impl KeyframeData {
    pub(crate) fn read(cursor: &mut Cursor<'_>) -> Result<Self> {
        let rotations = Track::read(cursor, false)?;
        let mut axes = Vec::new();
        if rotations.interpolation == Interpolation::Xyz {
            // Which axis is which, which nothing here needs: the three channels follow in order.
            cursor.skip(4)?;
            axes.reserve_exact(3);
            for _ in 0..3 {
                axes.push(Track::read(cursor, false)?);
            }
        }
        Ok(Self {
            rotations,
            axes,
            translations: Track::read(cursor, false)?,
            scales: Track::read(cursor, false)?,
        })
    }

    /// The last moment any channel says anything about, which is how long the animation runs.
    pub fn duration(&self) -> f32 {
        let rotation = self.rotations.keys.last().map_or(0.0, |key| key.time);
        let axes = self
            .axes
            .iter()
            .filter_map(|axis| axis.keys.last())
            .fold(0.0f32, |end, key| end.max(key.time));
        let translation = self.translations.keys.last().map_or(0.0, |key| key.time);
        let scale = self.scales.keys.last().map_or(0.0, |key| key.time);
        rotation.max(axes).max(translation).max(scale)
    }
}
