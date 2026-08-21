//! What slides a texture across the surface it is painted on.

use crate::cursor::{Cursor, Link};
use crate::error::Result;
use crate::keyframe_data::{FloatKey, Track};

/// A `NiUVController`: the clock, and the channels it drives.
///
/// **What makes Vivec's water run and Red Mountain's lava crawl.** Neither is animated geometry —
/// both are a flat sheet with a texture walked across it, which is the whole of how the original
/// engine drew a moving fluid. Fifty of the game's files carry one.
#[derive(Debug, Clone)]
pub struct UvController {
    /// Bit one is whether the clip repeats, which every shipped one does — a scroll that stopped
    /// at the end of its span would leave a waterfall frozen after a few seconds.
    pub flags: u16,
    /// How fast the clip runs and where in it this controller starts.
    pub frequency: f32,
    pub phase: f32,
    /// The span of the clip, in seconds. A scroll's channels are keyed across exactly this.
    pub start: f32,
    pub stop: f32,
    /// Which of the geometry's texture coordinate sets it moves. Morrowind's meshes have one.
    pub uv_set: u16,
    /// The `NiUVData` holding the four channels.
    pub data: Link,
}

impl UvController {
    /// Reads one, consuming exactly its bytes.
    pub(crate) fn read(cursor: &mut Cursor<'_>) -> Result<Self> {
        cursor.skip(4)?; // Next controller in the chain.
        let flags = cursor.u16()?;
        let controller = Self {
            flags,
            frequency: cursor.f32()?,
            phase: cursor.f32()?,
            start: cursor.f32()?,
            stop: cursor.f32()?,
            // The target node, which a controller is reached *from* rather than through — see
            // `TimeController`.
            uv_set: {
                cursor.skip(4)?;
                cursor.u16()?
            },
            data: cursor.link()?,
        };
        Ok(controller)
    }
}

/// A `NiUVData`: where the texture sits, and how many times it repeats, over the clip.
///
/// **Four channels, and only two of them ever move.** Nothing the game ships keys the tiling; what
/// it keys is the offset, and almost always as a straight ramp from zero to one across the clip,
/// which with the controller looping is a scroll that never stops.
#[derive(Debug, Clone, Default)]
pub struct UvData {
    /// Where the texture's origin sits, along each axis.
    pub offset_u: Track<FloatKey>,
    pub offset_v: Track<FloatKey>,
    /// How many times it repeats across the surface, along each axis.
    pub tiling_u: Track<FloatKey>,
    pub tiling_v: Track<FloatKey>,
}

impl UvData {
    /// Reads one, consuming exactly its bytes.
    pub(crate) fn read(cursor: &mut Cursor<'_>) -> Result<Self> {
        Ok(Self {
            offset_u: Track::read(cursor, false)?,
            offset_v: Track::read(cursor, false)?,
            tiling_u: Track::read(cursor, false)?,
            tiling_v: Track::read(cursor, false)?,
        })
    }

    /// How far the texture travels along each axis in one second, or zero where it does not move.
    ///
    /// **A rate rather than a curve, which is what the data supports.** A channel keyed as a
    /// straight line from one value to another over a looping clip *is* a constant speed, and
    /// reproducing it needs the speed and nothing else — no keys uploaded, no lookup per pixel, and
    /// no seam where the clip wraps. A channel keyed any other way is not a scroll and comes back
    /// as no motion rather than as a guess.
    pub fn scroll(&self, span: f32) -> [f32; 2] {
        let rate = |track: &Track<FloatKey>| match track.keys.as_slice() {
            [first, .., last] if span > 0.0 => (last.value - first.value) / span,
            _ => 0.0,
        };
        [rate(&self.offset_u), rate(&self.offset_v)]
    }
}

#[cfg(test)]
mod tests {
    use crate::keyframe_data::Interpolation;

    use super::*;

    /// A channel with the given keys, linear, as every shipped one is.
    fn keyed(values: &[(f32, f32)]) -> Track<FloatKey> {
        Track {
            interpolation: Interpolation::Linear,
            keys: values
                .iter()
                .map(|&(time, value)| FloatKey { time, value })
                .collect(),
        }
    }

    #[test]
    fn a_ramp_over_a_clip_is_a_speed_and_a_wobble_is_not() {
        // **`in_lava_256` writes its V channel as nought to two over forty-eight seconds**, which
        // is one twenty-fourth of the texture a second — the crawl every pool of lava in the game
        // has. Nothing is keyed but the offsets and nothing is keyed as anything but a line, so the
        // rate is the whole of what has to be carried.
        let lava = UvData {
            offset_v: keyed(&[(0.0, 0.0), (48.0, 2.0)]),
            ..UvData::default()
        };
        assert_eq!(lava.scroll(48.0), [0.0, 2.0 / 48.0]);

        // **A channel that comes back to where it started is a wobble, not a scroll**, and it has
        // to read as no motion rather than as a guess at one. Ghostgate's fences shiver their U
        // across four keys and drift their V across two, and only the second is a speed.
        let fence = UvData {
            offset_u: keyed(&[
                (0.0, 0.0),
                (2.0, 0.05),
                (4.0, 0.0),
                (6.0, -0.05),
                (8.0, 0.0),
            ]),
            offset_v: keyed(&[(0.0, 0.0), (8.0, 2.0)]),
            ..UvData::default()
        };
        assert_eq!(fence.scroll(8.0), [0.0, 0.25]);

        // An empty channel and a clip with no length are both no motion rather than a division.
        assert_eq!(UvData::default().scroll(8.0), [0.0, 0.0]);
        assert_eq!(lava.scroll(0.0), [0.0, 0.0]);
    }
}
