//! The names an animation carries at the moments they apply to.

use crate::cursor::Cursor;
use crate::error::Result;

/// One moment, and whatever the artist wrote against it.
///
/// **The text is not one thing.** A single key can hold several lines, and a line is anything from
/// an animation group's boundary — `idle: start`, `walkforward: loop stop` — to a sound to play or a
/// note to nobody. Reading it is the caller's business; this only carries it.
#[derive(Debug, Clone)]
pub struct TextKey {
    pub time: f32,
    pub text: String,
}

/// Every text key on one animation, in the order they occur.
#[derive(Debug, Clone, Default)]
pub struct TextKeys {
    pub keys: Vec<TextKey>,
}

impl TextKeys {
    pub(crate) fn read(cursor: &mut Cursor<'_>) -> Result<Self> {
        // The next extra-data block, which nothing follows: a model's text keys are reached from
        // the node that owns them.
        cursor.skip(4)?;
        cursor.skip(4)?; // Byte count, recomputed from the keys themselves.
        let count = cursor.u32()? as usize;
        let mut keys = Vec::with_capacity(count);
        for _ in 0..count {
            keys.push(TextKey {
                time: cursor.f32()?,
                text: cursor.string()?,
            });
        }
        Ok(Self { keys })
    }
}
