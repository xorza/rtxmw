//! Placement of a reference in world space.

/// A world position and rotation, as stored in a `DATA` subrecord.
///
/// Rotations are radians, applied about the negated axes in Z, Y, X order — the convention the
/// original engine used. Translating that into a matrix is the renderer's job, not this crate's.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Position {
    /// World units: +X east, +Y north, +Z up.
    pub translation: [f32; 3],
    /// Radians about X, Y, Z.
    pub rotation: [f32; 3],
}

impl Position {
    /// Bytes a `DATA` subrecord carrying a position occupies.
    pub const SIZE: usize = 24;

    /// Reads six little-endian floats, or `None` if the payload is the wrong width.
    pub fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < Self::SIZE {
            return None;
        }
        let float = |at: usize| f32::from_le_bytes(data[at..at + 4].try_into().unwrap());
        Some(Self {
            translation: [float(0), float(4), float(8)],
            rotation: [float(12), float(16), float(20)],
        })
    }
}
