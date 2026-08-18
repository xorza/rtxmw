//! Four-character record and subrecord tags.

/// The four-byte tag that opens every record and subrecord, such as `CELL` or `FRMR`.
///
/// Kept as raw bytes rather than a string: tags are compared constantly during traversal, and a
/// four-byte integer comparison is both faster and immune to encoding questions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RecordName(pub [u8; 4]);

impl RecordName {
    /// The tag written in a source file, e.g. `RecordName::new(b"CELL")`.
    pub const fn new(tag: &[u8; 4]) -> Self {
        Self(*tag)
    }

    /// The tag as text. Always ASCII in practice; non-ASCII bytes render as `?`.
    pub fn as_str(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.0)
    }
}

impl std::fmt::Display for RecordName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_str())
    }
}
