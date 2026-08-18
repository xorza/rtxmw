//! Errors raised while reading an ESM/ESP content file.

/// Anything that can go wrong parsing a content file.
///
/// Content files are untrusted input — a truncated or hand-edited plugin must be reported rather
/// than trusted into an out-of-bounds read.
#[derive(Debug)]
pub enum EsmError {
    /// The file does not begin with a `TES3` record.
    NotAContentFile,
    /// A record or subrecord claims more bytes than remain.
    Truncated {
        /// Byte offset the overrunning header started at.
        offset: usize,
        /// What it asked for.
        wanted: usize,
        /// What was left.
        available: usize,
    },
    /// A record or subrecord header was cut short.
    UnexpectedEnd { offset: usize },
    /// A fixed-size subrecord was not the size its type requires.
    BadSubrecordSize {
        name: crate::record_name::RecordName,
        wanted: usize,
        found: usize,
    },
}

impl std::fmt::Display for EsmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAContentFile => write!(f, "file does not start with a TES3 record"),
            Self::Truncated {
                offset,
                wanted,
                available,
            } => write!(
                f,
                "record at offset {offset} wants {wanted} bytes but only {available} remain"
            ),
            Self::UnexpectedEnd { offset } => {
                write!(f, "file ends mid-header at offset {offset}")
            }
            Self::BadSubrecordSize {
                name,
                wanted,
                found,
            } => write!(
                f,
                "subrecord {name} should be {wanted} bytes, found {found}"
            ),
        }
    }
}

impl std::error::Error for EsmError {}

/// Result alias for content-file parsing.
pub type Result<T> = std::result::Result<T, EsmError>;
