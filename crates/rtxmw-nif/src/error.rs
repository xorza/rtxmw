//! Errors raised while reading a NIF.

/// Anything that can go wrong parsing a NIF.
///
/// Morrowind-era blocks carry no size, so an unrecognised block type is fatal rather than
/// skippable: without knowing how many bytes it occupies there is no way to find the next one.
#[derive(Debug)]
pub enum NifError {
    /// The file does not begin with a recognised version line.
    NotANif,
    /// The file is a NIF, but from a version this reader does not handle.
    UnsupportedVersion { version: u32 },
    /// A read ran past the end of the file.
    UnexpectedEnd {
        offset: usize,
        wanted: usize,
        available: usize,
    },
    /// A block type that has no parser. Fatal — see the type-level note.
    UnknownBlock {
        index: usize,
        kind: String,
        offset: usize,
    },
    /// A block parsed, but left the cursor somewhere impossible.
    Malformed {
        index: usize,
        kind: String,
        reason: String,
    },
    /// A field held a value outside its enumeration. The enclosing block is attached separately.
    BadValue { what: &'static str, value: u32 },
    /// A root index points outside the block table.
    BadRoot { link: i32 },
    /// A failure inside a specific block. Blocks carry no size, so naming the one that went wrong
    /// is the only way to tell a bad parser from a bad file.
    InBlock {
        index: usize,
        kind: String,
        source: Box<NifError>,
    },
}

impl std::fmt::Display for NifError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotANif => write!(f, "not a NIF: missing the version line"),
            Self::UnsupportedVersion { version } => {
                write!(f, "unsupported NIF version {version:#010x}")
            }
            Self::UnexpectedEnd {
                offset,
                wanted,
                available,
            } => write!(
                f,
                "read at offset {offset} wants {wanted} bytes but only {available} remain"
            ),
            Self::UnknownBlock {
                index,
                kind,
                offset,
            } => write!(
                f,
                "block {index} at offset {offset} has type {kind:?}, which has no parser; \
                 Morrowind blocks carry no size, so parsing cannot continue past it"
            ),
            Self::Malformed {
                index,
                kind,
                reason,
            } => write!(f, "block {index} ({kind}): {reason}"),
            Self::BadValue { what, value } => write!(f, "unknown {what} {value}"),
            Self::BadRoot { link } => {
                write!(f, "root points at {link}, which is not a block")
            }
            Self::InBlock {
                index,
                kind,
                source,
            } => write!(f, "block {index} ({kind}): {source}"),
        }
    }
}

impl std::error::Error for NifError {}

/// Result alias for NIF parsing.
pub type Result<T> = std::result::Result<T, NifError>;
