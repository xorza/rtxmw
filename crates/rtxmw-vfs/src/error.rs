//! Errors raised while indexing or reading from the virtual file system.

use std::path::PathBuf;

/// Anything that can go wrong opening an archive or reading a file out of one.
#[derive(Debug)]
pub enum VfsError {
    /// The underlying file system refused a read.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// An archive's own structure is inconsistent. Archives are untrusted input: a truncated or
    /// hand-edited BSA must be reported, never trusted into an out-of-bounds read.
    Malformed { archive: PathBuf, reason: String },
    /// No archive in the index holds this path.
    NotFound(String),
}

impl VfsError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn malformed(archive: impl Into<PathBuf>, reason: impl Into<String>) -> Self {
        Self::Malformed {
            archive: archive.into(),
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for VfsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Malformed { archive, reason } => {
                write!(f, "{} is not a valid archive: {reason}", archive.display())
            }
            Self::NotFound(path) => write!(f, "no such file in the virtual file system: {path}"),
        }
    }
}

impl std::error::Error for VfsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Result alias for virtual file system operations.
pub type Result<T> = std::result::Result<T, VfsError>;
