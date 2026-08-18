//! Errors raised while assembling a scene.

use rtxmw_esm::EsmError;
use rtxmw_nif::NifError;
use rtxmw_vfs::VfsError;

/// Anything that can go wrong turning content files into placed geometry.
#[derive(Debug)]
pub enum SceneError {
    /// The named cell is not in the content file.
    NoSuchCell(String),
    /// A content file could not be read.
    Esm(EsmError),
    /// A mesh could not be read out of the virtual file system.
    Vfs(VfsError),
    /// A mesh failed to parse.
    Nif { path: String, source: NifError },
}

impl std::fmt::Display for SceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchCell(name) => write!(f, "no cell named {name:?}"),
            Self::Esm(e) => write!(f, "content file: {e}"),
            Self::Vfs(e) => write!(f, "virtual file system: {e}"),
            Self::Nif { path, source } => write!(f, "{path}: {source}"),
        }
    }
}

impl std::error::Error for SceneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Esm(e) => Some(e),
            Self::Vfs(e) => Some(e),
            Self::Nif { source, .. } => Some(source),
            Self::NoSuchCell(_) => None,
        }
    }
}

impl From<EsmError> for SceneError {
    fn from(value: EsmError) -> Self {
        Self::Esm(value)
    }
}

impl From<VfsError> for SceneError {
    fn from(value: VfsError) -> Self {
        Self::Vfs(value)
    }
}

/// Result alias for scene assembly.
pub type Result<T> = std::result::Result<T, SceneError>;
