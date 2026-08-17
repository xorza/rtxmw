//! A directory of loose files, indexed like an archive.

use std::path::{Path, PathBuf};

use crate::error::{Result, VfsError};
use crate::normalized_path::NormalizedPath;

/// A loose-file directory tree.
///
/// The on-disk path is kept alongside the normalized one: the index is case-folded but Linux file
/// systems are not, so the original spelling is the only way back to the file.
#[derive(Debug)]
pub struct DirectoryArchive {
    root: PathBuf,
    entries: Vec<DirectoryEntry>,
}

#[derive(Debug)]
struct DirectoryEntry {
    normalized: NormalizedPath,
    on_disk: PathBuf,
}

impl DirectoryArchive {
    /// Walks `root` recursively and indexes every file beneath it.
    pub fn open(root: &Path) -> Result<Self> {
        let mut entries = Vec::new();
        Self::walk(root, root, &mut entries)?;
        // Deterministic order regardless of what the file system hands back, so a duplicate that
        // differs only in case resolves the same way on every run.
        entries.sort_by(|a, b| a.normalized.cmp(&b.normalized));
        Ok(Self {
            root: root.to_path_buf(),
            entries,
        })
    }

    fn walk(root: &Path, directory: &Path, out: &mut Vec<DirectoryEntry>) -> Result<()> {
        let listing = std::fs::read_dir(directory).map_err(|e| VfsError::io(directory, e))?;
        for entry in listing {
            let entry = entry.map_err(|e| VfsError::io(directory, e))?;
            let path = entry.path();
            let kind = entry.file_type().map_err(|e| VfsError::io(&path, e))?;

            if kind.is_dir() {
                Self::walk(root, &path, out)?;
                continue;
            }
            // Symlinks to files are followed; `is_dir` above already covers directory links.
            let relative = path.strip_prefix(root).unwrap_or(&path);
            let Some(relative) = relative.to_str() else {
                // A non-UTF-8 name cannot be a Morrowind asset reference, so it can never be
                // looked up. Skipping is friendlier than failing the whole scan.
                continue;
            };
            out.push(DirectoryEntry {
                normalized: NormalizedPath::new(relative),
                on_disk: path,
            });
        }
        Ok(())
    }

    /// Number of files found.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the directory holds no files.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The directory that was indexed.
    pub fn path(&self) -> &Path {
        &self.root
    }

    /// The normalized name of entry `index`.
    pub fn entry_name(&self, index: usize) -> &str {
        self.entries[index].normalized.as_str()
    }

    /// Reads entry `index` in full.
    pub fn read(&self, index: usize) -> Result<Vec<u8>> {
        let path = &self.entries[index].on_disk;
        std::fs::read(path).map_err(|e| VfsError::io(path, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a throwaway tree and returns its root.
    fn tree(tag: &str, files: &[&str]) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "rtxmw-dir-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        for file in files {
            let path = root.join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, file.as_bytes()).unwrap();
        }
        root
    }

    #[test]
    fn indexes_nested_files_with_normalized_names() {
        let root = tree("nested", &["Meshes/A.NIF", "meshes/sub/B.nif", "top.txt"]);
        let archive = DirectoryArchive::open(&root).unwrap();

        assert_eq!(archive.len(), 3);
        let names: Vec<_> = (0..archive.len()).map(|i| archive.entry_name(i)).collect();
        // Sorted and normalized.
        assert_eq!(names, ["meshes/a.nif", "meshes/sub/b.nif", "top.txt"]);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn reads_contents_through_the_original_on_disk_spelling() {
        let root = tree("case", &["Meshes/A.NIF"]);
        let archive = DirectoryArchive::open(&root).unwrap();

        // Indexed lowercase, but the file on disk is not — reading must still find it.
        assert_eq!(archive.entry_name(0), "meshes/a.nif");
        assert_eq!(archive.read(0).unwrap(), b"Meshes/A.NIF");

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn an_empty_directory_indexes_to_nothing() {
        let root = tree("empty", &[]);
        std::fs::create_dir_all(&root).unwrap();
        let archive = DirectoryArchive::open(&root).unwrap();

        assert!(archive.is_empty());
        assert_eq!(archive.len(), 0);

        std::fs::remove_dir_all(&root).ok();
    }
}
