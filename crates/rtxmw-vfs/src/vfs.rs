//! The layered virtual file system.

use std::collections::HashMap;
use std::path::Path;

use crate::bsa_archive::BsaArchive;
use crate::directory_archive::DirectoryArchive;
use crate::error::{Result, VfsError};
use crate::normalized_path::NormalizedPath;

/// One indexed source of files.
#[derive(Debug)]
enum ArchiveSource {
    Bsa(BsaArchive),
    Directory(DirectoryArchive),
}

impl ArchiveSource {
    fn len(&self) -> usize {
        match self {
            Self::Bsa(a) => a.len(),
            Self::Directory(a) => a.len(),
        }
    }

    fn entry_name(&self, index: usize) -> &str {
        match self {
            Self::Bsa(a) => a.entry_name(index),
            Self::Directory(a) => a.entry_name(index),
        }
    }

    fn read(&self, index: usize) -> Result<Vec<u8>> {
        match self {
            Self::Bsa(a) => a.read(index),
            Self::Directory(a) => a.read(index),
        }
    }

    fn path(&self) -> &Path {
        match self {
            Self::Bsa(a) => a.path(),
            Self::Directory(a) => a.path(),
        }
    }
}

/// Where a path resolves to.
#[derive(Debug, Clone, Copy)]
struct FileLocation {
    source: u32,
    entry: u32,
}

/// A flat, case-insensitive view over layered archives and loose directories.
///
/// Sources are added in priority order, lowest first: a path present in more than one source
/// resolves to the **last** source that provided it. Morrowind's own load order works this way, and
/// it is what lets a loose file in `Data Files/` override the same path inside `Morrowind.bsa` —
/// which is how essentially every texture replacer works.
#[derive(Debug, Default)]
pub struct Vfs {
    sources: Vec<ArchiveSource>,
    index: HashMap<NormalizedPath, FileLocation>,
}

impl Vfs {
    /// An empty file system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a BSA, overriding paths from sources added earlier.
    pub fn add_bsa(&mut self, path: &Path) -> Result<()> {
        let archive = BsaArchive::open(path)?;
        self.push(ArchiveSource::Bsa(archive));
        Ok(())
    }

    /// Adds a loose-file directory, overriding paths from sources added earlier.
    pub fn add_directory(&mut self, path: &Path) -> Result<()> {
        let archive = DirectoryArchive::open(path)?;
        self.push(ArchiveSource::Directory(archive));
        Ok(())
    }

    fn push(&mut self, source: ArchiveSource) {
        let source_index = self.sources.len() as u32;
        self.index.reserve(source.len());
        for entry in 0..source.len() {
            // Archives normalize while indexing, so this only wraps.
            let path = NormalizedPath::from_normalized(source.entry_name(entry));
            self.index.insert(
                path,
                FileLocation {
                    source: source_index,
                    entry: entry as u32,
                },
            );
        }
        self.sources.push(source);
    }

    /// Number of distinct paths visible, after overrides.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether nothing is indexed.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Whether `path` resolves, in any case or separator spelling.
    pub fn contains(&self, path: &str) -> bool {
        self.index.contains_key(&NormalizedPath::new(path))
    }

    /// Reads `path` in full.
    pub fn read(&self, path: &str) -> Result<Vec<u8>> {
        let key = NormalizedPath::new(path);
        let location = self
            .index
            .get(&key)
            .ok_or_else(|| VfsError::NotFound(key.as_str().to_owned()))?;
        self.sources[location.source as usize].read(location.entry as usize)
    }

    /// The archive or directory `path` currently resolves to, for diagnosing override order.
    pub fn source_of(&self, path: &str) -> Option<&Path> {
        let location = self.index.get(&NormalizedPath::new(path))?;
        Some(self.sources[location.source as usize].path())
    }

    /// Every visible path, in unspecified order.
    pub fn paths(&self) -> impl Iterator<Item = &NormalizedPath> {
        self.index.keys()
    }

    /// Visible paths beneath `directory`, in unspecified order.
    pub fn paths_under(&self, directory: &str) -> impl Iterator<Item = &NormalizedPath> {
        let prefix = NormalizedPath::new(directory);
        self.index
            .keys()
            .filter(move |p| p.starts_with_directory(&prefix))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsa_archive::internals::{build_bsa, write_temp};

    #[test]
    fn resolves_regardless_of_case_or_separator() {
        let bsa = write_temp(&build_bsa(&[(r"Meshes\A.NIF", b"data")]), "vfs-case");
        let mut vfs = Vfs::new();
        vfs.add_bsa(&bsa).unwrap();

        assert!(vfs.contains(r"meshes\a.nif"));
        assert!(vfs.contains("MESHES/A.NIF"));
        assert!(vfs.contains("/meshes//a.nif"));
        assert_eq!(vfs.read(r"Meshes\A.NIF").unwrap(), b"data");

        std::fs::remove_file(&bsa).ok();
    }

    #[test]
    fn a_later_archive_overrides_an_earlier_one() {
        let first = write_temp(&build_bsa(&[("shared.txt", b"first")]), "vfs-first");
        let second = write_temp(
            &build_bsa(&[("shared.txt", b"second"), ("only-second.txt", b"x")]),
            "vfs-second",
        );

        let mut vfs = Vfs::new();
        vfs.add_bsa(&first).unwrap();
        vfs.add_bsa(&second).unwrap();

        // Two archives, three entries, but only two distinct paths.
        assert_eq!(vfs.len(), 2);
        assert_eq!(vfs.read("shared.txt").unwrap(), b"second");
        assert_eq!(vfs.source_of("shared.txt"), Some(second.as_path()));

        std::fs::remove_file(&first).ok();
        std::fs::remove_file(&second).ok();
    }

    #[test]
    fn a_loose_file_overrides_an_archive_when_added_after_it() {
        let bsa = write_temp(&build_bsa(&[(r"textures\t.dds", b"packed")]), "vfs-loose");

        let root = std::env::temp_dir().join(format!("rtxmw-vfs-loose-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("textures")).unwrap();
        std::fs::write(root.join("textures/T.DDS"), b"loose").unwrap();

        let mut vfs = Vfs::new();
        vfs.add_bsa(&bsa).unwrap();
        vfs.add_directory(&root).unwrap();

        assert_eq!(vfs.read("textures/t.dds").unwrap(), b"loose");
        assert_eq!(vfs.source_of("textures/t.dds"), Some(root.as_path()));

        // Reversing the order reverses the winner — the rule is call order, not source kind.
        let mut reversed = Vfs::new();
        reversed.add_directory(&root).unwrap();
        reversed.add_bsa(&bsa).unwrap();
        assert_eq!(reversed.read("textures/t.dds").unwrap(), b"packed");

        std::fs::remove_file(&bsa).ok();
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn missing_paths_report_the_normalized_name() {
        let vfs = Vfs::new();
        assert!(vfs.is_empty());

        let error = vfs
            .read(r"Meshes\Missing.NIF")
            .expect_err("should not resolve");
        match error {
            VfsError::NotFound(path) => assert_eq!(path, "meshes/missing.nif"),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn directory_listing_filters_to_whole_components() {
        let bsa = write_temp(
            &build_bsa(&[
                (r"meshes\a.nif", b"a"),
                (r"meshes\sub\b.nif", b"b"),
                (r"meshesx\c.nif", b"c"),
                (r"textures\d.dds", b"d"),
            ]),
            "vfs-under",
        );
        let mut vfs = Vfs::new();
        vfs.add_bsa(&bsa).unwrap();

        let mut under: Vec<_> = vfs.paths_under("Meshes").map(|p| p.to_string()).collect();
        under.sort();
        // "meshesx" must not match the "meshes" prefix.
        assert_eq!(under, ["meshes/a.nif", "meshes/sub/b.nif"]);

        assert_eq!(vfs.paths().count(), 4);

        std::fs::remove_file(&bsa).ok();
    }
}
