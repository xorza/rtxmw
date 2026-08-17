//! Reader for Morrowind-era BSA archives.
//!
//! Layout, from the header onward:
//!
//! ```text
//! u32 magic = 0x100
//! u32 dirsize          bytes in the directory block below
//! u32 numfiles
//! -- directory block, dirsize bytes total --
//! numfiles x { u32 size; u32 offset }   offset is relative to the data block
//! numfiles x   u32 name_offset          into the name buffer
//! name buffer, NUL-terminated strings, dirsize - 12 * numfiles bytes
//! -- end directory block --
//! numfiles x u64 hash                   unused here; names are the lookup key
//! data block                            file contents, uncompressed
//! ```
//!
//! The data block therefore begins at `12 + dirsize + 8 * numfiles`.
//!
//! Unix only: reads are positioned (`pread`) so one archive can be shared across threads without a
//! seek/read race. Windows has the same facility under a different name; add the shim when a
//! Windows build is actually wanted.

use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::error::{Result, VfsError};
use crate::normalized_path::NormalizedPath;

/// Identifies a Morrowind BSA. Later Bethesda archives use `BSA\0` and a different layout.
const MAGIC: u32 = 0x100;

/// Bytes of directory data each file costs: 4 size + 4 offset + 4 name offset.
const DIRECTORY_BYTES_PER_FILE: u64 = 12;

/// One file inside the archive.
#[derive(Debug)]
struct BsaEntry {
    /// Absolute offset into the archive file.
    offset: u64,
    size: u32,
    /// Range into [`BsaArchive::names`].
    name: Range<u32>,
}

/// A memory-resident index over a BSA, reading file contents on demand.
///
/// Only the directory is held; Morrowind.bsa is 310 MB of mostly texture data and there is no
/// reason for it to be resident.
#[derive(Debug)]
pub struct BsaArchive {
    file: File,
    path: PathBuf,
    entries: Vec<BsaEntry>,
    /// Every entry name, already normalized, concatenated into one buffer. Flat rather than a
    /// `Vec<String>`: Morrowind.bsa alone holds thousands of entries.
    names: String,
}

impl BsaArchive {
    /// Opens `path` and reads its directory.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| VfsError::io(path, e))?;
        let archive_size = file.metadata().map_err(|e| VfsError::io(path, e))?.len();

        let mut header = [0u8; 12];
        Self::read_at(&file, path, 0, &mut header)?;
        let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(VfsError::malformed(
                path,
                format!("expected magic {MAGIC:#x}, found {magic:#x}"),
            ));
        }
        let directory_size = u32::from_le_bytes(header[4..8].try_into().unwrap()) as u64;
        let file_count = u32::from_le_bytes(header[8..12].try_into().unwrap()) as u64;

        // Reject impossible geometry before any of it is used as an offset. Each file needs at
        // least its 12 directory bytes, 8 hash bytes and a 1-byte name.
        let offset_table_size = DIRECTORY_BYTES_PER_FILE * file_count;
        if directory_size < offset_table_size
            || archive_size < 12 + directory_size + 8 * file_count
            || file_count * 21 > archive_size.saturating_sub(12)
        {
            return Err(VfsError::malformed(
                path,
                format!(
                    "directory of {file_count} files does not fit in {archive_size} bytes \
                     (dirsize {directory_size})"
                ),
            ));
        }

        let file_count = file_count as usize;
        let mut offsets = vec![0u8; offset_table_size as usize];
        Self::read_at(&file, path, 12, &mut offsets)?;

        let mut name_buffer = vec![0u8; (directory_size - offset_table_size) as usize];
        Self::read_at(&file, path, 12 + offset_table_size, &mut name_buffer)?;

        let data_offset = 12 + directory_size + 8 * file_count as u64;

        let mut entries = Vec::with_capacity(file_count);
        let mut names = String::with_capacity(name_buffer.len());
        let read_u32 = |index: usize| {
            let at = index * 4;
            u32::from_le_bytes(offsets[at..at + 4].try_into().unwrap())
        };

        for index in 0..file_count {
            let size = read_u32(index * 2);
            let relative_offset = read_u32(index * 2 + 1) as u64;
            let name_offset = read_u32(2 * file_count + index) as usize;

            let offset = data_offset + relative_offset;
            if offset + size as u64 > archive_size {
                return Err(VfsError::malformed(
                    path,
                    format!("file {index} spans past the end of the archive"),
                ));
            }
            if name_offset >= name_buffer.len() {
                return Err(VfsError::malformed(
                    path,
                    format!("file {index} has a name offset outside the name buffer"),
                ));
            }
            let tail = &name_buffer[name_offset..];
            let Some(end) = tail.iter().position(|&b| b == 0) else {
                return Err(VfsError::malformed(
                    path,
                    format!("file {index} has an unterminated name"),
                ));
            };

            // Names are legacy Windows codepage bytes. Nothing in Morrowind's own archives is
            // outside ASCII, and normalization only touches ASCII, so the bytes pass through.
            let raw = String::from_utf8_lossy(&tail[..end]);
            let normalized = NormalizedPath::new(&raw);

            let start = names.len() as u32;
            names.push_str(normalized.as_str());
            entries.push(BsaEntry {
                offset,
                size,
                name: start..names.len() as u32,
            });
        }

        Ok(Self {
            file,
            path: path.to_path_buf(),
            entries,
            names,
        })
    }

    /// Number of files in the archive.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the archive holds no files.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Where the archive was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The normalized name of entry `index`.
    pub fn entry_name(&self, index: usize) -> &str {
        let entry = &self.entries[index];
        &self.names[entry.name.start as usize..entry.name.end as usize]
    }

    /// Size in bytes of entry `index`.
    pub fn entry_size(&self, index: usize) -> u32 {
        self.entries[index].size
    }

    /// Reads entry `index` in full.
    pub fn read(&self, index: usize) -> Result<Vec<u8>> {
        let entry = &self.entries[index];
        let mut out = vec![0u8; entry.size as usize];
        Self::read_at(&self.file, &self.path, entry.offset, &mut out)?;
        Ok(out)
    }

    fn read_at(file: &File, path: &Path, offset: u64, buffer: &mut [u8]) -> Result<()> {
        // Positioned reads rather than seek + read: the archive is shared across threads and a
        // seek/read pair would race.
        file.read_exact_at(buffer, offset)
            .map_err(|e| VfsError::io(path, e))
    }
}

#[cfg(any(test, feature = "internals"))]
pub mod internals {
    //! Synthetic archive construction, so the reader can be tested without game data.

    use crate::normalized_path::NormalizedPath;

    /// Builds a well-formed BSA image holding `files` as `(name, contents)`.
    pub fn build_bsa(files: &[(&str, &[u8])]) -> Vec<u8> {
        let count = files.len() as u32;

        let mut names = Vec::new();
        let mut name_offsets = Vec::with_capacity(files.len());
        for (name, _) in files {
            name_offsets.push(names.len() as u32);
            names.extend_from_slice(name.as_bytes());
            names.push(0);
        }

        let directory_size = 12 * count + names.len() as u32;

        let mut out = Vec::new();
        out.extend_from_slice(&super::MAGIC.to_le_bytes());
        out.extend_from_slice(&directory_size.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());

        let mut running = 0u32;
        for (_, contents) in files {
            out.extend_from_slice(&(contents.len() as u32).to_le_bytes());
            out.extend_from_slice(&running.to_le_bytes());
            running += contents.len() as u32;
        }
        for offset in &name_offsets {
            out.extend_from_slice(&offset.to_le_bytes());
        }
        out.extend_from_slice(&names);
        // Hashes are never read back, but the data block only starts after them.
        out.extend_from_slice(&vec![0u8; 8 * files.len()]);
        for (_, contents) in files {
            out.extend_from_slice(contents);
        }
        out
    }

    /// Writes `image` to a uniquely named file under the system temp directory.
    pub fn write_temp(image: &[u8], tag: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "rtxmw-{tag}-{}-{:?}.bsa",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, image).expect("could not write the test archive");
        path
    }

    /// Names as the archive stores them, normalized, for assertion.
    pub fn normalized(names: &[&str]) -> Vec<String> {
        names
            .iter()
            .map(|n| NormalizedPath::new(n).as_str().to_owned())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::internals::{build_bsa, normalized, write_temp};
    use super::*;

    const FILES: &[(&str, &[u8])] = &[
        (r"meshes\a.nif", b"first"),
        (r"Textures\B.DDS", b"second contents"),
        ("readme.txt", b""),
    ];

    #[test]
    fn reads_names_sizes_and_contents() {
        let path = write_temp(&build_bsa(FILES), "reads");
        let archive = BsaArchive::open(&path).expect("archive should open");

        assert_eq!(archive.len(), 3);
        assert!(!archive.is_empty());

        let expected = normalized(&[r"meshes\a.nif", r"Textures\B.DDS", "readme.txt"]);
        for (index, want) in expected.iter().enumerate() {
            assert_eq!(archive.entry_name(index), want);
        }

        assert_eq!(archive.entry_size(0), 5);
        assert_eq!(archive.read(0).unwrap(), b"first");
        assert_eq!(archive.read(1).unwrap(), b"second contents");
        // A zero-length entry is legal and must read back as empty rather than failing.
        assert_eq!(archive.entry_size(2), 0);
        assert_eq!(archive.read(2).unwrap(), b"");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_a_foreign_magic() {
        let mut image = build_bsa(FILES);
        // "BSA\0" — a TES4-era archive, which this reader must not attempt.
        image[0..4].copy_from_slice(&0x0041_5342u32.to_le_bytes());
        let path = write_temp(&image, "magic");

        let error = BsaArchive::open(&path).expect_err("foreign magic must be rejected");
        assert!(
            matches!(error, VfsError::Malformed { .. }),
            "expected Malformed, got {error:?}"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_a_file_count_that_cannot_fit() {
        let mut image = build_bsa(FILES);
        image[8..12].copy_from_slice(&1_000_000u32.to_le_bytes());
        let path = write_temp(&image, "count");

        let error = BsaArchive::open(&path).expect_err("impossible file count must be rejected");
        assert!(matches!(error, VfsError::Malformed { .. }));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_an_entry_pointing_past_the_end() {
        let mut image = build_bsa(FILES);
        // The first entry's data offset sits right after the 12-byte header.
        image[12 + 4..12 + 8].copy_from_slice(&0xFFFF_0000u32.to_le_bytes());
        let path = write_temp(&image, "offset");

        let error = BsaArchive::open(&path).expect_err("out-of-range offset must be rejected");
        assert!(matches!(error, VfsError::Malformed { .. }));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_a_truncated_archive() {
        let image = build_bsa(FILES);
        let path = write_temp(&image[..image.len() / 2], "truncated");

        let error = BsaArchive::open(&path).expect_err("truncation must be rejected");
        assert!(
            matches!(error, VfsError::Malformed { .. } | VfsError::Io { .. }),
            "expected Malformed or Io, got {error:?}"
        );

        std::fs::remove_file(&path).ok();
    }
}
