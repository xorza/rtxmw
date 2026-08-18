//! Record and subrecord traversal over a content file.
//!
//! An ESM/ESP is a flat sequence of records, each `NAME(4) | u32 size | u32 unused | u32 flags`
//! followed by `size` bytes of subrecords, each `NAME(4) | u32 size | size bytes`. There is no
//! index and no nesting: everything is found by walking forward.
//!
//! The whole file is held in memory and every record and subrecord is a borrowed slice of it, so
//! traversal allocates nothing. Morrowind's three content files total roughly 94 MB, which is not
//! worth the complexity of streaming.

use crate::error::{EsmError, Result};
use crate::record_name::RecordName;

/// Bytes in a record header: tag, size, unused, flags.
const RECORD_HEADER: usize = 16;
/// Bytes in a subrecord header: tag, size.
const SUBRECORD_HEADER: usize = 8;

/// Record flag marking the record as deleted by a later plugin.
const FLAG_DELETED: u32 = 0x20;
/// Record flag marking the record persistent across cell unloads.
const FLAG_PERSISTENT: u32 = 0x400;
/// Record flag telling the engine to ignore the record entirely.
const FLAG_IGNORED: u32 = 0x1000;
/// Record flag marking the record blocked from being overridden.
const FLAG_BLOCKED: u32 = 0x2000;

/// One top-level record, borrowed from the file.
#[derive(Debug, Clone, Copy)]
pub struct Record<'a> {
    name: RecordName,
    flags: u32,
    data: &'a [u8],
    offset: usize,
}

impl<'a> Record<'a> {
    /// The record's four-character tag.
    pub fn name(&self) -> RecordName {
        self.name
    }

    /// The raw record flags.
    pub fn flags(&self) -> u32 {
        self.flags
    }

    /// Byte offset of this record's header within the file, for diagnostics.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Whether a later plugin deleted this record.
    ///
    /// Morrowind marks deletion two different ways and both occur in the shipped data: a header
    /// flag, and a `DELE` subrecord. Checking only one misses real deletions.
    pub fn is_deleted(&self) -> bool {
        if self.flags & FLAG_DELETED != 0 {
            return true;
        }
        self.subrecords()
            .flatten()
            .any(|sub| sub.name() == RecordName::new(b"DELE"))
    }

    /// Whether the engine should skip this record outright.
    pub fn is_ignored(&self) -> bool {
        self.flags & FLAG_IGNORED != 0
    }

    /// Whether the record is persistent across cell unloads.
    pub fn is_persistent(&self) -> bool {
        self.flags & FLAG_PERSISTENT != 0
    }

    /// Whether the record is protected from being overridden.
    pub fn is_blocked(&self) -> bool {
        self.flags & FLAG_BLOCKED != 0
    }

    /// The record's payload, before subrecord framing.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Walks the record's subrecords.
    pub fn subrecords(&self) -> SubrecordIter<'a> {
        SubrecordIter {
            data: self.data,
            position: 0,
            base: self.offset + RECORD_HEADER,
        }
    }
}

/// One subrecord, borrowed from the file.
#[derive(Debug, Clone, Copy)]
pub struct Subrecord<'a> {
    name: RecordName,
    data: &'a [u8],
}

impl<'a> Subrecord<'a> {
    /// The subrecord's four-character tag.
    pub fn name(&self) -> RecordName {
        self.name
    }

    /// The subrecord's payload.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// The payload as text, with the trailing NUL and any padding removed.
    ///
    /// Strings are legacy Windows codepage bytes. Morrowind's own content is ASCII, where that
    /// coincides with UTF-8; anything outside it degrades to a replacement character rather than
    /// failing the parse.
    pub fn as_str(&self) -> std::borrow::Cow<'a, str> {
        let end = self
            .data
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.data.len());
        String::from_utf8_lossy(&self.data[..end])
    }

    /// The payload as a little-endian `u32`.
    pub fn as_u32(&self) -> Result<u32> {
        self.fixed::<4>().map(u32::from_le_bytes)
    }

    /// The payload as a little-endian `i32`.
    pub fn as_i32(&self) -> Result<i32> {
        self.fixed::<4>().map(i32::from_le_bytes)
    }

    /// The payload as a little-endian `f32`.
    pub fn as_f32(&self) -> Result<f32> {
        self.fixed::<4>().map(f32::from_le_bytes)
    }

    fn fixed<const N: usize>(&self) -> Result<[u8; N]> {
        self.data
            .try_into()
            .map_err(|_| EsmError::BadSubrecordSize {
                name: self.name,
                wanted: N,
                found: self.data.len(),
            })
    }
}

/// Iterator over a record's subrecords.
#[derive(Debug, Clone)]
pub struct SubrecordIter<'a> {
    data: &'a [u8],
    position: usize,
    /// File offset the payload starts at, so errors can point at the file rather than the record.
    base: usize,
}

impl<'a> Iterator for SubrecordIter<'a> {
    type Item = Result<Subrecord<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.data.len() {
            return None;
        }
        let at = self.position;
        if self.data.len() - at < SUBRECORD_HEADER {
            self.position = self.data.len();
            return Some(Err(EsmError::UnexpectedEnd {
                offset: self.base + at,
            }));
        }

        let name = RecordName(self.data[at..at + 4].try_into().unwrap());
        let size = u32::from_le_bytes(self.data[at + 4..at + 8].try_into().unwrap()) as usize;
        let start = at + SUBRECORD_HEADER;
        let available = self.data.len() - start;
        if size > available {
            self.position = self.data.len();
            return Some(Err(EsmError::Truncated {
                offset: self.base + at,
                wanted: size,
                available,
            }));
        }

        self.position = start + size;
        Some(Ok(Subrecord {
            name,
            data: &self.data[start..start + size],
        }))
    }
}

/// A content file held in memory, with its header parsed.
#[derive(Debug)]
pub struct EsmReader<'a> {
    data: &'a [u8],
    header: crate::header::Header,
    body: usize,
}

impl<'a> EsmReader<'a> {
    /// Parses the `TES3` header and positions at the first record after it.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let mut records = RecordIter { data, position: 0 };
        let first = records.next().ok_or(EsmError::NotAContentFile)??;
        if first.name() != RecordName::new(b"TES3") {
            return Err(EsmError::NotAContentFile);
        }
        let header = crate::header::Header::parse(&first)?;
        Ok(Self {
            data,
            header,
            body: records.position,
        })
    }

    /// The file's `TES3` header.
    pub fn header(&self) -> &crate::header::Header {
        &self.header
    }

    /// The record starting at `offset`, which an earlier pass reported as [`Record::offset`].
    ///
    /// Records are found by walking and a content file is tens of megabytes, so an index of
    /// offsets is what turns "load this cell" from a pass over the whole file into two reads. An
    /// offset that did not come from a record parses whatever bytes are there as a header, which
    /// is why this takes one rather than searching for the nearest.
    pub fn record_at(&self, offset: usize) -> Result<Record<'a>> {
        RecordIter {
            data: self.data,
            position: offset,
        }
        .next()
        .ok_or(EsmError::UnexpectedEnd { offset })?
    }

    /// Walks every record after the header.
    pub fn records(&self) -> RecordIter<'a> {
        RecordIter {
            data: self.data,
            position: self.body,
        }
    }
}

/// Iterator over a content file's records.
#[derive(Debug, Clone)]
pub struct RecordIter<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Iterator for RecordIter<'a> {
    type Item = Result<Record<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.data.len() {
            return None;
        }
        let at = self.position;
        if self.data.len() - at < RECORD_HEADER {
            self.position = self.data.len();
            return Some(Err(EsmError::UnexpectedEnd { offset: at }));
        }

        let name = RecordName(self.data[at..at + 4].try_into().unwrap());
        let size = u32::from_le_bytes(self.data[at + 4..at + 8].try_into().unwrap()) as usize;
        // Bytes 8..12 are unused in Morrowind-era files.
        let flags = u32::from_le_bytes(self.data[at + 12..at + 16].try_into().unwrap());

        let start = at + RECORD_HEADER;
        let available = self.data.len() - start;
        if size > available {
            self.position = self.data.len();
            return Some(Err(EsmError::Truncated {
                offset: at,
                wanted: size,
                available,
            }));
        }

        self.position = start + size;
        Some(Ok(Record {
            name,
            flags,
            data: &self.data[start..start + size],
            offset: at,
        }))
    }
}

#[cfg(any(test, feature = "internals"))]
pub mod internals {
    //! Synthetic content-file construction, so the reader can be tested without game data.

    /// One subrecord to write.
    #[derive(Debug, Clone, Copy)]
    pub struct SubrecordSpec<'a> {
        pub name: &'a [u8; 4],
        pub data: &'a [u8],
    }

    /// Appends a record with the given tag, flags and subrecords.
    pub fn push_record(out: &mut Vec<u8>, tag: &[u8; 4], flags: u32, subs: &[SubrecordSpec<'_>]) {
        let mut payload = Vec::new();
        for sub in subs {
            payload.extend_from_slice(sub.name);
            payload.extend_from_slice(&(sub.data.len() as u32).to_le_bytes());
            payload.extend_from_slice(sub.data);
        }
        out.extend_from_slice(tag);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&payload);
    }

    /// A minimal but well-formed `TES3` header record.
    pub fn push_header(out: &mut Vec<u8>) {
        let mut hedr = Vec::new();
        hedr.extend_from_slice(&1.3f32.to_le_bytes());
        hedr.extend_from_slice(&0u32.to_le_bytes());
        hedr.extend_from_slice(&[b'a'; 32]);
        hedr.extend_from_slice(&[b'd'; 256]);
        hedr.extend_from_slice(&7u32.to_le_bytes());
        push_record(
            out,
            b"TES3",
            0,
            &[SubrecordSpec {
                name: b"HEDR",
                data: &hedr,
            }],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::internals::{SubrecordSpec, push_header, push_record};
    use super::*;

    fn file_with_two_records() -> Vec<u8> {
        let mut out = Vec::new();
        push_header(&mut out);
        push_record(
            &mut out,
            b"STAT",
            0,
            &[
                SubrecordSpec {
                    name: b"NAME",
                    data: b"in_de_shack\0",
                },
                SubrecordSpec {
                    name: b"MODL",
                    data: b"i\\in_de_shack.nif\0",
                },
            ],
        );
        push_record(
            &mut out,
            b"MISC",
            FLAG_DELETED,
            &[SubrecordSpec {
                name: b"NAME",
                data: b"gone\0",
            }],
        );
        out
    }

    #[test]
    fn walks_records_and_subrecords_without_the_header() {
        let file = file_with_two_records();
        let esm = EsmReader::new(&file).expect("should parse");

        let records: Vec<_> = esm.records().map(|r| r.unwrap()).collect();
        assert_eq!(records.len(), 2, "the TES3 header must not be yielded");
        assert_eq!(records[0].name(), RecordName::new(b"STAT"));
        assert_eq!(records[1].name(), RecordName::new(b"MISC"));

        let subs: Vec<_> = records[0].subrecords().map(|s| s.unwrap()).collect();
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0].name(), RecordName::new(b"NAME"));
        assert_eq!(subs[0].as_str(), "in_de_shack");
        assert_eq!(subs[1].as_str(), r"i\in_de_shack.nif");
    }

    #[test]
    fn deletion_is_detected_from_the_flag_and_from_a_dele_subrecord() {
        let file = file_with_two_records();
        let esm = EsmReader::new(&file).unwrap();
        let records: Vec<_> = esm.records().map(|r| r.unwrap()).collect();
        assert!(!records[0].is_deleted());
        assert!(records[1].is_deleted(), "header flag should count");

        // The other spelling: no flag, but a DELE subrecord.
        let mut out = Vec::new();
        push_header(&mut out);
        push_record(
            &mut out,
            b"STAT",
            0,
            &[
                SubrecordSpec {
                    name: b"NAME",
                    data: b"x\0",
                },
                SubrecordSpec {
                    name: b"DELE",
                    data: &0u32.to_le_bytes(),
                },
            ],
        );
        let esm = EsmReader::new(&out).unwrap();
        let record = esm.records().next().unwrap().unwrap();
        assert!(record.is_deleted(), "DELE subrecord should count");
    }

    #[test]
    fn typed_subrecord_accessors_check_their_width() {
        let mut out = Vec::new();
        push_header(&mut out);
        push_record(
            &mut out,
            b"TEST",
            0,
            &[
                SubrecordSpec {
                    name: b"INTV",
                    data: &(-7i32).to_le_bytes(),
                },
                SubrecordSpec {
                    name: b"FLTV",
                    data: &1.5f32.to_le_bytes(),
                },
                SubrecordSpec {
                    name: b"SHRT",
                    data: &[1, 2],
                },
            ],
        );
        let esm = EsmReader::new(&out).unwrap();
        let record = esm.records().next().unwrap().unwrap();
        let subs: Vec<_> = record.subrecords().map(|s| s.unwrap()).collect();

        assert_eq!(subs[0].as_i32().unwrap(), -7);
        assert_eq!(subs[1].as_f32().unwrap(), 1.5);
        // A two-byte payload is not an i32, and must say so rather than read past itself.
        assert!(matches!(
            subs[2].as_i32(),
            Err(EsmError::BadSubrecordSize {
                wanted: 4,
                found: 2,
                ..
            })
        ));
    }

    #[test]
    fn a_file_not_starting_with_tes3_is_rejected() {
        let mut out = Vec::new();
        push_record(&mut out, b"STAT", 0, &[]);
        assert!(matches!(
            EsmReader::new(&out),
            Err(EsmError::NotAContentFile)
        ));
        assert!(matches!(
            EsmReader::new(&[]),
            Err(EsmError::NotAContentFile)
        ));
    }

    #[test]
    fn a_record_claiming_more_bytes_than_remain_is_rejected() {
        let mut file = file_with_two_records();
        // Header record is 16 bytes + payload; the STAT record follows. Overstate its size.
        let stat_at = file
            .windows(4)
            .position(|w| w == b"STAT")
            .expect("STAT should be present");
        file[stat_at + 4..stat_at + 8].copy_from_slice(&0xFFFF_0000u32.to_le_bytes());

        let esm = EsmReader::new(&file).unwrap();
        let error = esm
            .records()
            .find_map(|r| r.err())
            .expect("truncation should be reported");
        assert!(matches!(error, EsmError::Truncated { .. }), "got {error:?}");
    }

    #[test]
    fn a_subrecord_claiming_more_bytes_than_remain_is_rejected() {
        let mut file = file_with_two_records();
        let modl_at = file
            .windows(4)
            .position(|w| w == b"MODL")
            .expect("MODL should be present");
        file[modl_at + 4..modl_at + 8].copy_from_slice(&9999u32.to_le_bytes());

        let esm = EsmReader::new(&file).unwrap();
        let record = esm.records().next().unwrap().unwrap();
        let error = record
            .subrecords()
            .find_map(|s| s.err())
            .expect("truncation should be reported");
        assert!(matches!(error, EsmError::Truncated { .. }), "got {error:?}");
    }
}
