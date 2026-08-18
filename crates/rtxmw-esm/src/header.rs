//! The `TES3` header that opens every content file.

use crate::error::Result;
use crate::esm_reader::Record;
use crate::record_name::RecordName;

/// Whether a content file is a master or a plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    /// `.esm` — a master file other files may depend on.
    Master,
    /// `.esp` — a plugin.
    Plugin,
}

/// A content file this one depends on, in load order.
#[derive(Debug, Clone)]
pub struct MasterFile {
    pub name: String,
    /// Size in bytes the author's copy had, used to detect a mismatched master.
    pub size: u64,
}

/// The parsed `TES3` record.
#[derive(Debug, Clone)]
pub struct Header {
    pub version: f32,
    pub kind: FileKind,
    pub author: String,
    pub description: String,
    /// Records following the header, as the writer counted them.
    pub record_count: u32,
    pub masters: Vec<MasterFile>,
}

impl Header {
    pub(crate) fn parse(record: &Record<'_>) -> Result<Self> {
        let mut header = Self {
            version: 0.0,
            kind: FileKind::Plugin,
            author: String::new(),
            description: String::new(),
            record_count: 0,
            masters: Vec::new(),
        };

        // A master's name always precedes its size, so the size attaches to the last name seen.
        let mut pending: Option<String> = None;
        for sub in record.subrecords() {
            let sub = sub?;
            match sub.name() {
                n if n == RecordName::new(b"HEDR") => {
                    let data = sub.data();
                    // 4 version + 4 kind + 32 author + 256 description + 4 record count.
                    if data.len() >= 300 {
                        header.version = f32::from_le_bytes(data[0..4].try_into().unwrap());
                        let kind = u32::from_le_bytes(data[4..8].try_into().unwrap());
                        header.kind = if kind == 1 {
                            FileKind::Master
                        } else {
                            FileKind::Plugin
                        };
                        header.author = nul_terminated(&data[8..40]);
                        header.description = nul_terminated(&data[40..296]);
                        header.record_count =
                            u32::from_le_bytes(data[296..300].try_into().unwrap());
                    }
                }
                n if n == RecordName::new(b"MAST") => {
                    if let Some(name) = pending.take() {
                        // Two names in a row means the previous had no size; keep it anyway.
                        header.masters.push(MasterFile { name, size: 0 });
                    }
                    pending = Some(sub.as_str().into_owned());
                }
                n if n == RecordName::new(b"DATA") => {
                    if let Some(name) = pending.take() {
                        let size = sub
                            .data()
                            .get(..8)
                            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                            .unwrap_or(0);
                        header.masters.push(MasterFile { name, size });
                    }
                }
                _ => {}
            }
        }
        if let Some(name) = pending {
            header.masters.push(MasterFile { name, size: 0 });
        }
        Ok(header)
    }
}

/// Trims a fixed-width field at its first NUL and decodes it lossily.
fn nul_terminated(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}
