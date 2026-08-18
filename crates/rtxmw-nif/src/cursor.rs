//! Little-endian cursor over a NIF's bytes.

use crate::error::{NifError, Result};

/// A block index. `-1` means "no block".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Link(pub i32);

impl Default for Link {
    /// A link to nothing, which is how absent references are stored.
    fn default() -> Self {
        Self(-1)
    }
}

impl Link {
    /// Whether the link points at nothing.
    pub fn is_none(self) -> bool {
        self.0 < 0
    }

    /// The index, when the link points somewhere.
    pub fn index(self) -> Option<usize> {
        (self.0 >= 0).then_some(self.0 as usize)
    }
}

/// Reads primitives out of a NIF, tracking position so a desync can be located.
///
/// Every read is bounds-checked. Blocks carry no size in this file version, so a parser that reads
/// one field too many silently shifts every subsequent block; the bounds check turns the eventual
/// failure into an error that names an offset instead of nonsense geometry.
#[derive(Debug)]
pub struct Cursor<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    /// Starts at the beginning of `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    /// Current byte offset.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Bytes left unread.
    pub fn remaining(&self) -> usize {
        self.data.len() - self.position
    }

    /// Whether every byte has been consumed.
    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        if count > self.remaining() {
            return Err(NifError::UnexpectedEnd {
                offset: self.position,
                wanted: count,
                available: self.remaining(),
            });
        }
        let at = self.position;
        self.position += count;
        Ok(&self.data[at..at + count])
    }

    /// Reads a `u8`.
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    /// Reads a `u16`.
    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    /// Reads a `u32`.
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    /// Reads an `i32`.
    pub fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    /// Reads an `f32`.
    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    /// Reads a boolean.
    ///
    /// Four bytes before version 4.1.0.0, one after. Morrowind is 4.0.0.2, so four — getting this
    /// wrong desyncs every block that follows.
    pub fn bool(&mut self, version: u32) -> Result<bool> {
        if version < crate::nif_file::version(4, 1, 0, 0) {
            Ok(self.i32()? != 0)
        } else {
            Ok(self.u8()? != 0)
        }
    }

    /// Reads a block index.
    pub fn link(&mut self) -> Result<Link> {
        Ok(Link(self.i32()?))
    }

    /// Reads `count` block indices.
    pub fn links(&mut self, count: usize) -> Result<Vec<Link>> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.link()?);
        }
        Ok(out)
    }

    /// Reads a length-prefixed string.
    ///
    /// Legacy Windows codepage bytes; Morrowind's own content is ASCII, and anything outside it
    /// degrades to a replacement character rather than failing the parse.
    pub fn string(&mut self) -> Result<String> {
        let length = self.u32()? as usize;
        let bytes = self.take(length)?;
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
    }

    /// Reads a line terminated by `\n`, for the version header.
    pub fn line(&mut self) -> Result<String> {
        let start = self.position;
        while self.position < self.data.len() && self.data[self.position] != b'\n' {
            self.position += 1;
        }
        if self.position >= self.data.len() {
            return Err(NifError::UnexpectedEnd {
                offset: start,
                wanted: 1,
                available: 0,
            });
        }
        let line = String::from_utf8_lossy(&self.data[start..self.position]).into_owned();
        self.position += 1;
        Ok(line.trim_end_matches('\r').to_owned())
    }

    /// Reads three floats.
    pub fn vec3(&mut self) -> Result<[f32; 3]> {
        Ok([self.f32()?, self.f32()?, self.f32()?])
    }

    /// Reads two floats.
    pub fn vec2(&mut self) -> Result<[f32; 2]> {
        Ok([self.f32()?, self.f32()?])
    }

    /// Reads three floats as a colour.
    pub fn color3(&mut self) -> Result<[f32; 3]> {
        self.vec3()
    }

    /// Reads four floats as a colour.
    pub fn color4(&mut self) -> Result<[f32; 4]> {
        Ok([self.f32()?, self.f32()?, self.f32()?, self.f32()?])
    }

    /// Reads a 3x3 matrix, row-major as stored.
    pub fn matrix3(&mut self) -> Result<[[f32; 3]; 3]> {
        Ok([self.vec3()?, self.vec3()?, self.vec3()?])
    }

    /// Reads `count` `u16`s.
    pub fn u16s(&mut self, count: usize) -> Result<Vec<u16>> {
        let bytes = self.take(count * 2)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// Skips `count` bytes.
    pub fn skip(&mut self, count: usize) -> Result<()> {
        self.take(count).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_primitives_in_little_endian_order() {
        let bytes = [0x01u8, 0x02, 0x03, 0x04, 0x00, 0x00, 0x80, 0x3f];
        let mut cursor = Cursor::new(&bytes);
        assert_eq!(cursor.u32().unwrap(), 0x0403_0201);
        assert_eq!(cursor.f32().unwrap(), 1.0);
        assert!(cursor.is_empty());
    }

    #[test]
    fn bool_width_follows_the_file_version() {
        let bytes = [1u8, 0, 0, 0];
        // Morrowind: four bytes.
        let mut old = Cursor::new(&bytes);
        assert!(old.bool(crate::nif_file::version(4, 0, 0, 2)).unwrap());
        assert_eq!(old.remaining(), 0);

        // Later versions: one byte, leaving three.
        let mut new = Cursor::new(&bytes);
        assert!(new.bool(crate::nif_file::version(20, 0, 0, 5)).unwrap());
        assert_eq!(new.remaining(), 3);
    }

    #[test]
    fn strings_are_length_prefixed_and_stop_at_a_nul() {
        let mut bytes = 5u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"ab\0cd");
        let mut cursor = Cursor::new(&bytes);
        assert_eq!(cursor.string().unwrap(), "ab");
        // The whole declared length is consumed even though the text stopped early.
        assert!(cursor.is_empty());
    }

    #[test]
    fn a_read_past_the_end_names_the_offset() {
        let mut cursor = Cursor::new(&[0u8, 1]);
        let error = cursor.u32().expect_err("should not fit");
        assert!(matches!(
            error,
            NifError::UnexpectedEnd {
                offset: 0,
                wanted: 4,
                available: 2
            }
        ));
    }

    #[test]
    fn links_distinguish_none_from_an_index() {
        let mut bytes = (-1i32).to_le_bytes().to_vec();
        bytes.extend_from_slice(&7i32.to_le_bytes());
        let mut cursor = Cursor::new(&bytes);

        let none = cursor.link().unwrap();
        assert!(none.is_none());
        assert_eq!(none.index(), None);

        let some = cursor.link().unwrap();
        assert!(!some.is_none());
        assert_eq!(some.index(), Some(7));
    }

    #[test]
    fn line_stops_at_the_newline_and_trims_carriage_return() {
        let bytes = b"NetImmerse File Format, Version 4.0.0.2\r\nrest";
        let mut cursor = Cursor::new(bytes);
        assert_eq!(
            cursor.line().unwrap(),
            "NetImmerse File Format, Version 4.0.0.2"
        );
        assert_eq!(cursor.remaining(), 4);
    }
}
