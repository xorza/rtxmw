//! Reading Direct3D 9 `.dds` files, which is nearly everything the game ships.

use crate::error::{Result, TextureError};
use crate::texture::{Texture, TextureFormat, describe_levels};

/// The four bytes every DDS starts with.
pub(crate) const MAGIC: &[u8] = b"DDS ";

/// Magic plus a 124-byte header; pixel data follows immediately.
const DATA_START: usize = 128;

/// `DDPF_FOURCC` — the pixel format names a compression scheme rather than channel masks.
const FOURCC: u32 = 0x4;

/// Decodes a DDS into its whole mip chain.
///
/// Only the Direct3D 9 header is handled. `DX10` extended headers exist in the format but no
/// Morrowind asset uses one, and guessing at a header this parser has never seen would produce
/// plausible-looking garbage rather than an error.
pub(crate) fn decode(bytes: &[u8]) -> Result<Texture> {
    if bytes.len() < DATA_START {
        return Err(TextureError::Truncated {
            wanted: DATA_START,
            available: bytes.len(),
        });
    }

    let field = |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let height = field(12);
    let width = field(16);
    // Zero means "no chain declared", which is one level rather than none.
    let declared_mips = field(28).max(1);
    let pixel_flags = field(80);

    if width == 0 || height == 0 {
        return Err(TextureError::EmptyImage { width, height });
    }

    let format = if pixel_flags & FOURCC != 0 {
        let fourcc: [u8; 4] = bytes[84..88].try_into().unwrap();
        match &fourcc {
            b"DXT1" => TextureFormat::Bc1,
            b"DXT3" => TextureFormat::Bc2,
            _ => return Err(TextureError::UnsupportedFourCc(fourcc)),
        }
    } else {
        // The only uncompressed layout the game ships is 32-bit with alpha in the top byte, which
        // is `A8R8G8B8` — blue first once it is bytes in a file.
        let bit_count = field(88);
        let red_mask = field(92);
        let green_mask = field(96);
        let blue_mask = field(100);
        let alpha_mask = field(104);
        if bit_count != 32
            || red_mask != 0x00FF_0000
            || green_mask != 0x0000_FF00
            || blue_mask != 0x0000_00FF
            || alpha_mask != 0xFF00_0000
        {
            return Err(TextureError::UnsupportedPixelFormat {
                bit_count,
                masks: [red_mask, green_mask, blue_mask, alpha_mask],
            });
        }
        TextureFormat::Bgra8
    };

    // A file may hold fewer levels than it declares. Trusting the count would read past the end;
    // trusting the data alone would silently drop levels a correct file does have, so take the
    // smaller and let the length check in `Texture::new` reject anything still inconsistent.
    let available = bytes.len() - DATA_START;
    let mut levels = describe_levels(format, width, height, declared_mips);
    while levels
        .last()
        .is_some_and(|l| l.offset as usize + l.size as usize > available)
        && levels.len() > 1
    {
        levels.pop();
    }

    let end = levels
        .last()
        .map_or(0, |l| l.offset as usize + l.size as usize);
    let data = bytes[DATA_START..DATA_START + end.min(available)].to_vec();
    Texture::new(format, data, levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal Direct3D 9 DDS around `payload`.
    fn dds(width: u32, height: u32, mips: u32, fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0u8; DATA_START];
        bytes[..4].copy_from_slice(MAGIC);
        bytes[4..8].copy_from_slice(&124u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&height.to_le_bytes());
        bytes[16..20].copy_from_slice(&width.to_le_bytes());
        bytes[28..32].copy_from_slice(&mips.to_le_bytes());
        bytes[80..84].copy_from_slice(&FOURCC.to_le_bytes());
        bytes[84..88].copy_from_slice(fourcc);
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn a_dxt1_file_decodes_to_bc1_with_its_whole_chain() {
        // 8x8 then 4x4 then 2x2 then 1x1: 32 + 8 + 8 + 8 bytes.
        let payload = vec![0xABu8; 56];
        let texture = decode(&dds(8, 8, 4, b"DXT1", &payload)).expect("should decode");

        assert_eq!(texture.format(), TextureFormat::Bc1);
        assert_eq!((texture.width(), texture.height()), (8, 8));
        assert_eq!(texture.levels().len(), 4);
        assert_eq!(texture.level_data(0).len(), 32);
        assert_eq!(texture.level_data(3).len(), 8);
        assert_eq!(texture.data(), payload.as_slice());
    }

    #[test]
    fn dxt3_is_bc2_and_costs_twice_as_much_per_block() {
        let texture = decode(&dds(4, 4, 1, b"DXT3", &[0u8; 16])).expect("should decode");
        assert_eq!(texture.format(), TextureFormat::Bc2);
        assert_eq!(texture.level_data(0).len(), 16);
    }

    #[test]
    fn a_missing_mip_count_means_one_level_not_none() {
        // The game ships 364 files with the count left at zero.
        let texture = decode(&dds(4, 4, 0, b"DXT1", &[0u8; 8])).expect("should decode");
        assert_eq!(texture.levels().len(), 1);
    }

    #[test]
    fn a_chain_longer_than_the_file_is_trimmed_rather_than_read_past() {
        // Declares four levels but only carries the first two.
        let texture = decode(&dds(8, 8, 4, b"DXT1", &[0u8; 40])).expect("should decode");
        assert_eq!(texture.levels().len(), 2);
        assert_eq!(texture.data().len(), 40);
    }

    #[test]
    fn an_unknown_compression_scheme_names_itself() {
        // DXT5 does not appear anywhere in the shipped data, so meeting one means an added asset.
        let error = decode(&dds(4, 4, 1, b"DXT5", &[0u8; 16])).expect_err("should reject");
        assert!(matches!(error, TextureError::UnsupportedFourCc(f) if &f == b"DXT5"));
    }

    #[test]
    fn a_truncated_header_is_rejected_before_any_field_is_read() {
        let error = decode(&[0u8; 64]).expect_err("should reject");
        assert!(matches!(
            error,
            TextureError::Truncated {
                wanted: 128,
                available: 64
            }
        ));
    }
}
