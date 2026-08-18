//! Reading the handful of `.tga` files that escaped the conversion to DDS.

use crate::error::{Result, TextureError};
use crate::texture::{MipLevel, Texture, TextureFormat};

/// Fixed-size header before the optional id and colour-map blocks.
const HEADER: usize = 18;

/// Uncompressed true colour — the only image type the game ships.
const UNCOMPRESSED_TRUE_COLOUR: u8 = 2;

/// Bit 5 of the descriptor byte: set means the first row stored is the top one.
const TOP_ORIGIN: u8 = 0x20;

/// Decodes a TGA, expanding to `Rgba8`.
///
/// Only eleven files in the shipped data are TGA and every one is uncompressed 24-bit, so run-length
/// encoding and colour maps are rejected rather than guessed at. They are cheap to add if a mod
/// needs them; inventing them untested is not.
pub(crate) fn decode(bytes: &[u8]) -> Result<Texture> {
    if bytes.len() < HEADER {
        return Err(TextureError::Truncated {
            wanted: HEADER,
            available: bytes.len(),
        });
    }

    let id_length = bytes[0] as usize;
    let colour_map = bytes[1];
    let image_type = bytes[2];
    let width = u16::from_le_bytes([bytes[12], bytes[13]]) as u32;
    let height = u16::from_le_bytes([bytes[14], bytes[15]]) as u32;
    let bits = bytes[16];
    let descriptor = bytes[17];

    if image_type != UNCOMPRESSED_TRUE_COLOUR || colour_map != 0 {
        return Err(TextureError::UnsupportedTga { image_type, bits });
    }
    if bits != 24 && bits != 32 {
        return Err(TextureError::UnsupportedTga { image_type, bits });
    }
    if width == 0 || height == 0 {
        return Err(TextureError::EmptyImage { width, height });
    }

    let source_bytes = bits as usize / 8;
    let start = HEADER + id_length;
    let wanted = start + width as usize * height as usize * source_bytes;
    if bytes.len() < wanted {
        return Err(TextureError::Truncated {
            wanted,
            available: bytes.len(),
        });
    }

    // TGA stores blue first and, by default, the bottom row first. Both are flipped here so every
    // decoded texture reaches the GPU in one orientation and one channel order.
    let mut data = Vec::with_capacity(width as usize * height as usize * 4);
    let top_first = descriptor & TOP_ORIGIN != 0;
    for row in 0..height as usize {
        let source_row = if top_first {
            row
        } else {
            height as usize - 1 - row
        };
        let offset = start + source_row * width as usize * source_bytes;
        for texel in 0..width as usize {
            let at = offset + texel * source_bytes;
            data.extend_from_slice(&[bytes[at + 2], bytes[at + 1], bytes[at]]);
            data.push(if source_bytes == 4 {
                bytes[at + 3]
            } else {
                0xFF
            });
        }
    }

    let levels = vec![MipLevel {
        offset: 0,
        size: data.len() as u32,
        width,
        height,
    }];
    Texture::new(TextureFormat::Rgba8, data, levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `width` x `height` uncompressed TGA whose texels are blue-green-red as stored.
    fn tga(width: u16, height: u16, descriptor: u8, texels: &[[u8; 3]]) -> Vec<u8> {
        let mut bytes = vec![0u8; HEADER];
        bytes[2] = UNCOMPRESSED_TRUE_COLOUR;
        bytes[12..14].copy_from_slice(&width.to_le_bytes());
        bytes[14..16].copy_from_slice(&height.to_le_bytes());
        bytes[16] = 24;
        bytes[17] = descriptor;
        for texel in texels {
            bytes.extend_from_slice(texel);
        }
        bytes
    }

    #[test]
    fn stored_bgr_comes_out_rgba_with_opaque_alpha() {
        // One texel stored blue=1, green=2, red=3.
        let texture = decode(&tga(1, 1, TOP_ORIGIN, &[[1, 2, 3]])).expect("should decode");
        assert_eq!(texture.format(), TextureFormat::Rgba8);
        assert_eq!(texture.data(), &[3, 2, 1, 255]);
    }

    #[test]
    fn a_bottom_origin_image_is_flipped_to_top_first() {
        // Two rows of one texel: red then green as stored, bottom row first.
        let bottom_first = tga(1, 2, 0, &[[0, 0, 0xFF], [0, 0xFF, 0]]);
        let texture = decode(&bottom_first).expect("should decode");
        // Row zero of the result must be the *last* row stored.
        assert_eq!(&texture.data()[..4], &[0, 0xFF, 0, 255]);
        assert_eq!(&texture.data()[4..], &[0xFF, 0, 0, 255]);

        // The same texels declared top-first must come back in the order they were written.
        let top_first = tga(1, 2, TOP_ORIGIN, &[[0, 0, 0xFF], [0, 0xFF, 0]]);
        let texture = decode(&top_first).expect("should decode");
        assert_eq!(&texture.data()[..4], &[0xFF, 0, 0, 255]);
    }

    #[test]
    fn run_length_encoding_is_rejected_rather_than_misread() {
        let mut bytes = tga(1, 1, TOP_ORIGIN, &[[1, 2, 3]]);
        bytes[2] = 10; // RLE true colour.
        let error = decode(&bytes).expect_err("should reject");
        assert!(matches!(
            error,
            TextureError::UnsupportedTga {
                image_type: 10,
                bits: 24
            }
        ));
    }

    #[test]
    fn a_short_pixel_block_is_rejected() {
        let mut bytes = tga(4, 4, TOP_ORIGIN, &[]);
        bytes.extend_from_slice(&[0u8; 8]);
        let error = decode(&bytes).expect_err("should reject");
        assert!(matches!(error, TextureError::Truncated { wanted: 66, .. }));
    }
}
