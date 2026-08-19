//! Expanding a texture to plain bytes, for something that has to look at it rather than sample it.

use crate::texture::{Texture, TextureFormat};

/// A 5-6-5 endpoint as eight bits per channel.
///
/// Scaled so that all-ones reaches 255 rather than 248, which is what a bare shift would give.
fn endpoint(packed: u16) -> [u8; 3] {
    let (r, g, b) = (packed >> 11, (packed >> 5) & 0x3F, packed & 0x1F);
    [
        ((r * 255 + 15) / 31) as u8,
        ((g * 255 + 31) / 63) as u8,
        ((b * 255 + 15) / 31) as u8,
    ]
}

fn lerp(a: [u8; 3], b: [u8; 3], numerator: u16, denominator: u16) -> [u8; 3] {
    let mix = |x: u8, y: u8| {
        ((u16::from(x) * (denominator - numerator) + u16::from(y) * numerator) / denominator) as u8
    };
    [mix(a[0], b[0]), mix(a[1], b[1]), mix(a[2], b[2])]
}

/// The four colours a BC1 block's texels choose between, and whether the fourth is transparent.
///
/// **The endpoints' order is the mode bit.** With the first above the second there are four opaque
/// colours; otherwise the third is their midpoint and the fourth is transparent black, which is how
/// BC1 spells one-bit alpha.
fn palette(block: &[u8], always_opaque: bool) -> ([[u8; 3]; 4], bool) {
    let first = u16::from(block[0]) | (u16::from(block[1]) << 8);
    let second = u16::from(block[2]) | (u16::from(block[3]) << 8);
    let (a, b) = (endpoint(first), endpoint(second));
    if always_opaque || first > second {
        ([a, b, lerp(a, b, 1, 3), lerp(a, b, 2, 3)], true)
    } else {
        ([a, b, lerp(a, b, 1, 2), [0, 0, 0]], false)
    }
}

/// The top level as `RGBA8`, whatever the texture is stored as.
///
/// See [`Texture::to_rgba8`], which is how a caller reaches this.
pub(crate) fn expand(texture: &Texture) -> Vec<u8> {
    let (width, height) = (texture.width().max(1), texture.height().max(1));
    let mut out = vec![0u8; (width * height * 4) as usize];
    let level = texture.level_data(0);
    let mut put = |x: u32, y: u32, rgb: [u8; 3], alpha: u8| {
        if x < width && y < height {
            let at = ((y * width + x) * 4) as usize;
            out[at..at + 4].copy_from_slice(&[rgb[0], rgb[1], rgb[2], alpha]);
        }
    };

    match texture.format() {
        format @ (TextureFormat::Bc1 | TextureFormat::Bc2) => {
            let stride = format.unit_size() as usize;
            let colour_at = usize::from(format == TextureFormat::Bc2) * 8;
            let blocks_wide = width.div_ceil(4).max(1);
            for (index, block) in level.chunks_exact(stride).enumerate() {
                let (bx, by) = (index as u32 % blocks_wide, index as u32 / blocks_wide);
                let colour = &block[colour_at..colour_at + 8];
                // BC2 carries alpha of its own, so its colour block never spends a slot on it.
                let (colours, opaque) = palette(colour, colour_at > 0);
                for texel in 0..16u32 {
                    let choice = (colour[4 + (texel / 4) as usize] >> ((texel % 4) * 2)) & 0b11;
                    let alpha = match colour_at {
                        // Four bits per texel, low nibble first, spread back over the byte.
                        0 if opaque || choice != 3 => 255,
                        0 => 0,
                        _ => {
                            let packed = block[(texel / 2) as usize];
                            let nibble = if texel % 2 == 0 {
                                packed & 0xF
                            } else {
                                packed >> 4
                            };
                            nibble * 17
                        }
                    };
                    put(
                        bx * 4 + texel % 4,
                        by * 4 + texel / 4,
                        colours[choice as usize],
                        alpha,
                    );
                }
            }
        }
        format => {
            let (r, b) = if format == TextureFormat::Bgra8 {
                (2, 0)
            } else {
                (0, 2)
            };
            for (index, texel) in level.chunks_exact(4).enumerate() {
                let (x, y) = (index as u32 % width, index as u32 / width);
                put(x, y, [texel[r], texel[1], texel[b]], texel[3]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One BC1 block: white and black endpoints, one texel of each palette entry per row.
    fn block(first: u16, second: u16) -> Vec<u8> {
        let mut bytes = vec![
            (first & 0xFF) as u8,
            (first >> 8) as u8,
            (second & 0xFF) as u8,
            (second >> 8) as u8,
        ];
        // 0b11_10_01_00: indices 0, 1, 2, 3 across each row of four.
        bytes.extend([0b11_10_01_00u8; 4]);
        bytes
    }

    #[test]
    fn a_bc1_block_expands_to_its_palette_in_order() {
        // White above black, so the four-colour palette: the endpoints and two thirds between them.
        let texture = Texture::from_pixels(TextureFormat::Bc1, 4, 4, block(0xFFFF, 0x0000));
        let rgba = texture.to_rgba8();
        let red = |texel: usize| rgba[texel * 4];
        assert_eq!(
            [red(0), red(1), red(2), red(3)],
            [255, 0, 170, 85],
            "the palette should run white, black, two thirds, one third"
        );
        assert!(
            rgba.chunks_exact(4).all(|t| t[3] == 255),
            "the four-colour mode has no transparent entry"
        );
        // Every row of the block chose the same indices, so every row expands the same way.
        assert_eq!(red(4), 255, "the second row should repeat the first");
    }

    #[test]
    fn the_three_colour_mode_spells_alpha_with_its_fourth_entry() {
        // Black *below* white this time, which is the mode bit: the third entry becomes the
        // midpoint and the fourth is transparent rather than an interpolant.
        let texture = Texture::from_pixels(TextureFormat::Bc1, 4, 4, block(0x0000, 0xFFFF));
        let rgba = texture.to_rgba8();
        assert_eq!(
            [rgba[0], rgba[4], rgba[8]],
            [0, 255, 127],
            "the palette should run black, white, their midpoint"
        );
        assert_eq!(rgba[3 * 4 + 3], 0, "the fourth entry is transparent");
        assert_eq!(rgba[7], 255, "and the others are not");
    }

    #[test]
    fn an_uncompressed_texture_comes_back_as_it_went_in() {
        // `Bgra8` is how a `.dds` stores it, and the swap is the only thing this does to it.
        let texture = Texture::from_pixels(TextureFormat::Bgra8, 1, 1, vec![10, 20, 30, 40]);
        assert_eq!(texture.to_rgba8(), vec![30, 20, 10, 40]);
        let texture = Texture::from_pixels(TextureFormat::Rgba8, 1, 1, vec![10, 20, 30, 40]);
        assert_eq!(texture.to_rgba8(), vec![10, 20, 30, 40]);
    }
}
