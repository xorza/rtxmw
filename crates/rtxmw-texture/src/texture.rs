//! A decoded texture: pixel data plus the shape needed to upload it.

use crate::error::{Result, TextureError};
use crate::rgba8;
use crate::shading_map;
use crate::{dds, tga};

/// How a texture's bytes are encoded.
///
/// Deliberately not a `VkFormat`: nothing below `rtxmw-gpu` knows about Vulkan, and the *colour
/// space* is not the decoder's business either. Morrowind's textures are albedo and want an sRGB
/// view, but a replacer pack's normal map is the same BC1 bytes wanting a UNORM one — so the
/// consumer picks, and this only says what the bytes are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    /// DXT1. Eight bytes per 4x4 block, RGB with an optional one-bit alpha.
    Bc1,
    /// DXT3. Sixteen bytes per 4x4 block: explicit four-bit alpha, then a BC1 colour block.
    Bc2,
    /// Uncompressed, blue-green-red-alpha byte order — Direct3D's `A8R8G8B8` as it sits in a file.
    Bgra8,
    /// Uncompressed, red-green-blue-alpha byte order.
    Rgba8,
}

impl TextureFormat {
    /// Bytes each unit of the format occupies: one 4x4 block, or one texel.
    pub fn unit_size(self) -> u32 {
        match self {
            Self::Bc1 => 8,
            Self::Bc2 => 16,
            Self::Bgra8 | Self::Rgba8 => 4,
        }
    }

    /// Texels along each axis of one unit — four for block formats, one otherwise.
    pub fn unit_extent(self) -> u32 {
        match self {
            Self::Bc1 | Self::Bc2 => 4,
            Self::Bgra8 | Self::Rgba8 => 1,
        }
    }

    /// Bytes one mip level of `width` x `height` occupies.
    ///
    /// Block formats round up, so a 2x2 mip of a BC1 chain still costs a whole 8-byte block. Losing
    /// that rounding truncates every level below 4x4 and desynchronises the rest of the chain.
    pub fn level_size(self, width: u32, height: u32) -> u32 {
        let unit = self.unit_extent();
        width.div_ceil(unit) * height.div_ceil(unit) * self.unit_size()
    }
}

/// Where one mip level sits inside a texture's single data buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MipLevel {
    pub offset: u32,
    pub size: u32,
    pub width: u32,
    pub height: u32,
}

/// One decoded texture and its whole mip chain.
///
/// The levels share one buffer with a range table beside it rather than being a `Vec<Vec<u8>>`,
/// which would allocate per level for the eleven-deep chains the game ships. It is also the shape
/// an upload wants: one staging buffer and one copy region per level.
#[derive(Debug, Clone)]
pub struct Texture {
    format: TextureFormat,
    data: Vec<u8>,
    /// Never empty, and level zero is the full image — so it is also where the size comes from.
    levels: Vec<MipLevel>,
}

impl Texture {
    /// Decodes a texture from file bytes, choosing the container by content rather than by name.
    ///
    /// Sniffing rather than trusting the extension is not fussiness: the original engine forces a
    /// `.dds` extension onto paths that name a `.tga`, so a file's name routinely disagrees with
    /// what is inside it.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.starts_with(dds::MAGIC) {
            dds::decode(bytes)
        } else {
            tga::decode(bytes)
        }
    }

    /// Builds a single-level texture from pixels held in memory.
    ///
    /// For images the renderer synthesises rather than loads — a fallback for a missing file being
    /// the one that matters.
    pub fn from_pixels(format: TextureFormat, width: u32, height: u32, data: Vec<u8>) -> Self {
        let levels = vec![MipLevel {
            offset: 0,
            size: format.level_size(width, height),
            width,
            height,
        }];
        Self::new(format, data, levels).expect("synthesised pixels match their own level table")
    }

    /// Builds a texture from already-decoded bytes and its level table.
    ///
    /// A short buffer is a malformed file and comes back as an error; an empty level table is a
    /// caller that built one wrong, which no input can cause.
    pub(crate) fn new(format: TextureFormat, data: Vec<u8>, levels: Vec<MipLevel>) -> Result<Self> {
        assert!(!levels.is_empty(), "a texture needs at least one mip level");

        let last = levels[levels.len() - 1];
        let end = last.offset as usize + last.size as usize;
        if end > data.len() {
            return Err(TextureError::Truncated {
                wanted: end,
                available: data.len(),
            });
        }
        Ok(Self {
            format,
            data,
            levels,
        })
    }

    /// How the bytes are encoded.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// Width of the largest level.
    pub fn width(&self) -> u32 {
        self.levels[0].width
    }

    /// Height of the largest level.
    pub fn height(&self) -> u32 {
        self.levels[0].height
    }

    /// The same texture with a full mip chain built under it, for one that shipped without.
    ///
    /// **Uncompressed formats only**, which is what the callers that need this have: Morrowind's
    /// `tx_sky_*` sheets are `Bgra8` with a single level, and a shader minifying one — a cloud layer
    /// repeats its tile forty times across the last degrees above the horizon — has nothing to fall
    /// back to and aliases. Block formats already ship their chains, and downsampling one would mean
    /// decoding and re-encoding it.
    ///
    /// A box filter, which is what a mip chain is: each level is the average of four texels of the
    /// one above, in whatever byte order the format keeps. Returns the texture unchanged where it
    /// already has a chain or the format is compressed.
    pub fn with_mips(self) -> Self {
        let unit = self.format.unit_size() as usize;
        let compressed = self.format.unit_extent() != 1;
        if self.levels.len() > 1 || compressed {
            return self;
        }
        let (width, height) = (self.width(), self.height());
        let wanted = describe_levels(self.format, width, height, mip_count(width, height));
        let total = wanted.last().map_or(0, |l| (l.offset + l.size) as usize);
        let mut data = Vec::with_capacity(total);
        data.extend_from_slice(&self.data[..wanted[0].size as usize]);

        for pair in wanted.windows(2) {
            let (above, level) = (pair[0], pair[1]);
            let base = above.offset as usize;
            for y in 0..level.height {
                for x in 0..level.width {
                    // The four texels above this one, clamped where a level is one texel wide.
                    let sx = (2 * x).min(above.width - 1);
                    let sy = (2 * y).min(above.height - 1);
                    let ox = (sx + 1).min(above.width - 1);
                    let oy = (sy + 1).min(above.height - 1);
                    let at = |px: u32, py: u32| base + ((py * above.width + px) as usize) * unit;
                    let corners = [at(sx, sy), at(ox, sy), at(sx, oy), at(ox, oy)];
                    for byte in 0..unit {
                        let sum: u32 = corners.iter().map(|&c| data[c + byte] as u32).sum();
                        data.push(((sum + 2) / 4) as u8);
                    }
                }
            }
        }
        Self::new(self.format, data, wanted).expect("a box-filtered chain matches its own table")
    }

    /// Every mip level, largest first.
    pub fn levels(&self) -> &[MipLevel] {
        &self.levels
    }

    /// The lighting this texture appears to have been painted with, at low resolution.
    ///
    /// **Normalised, so the correction is a redistribution rather than a brightness change.**
    /// Dividing a texture by this leaves its average colour where it was and flattens only the
    /// variation across it, which is what separates baked shading from base colour — see
    /// `docs/design.md` §5.1 for why vanilla assets need it at all.
    ///
    /// Returned as a texture because that is how it reaches the GPU: the bindless array already
    /// takes one of these per entry, so a map needs no binding, no format and no upload path of its
    /// own.
    pub fn shading_map(&self) -> Texture {
        shading_map::estimate(self)
    }

    /// The top level as `RGBA8`, whatever this is stored as.
    ///
    /// **For inspection, not for rendering.** Every target samples BC natively, so a frame never
    /// wants this — what does is a tool that has to draw a texture into an image of its own.
    pub fn to_rgba8(&self) -> Vec<u8> {
        rgba8::expand(self)
    }

    /// What one byte of a [`Texture::shading_map`] means, as the multiplier to divide a texel by.
    ///
    /// Exposed because a tool that draws the correction has to do it the way the sampler does; see
    /// `SHADING_SCALE` in `surface.glsl` for the shader's own copy of the same number.
    pub fn shading_multiplier(encoded: u8) -> f32 {
        shading_map::multiplier(encoded)
    }

    /// A shading map that changes nothing, for a material whose texture never loaded.
    pub fn neutral_shading() -> Texture {
        shading_map::neutral()
    }

    /// The whole chain's bytes, which the level table indexes into.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The bytes of one level.
    pub fn level_data(&self, level: usize) -> &[u8] {
        let level = self.levels[level];
        &self.data[level.offset as usize..level.offset as usize + level.size as usize]
    }
}

/// How many levels a full chain from `width` x `height` has, down to a single texel.
pub(crate) fn mip_count(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

/// The chain of `count` levels starting at `width` x `height`.
///
/// Each level halves, floored at one texel, which is the convention every mipped format uses.
pub(crate) fn describe_levels(
    format: TextureFormat,
    width: u32,
    height: u32,
    count: u32,
) -> Vec<MipLevel> {
    let mut levels = Vec::with_capacity(count as usize);

    let mut offset = 0;
    let (mut w, mut h) = (width, height);
    for _ in 0..count {
        let size = format.level_size(w, h);
        levels.push(MipLevel {
            offset,
            size,
            width: w,
            height: h,
        });
        offset += size;
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_formats_round_a_level_up_to_whole_blocks() {
        // 4x4 is exactly one block; anything smaller still costs a whole one.
        assert_eq!(TextureFormat::Bc1.level_size(4, 4), 8);
        assert_eq!(TextureFormat::Bc1.level_size(2, 2), 8);
        assert_eq!(TextureFormat::Bc1.level_size(1, 1), 8);
        // 256x256 is 64x64 blocks at 8 bytes each.
        assert_eq!(TextureFormat::Bc1.level_size(256, 256), 64 * 64 * 8);
        // BC2 is the same block count at twice the size.
        assert_eq!(TextureFormat::Bc2.level_size(256, 256), 64 * 64 * 16);
        // Non-multiples round up: 5 texels need two blocks.
        assert_eq!(TextureFormat::Bc1.level_size(5, 5), 2 * 2 * 8);

        // Uncompressed is one unit per texel.
        assert_eq!(TextureFormat::Bgra8.level_size(5, 5), 100);
    }

    #[test]
    fn a_sheet_with_no_chain_gets_a_box_filtered_one() {
        // **Four texels averaging to one**, checked on a pattern whose answer is arithmetic: a 2x2
        // of 0, 100, 200 and 255 in every channel comes to 139 — `(555 + 2) / 4`.
        let pixels: Vec<u8> = [0u8, 100, 200, 255]
            .iter()
            .flat_map(|&v| [v, v, v, v])
            .collect();
        let flat = Texture::from_pixels(TextureFormat::Rgba8, 2, 2, pixels);
        assert_eq!(flat.levels().len(), 1, "the fixture starts with no chain");

        let mipped = flat.with_mips();
        assert_eq!(mipped.levels().len(), 2, "2x2 has one level under it");
        let one = mipped.levels()[1];
        assert_eq!((one.width, one.height), (1, 1));
        assert_eq!(&mipped.level_data(1)[..4], &[139, 139, 139, 139]);
        // The top level is untouched, which is the half a caller can least afford to lose.
        assert_eq!(&mipped.level_data(0)[..4], &[0, 0, 0, 0]);

        // **A chain that already exists is left alone**, and so is a block format, which would have
        // to be decoded and re-encoded to filter.
        assert_eq!(mipped.clone().with_mips().levels().len(), 2);
        let block = Texture::from_pixels(TextureFormat::Bc1, 4, 4, vec![0; 8]);
        assert_eq!(block.with_mips().levels().len(), 1);
    }

    #[test]
    fn a_mip_chain_halves_to_one_texel_and_packs_end_to_end() {
        let levels = describe_levels(TextureFormat::Bgra8, 8, 4, 4);

        // 8x4, 4x2, 2x1, 1x1 — height floors at one and stays there while width keeps halving.
        assert_eq!(
            levels,
            vec![
                MipLevel {
                    offset: 0,
                    size: 128,
                    width: 8,
                    height: 4
                },
                MipLevel {
                    offset: 128,
                    size: 32,
                    width: 4,
                    height: 2
                },
                MipLevel {
                    offset: 160,
                    size: 8,
                    width: 2,
                    height: 1
                },
                MipLevel {
                    offset: 168,
                    size: 4,
                    width: 1,
                    height: 1
                },
            ]
        );
    }

    #[test]
    fn a_texture_shorter_than_its_level_table_is_rejected() {
        let levels = describe_levels(TextureFormat::Bc1, 8, 8, 2);
        // 8x8 is 32 bytes, 4x4 is 8: the chain wants 40.
        assert_eq!(levels.iter().map(|l| l.size).sum::<u32>(), 40);

        let short = Texture::new(TextureFormat::Bc1, vec![0; 39], levels.clone());
        assert!(matches!(
            short,
            Err(TextureError::Truncated {
                wanted: 40,
                available: 39
            })
        ));
        let whole = Texture::new(TextureFormat::Bc1, vec![0; 40], levels).expect("fits");
        // The size comes from level zero rather than being carried alongside it, so the two cannot
        // drift apart.
        assert_eq!((whole.width(), whole.height()), (8, 8));
    }
}

#[cfg(any(test, feature = "internals"))]
pub mod internals {
    //! Building a texture by hand, for tests that need a mip chain a decoder would have supplied.

    use super::{MipLevel, Texture, TextureFormat};

    impl Texture {
        /// A texture whose levels are given outright, finest first.
        ///
        /// [`Texture::from_pixels`] makes a single level, which is all a fixture needs to check a
        /// colour — but anything about *minification* is invisible without a chain, because a
        /// sampler asked for a level a texture does not have simply returns the one it has.
        pub fn from_levels(
            format: TextureFormat,
            width: u32,
            height: u32,
            levels: &[&[u8]],
        ) -> Self {
            let mut data = Vec::new();
            let mut chain = Vec::with_capacity(levels.len());
            for (index, bytes) in levels.iter().enumerate() {
                let scale = 1u32 << index;
                chain.push(MipLevel {
                    offset: data.len() as u32,
                    size: bytes.len() as u32,
                    width: (width / scale).max(1),
                    height: (height / scale).max(1),
                });
                data.extend_from_slice(bytes);
            }
            Self {
                format,
                levels: chain,
                data,
            }
        }
    }
}
