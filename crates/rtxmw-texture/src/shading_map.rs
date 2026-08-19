//! Estimating the lighting a vanilla texture was painted with.

use crate::texture::{Texture, TextureFormat};

/// Side of the estimate, in texels.
///
/// **Small on purpose.** What is being separated out is ambient occlusion and directional shading
/// painted across a surface, which varies over the whole texture rather than between neighbouring
/// texels. Anything finer starts describing the painted detail instead, and dividing that out is the
/// over-correction `docs/design.md` §5.1 names as the failure to watch for — flat, washed-out output
/// where the algorithm removed what the artist meant.
const SIDE: u32 = 32;

/// How far the estimate is allowed from neutral.
///
/// A texture with a black region would otherwise divide by nearly zero and blow the corrected
/// albedo past white. The clamp costs nothing on a surface whose shading is gentle, which is all of
/// them that this is meant for.
const RANGE: std::ops::RangeInclusive<f32> = 0.5..=2.0;

/// Rec. 709, matching the primaries the textures are decoded to.
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// The lighting `texture` appears to have been painted with, as a [`SIDE`]-square map whose mean is
/// one. See [`Texture::shading_map`], which is how a caller reaches this.
pub(crate) fn estimate(texture: &Texture) -> Texture {
    let coarse = coarse_luminance(texture);
    let smoothed = blur(&coarse);

    let mean = smoothed.iter().sum::<f32>() / smoothed.len() as f32;
    // A texture with no light in it at all has nothing to estimate, and normalising by its mean
    // would be a division by zero. Neutral is the honest answer.
    let scale = if mean > 1e-4 { 1.0 / mean } else { 0.0 };

    let mut data = Vec::with_capacity((SIDE * SIDE * 4) as usize);
    for value in smoothed {
        let shading = if scale == 0.0 {
            1.0
        } else {
            (value * scale).clamp(*RANGE.start(), *RANGE.end())
        };
        // **Through the sRGB transfer, because the array binds these as an sRGB format.** The
        // hardware decodes every sample it takes, so a value written linearly would come back
        // gamma-corrected; encoding it here means the decode gives back exactly what was meant.
        let byte = (to_srgb(shading / RANGE.end()) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8;
        data.extend([byte, byte, byte, 255]);
    }
    Texture::from_pixels(TextureFormat::Rgba8, SIDE, SIDE, data)
}

/// Mean luminance over a [`SIDE`]-square grid of `texture`'s top level.
///
/// **Block averages where the format has them.** A BC1 or BC2 block carries two endpoint colours and
/// sixteen two-bit indices, so its mean is arithmetic on twenty bytes — there is no need to
/// decompress, which this crate deliberately never does.
fn coarse_luminance(texture: &Texture) -> Vec<f32> {
    let (width, height) = (texture.width().max(1), texture.height().max(1));
    let mut total = vec![0.0f32; (SIDE * SIDE) as usize];
    let mut count = vec![0u32; (SIDE * SIDE) as usize];
    let mut add = |x: u32, y: u32, luminance: f32| {
        let cell = (y * SIDE / height).min(SIDE - 1) * SIDE + (x * SIDE / width).min(SIDE - 1);
        total[cell as usize] += luminance;
        count[cell as usize] += 1;
    };

    let level = texture.level_data(0);
    match texture.format() {
        TextureFormat::Bc1 | TextureFormat::Bc2 => {
            let stride = texture.format().unit_size() as usize;
            let colour_at = usize::from(texture.format() == TextureFormat::Bc2) * 8;
            for (index, block) in level.chunks_exact(stride).enumerate() {
                let blocks_wide = width.div_ceil(4).max(1);
                let (bx, by) = (index as u32 % blocks_wide, index as u32 / blocks_wide);
                add(
                    bx * 4,
                    by * 4,
                    block_luminance(&block[colour_at..colour_at + 8]),
                );
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
                if y >= height {
                    break;
                }
                add(x, y, luminance(texel[r], texel[1], texel[b]));
            }
        }
    }

    // **A cell with no texel in it takes the average rather than zero.** A texture smaller than the
    // grid — and Morrowind has plenty, down to single texels — lands in a handful of cells and
    // leaves the rest empty; reading those as black makes the estimate a spike on a floor, and
    // normalising by its mean then sends the correction to the clamps. Filling them with the mean
    // makes such a texture come out neutral, which is the right answer: too few texels to resolve
    // any shading is the same as having none.
    let sampled: f32 = total
        .iter()
        .zip(&count)
        .filter(|(_, n)| **n > 0)
        .map(|(sum, n)| sum / *n as f32)
        .sum();
    let filled = count.iter().filter(|n| **n > 0).count();
    let mean = if filled == 0 {
        0.0
    } else {
        sampled / filled as f32
    };

    total
        .iter()
        .zip(&count)
        .map(|(sum, n)| if *n == 0 { mean } else { sum / *n as f32 })
        .collect()
}

/// A map that changes nothing, for a material whose texture is missing.
///
/// **Not the array's fallback.** An absent shading map would leave the slot holding the magenta
/// stand-in, whose red channel decodes to the top of the range — so every untextured surface would
/// be divided by two.
pub(crate) fn neutral() -> Texture {
    let byte = (to_srgb(1.0 / RANGE.end()) * 255.0).round() as u8;
    Texture::from_pixels(TextureFormat::Rgba8, 1, 1, vec![byte, byte, byte, 255])
}

/// The sRGB transfer function, as the IEC standard defines it.
fn to_srgb(linear: f32) -> f32 {
    let linear = linear.clamp(0.0, 1.0);
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// Its inverse, which is what the sampler applies. Here for the tests that read a map back.
#[cfg(test)]
fn to_linear(encoded: f32) -> f32 {
    if encoded <= 0.040_45 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

fn luminance(r: u8, g: u8, b: u8) -> f32 {
    (LUMA[0] * f32::from(r) + LUMA[1] * f32::from(g) + LUMA[2] * f32::from(b)) / 255.0
}

/// Mean luminance of one BC1 colour block, from its endpoints and indices.
fn block_luminance(block: &[u8]) -> f32 {
    let endpoint = |at: usize| {
        let packed = u16::from(block[at]) | (u16::from(block[at + 1]) << 8);
        // 5-6-5, expanded so that all-ones maps to 255 rather than to 248.
        let (r, g, b) = (packed >> 11, (packed >> 5) & 0x3F, packed & 0x1F);
        [
            ((r * 255 + 15) / 31) as u8,
            ((g * 255 + 31) / 63) as u8,
            ((b * 255 + 15) / 31) as u8,
        ]
    };
    let (first, second) = (endpoint(0), endpoint(2));
    let (a, b) = (
        luminance(first[0], first[1], first[2]),
        luminance(second[0], second[1], second[2]),
    );
    // The four-colour palette when the first endpoint sorts above the second, and the three-colour
    // one with a transparent fourth entry otherwise — which is how BC1 spells one-bit alpha.
    let opaque = (u16::from(block[1]) << 8 | u16::from(block[0]))
        > (u16::from(block[3]) << 8 | u16::from(block[2]));
    let palette = if opaque {
        [a, b, (2.0 * a + b) / 3.0, (a + 2.0 * b) / 3.0]
    } else {
        [a, b, (a + b) / 2.0, 0.0]
    };

    let mut total = 0.0;
    for texel in 0..16 {
        let index = (block[4 + texel / 4] >> ((texel % 4) * 2)) & 0b11;
        total += palette[index as usize];
    }
    total / 16.0
}

/// Three box passes over the grid, which is a close enough Gaussian for a field this coarse.
///
/// Without it the estimate still carries whatever landed inside one cell, and dividing by that would
/// print the grid onto the corrected texture.
fn blur(grid: &[f32]) -> Vec<f32> {
    let mut current = grid.to_vec();
    let mut next = current.clone();
    for _ in 0..3 {
        for y in 0..SIDE as i32 {
            for x in 0..SIDE as i32 {
                let mut total = 0.0;
                let mut count = 0.0;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (sx, sy) = (x + dx, y + dy);
                        if (0..SIDE as i32).contains(&sx) && (0..SIDE as i32).contains(&sy) {
                            total += current[(sy * SIDE as i32 + sx) as usize];
                            count += 1.0;
                        }
                    }
                }
                next[(y * SIDE as i32 + x) as usize] = total / count;
            }
        }
        std::mem::swap(&mut current, &mut next);
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An `Rgba8` texture whose grey level at each texel is `shade(x, y)` on `0..1`.
    fn grey(side: u32, shade: impl Fn(f32, f32) -> f32) -> Texture {
        let mut data = Vec::with_capacity((side * side * 4) as usize);
        for y in 0..side {
            for x in 0..side {
                let level = (shade(x as f32 / side as f32, y as f32 / side as f32) * 255.0)
                    .clamp(0.0, 255.0) as u8;
                data.extend([level, level, level, 255]);
            }
        }
        Texture::from_pixels(TextureFormat::Rgba8, side, side, data)
    }

    /// The map's values as the multipliers they encode, decoded the way the sampler will.
    fn shading(map: &Texture) -> Vec<f32> {
        map.data()
            .chunks_exact(4)
            .map(|texel| to_linear(f32::from(texel[0]) / 255.0) * RANGE.end())
            .collect()
    }

    #[test]
    fn the_shader_scales_a_map_by_the_same_number_this_does() {
        // `surface.glsl` spells this out as `SHADING_SCALE`, because a GLSL shader cannot see a Rust
        // constant. Nothing links the two but this: change the range and the shader reads every map
        // wrongly, with no error anywhere and only a shift in brightness to show for it.
        assert_eq!(*RANGE.end(), 2.0);
    }

    #[test]
    fn a_flat_texture_has_no_shading_to_remove() {
        // **The case that must not be touched.** A surface painted one colour carries no baked
        // lighting, and a de-lighting pass that alters it is subtracting something that was never
        // there — which is over-correction in its purest form.
        for level in [0.05, 0.5, 0.95] {
            let map = grey(64, |_, _| level).shading_map();
            let values = shading(&map);
            let worst = values
                .iter()
                .map(|v| (v - 1.0).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst < 0.01,
                "a flat texture at {level} produced a correction of up to {worst:.3} away from \
                 neutral, so de-lighting it would change a surface with no shading in it"
            );
        }
    }

    #[test]
    fn a_painted_gradient_is_recovered_and_averages_to_neutral() {
        // Luminance ramping from a quarter to three quarters across the texture, which is what a
        // wall with light falling across it looks like once an artist has painted it in.
        let map = grey(64, |x, _| 0.25 + 0.5 * x).shading_map();
        let values = shading(&map);

        // **Its mean is one by construction**, so dividing by it moves no brightness overall — the
        // correction redistributes rather than darkens.
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        assert!(
            (mean - 1.0).abs() < 0.02,
            "the estimate averages {mean:.3} rather than one, so dividing by it would shift the \
             texture's overall brightness"
        );

        // Left edge sits at 0.25 of the ramp and the right at 0.75, a ratio of three; the blur
        // pulls the ends in, so the recovered ratio is somewhat less. It has to be clearly above
        // one and in the right direction.
        let row = SIDE as usize / 2 * SIDE as usize;
        let (left, right) = (values[row + 2], values[row + SIDE as usize - 3]);
        println!("gradient recovered as {left:.3} to {right:.3}, mean {mean:.3}");
        assert!(
            right / left > 1.8,
            "a ramp of three to one came back as {right:.3} over {left:.3}, which is too flat to \
             be removing the shading that is there"
        );
    }

    #[test]
    fn painted_detail_is_left_alone() {
        // **The guard §5.1 asks for.** A checkerboard is detail, not shading: it alternates every
        // texel, so no lighting could have produced it. An estimate that follows it would divide
        // the artist's work out of the texture and leave it washed out.
        let map = grey(64, |x, y| {
            let cell = |v: f32| (v * 32.0) as u32 % 2;
            if cell(x) == cell(y) { 0.2 } else { 0.8 }
        })
        .shading_map();
        let values = shading(&map);
        let worst = values
            .iter()
            .map(|v| (v - 1.0).abs())
            .fold(0.0f32, f32::max);
        println!("checkerboard produced at most {worst:.3} away from neutral");
        assert!(
            worst < 0.05,
            "detail alternating every texel was read as shading, up to {worst:.3} from neutral; \
             dividing that out is the over-correction that flattens a texture"
        );
    }

    #[test]
    fn a_texture_too_small_to_resolve_shading_comes_back_neutral() {
        // **Smaller than the grid, which Morrowind has plenty of.** A single texel lands in one cell
        // of the thirty-two square and leaves the rest with nothing in them; reading those as black
        // makes the estimate a spike on a floor, and normalising by its mean drives the correction
        // to the clamps — a flat texture would come out divided by two at one corner and doubled
        // everywhere else.
        for side in [1, 2, 8] {
            let map = grey(side, |_, _| 0.6).shading_map();
            let worst = shading(&map)
                .iter()
                .map(|v| (v - 1.0).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst < 0.01,
                "a {side}x{side} texture produced a correction of up to {worst:.3} from neutral, \
                 when it has too few texels for any shading to be visible in"
            );
        }
    }

    #[test]
    fn a_missing_texture_gets_a_map_that_changes_nothing() {
        // The array's own stand-in is magenta, whose red channel is the top of the range — so a
        // slot left empty would divide the surface by two rather than leave it alone.
        let values = shading(&Texture::neutral_shading());
        assert!(
            values.iter().all(|v| (v - 1.0).abs() < 0.01),
            "the neutral map is not neutral: {values:?}"
        );
    }

    #[test]
    fn a_compressed_block_averages_its_palette() {
        // BC1's two endpoints and sixteen indices, without decompressing: the mean is arithmetic on
        // the block, and getting it wrong would misread every shipped texture, all of which are
        // compressed.
        //
        // Endpoints white and black, with `c0 > c1` so the palette is the four-colour one:
        // white, black, two thirds white, one third white. Indices chosen so each appears four
        // times, which averages to exactly the mean of the four.
        let mut block = vec![0xFF, 0xFF, 0x00, 0x00];
        // Four texels of each index, packed two bits apiece: 0b11_10_01_00 per row.
        block.extend([0b11_10_01_00u8; 4]);
        let expected = (1.0 + 0.0 + 2.0 / 3.0 + 1.0 / 3.0) / 4.0;
        let measured = block_luminance(&block);
        println!("block averages {measured:.4}, hand-computed {expected:.4}");
        assert!(
            (measured - expected).abs() < 0.01,
            "a block of white, black and the two interpolants averaged {measured} rather than \
             {expected}"
        );

        // And the three-colour palette, where the fourth entry is transparent black rather than an
        // interpolant — which is how BC1 spells one-bit alpha.
        let mut cutout = vec![0x00, 0x00, 0xFF, 0xFF];
        cutout.extend([0b11_10_01_00u8; 4]);
        let expected = (0.0 + 1.0 + 0.5 + 0.0) / 4.0;
        assert!(
            (block_luminance(&cutout) - expected).abs() < 0.01,
            "the three-colour palette averaged {} rather than {expected}",
            block_luminance(&cutout)
        );
    }
}
