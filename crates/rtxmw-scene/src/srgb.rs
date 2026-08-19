//! Decoding the colours Morrowind stores, and what the eye makes of them once decoded.

use glam::Vec3;

/// What the eye makes of a linear colour, which is the only sense in which one has a brightness.
///
/// Rec. 709, matching the primaries every texture in the game is decoded to.
pub(crate) const LUMA: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// Converts a packed `0xAABBGGRR` colour to linear RGB.
///
/// Everything the game stores as a colour — light tints, cell ambient, material colours — was
/// authored against a fixed-function renderer and is sRGB-encoded. Using those bytes directly in a
/// renderer that works in linear light makes every one of them too bright and too washed out, the
/// same error as sampling an albedo texture through a UNORM view. The difference is largest in the
/// midtones and vanishes at both ends, which is why it reads as "a bit off" rather than as broken.
pub fn to_linear(packed: u32) -> Vec3 {
    let channel = |shift: u32| {
        let encoded = ((packed >> shift) & 0xFF) as f32 / 255.0;
        if encoded <= 0.04045 {
            encoded / 12.92
        } else {
            ((encoded + 0.055) / 1.055).powf(2.4)
        }
    };
    Vec3::new(channel(0), channel(8), channel(16))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_packed_colour_unpacks_red_first_and_decodes_to_linear() {
        // 0xAABBGGRR, so the low byte is red and the high one is alpha, which is ignored.
        let red = to_linear(0x0000_00FF);
        assert!((red.x - 1.0).abs() < 1e-5);
        assert_eq!((red.y, red.z), (0.0, 0.0));
        assert_eq!(to_linear(0xFF00_0000), Vec3::ZERO);

        let blue = to_linear(0x00FF_0000);
        assert!((blue.z - 1.0).abs() < 1e-5);

        // Mid grey is where the two spaces diverge most: 0.502 encoded is 0.216 linear. White and
        // black are identical in both, so neither can detect a missing conversion.
        let grey = to_linear(0x0080_8080);
        assert!((grey.x - 0.2158).abs() < 1e-3, "got {}", grey.x);
        for channel in to_linear(0x00FF_FFFF).to_array() {
            assert!((channel - 1.0).abs() < 1e-5);
        }
    }
}
