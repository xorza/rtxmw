//! The colour space every texture is decoded into, and what the eye makes of one.

/// Rec. 709 luminance weights, matching the primaries every texture is decoded to.
///
/// **The one copy on the host**, which is why it sits in the lowest crate that needs it rather than
/// beside either caller: `shading_map` weighs a texel with it to estimate what lighting was
/// painted into a texture, and `rtxmw_scene::LUMA` is this, widened to a `Vec3`. A plain array
/// because this crate has no dependencies at all and adding a maths one for three numbers would be
/// the tail wagging the dog.
///
/// `shaders/colour.glsl` is the same three numbers in GLSL, which no host constant can reach.
pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// The sRGB transfer function, as the IEC standard defines it.
pub(crate) fn to_srgb(linear: f32) -> f32 {
    let linear = linear.clamp(0.0, 1.0);
    match linear <= 0.003_130_8 {
        true => linear * 12.92,
        false => 1.055 * linear.powf(1.0 / 2.4) - 0.055,
    }
}

/// Its inverse, which is what a sampler applies to an sRGB view.
pub(crate) fn to_linear(encoded: f32) -> f32 {
    match encoded <= 0.040_45 {
        true => encoded / 12.92,
        false => ((encoded + 0.055) / 1.055).powf(2.4),
    }
}

/// `to_linear` for a whole byte, which is the shape almost every caller has.
///
/// **What anything holding loose bytes wants**, here rather than open-coded beside each of them: a
/// decoded texture is `[u8]`, an ini colour is three of them, and a Morrowind record packs four
/// into a word. Every copy of a curve with four constants in it is another place to mistype one.
pub fn channel_to_linear(encoded: u8) -> f32 {
    to_linear(f32::from(encoded) / 255.0)
}

/// [`channel_to_linear`] the other way, for something writing a byte a sampler will decode.
///
/// The clamp is [`to_srgb`]'s own, so the product cannot leave `0..=255` and a caller needs no
/// second one — which is what two of these used to carry between them and one did not.
pub(crate) fn channel_to_srgb(linear: f32) -> u8 {
    (to_srgb(linear) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_curve_is_its_own_inverse_and_meets_its_linear_toe_where_the_standard_says() {
        // Mid grey is where the two spaces diverge most, and the figure everything else is checked
        // against: 128/255 encoded is 0.2159 linear.
        assert!((channel_to_linear(128) - 0.2158).abs() < 1e-3);
        // Both ends are identical in either space, which is why a missing conversion hides there.
        assert_eq!(channel_to_linear(0), 0.0);
        assert!((channel_to_linear(255) - 1.0).abs() < 1e-6);

        // The knee: below 0.04045 encoded the curve is a straight 1/12.92, and the two halves meet
        // there rather than stepping.
        let knee = 0.040_45;
        assert!((to_linear(knee) - knee / 12.92).abs() < 1e-9);
        assert!((to_linear(knee + 1e-6) - knee / 12.92).abs() < 1e-6);

        // And back again, for every byte there is — exactly, which is what makes a shading
        // map a lossless round trip through the array rather than a slow drift.
        for encoded in 0..=255u8 {
            assert_eq!(channel_to_srgb(channel_to_linear(encoded)), encoded);
        }
    }
}
