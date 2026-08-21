//! What something glowing at a temperature looks like, from Planck's law and the eye's own curves.

use glam::Vec3;

/// The visible band, and how finely it is walked, in nanometres.
///
/// Five-nanometre steps over 380 to 780 is the sampling the CIE tables themselves are published at,
/// so nothing is gained by going finer and the fits below are stated against it.
const SHORTEST: f32 = 380.0;
const LONGEST: f32 = 780.0;
const STEP: f32 = 5.0;

/// Planck's law, in SI: `2hc^2` and `hc/k`.
///
/// Written as the two combinations that actually appear rather than as three constants, because the
/// first is a scale that the normalisation below divides straight back out and the second is the
/// only place the temperature enters.
const TWO_H_C_SQUARED: f64 = 1.191_042_97e-16;
const H_C_OVER_K: f64 = 1.438_776_88e-2;

/// One lobe of the piecewise-Gaussian fits to the CIE 1931 colour matching functions.
///
/// **Wyman, Sloan and Shirley, JCGT 2013.** Each of the three curves is a sum of one to three of
/// these, and each lobe is a Gaussian with a different width either side of its peak — which is what
/// lets two lobes carry `x̄`'s long red tail and its short blue one at once. Accurate to under a
/// percent of the tabulated curves, against a table this would otherwise have to carry.
#[derive(Debug, Clone, Copy)]
struct Lobe {
    weight: f32,
    peak: f32,
    /// Standard deviation below the peak, and above it.
    below: f32,
    above: f32,
}

impl Lobe {
    const fn new(weight: f32, peak: f32, below: f32, above: f32) -> Self {
        Self {
            weight,
            peak,
            below,
            above,
        }
    }

    /// This lobe's contribution at `wavelength`, in nanometres.
    fn at(self, wavelength: f32) -> f32 {
        let spread = if wavelength < self.peak {
            self.below
        } else {
            self.above
        };
        let t = (wavelength - self.peak) / spread;
        self.weight * (-0.5 * t * t).exp()
    }
}

/// `x̄`, `ȳ` and `z̄`, as the lobes they are summed from.
const MATCHING: [&[Lobe]; 3] = [
    &[
        Lobe::new(1.056, 599.8, 37.9, 31.0),
        Lobe::new(0.362, 442.0, 16.0, 26.7),
        Lobe::new(-0.065, 501.1, 20.4, 26.2),
    ],
    &[
        Lobe::new(0.821, 568.8, 46.9, 40.5),
        Lobe::new(0.286, 530.9, 16.3, 31.1),
    ],
    &[
        Lobe::new(1.217, 437.0, 11.8, 36.0),
        Lobe::new(0.681, 459.0, 26.0, 13.8),
    ],
];

/// CIE XYZ to linear sRGB, which is the space everything in this renderer is lit in.
const TO_LINEAR_SRGB: [[f32; 3]; 3] = [
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
];

/// The colour of a blackbody at `kelvin`, in linear sRGB, scaled to unit luminance.
///
/// **Hue and not level.** What comes back is what the temperature says the *colour* is; how bright
/// the thing glowing is belongs to whatever is glowing. Normalising by luminance rather than by the
/// largest channel is what makes that split clean: two temperatures at the same returned brightness
/// differ only in where the energy sits across the three channels.
///
/// **Below about 1900 K a monitor cannot say it.** The red primary of sRGB is not a deep enough
/// red, so the deepest embers come back on the edge of the gamut with a negative green or blue,
/// which is clamped. That is a property of the display rather than of this: the answer is right and
/// the screen cannot print it.
pub fn colour(kelvin: f32) -> Vec3 {
    let mut xyz = Vec3::ZERO;
    let mut wavelength = SHORTEST;
    while wavelength <= LONGEST {
        let power = planck(wavelength, kelvin);
        for (channel, lobes) in MATCHING.iter().enumerate() {
            xyz[channel] += power * lobes.iter().map(|lobe| lobe.at(wavelength)).sum::<f32>();
        }
        wavelength += STEP;
    }

    let linear = Vec3::from_array(std::array::from_fn(|channel| {
        Vec3::from_array(TO_LINEAR_SRGB[channel]).dot(xyz)
    }));
    // A hot enough blackbody leaves the gamut on the blue side and a cold one on the red; the
    // clamp is where a colour a monitor cannot show becomes the nearest one it can.
    let shown = linear.max(Vec3::ZERO);
    // `ȳ` *is* luminance, so `xyz.y` is the number to divide by — and it is taken from the XYZ
    // rather than from the clamped RGB so that clamping cannot change how bright the answer is.
    shown / xyz.y.max(1e-30)
}

/// Spectral radiance at `wavelength` nanometres from a blackbody at `kelvin`, in SI units.
///
/// Computed in double precision because the exponent runs to eighty at the red end of a candle
/// flame, and `exp` of that in single precision has already lost most of its mantissa.
fn planck(wavelength: f32, kelvin: f32) -> f32 {
    let metres = f64::from(wavelength) * 1e-9;
    let power = TWO_H_C_SQUARED
        / (metres.powi(5) * ((H_C_OVER_K / (metres * f64::from(kelvin))).exp() - 1.0));
    power as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative luminance of a linear sRGB colour, which is what [`colour`] normalises by.
    fn luminance(rgb: Vec3) -> f32 {
        rgb.dot(Vec3::new(0.2126, 0.7152, 0.0722))
    }

    #[test]
    fn a_blackbody_reddens_as_it_cools_and_goes_white_at_the_suns_temperature() {
        // **6500 K is where sRGB's own white sits.** The standard's white point is D65, which is
        // daylight rather than a blackbody, but the two agree closely enough that a 6500 K body has
        // to come back very nearly neutral — and that is the one point on this curve that can be
        // checked against something outside it.
        let white = colour(6500.0);
        let spread = white.max_element() - white.min_element();
        assert!(
            spread < 0.12,
            "6500 K should be near neutral; it came back {white} spanning {spread}"
        );

        // **A candle is 1900 K and comes out orange**: far more red than green, and very little
        // blue. Every one of these is an ordering rather than a level, which is what survives the
        // normalisation.
        let candle = colour(1900.0);
        assert!(
            candle.x > candle.y && candle.y > candle.z,
            "a candle should be red over green over blue; it came back {candle}"
        );
        assert!(
            candle.z < 0.25 * candle.x,
            "a candle's blue is a fraction of its red; it came back {candle}"
        );

        // **Below about 2100 K sRGB has no blue to give**, which is a fact about the display and
        // not about this: the deepest embers are a redder red than the standard's red primary, so
        // the answer leaves the gamut and what comes back is the nearest colour a monitor has. The
        // whole of a candle's range — 1100 K at the tip to 1900 K at the base — sits inside that.
        for kelvin in [1100.0, 1500.0, 1900.0] {
            assert_eq!(colour(kelvin).z, 0.0, "{kelvin} K");
        }
        assert!(colour(2200.0).z > 0.0, "2200 K should be inside the gamut");

        // **Green rises against red over the whole range**, which is the shape of the Planckian
        // locus and what a wrong matching function would break. Taken as a ratio because that is
        // what survives the normalisation, and over green rather than blue because blue is clamped
        // away exactly where a flame lives.
        //
        //  1100 K  (4.05, 0.20, 0.00)      1900 K  (2.59, 0.63, 0.00)
        //  2600 K  (1.98, 0.79, 0.16)      6500 K  (1.04, 0.98, 1.03)
        let mut previous = 0.0;
        for kelvin in [1100.0, 1500.0, 1900.0, 2600.0, 4000.0, 6500.0, 9000.0] {
            let rgb = colour(kelvin);
            let warmth = rgb.y / rgb.x;
            assert!(
                warmth > previous,
                "{kelvin} K is no less red than the step below it: {rgb}"
            );
            previous = warmth;
        }

        // Unit luminance is what makes the colour a hue rather than a brightness, and it holds
        // across the whole range including the temperatures the gamut cannot show.
        for kelvin in [1100.0, 1900.0, 4000.0, 9000.0] {
            let level = luminance(colour(kelvin));
            assert!(
                (level - 1.0).abs() < 0.06,
                "{kelvin} K came back at luminance {level}"
            );
        }
    }
}
