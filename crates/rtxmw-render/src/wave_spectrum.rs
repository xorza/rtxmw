//! Turning an empirical sea spectrum into the sinusoids the shader sums.
//!
//! The surface is a sum of plane waves, and what decides whether it looks like water is which waves
//! are in the sum. This builds that list from the **TMA** spectrum — JONSWAP with Kitaigorodskii's
//! shallow-water attenuation — spread over directions by **Donelan-Banner**, which is the pairing
//! Horvath's *Empirical Directional Wave Spectra for Computer Graphics* settled on for film work and
//! which the real-time literature has followed since.
//!
//! It replaces a geometric series picked by eye. Two things make that worth the trouble beyond
//! tidiness: TMA's depth term is exactly the coastal-shelf correction Vvardenfell's water needs, and
//! Donelan-Banner's spread is frequency-dependent — narrow at the swell, broad at the chop — which
//! is the shape a sum of plane waves has to have to avoid drawing a lattice.
//!
//! The whole table is built once on the host. Nothing here runs per pixel.

use bytemuck::{Pod, Zeroable};
use glam::Vec2;

/// Morrowind's gravity in world units per second squared, matching the shader's own.
const GRAVITY: f32 = 627.1;

/// How many sinusoids the surface is summed from.
///
/// Declared again in `primary_visibility.comp`, and pinned by a test: the shader walks this table
/// with a loop bound of its own.
pub(crate) const WAVE_COUNT: usize = 32;

/// How many wavenumber bands the spectrum is sampled at.
const BANDS: usize = 8;

/// How many directions each band is split into.
const PER_BAND: usize = WAVE_COUNT / BANDS;

/// The golden angle, turning each band off the last so no two share a direction.
const GOLDEN: f32 = 2.399_963_2;

/// The shortest wave the spectrum carries, in world units.
///
/// **A band limit in time as much as in space.** Curvature climbs with wavenumber, so the shortest
/// waves decide the caustics — and a wave's period falls with its length, so they also decide how
/// fast the pattern on the seabed reshuffles. Carried down to a quarter of this, the light below
/// changed by three quarters of its own contrast every twelfth of a second, which reads as stripes
/// tearing across the bottom rather than as water. Half a metre puts the change back to half its
/// contrast, which is where it was before any of this and where nobody complained about it.
///
/// The trade is exactly that and cannot be had both ways: shorter waves focus harder *and* move
/// faster, because they are the same waves.
///
/// It is where the bands are *centred*, not quite where they end: the lowest sits below the peak
/// and the highest half a band above this, so the table actually spans about 980 down to 27 units.
const SHORTEST: f32 = 32.0;

/// One sinusoid of the sea, as the shader reads it.
///
/// Scalar layout, so this is twenty tightly packed bytes and the shader's `Wave` must match field
/// for field.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuWave {
    /// Unit vector the wave travels along.
    pub(crate) direction: [f32; 2],
    /// Radians of phase per world unit.
    pub(crate) wavenumber: f32,
    /// Half the crest-to-trough height, in world units.
    pub(crate) amplitude: f32,
    /// Radians of phase per second, from the dispersion relation at this depth.
    pub(crate) speed: f32,
}

/// What the sea is doing, in the four numbers a spectrum needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SeaState {
    /// The average height of the highest third of the waves, in world units — the figure
    /// oceanography quotes, and the one that decides how rough this looks.
    pub(crate) significant_height: f32,
    /// The wavelength carrying the most energy, in world units.
    pub(crate) peak_wavelength: f32,
    /// Depth of the shelf the spectrum is attenuated against.
    pub(crate) depth: f32,
    /// Which way the wind is blowing, in radians about `+Z`.
    pub(crate) bearing: f32,
}

impl Default for SeaState {
    /// A sheltered bay: a six-metre swell a hand's breadth high over a shelf a few metres down.
    fn default() -> Self {
        Self {
            significant_height: 9.4,
            peak_wavelength: 420.0,
            depth: 300.0,
            bearing: 0.6,
        }
    }
}

impl SeaState {
    /// The sinusoids this sea is made of.
    ///
    /// Sampled by *quantile* in direction rather than at even angles: the share of a band's energy
    /// between two quantiles of its spread is the same by construction, so every component carries
    /// the same amplitude and the spread's shape is exact however few directions are taken.
    pub(crate) fn waves(&self) -> [GpuWave; WAVE_COUNT] {
        let peak = self.frequency_of(std::f32::consts::TAU / self.peak_wavelength);
        // From below the peak, where the swell is, up to the shortest wave worth carrying.
        let lowest = 0.7 * peak;
        let highest = self.frequency_of(std::f32::consts::TAU / SHORTEST);
        let step = (highest / lowest).powf(1.0 / (BANDS - 1) as f32);

        let mut waves = [GpuWave::default(); WAVE_COUNT];
        for band in 0..BANDS {
            let frequency = lowest * step.powi(band as i32);
            // The band's width, taken symmetrically about it in the geometric spacing.
            let width = frequency * (step.sqrt() - 1.0 / step.sqrt());
            let energy = self.energy_at(frequency, peak) * width / PER_BAND as f32;
            let spread = spread_of(frequency / peak);

            for step_in_band in 0..PER_BAND {
                let quantile = (step_in_band as f32 + 0.5) / PER_BAND as f32;
                let angle = self.bearing + GOLDEN * band as f32 + turn_of(quantile, spread);
                // **Each direction takes its own place in the band, not the band's middle.** A band
                // stands for a range of wavelengths, and giving all of its directions the same one
                // gives them the same speed too — plane waves of a single length travelling
                // together do not interfere into a sea, they translate as a rigid pattern, and the
                // seabed below reads that as stripes sliding across it. Spread across the band they
                // beat against one another instead, which is the whole point of a spectrum.
                let within = frequency * step.powf(quantile - 0.5);
                waves[band * PER_BAND + step_in_band] = GpuWave {
                    direction: Vec2::from_angle(angle).to_array(),
                    wavenumber: self.wavenumber_of(within),
                    // Held back until the whole table is known, so it can be scaled to the height
                    // that was actually asked for.
                    amplitude: (2.0 * energy).max(0.0).sqrt(),
                    speed: within,
                };
            }
        }

        // **The spectrum's own scale is thrown away and the asked-for height put in its place.**
        // JONSWAP's `alpha` is a fetch-and-wind parameter that nothing here knows, and every term
        // in it is a constant multiplier — so it cancels, and what is left is a shape to be scaled
        // by the one number a person can picture.
        let variance: f32 = waves.iter().map(|w| w.amplitude * w.amplitude / 2.0).sum();
        if variance > 0.0 {
            let scale = self.significant_height / (4.0 * variance.sqrt());
            for wave in &mut waves {
                wave.amplitude *= scale;
            }
        }
        waves
    }

    /// The dispersion relation at this depth: `omega^2 = g k tanh(k h)`.
    ///
    /// Deep water's `sqrt(g k)` is its limit, and a wave whose length approaches the depth falls
    /// behind it — which is why a swell slows and steepens as it reaches a shore.
    fn frequency_of(&self, wavenumber: f32) -> f32 {
        (GRAVITY * wavenumber * (wavenumber * self.depth).tanh()).sqrt()
    }

    /// The same relation the other way round, by Newton from the deep-water guess.
    fn wavenumber_of(&self, frequency: f32) -> f32 {
        let target = frequency * frequency;
        let mut wavenumber = target / GRAVITY;
        for _ in 0..12 {
            let depth = wavenumber * self.depth;
            let tanh = depth.tanh();
            let value = GRAVITY * wavenumber * tanh - target;
            let slope = GRAVITY * tanh + GRAVITY * wavenumber * self.depth * (1.0 - tanh * tanh);
            wavenumber -= value / slope;
        }
        wavenumber.max(1.0e-6)
    }

    /// The TMA spectrum: JONSWAP shaped by how much of it a shelf this deep will carry.
    ///
    /// `alpha` is left at one — see [`Self::waves`], which scales the result to a height instead.
    fn energy_at(&self, frequency: f32, peak: f32) -> f32 {
        // JONSWAP: a Pierson-Moskowitz tail under a peak sharpened by `gamma`.
        let width = if frequency <= peak { 0.07 } else { 0.09 };
        let offset = (frequency - peak) / (width * peak);
        let sharpening = 3.3f32.powf((-0.5 * offset * offset).exp());
        let tail = (-1.25 * (peak / frequency).powi(4)).exp();
        let jonswap = GRAVITY * GRAVITY / frequency.powi(5) * tail * sharpening;
        jonswap * self.depth_factor(frequency)
    }

    /// Kitaigorodskii's attenuation, which is what makes this TMA rather than JONSWAP.
    ///
    /// A shelf cannot carry a wave whose orbit reaches the bottom, so the spectrum is cut where the
    /// water is too shallow for it — running from nothing, through a quadratic knee, to unchanged
    /// once the wave no longer feels the ground.
    fn depth_factor(&self, frequency: f32) -> f32 {
        let scaled = frequency * (self.depth / GRAVITY).sqrt();
        if scaled <= 1.0 {
            0.5 * scaled * scaled
        } else if scaled < 2.0 {
            1.0 - 0.5 * (2.0 - scaled) * (2.0 - scaled)
        } else {
            1.0
        }
    }
}

/// Donelan-Banner's spread parameter, against how far above the peak a band sits.
///
/// **Large is narrow.** The swell arrives as near-parallel trains and comes out around two and a
/// half; the chop well above the peak settles near four tenths, which is a fan wide enough that a
/// sum of it does not draw a grain.
fn spread_of(relative: f32) -> f32 {
    if relative < 0.95 {
        2.61 * relative.powf(1.3)
    } else if relative < 1.6 {
        2.28 * relative.powf(-1.3)
    } else {
        let exponent = -0.4 + 0.8393 * (-0.567 * (relative * relative).ln()).exp();
        10.0f32.powf(exponent)
    }
}

/// Where a given share of a `sech^2` spread lies, in radians off the wind.
///
/// The spread integrates to `tanh`, so its quantiles are an `atanh` — no search and no table.
fn turn_of(quantile: f32, spread: f32) -> f32 {
    let edge = (spread * std::f32::consts::PI).tanh();
    ((2.0 * quantile - 1.0) * edge).atanh() / spread
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shelf_carries_only_what_it_is_deep_enough_for() {
        // Kitaigorodskii's term, at its three parts. A wave far too long for the depth is cut to
        // nothing, one short enough not to feel the bottom passes untouched, and between them the
        // knee is continuous — a step there would draw a line across the water.
        let shelf = SeaState {
            depth: 300.0,
            ..SeaState::default()
        };
        let scaled = |s: f32| s / (shelf.depth / GRAVITY).sqrt();

        assert!(
            shelf.depth_factor(scaled(0.2)) < 0.03,
            "a long swell is stopped"
        );
        assert_eq!(
            shelf.depth_factor(scaled(2.5)),
            1.0,
            "short waves pass whole"
        );
        // Both halves agree where they meet, and the curve only ever climbs.
        let knee = shelf.depth_factor(scaled(1.0));
        assert!(
            (knee - 0.5).abs() < 1e-5,
            "the knee is at a half, got {knee}"
        );
        let mut last = 0.0;
        for step in 0..=40 {
            let here = shelf.depth_factor(scaled(step as f32 * 0.075));
            assert!(here >= last - 1e-6, "the attenuation fell at {step}");
            last = here;
        }
    }

    #[test]
    fn the_spread_is_narrow_at_the_swell_and_broad_above_it() {
        // The property §7.15 arrived at from a symptom, in its empirical form: a swell arrives as
        // parallel trains and the chop riding on it does not. Large is narrow.
        let at_peak = spread_of(1.0);
        let chop = spread_of(6.0);
        assert!(
            at_peak > 2.0,
            "the swell should be a narrow fan, got {at_peak}"
        );
        assert!(chop < 0.6, "the chop should be a wide one, got {chop}");
        assert!(at_peak > 3.0 * chop);

        // The quantiles of that spread put half the energy either side of the wind, and put more of
        // it off-axis when the fan is wider.
        assert!(
            turn_of(0.5, at_peak).abs() < 1e-6,
            "the median is the wind itself"
        );
        assert!(turn_of(0.9, chop).abs() > turn_of(0.9, at_peak).abs());
    }

    #[test]
    fn the_waves_carry_the_height_and_the_dispersion_they_were_asked_for() {
        let sea = SeaState::default();
        let waves = sea.waves();

        // Significant height is four times the root of the variance, which is the sum of `A^2 / 2`
        // over independent sinusoids. Building the spectrum and then scaling to this is the whole
        // reason `alpha` never appears.
        let variance: f32 = waves.iter().map(|w| w.amplitude * w.amplitude / 2.0).sum();
        let height = 4.0 * variance.sqrt();
        assert!(
            (height - sea.significant_height).abs() < 1e-3,
            "asked for {} and built {height}",
            sea.significant_height
        );

        for wave in &waves {
            assert!(wave.amplitude > 0.0, "a component carries nothing");
            let direction = Vec2::from(wave.direction);
            assert!((direction.length() - 1.0).abs() < 1e-5);
            // Every component obeys the relation its own speed was taken from.
            let expected =
                (GRAVITY * wave.wavenumber * (wave.wavenumber * sea.depth).tanh()).sqrt();
            assert!(
                (wave.speed - expected).abs() < 1e-3 * expected,
                "speed {} against {expected} for k {}",
                wave.speed,
                wave.wavenumber
            );
        }
    }

    #[test]
    fn shallow_water_slows_a_wave_and_deep_water_does_not() {
        // The reason this is TMA rather than JONSWAP. A wave whose orbit reaches the bottom is held
        // back; one that never feels it travels at the deep-water speed whatever the chart says.
        let long = std::f32::consts::TAU / 420.0;
        let short = std::f32::consts::TAU / 8.0;
        let deep = SeaState {
            depth: 4000.0,
            ..SeaState::default()
        };
        // Half a metre of water under a six-metre swell, which is a fourteenth of its own length.
        let shallow = SeaState {
            depth: 30.0,
            ..SeaState::default()
        };

        let open = (GRAVITY * long).sqrt();
        assert!((deep.frequency_of(long) - open).abs() < 1e-3 * open);
        // A swell in water a fourteenth of its length deep travels at about two thirds of its open
        // speed; a tenth of its length costs it a sixth, and half a length costs it nothing at all.
        let dragged = shallow.frequency_of(long) / open;
        assert!(
            (dragged - 0.649).abs() < 0.01,
            "expected about two thirds of the open speed, got {dragged}"
        );
        // A short wave cannot tell the two apart.
        assert!(
            (shallow.frequency_of(short) - deep.frequency_of(short)).abs()
                < 1e-3 * deep.frequency_of(short)
        );
    }

    #[test]
    fn the_table_matches_the_one_the_shader_walks() {
        // The shader loops to a bound of its own and reads a struct laid out by hand. A count that
        // disagreed would walk off the end of the block; a layout that disagreed would read a
        // direction out of an amplitude, and neither shows up as anything but wrong-looking water.
        assert_eq!(WAVE_COUNT, 32);
        assert_eq!(size_of::<GpuWave>(), 20);
        assert_eq!(WAVE_COUNT % BANDS, 0, "every band takes the same share");
    }

    #[test]
    fn the_swell_arrives_in_a_narrow_fan_and_the_chop_in_a_wide_one() {
        // The lattice §7.15 was about comes from a handful of components sharing directions, and
        // what prevents it is the spread being *wide where it matters*: curvature is dominated by
        // the shortest waves, so those are the ones that have to disagree about which way they are
        // going. Donelan-Banner says they do, and this is that end to end.
        let waves = SeaState::default().waves();
        let fan_of = |band: usize| {
            let slice = &waves[band * PER_BAND..(band + 1) * PER_BAND];
            let mut widest = 0.0f32;
            for (index, wave) in slice.iter().enumerate() {
                for other in &slice[index + 1..] {
                    let cosine = Vec2::from(wave.direction)
                        .dot(Vec2::from(other.direction))
                        .clamp(-1.0, 1.0);
                    widest = widest.max(cosine.acos().to_degrees());
                }
            }
            widest
        };

        let swell = fan_of(0);
        let chop = fan_of(BANDS - 1);
        // Sixty-eight degrees across four samples of the lowest band, which sits below the peak
        // where the spread has already begun to open; the chop above it fans past a half turn.
        assert!(
            swell < 80.0,
            "the swell should arrive in a narrow fan, got {swell} degrees"
        );
        assert!(
            chop > 120.0,
            "the chop should be a wide one, got {chop} degrees"
        );

        // And nothing is a duplicate of anything, in any band.
        for (index, wave) in waves.iter().enumerate() {
            for other in &waves[index + 1..] {
                assert_ne!(wave.direction, other.direction);
            }
        }
    }
}
