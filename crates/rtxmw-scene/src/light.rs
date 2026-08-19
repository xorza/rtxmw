//! A light placed in a cell.

use glam::Vec3;

/// One point light, in world space.
///
/// Morrowind stores a colour and a radius but no intensity: the original renderer had a fixed
/// attenuation curve and no physical units, so brightness fell out of the curve. The radius is
/// therefore the only control the data gives, and whatever turns this into radiance has to supply
/// the scale itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    pub position: Vec3,
    /// Linear-space RGB in `0..1`.
    pub colour: Vec3,
    /// The radius the record carries, in world units.
    ///
    /// **Not how far the light reaches.** It is the only number `LIGH` gives, so everything about a
    /// lamp is derived from it, and a renderer is free to derive reach as something other than the
    /// value itself — this one does, because Morrowind's radii light a lantern's own post and
    /// nothing else.
    pub radius: f32,
}

/// A cell's fixed lighting, which for an interior is most of its illumination.
///
/// Morrowind interiors were authored against a renderer with no global illumination, so the ambient
/// term stands in for every bounce the original engine could not compute. Applying it unchanged on
/// top of real light double-counts — the same problem the pre-lit albedo has, and recorded in
/// `docs/design.md` §5.1 alongside it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Ambient {
    /// Linear-space RGB.
    pub colour: Vec3,
    /// Directional term the original engine used for interiors faking a sun.
    pub sunlight: Vec3,
    pub fog: Vec3,
    /// How thickly that fog sits, as the cell records it.
    ///
    /// Morrowind's own number, and its scale is the original engine's rather than anything physical
    /// — the renderer reads it as a relative thickness, not as an extinction coefficient.
    pub fog_density: f32,
}

impl Ambient {
    /// The lighting an interior's `AMBI` record describes.
    pub fn from_record(record: rtxmw_esm::CellAmbient) -> Self {
        Self {
            colour: crate::srgb::to_linear(record.ambient),
            sunlight: crate::srgb::to_linear(record.sunlight),
            fog: crate::srgb::to_linear(record.fog),
            // **The density is the float's own bits**, which is how the record carries it — and any
            // four bytes are *some* float, so a malformed or modded cell can hand over a NaN or a
            // negative. Neither is an assert: this is a file on disk that something else wrote, and
            // one bad cell should be fog-free rather than fatal.
            fog_density: match f32::from_bits(record.fog_density) {
                density if density.is_finite() && density >= 0.0 => density,
                _ => 0.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(fog_density: u32) -> rtxmw_esm::CellAmbient {
        rtxmw_esm::CellAmbient {
            ambient: 0,
            sunlight: 0,
            fog: 0,
            fog_density,
        }
    }

    #[test]
    fn a_malformed_fog_density_reads_as_no_fog_rather_than_as_a_nan() {
        // Any four bytes are some float, so a mod or a corrupt file can carry one of these. A NaN
        // would run through the renderer's `exp` and turn every pixel of that cell into something
        // the tone curve cannot map, and a negative density is fog that thickens what it passes.
        for bits in [
            f32::NAN.to_bits(),
            f32::INFINITY.to_bits(),
            (-1.0f32).to_bits(),
        ] {
            assert_eq!(
                Ambient::from_record(record(bits)).fog_density,
                0.0,
                "a density stored as {bits:#010x} should read as no fog"
            );
        }
        // And a real one comes through as itself rather than being sanitised into a default.
        assert_eq!(
            Ambient::from_record(record(0.75f32.to_bits())).fog_density,
            0.75
        );
    }
}
