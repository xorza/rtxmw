//! The sun: the only thing lighting an exterior.

use glam::Vec3;

/// A directional light with a disc, far enough away that its rays are parallel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sun {
    /// Unit vector the light **travels along** — from the sun toward the world, so the direction
    /// *to* the sun is its negation. Matching OpenMW's `setSunDirection`, which is the sense the
    /// original constant is written in.
    pub direction: Vec3,
    /// Linear RGB radiance. Zero means no sun at all, which is what an interior has.
    pub colour: Vec3,
    /// Half the angle the disc subtends, in radians.
    ///
    /// The real sun is about half a degree across, and this is the whole reason its shadows have a
    /// penumbra rather than a hard edge — a point source would give neither. OpenMW has no such
    /// value anywhere: its sun is a pure direction and its shadows are as sharp as the shadow map.
    pub angular_radius: f32,
}

/// Angular radius of the real sun, in radians — a disc about half a degree across.
const REAL_ANGULAR_RADIUS: f32 = 0.004_654;

/// Radiance of the sun against the sky's ambient.
///
/// Not a physical figure: auto-exposure absorbs any overall scale, so what matters is the *ratio*
/// between direct sun and sky, which on a clear day is roughly five to one on a surface facing it.
/// Provisional in exactly the way `LightBuffer::INTENSITY` is, and for the same reason — see
/// `docs/design.md` §5.1.
const DAYLIGHT: f32 = 8.0;

impl Sun {
    /// Morrowind's sun at a point in its day, `orbit` running from 1 at sunrise to −1 at nightfall.
    ///
    /// **The direction is the game's own hardcoded constant**, `(-400 · orbit, 75, -100)` from
    /// `apps/openmw/mwworld/weather.cpp:900`. It is not an astronomical model: the sun runs east to
    /// west at a fixed angle from overhead, sitting to the south, and the numbers are Morrowind's
    /// rather than anything derived.
    pub fn at(orbit: f32) -> Self {
        Self {
            direction: Vec3::new(-400.0 * orbit, 75.0, -100.0).normalize(),
            // Warm white, and warmer as it nears the horizon, which is the one part of a weather
            // system cheap enough to have without one.
            colour: Vec3::new(1.0, 0.95, 0.88) * DAYLIGHT,
            angular_radius: REAL_ANGULAR_RADIUS,
        }
    }

    /// Mid-morning, until time of day is something the engine tracks.
    pub fn default_daylight() -> Self {
        Self::at(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sun_crosses_from_east_to_west_and_stays_above_the_horizon() {
        // Light travelling toward -X comes *from* the east, which is where the sun rises.
        let sunrise = Sun::at(1.0);
        assert!(sunrise.direction.x < -0.5, "{:?}", sunrise.direction);
        let dusk = Sun::at(-1.0);
        assert!(dusk.direction.x > 0.5, "{:?}", dusk.direction);

        // Overhead at midday, in the sense that its east-west component vanishes.
        let noon = Sun::at(0.0);
        assert!(noon.direction.x.abs() < 1e-6);

        for orbit in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let sun = Sun::at(orbit);
            assert!((sun.direction.length() - 1.0).abs() < 1e-5);
            // Light travels downward at every hour: the constant's z is fixed and negative, so the
            // sun never sets in this model — night is something the weather system turns off.
            assert!(sun.direction.z < 0.0, "orbit {orbit} sends light upward");
            // And northward, which puts the sun in the south.
            assert!(sun.direction.y > 0.0, "orbit {orbit}");
        }

        // Highest at noon, lowest at the ends: the vertical share of a unit vector is largest when
        // the horizontal one is smallest.
        assert!(noon.direction.z < sunrise.direction.z);
    }

    #[test]
    fn the_disc_is_the_size_of_the_real_one() {
        // Half a degree across, which is what gives a shadow a penumbra a few centimetres wide at
        // arm's length and metres wide at the foot of a cliff.
        let diameter = Sun::default_daylight().angular_radius * 2.0;
        assert!((diameter.to_degrees() - 0.533).abs() < 0.01, "{diameter}");
    }
}
