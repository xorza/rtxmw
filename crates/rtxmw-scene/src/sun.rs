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

/// Radiance of the sun against the sky's ambient.
///
/// Not a physical figure: auto-exposure absorbs any overall scale, so what matters is the *ratio*
/// between direct sun and sky, which on a clear day is roughly five to one on a surface facing it.
/// Provisional in exactly the way `LightBuffer::INTENSITY` is, and for the same reason — see
/// `docs/design.md` §5.1.
///
/// **Before any air, and so no longer that ratio at every hour.** [`crate::Sky`] attenuates this by
/// the atmosphere a beam crossed and derives the sky from what it lost, so the two move apart as the
/// sun descends — which is the point. This is the figure at the top of the atmosphere.
///
/// **`EMISSIVE_INTENSITY` in `primary_visibility.comp` is a fraction of this.** A Morrowind material's
/// emissive means "as bright as a fully lit surface" in a renderer whose light never passed one, so
/// it has to be carried onto this scale; change the number here and every glow in the world drifts
/// with it.
const DAYLIGHT: f32 = 8.0;

impl Sun {
    /// Angular radius of the real sun, in radians — a disc about half a degree across.
    ///
    /// Public because it is the width every soft shadow in the game is judged against, and the
    /// tests that measure a penumbra need the number the renderer actually used rather than one
    /// written down twice.
    pub const REAL_ANGULAR_RADIUS: f32 = 0.004_654;

    /// Morrowind's sun at a point in its day, `orbit` running from 1 at sunrise to −1 at nightfall.
    ///
    /// **The horizontal swing is the game's own hardcoded constant**, `(-400 · orbit, 75, ...)` from
    /// `apps/openmw/mwworld/weather.cpp:900`. It is not an astronomical model: the sun runs east to
    /// west and sits to the south, and the numbers are Morrowind's rather than anything derived.
    ///
    /// **The vertical is not, and it is a deliberate departure.** The game's third component is the
    /// constant −100, so its sun swings east to west without ever descending: at sunrise it stands
    /// fourteen degrees up and it has stood there since the world began. Morrowind has no sunrise.
    /// Scaling that component by `sqrt(1 - orbit²)` keeps its noon exactly — 100 of 125, the same
    /// 53 degrees — and lets the ends of the day reach the horizon they are named for.
    ///
    /// Full strength, before any air. What a beam has left by the time it arrives depends on how
    /// much atmosphere it crossed, which is [`crate::Sky`]'s to say.
    pub fn at(orbit: f32) -> Self {
        let arc = (1.0 - orbit * orbit).max(0.0).sqrt();
        Self {
            direction: Vec3::new(-400.0 * orbit, 75.0, -100.0 * arc).normalize(),
            colour: Vec3::new(1.0, 0.95, 0.88) * DAYLIGHT,
            angular_radius: Self::REAL_ANGULAR_RADIUS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sun_crosses_from_east_to_west_and_reaches_the_horizon_at_both_ends() {
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
            // Never upward, and northward throughout, which puts the sun in the south.
            assert!(sun.direction.z <= 0.0, "orbit {orbit} sends light upward");
            assert!(sun.direction.y > 0.0, "orbit {orbit}");
        }

        // **On the horizon at both ends of the day, which is the departure from the game's own
        // constant.** Morrowind's third component is a fixed −100, so its sun swings across the sky
        // without ever descending and stands fourteen degrees up at the moment it is said to rise.
        // Scaling by `sqrt(1 - orbit²)` leaves noon exactly where the constant put it and lets
        // sunrise and sunset mean what they say.
        assert_eq!(sunrise.direction.z, 0.0);
        assert_eq!(dusk.direction.z, 0.0);
        // Morrowind's noon out of its own `(0, 75, -100)`: 100 of 125 of the way down.
        assert!(
            (noon.direction.z + 0.8).abs() < 1e-6,
            "{:?}",
            noon.direction
        );
        // Highest at noon, and monotonic between.
        assert!(noon.direction.z < Sun::at(0.5).direction.z);
        assert!(Sun::at(0.5).direction.z < sunrise.direction.z);
    }

    #[test]
    fn the_disc_is_the_size_of_the_real_one() {
        // Half a degree across, which is what gives a shadow a penumbra a few centimetres wide at
        // arm's length and metres wide at the foot of a cliff.
        let diameter = Sun::REAL_ANGULAR_RADIUS * 2.0;
        assert!((diameter.to_degrees() - 0.533).abs() < 0.01, "{diameter}");
    }
}
