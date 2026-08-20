//! The sun: the only thing lighting an exterior.

use std::f32::consts::FRAC_PI_2;

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

/// Morrowind's own sun, out of `apps/openmw/mwworld/weather.cpp:900`'s `(-400 * orbit, 75, -100)`.
///
/// How far it swings east to west, how far north it sits, and how high it climbs. Named because the
/// night needs the first two as well — the bearing it sets on is `atan2(NORTHING, SWING)` — and a
/// second copy of the pair is a second thing to move if this ever does. The night's ends meet the
/// day's exactly, and that is only true while both read the same numbers.
const SWING: f32 = 400.0;
const NORTHING: f32 = 75.0;
/// How far down the sun reaches at noon, against [`SWING`] — 100 of 125, which is 53.13 degrees.
const CLIMB: f32 = 100.0;

/// How far under the sun gets at midnight, in radians.
///
/// Seventy degrees, reached over the six hours from sunset — about ten an hour, which is the rate
/// the sun actually leaves the sky at and close to the rate it climbs at on the other side of
/// sunrise. It is what makes the night dark rather than merely dim.
const NIGHT_DESCENT: f32 = 70.0 * (std::f32::consts::PI / 180.0);

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
    /// Scaling that component keeps its noon exactly — 100 of 125, the same 53 degrees — and lets the
    /// ends of the day reach the horizon they are named for.
    ///
    /// **By a cosine, and it was `sqrt(1 - orbit²)` first.** Both satisfy those two, and only one is
    /// smooth at the ends: a circle's edge is vertical there, so the sun fell from 0.57 degrees to
    /// nothing in the last eighteen seconds of the day and *everything* keyed to its elevation
    /// stepped with it — the air mass runs 30.5 to 38.0 across those seconds, and the ground's
    /// lighting dropped by 65% in two minutes of game time. A cosine of an angle that is itself
    /// linear in time has a finite slope at the horizon, which is also what a real sun's elevation
    /// does. The circle was the mistake of feeding a quantity linear in time to a function expecting
    /// one that was already a cosine.
    ///
    /// **And past the ends it keeps going, which it has to.** Clamping there leaves the sun parked
    /// *on* the horizon all night with nothing but an invented fade to say the difference, which is
    /// how the sky came to be darkest at the moment of sunrise.
    ///
    /// Under the horizon it descends at a **steady angle**, reaching `NIGHT_DESCENT` at midnight and
    /// coming back up by sunrise. That is what a real sun does — the hour angle turns at a constant
    /// rate — and it is not what continuing the circle into a hyperbola does: that plunges infinitely
    /// fast at the boundary and then flattens, reaching only twelve degrees down at midnight, which
    /// leaves a seventh of the sunset glow burning all night and turns midnight pink. Ten degrees an
    /// hour puts the sun seventy under at midnight and the glow is gone by an hour after dusk.
    ///
    /// **And it goes round rather than back.** Its bearing sweeps from where it set, through due
    /// north at midnight, to where it rises — because that is where a body going round once a day
    /// is, and because the alternative is a discontinuity: `orbit` wraps from −2 to +2 at midnight,
    /// so a heading read off it flips the sun across the sky in a minute.
    ///
    /// Full strength, before any air. What a beam has left by the time it arrives depends on how
    /// much atmosphere it crossed, which is [`crate::Sky`]'s to say.
    pub fn at(orbit: f32) -> Self {
        let direction = if orbit.abs() <= 1.0 {
            // Clamped at nought because `cos` of a quarter turn is not exactly zero in `f32` — it
            // is −4.4e-8, which would tip the setting sun a hair *above* the horizon and send its
            // light upward. `sqrt(1 - 1)` was exact and needed no such guard.
            Vec3::new(
                -SWING * orbit,
                NORTHING,
                -CLIMB * (orbit * FRAC_PI_2).cos().max(0.0),
            )
        } else {
            // **How far through the night, nought at sunset and one at sunrise.** The orbit runs off
            // the end of the day at both ends *and wraps at midnight* — it reaches −2 at one minute
            // to twelve and +2 at one minute past — so it is not itself a continuous account of the
            // night. This is.
            let night = match orbit < 0.0 {
                true => (-orbit - 1.0) * 0.5,
                false => 1.0 - (orbit - 1.0) * 0.5,
            };
            // Deepest at midnight, and the same ramp the orbit gave: `|orbit| - 1` in the old
            // arithmetic, which this reproduces exactly.
            let below = (1.0 - (2.0 * night - 1.0).abs()) * NIGHT_DESCENT;

            // **And the heading keeps turning the way it was turning, through due north at midnight
            // rather than doubling back.** Reading it off the orbit the way the daylit half does is
            // what made the sun flip forty degrees across the wrap, from west of north to east of
            // it, in a minute — invisible for as long as nothing at night read the sun's direction,
            // and the first thing that did was a moon's terminator.
            //
            // Sunset and sunrise sit symmetrically either side of north, so half the sweep lands
            // exactly there and both ends meet the day's own formula to the bit.
            let set = NORTHING.atan2(SWING);
            let angle = set - night * (std::f32::consts::PI + 2.0 * set);
            let (sin, cos) = angle.sin_cos();
            Vec3::new(cos, sin, 0.0) * below.cos() + Vec3::Z * below.sin()
        };
        Self {
            direction: direction.normalize(),
            colour: Vec3::new(1.0, 0.95, 0.88) * DAYLIGHT,
            angular_radius: Self::REAL_ANGULAR_RADIUS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::world_time::WorldTime;

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
    fn the_sun_goes_round_through_the_night_without_a_step_at_midnight() {
        // **Nothing at night read the sun's direction until a moon's terminator did**, and what it
        // found was a forty-degree jump in a game minute: the orbit reaches −2 a moment before
        // midnight and +2 a moment after, and a heading read off it flips from west of north to
        // east of it. So this walks the whole night and asks for no step anywhere.
        let at = |hour: f32| Sun::at(WorldTime::hours(hour).orbit()).direction;
        let mut worst: (f32, f32) = (0.0, 0.0);
        let mut previous = at(18.0);
        for step in 1..=(12 * 600) {
            let hour = 18.0 + step as f32 / 600.0;
            let now = at(hour);
            let moved = previous.dot(now).clamp(-1.0, 1.0).acos().to_degrees();
            if moved > worst.0 {
                worst = (moved, hour);
            }
            previous = now;
        }
        // A tenth of a minute of a twelve-hour night is a fifth of a degree at ten degrees an hour.
        assert!(
            worst.0 < 0.05,
            "the sun jumps {:.1} degrees at {:.3}",
            worst.0,
            worst.1
        );

        // And it goes *round*: due north at midnight, which is where a body halfway between setting
        // in the west and rising in the east has to be.
        let midnight = at(24.0);
        assert!(midnight.x.abs() < 1e-3, "{midnight:?}");
        assert!(
            midnight.y < 0.0,
            "the midnight sun is not to the north: {midnight:?}"
        );
        // Seventy degrees under, which `NIGHT_DESCENT` is.
        assert!(
            (midnight.z.asin().to_degrees() - 70.0).abs() < 0.01,
            "{midnight:?}"
        );

        // **Both ends meet the daylit formula exactly**, so nothing about the day moved: the night
        // branch at sunset and sunrise is the same vector the ellipse gives there.
        // Sampled on the night's own side of each: past sunset, and before sunrise.
        for (hour, orbit) in [(18.0 + 1e-4, -1.0), (6.0 - 1e-4, 1.0)] {
            let night = at(hour);
            let day = Sun::at(orbit).direction;
            assert!(
                night.dot(day) > 0.999_999,
                "{hour}: {night:?} against {day:?}"
            );
        }
    }

    #[test]
    fn the_disc_is_the_size_of_the_real_one() {
        // Half a degree across, which is what gives a shadow a penumbra a few centimetres wide at
        // arm's length and metres wide at the foot of a cliff.
        let diameter = Sun::REAL_ANGULAR_RADIUS * 2.0;
        assert!((diameter.to_degrees() - 0.533).abs() < 0.01, "{diameter}");
    }
}
