//! What lights an exterior, at an hour.

use glam::Vec3;

use crate::sun::Sun;
use crate::time_of_day::TimeOfDay;

/// Optical depth of the whole atmosphere looking straight up, per channel.
///
/// Rayleigh scattering goes as the inverse fourth power of wavelength, so blue is taken out of a
/// beam five times faster than red. These are the coefficients everyone uses — 5.8, 13.5 and
/// 33.1 per micrometre-to-the-fourth, times an eight-kilometre scale height — and they are the whole
/// reason a low sun is orange: at noon a beam crosses one atmosphere and loses a quarter of its
/// blue, and at sunset it crosses thirty and loses all of it.
const ZENITH_DEPTH: Vec3 = Vec3::new(0.0464, 0.1080, 0.2648);

/// How much of the sky's colour is mixed back toward grey.
///
/// **Single scattering is more saturated than a real sky.** The light in one has bounced more than
/// once, and every bounce mixes the channels back together; a model with one scattering event gives
/// a cobalt noon and a monochrome dusk. This stands in for the bounces until there is an atmosphere
/// LUT to do it properly.
const SKY_GREYING: f32 = 0.4;

/// The sky's radiance against what the arithmetic below produces.
///
/// Pure scale, set so the default hour lands on the fixed overcast blue this renderer used before
/// the sky moved — which keeps every image made against that one comparable. `ZENITH_DEPTH` is the
/// only constant in this file that came from anywhere but an eye; this, the greying, the twilight
/// width and the night floor are all tuned, and say so.
const SKY_STRENGTH: f32 = 4.15;

/// What the sky radiates once the sun is properly down.
///
/// Not moonlight — Morrowind has two moons and a night sky texture and this renderer draws neither.
/// It is the floor that keeps a night exterior legible rather than black, and it is blue because
/// scotopic vision is.
const NIGHT_SKY: Vec3 = Vec3::new(0.010, 0.014, 0.026);

/// How far past the horizon the sky keeps some of its daylight, in orbit.
///
/// Twilight: the sun lights the air above an observer for a while after it has stopped lighting the
/// observer. Without this the sky would switch off between one hour and the next.
const TWILIGHT: f32 = 0.2;

/// The sun and the sky above an exterior cell at one moment, which together are all its light.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sky {
    /// Black once the sun is down, which is how a cell with no sun says so.
    pub sun: Sun,
    /// What the sky itself radiates, and so what fills every shadow out of doors.
    pub ambient: Vec3,
}

impl Sky {
    /// The light above an exterior at `time`.
    pub fn at(time: TimeOfDay) -> Self {
        let orbit = time.orbit();
        let bare = Sun::at(orbit);
        // The sun's direction travels downward, so its climb above the horizon is the negation.
        let climb = (-bare.direction.z).max(0.0);
        let transmitted = Self::transmittance(climb);

        // **What the air took out of the beam is what the sky gives back**, and it has to get in
        // through the same air to do it — so the sky's colour is the product of the two. At noon
        // that is what was lost, which is blue; at dusk it is what survived thirty atmospheres of
        // it, which is red. One expression covers both, and neither is a tint chosen by hand.
        let scattered = (Vec3::ONE - transmitted) * transmitted;
        let grey = Vec3::splat((scattered.x + scattered.y + scattered.z) / 3.0);
        // The square root, not the climb itself: a sky an hour before sunset is dimmer than one at
        // noon but nothing like as much dimmer as the sun's own angle would make it, because most
        // of what it radiates is lit air that is nowhere near the horizon.
        let daylight = scattered.lerp(grey, SKY_GREYING) * SKY_STRENGTH * climb.sqrt();

        // Past the horizon the sun is gone but its light is not, for a while.
        let dusk = ((orbit.abs() - 1.0) / TWILIGHT).clamp(0.0, 1.0);
        Self {
            sun: Sun {
                colour: bare.colour * transmitted * (1.0 - dusk),
                ..bare
            },
            ambient: daylight.lerp(NIGHT_SKY, dusk),
        }
    }

    /// What is left of a beam that crossed the atmosphere to a sun `climb` above the horizon.
    ///
    /// `climb` is the sine of the elevation. The air mass is Kasten and Young's fit rather than the
    /// flat-earth `1/sin`, which is off by 10% at ten degrees and diverges at the horizon where all
    /// of this matters most — theirs stays finite, reaching 38 atmospheres at zero.
    fn transmittance(climb: f32) -> Vec3 {
        let degrees = climb.clamp(0.0, 1.0).asin().to_degrees();
        let air_mass = 1.0 / (climb + 0.50572 * (degrees + 6.07995).powf(-1.6364));
        (-ZENITH_DEPTH * air_mass).exp()
    }
}

impl Default for Sky {
    fn default() -> Self {
        Self::at(TimeOfDay::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sine of the sun's elevation, which is what every claim below is really about.
    fn climb(hour: f32) -> f32 {
        -Sky::at(TimeOfDay::hours(hour)).sun.direction.z
    }

    #[test]
    fn the_sun_climbs_to_noon_and_returns_to_the_horizon() {
        // Sunrise and sunset put it *on* the horizon, which Morrowind's own constant never does —
        // its vertical component is fixed, so its sun cannot fall below fourteen degrees.
        assert!(climb(6.0).abs() < 1e-6, "{}", climb(6.0));
        assert!(climb(20.0).abs() < 1e-6, "{}", climb(20.0));

        // Highest at the middle of the day, and symmetric either side of it.
        assert!(climb(13.0) > climb(10.0) && climb(10.0) > climb(7.0));
        assert!((climb(10.0) - climb(16.0)).abs() < 1e-6);
        // Morrowind's noon, out of its own `(0, 75, -100)`: 100/125 of the way up.
        assert!((climb(13.0) - 0.8).abs() < 1e-6, "{}", climb(13.0));
    }

    #[test]
    fn the_sun_reddens_and_dims_as_it_sets() {
        let noon = Sky::at(TimeOfDay::hours(13.0)).sun.colour;
        let dusk = Sky::at(TimeOfDay::hours(19.5)).sun.colour;

        // **Hand-computed from the constants above.** At noon the sun stands at 53.1 degrees, which
        // is 1.249 air masses, so `exp(-0.0464 * 1.249)` = 0.944 of the red survives and
        // `exp(-0.2648 * 1.249)` = 0.718 of the blue. The sun's own colour is `(1, 0.95, 0.88)`
        // times a daylight scale of 8, so red comes out at `8 * 0.944` = 7.55.
        assert!((noon.x - 7.55).abs() < 0.05, "{noon:?}");
        assert!((noon.z - 8.0 * 0.88 * 0.718).abs() < 0.05, "{noon:?}");

        // Setting, it is both dimmer and redder — and *redder* is the claim, so it is the ratio
        // that has to move, not just the magnitude. Blue falls away far faster than red.
        assert!(dusk.x < noon.x * 0.75, "{dusk:?} against {noon:?}");
        assert!(
            dusk.z / dusk.x < 0.25 * (noon.z / noon.x),
            "{dusk:?} against {noon:?}"
        );
    }

    #[test]
    fn the_sky_is_blue_by_day_and_warm_at_dusk_and_gone_at_night() {
        let noon = Sky::at(TimeOfDay::hours(13.0)).ambient;
        // Blue above green above red is what Rayleigh scattering means, and the reason the sky has
        // a colour at all.
        assert!(noon.z > noon.y && noon.y > noon.x, "{noon:?}");

        // Against the fixed overcast blue this replaced, which `SKY_STRENGTH` is set to match.
        assert!(
            (noon.length() - Vec3::new(0.35, 0.42, 0.55).length()).abs() < 0.12,
            "{noon:?}"
        );

        // At dusk the order inverts: the blue never got through the air to be scattered.
        let dusk = Sky::at(TimeOfDay::hours(19.8)).ambient;
        assert!(dusk.x > dusk.z, "{dusk:?}");
        assert!(dusk.length() < noon.length(), "{dusk:?} against {noon:?}");

        // And at night there is no sun at all, so nothing casts and only the floor is left.
        let night = Sky::at(TimeOfDay::hours(1.0));
        assert_eq!(night.sun.colour, Vec3::ZERO);
        assert_eq!(night.ambient, NIGHT_SKY);
    }
}
