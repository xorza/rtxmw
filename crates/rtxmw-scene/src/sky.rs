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

/// How much of the sky's colour is mixed back toward grey with the sun overhead.
///
/// **Single scattering is more saturated than a real sky.** The light in one has bounced more than
/// once, and every bounce mixes the channels back together; a model with one scattering event gives
/// a cobalt noon and a lurid dusk. This stands in for the bounces until there is an atmosphere LUT
/// to do it properly.
const SKY_GREYING: f32 = 0.4;

/// And how much with it on the horizon.
///
/// **That it rises at all is the part which is not a fudge.** More air is more bouncing, so less of
/// the colour survives, and a sun on the horizon is looking through thirty times what one overhead
/// is. Held level with `SKY_GREYING` the twilight tint came out three quarters saturated and an hour
/// after sunset was a flat magenta; letting it rise to here halves that and leaves noon alone.
const HORIZON_GREYING: f32 = 0.8;

/// The sky's radiance against what the arithmetic below produces.
///
/// Pure scale, set so the **default hour** comes out exactly where the model before it did — a
/// luminance of 0.518 — which keeps every image made against that one comparable. It is not the
/// fixed overcast blue that preceded all of this: that was 0.414, and the first version of the sky
/// matched it at *noon* while landing a quarter above it at half past nine. A test pins the hour
/// rather than the claim now.
///
/// `ZENITH_DEPTH` is the only constant in this file that came from anywhere but an eye; this, the
/// greying, the twilight width and the night floor are all tuned, and say so.
const SKY_STRENGTH: f32 = 2.361;

/// What the sky radiates once the sun is properly down.
///
/// Not moonlight — Morrowind has two moons and a night sky texture and this renderer draws neither.
/// It is the floor that keeps a night exterior legible rather than black, and it is blue because
/// scotopic vision is.
///
/// **Added to the daylight term rather than crossfaded with it**, which is not a detail: starlight
/// is another source and not a replacement for the sun, and a sum of two positive things cannot come
/// out below either of them. Crossfading did, and that was the bug — see the note by `HORIZON_SKY`.
const NIGHT_SKY: Vec3 = Vec3::new(0.010, 0.014, 0.026);

/// What the sky keeps when the sun is *on* the horizon, against a sun overhead.
///
/// **A sunset sky is the brightest sky there is, and this used to be zero.** The daylight term was
/// scaled by the sine of the sun's elevation, which is exactly zero at the horizon — so the sky went
/// black at the precise minute of sunrise, and every minute either side of it crossfaded between that
/// black and the night floor and so came out *below* the floor. Twilight was eighty times darker than
/// midnight, and the darkest frame of the whole day was the one the sun came up in.
///
/// The sine was the wrong function because it asks where the sun is relative to the *observer*. What
/// lights a sky is sunlit air, and the air above an observer is still in sunlight when the sun has
/// left their own horizon — which is what twilight is.
const HORIZON_SKY: f32 = 0.20;

/// How far below the horizon the sun gets before the sky has lost `1/e` of what it had there.
///
/// Six degrees is civil twilight, the point at which the brightest stars come out and a newspaper
/// stops being readable — which makes it the right e-fold: the glow is a quarter gone by nautical
/// twilight at twelve degrees and into the rounding by astronomical at eighteen.
///
/// Held as the sine of the angle, because that is what the sun's direction gives — and as a literal,
/// because `sin` is not a const function. A test below is what keeps it honest.
const TWILIGHT_FALL: f32 = 0.104_528;

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
        let bare = Sun::at(time.orbit());
        // The sun's direction travels downward, so its climb above the horizon is the negation —
        // and it is signed, because after sunset there is a good deal of sky left to light and how
        // far under the sun has gone is the only thing that says how much.
        let climb = -bare.direction.z;
        let transmitted = Self::transmittance(climb.max(0.0));

        // **The sky's colour runs between two spectra and never leaves the line between them.** One
        // is what the air scatters, which is blue; the other is what survived the air, which reddens
        // as the sun descends. How far along is how much the air took: nothing at noon, nearly all of
        // it at the horizon.
        //
        // The obvious expression is `(1 - T) * T` — light that got in and then scattered out — and it
        // is wrong in a way that is invisible until you plot it. Each channel peaks where its own
        // transmittance is a half, and green's half lands at six and a half air masses, which is an
        // elevation of nine degrees: every evening passed through **green** on its way from blue to
        // red. A sky does not do that. Two endpoints and a mix cannot, either — the line from blue to
        // red runs through grey.
        // **Blue, always**: scattering *out* of a beam and scattering *into* the sky are the same
        // event seen from two sides, so the sky's own colour is the depths above — normalised here
        // rather than written down again, because two copies of one spectrum drift apart.
        let rayleigh = ZENITH_DEPTH / (ZENITH_DEPTH.x + ZENITH_DEPTH.y + ZENITH_DEPTH.z);
        let survived = transmitted.x + transmitted.y + transmitted.z;
        let warmth = 1.0 - survived / 3.0;
        let scattered = rayleigh.lerp(transmitted / survived.max(1e-6), warmth);
        let grey = Vec3::splat((scattered.x + scattered.y + scattered.z) / 3.0);
        let greying = SKY_GREYING + (HORIZON_GREYING - SKY_GREYING) * warmth;

        // **How much of the sky is still lit**, in two halves that meet at the horizon. Above it the
        // square root of the climb rather than the climb: a sky an hour before sunset is dimmer than
        // one at noon but nothing like as much dimmer as the sun's own angle would make it. Below it,
        // an exponential decay with how far under the sun has gone, which is twilight.
        let lit = (HORIZON_SKY + (1.0 - HORIZON_SKY) * climb.max(0.0).sqrt())
            * (climb.min(0.0) / TWILIGHT_FALL).exp();

        // **The disc takes its own width to set**, half a degree, and that is the whole of what puts
        // the sun out — no separate fade, and nothing that can disagree with where the sun is. By
        // the time it matters the beam has crossed thirty-eight atmospheres and is a deep red
        // already, so this is the last of a light that has nearly gone rather than a cut.
        let radius = Sun::REAL_ANGULAR_RADIUS;
        let above = ((climb + radius) / (2.0 * radius)).clamp(0.0, 1.0);
        let showing = above * above * (3.0 - 2.0 * above);
        Self {
            sun: Sun {
                colour: bare.colour * transmitted * showing,
                ..bare
            },
            ambient: scattered.lerp(grey, greying) * SKY_STRENGTH * lit + NIGHT_SKY,
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

    /// What the eye makes of a colour, which is the only thing a "darker" claim can mean.
    fn luminance(colour: Vec3) -> f32 {
        colour.dot(Vec3::new(0.2126, 0.7152, 0.0722))
    }

    #[test]
    fn the_constants_are_the_angles_they_claim_to_be() {
        // Two of these are trigonometry that `const` cannot evaluate, so they are written out — and
        // a number written out is a number that can drift from the sentence above it.
        assert!(
            (TWILIGHT_FALL - 6.0f32.to_radians().sin()).abs() < 1e-6,
            "the twilight e-fold is not six degrees: {TWILIGHT_FALL}"
        );
        // And the sun's own disc, which the same file's disc-setting fade is measured in.
        assert!(
            (Sun::REAL_ANGULAR_RADIUS.to_degrees() * 2.0 - 0.533).abs() < 0.002,
            "the sun is not half a degree across: {}",
            Sun::REAL_ANGULAR_RADIUS
        );
    }

    #[test]
    fn the_sky_is_never_green() {
        // **Reported from the window, and the reason the tint is a mix of two spectra.** With
        // `(1 - T) * T` each channel peaked where its own transmittance was a half, and green's half
        // lands at an elevation of nine degrees — so every evening went blue, then *green*, then red.
        // A line between two endpoints cannot do that, and blue to red passes through grey.
        let mut hour = 0.0;
        while hour < 24.0 {
            let sky = Sky::at(TimeOfDay::hours(hour)).ambient;
            assert!(
                sky.y <= sky.x.max(sky.z),
                "the sky is green at {hour:.2}: {sky:?}"
            );
            hour += 1.0 / 60.0;
        }
    }

    #[test]
    fn the_night_is_blue_and_sits_on_its_floor() {
        // **Also reported: midnight came out pink.** The sun only reached twelve degrees under with
        // the old continuation, so a seventh of the sunset glow — deep red — burned all night on top
        // of the floor. Descending at a steady rate puts it sixty degrees under instead, where there
        // is nothing left to add.
        for hour in [23.0, 0.0, 1.0, 2.0, 3.0] {
            let night = Sky::at(TimeOfDay::hours(hour)).ambient;
            assert!(night.z > night.x, "{hour} is not blue: {night:?}");
            assert!(
                luminance(night) < 1.1 * luminance(NIGHT_SKY),
                "{hour} has daylight left in it: {night:?}"
            );
        }
    }

    #[test]
    fn the_sky_never_goes_darker_than_its_own_night() {
        // **The bug this exists for.** The daylight term was scaled by the sine of the sun's
        // elevation, so it hit exactly zero at the horizon, and the night floor was *crossfaded*
        // with it rather than added — which made 05:59 eighty times darker than midnight and 06:00
        // exactly black. The darkest frame of the day was the one the sun rose in.
        let floor = luminance(NIGHT_SKY);
        let mut hour = 0.0;
        while hour < 24.0 {
            let sky = Sky::at(TimeOfDay::hours(hour));
            assert!(
                luminance(sky.ambient) >= floor,
                "{hour:.2} is darker than midnight: {:?} against a floor of {NIGHT_SKY:?}",
                sky.ambient
            );
            hour += 1.0 / 60.0;
        }
    }

    #[test]
    fn the_sky_brightens_all_the_way_through_sunrise() {
        // Not merely never-black: never *dips*. A minute either side of sunrise used to differ by a
        // factor of three hundred in the wrong direction, and a curve that only avoids zero could
        // still do that.
        // **Through the night and up, and no further.** Past mid-morning the luminance eases off
        // again on its way to noon, and that is the model rather than a fault: `(1 - T) * T` is
        // light that got into the air and then scattered out of it, which is largest at middling
        // optical depth, and a noon sky is deep blue where blue carries little luminance. The
        // defect was never up there.
        let at = |hour: f32| luminance(Sky::at(TimeOfDay::hours(hour)).ambient);
        let mut hour = 0.0;
        while hour < 8.0 {
            let (before, after) = (at(hour), at(hour + 1.0 / 60.0));
            assert!(
                after >= before,
                "the sky dimmed between {hour:.2} and the minute after it: {before} to {after}"
            );
            hour += 1.0 / 60.0;
        }
        // And the day is worth something: noon is far above the night it came from, and the minute
        // sunrise happens in is above the minute before it rather than three hundred times below.
        assert!(
            at(13.0) > 20.0 * at(0.0),
            "{} against {}",
            at(13.0),
            at(0.0)
        );
        assert!(at(6.0) > at(5.0) && at(6.017) > at(6.0));
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
    fn the_sun_goes_out_over_its_own_width_and_not_before() {
        // Full daylight either side of the horizon by more than the disc is wide, and nothing at all
        // below it: the disc setting is the whole of what puts the sun out, so there is no second
        // fade to disagree with where the sun actually is.
        assert!(Sky::at(TimeOfDay::hours(6.2)).sun.colour.x > 0.0);
        assert_eq!(Sky::at(TimeOfDay::hours(5.5)).sun.colour, Vec3::ZERO);
        assert_eq!(Sky::at(TimeOfDay::hours(20.5)).sun.colour, Vec3::ZERO);

        // Half a degree of disc against fourteen hours of day is a minute or so of setting, and the
        // beam has crossed thirty-eight atmospheres by then — so what goes out is already a deep
        // red, and the blue left it long before.
        let setting = Sky::at(TimeOfDay::hours(20.0)).sun.colour;
        assert!(setting.x > setting.z * 100.0, "{setting:?}");
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

        // **At the default hour**, not at noon, because that is where `SKY_STRENGTH` is set: it
        // lands where the model before it did, so every image made against that one is still
        // comparable. Noon is brighter than mid-morning now, which it should always have been —
        // the old tint fell away at low air mass and took the middle of the day down with it.
        let default = Sky::at(TimeOfDay::default()).ambient;
        assert!((luminance(default) - 0.518).abs() < 0.01, "{default:?}");
        assert!(
            luminance(noon) > luminance(default),
            "{noon:?} against {default:?}"
        );

        // At dusk the order inverts: the blue never got through the air to be scattered.
        let dusk = Sky::at(TimeOfDay::hours(19.8)).ambient;
        assert!(dusk.x > dusk.z, "{dusk:?}");
        assert!(dusk.length() < noon.length(), "{dusk:?} against {noon:?}");

        // And at night there is no sun at all, so nothing casts.
        let night = Sky::at(TimeOfDay::hours(1.0));
        assert_eq!(night.sun.colour, Vec3::ZERO);
        // The floor is what is left, plus whatever twilight still reaches an hour past midnight —
        // which is little enough that the night is still blue.
        assert!(night.ambient.z > night.ambient.x, "{:?}", night.ambient);
        assert!(
            luminance(night.ambient) < 2.0 * luminance(NIGHT_SKY),
            "{:?}",
            night.ambient
        );
    }
}
