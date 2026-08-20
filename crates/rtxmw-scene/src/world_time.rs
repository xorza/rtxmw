//! How long the world has been running, which is the hour and the date together.

/// When the sun clears the horizon and when it returns to it, on the game's own clock.
///
/// Morrowind's, out of `Morrowind.ini`'s own `[Weather]` section rather than anything astronomical:
/// the day is twelve hours long and the same length all year, because the game has no seasons and no
/// latitude.
///
/// **Sunset was 20 here until the ini was read.** The doc claimed these came from the game and only
/// one of them did — `Sunset Time=18` is what it actually ships, and a two-hour-long day was invented
/// to go with it. The default hour moved from 9.5 to 9 at the same time, which is what keeps
/// `orbit()` at the 0.5 every image in this project has been made at: the same sun, reached over a
/// shorter day.
pub(crate) const SUNRISE: f32 = 6.0;
/// And when it returns to it, out of the same section.
pub(crate) const SUNSET: f32 = 18.0;

/// When the stars come out and when they go in, all out of the same `[Weather]` section.
///
/// `Stars Post-Sunset Start=1` — they begin an hour after the sun has gone; `Stars Fading
/// Duration=2` — over two hours; `Stars Pre-Sunrise Finish=2` — and they are gone two hours before it
/// returns. Morrowind's own schedule, and the reason this is not a curve chosen by eye.
const STARS_AFTER_SUNSET: f32 = 1.0;
/// `Stars Pre-Sunrise Finish=2` — and gone two hours before it returns.
const STARS_BEFORE_SUNRISE: f32 = 2.0;
/// `Stars Fading Duration=2` — each of those two changes taking two hours.
const STARS_FADE: f32 = 2.0;

/// How many hours are in a day, which is not how many the sun is up for.
const DAY: f32 = 24.0;

/// A reading off the world's clock: hours since it started running.
///
/// **Unwrapped, and that is what the moons need.** The hour of the day is all the sun ever asked
/// for, so this held a wrapped hour until there was something in the sky whose appearance depends on
/// *which* day it is — a moon's phase advances between one midnight and the next, and a clock that
/// forgets the date cannot say which phase. [`Self::hour`] is the clock face and [`Self::day`] the
/// date; nothing reads the total but those two.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct WorldTime {
    hours: f32,
}

impl WorldTime {
    /// A time in hours since the world started, which may be any number of days in.
    pub fn hours(hours: f32) -> Self {
        Self { hours }
    }

    /// This time `hours` later, which may be negative to step back.
    ///
    /// The running total stays private: an hour of the day and a date are the two things anything
    /// outside here has a use for, and a caller holding the total would be a second place for the
    /// wrapping to be got wrong.
    pub fn advanced(self, hours: f32) -> Self {
        Self {
            hours: self.hours + hours,
        }
    }

    /// The hour on the clock face, from midnight, wrapped into a day.
    pub fn hour(self) -> f32 {
        self.hours.rem_euclid(DAY)
    }

    /// Which day it is, counted from the one the world started on.
    ///
    /// **Fractional, and deliberately so.** A moon's phase is the only thing that reads this, and a
    /// phase that stepped at midnight would jump a whole eighth of a cycle between two frames. The
    /// fraction is the day's own progress, so the terminator crawls instead.
    pub fn day(self) -> f32 {
        self.hours / DAY
    }

    /// How far the sky has turned since the sun rose, in whole turns.
    ///
    /// **What a moon rides on.** Everything in a sky travels a circle once a day, and this is where
    /// round that circle the day has got to — zero as the sun clears the horizon, a quarter at noon,
    /// a half at sunset. [`Self::orbit`] is the same journey measured the way the *sun* wants it,
    /// running 1 to −1 across the daylit half; a moon is up for the other half too and needs the
    /// whole turn.
    pub fn turns_since_sunrise(self) -> f32 {
        (self.hour() - SUNRISE) / DAY
    }

    /// Where the sun stands in its arc: 1 at sunrise, 0 at noon, −1 at sunset.
    ///
    /// **It runs past both ends rather than stopping there**, and that is what carries the night:
    /// midnight is 1.86 and the sky reads whatever is beyond ±1 as the sun being down. Keeping it a
    /// continuous function of the hour rather than clamping is what lets dusk fade instead of
    /// switching off.
    pub fn orbit(self) -> f32 {
        1.0 - 2.0 * (self.hour() - SUNRISE) / (SUNSET - SUNRISE)
    }

    /// How much of the star field is out, from none to all of it.
    ///
    /// **Not a function of the sun's height**, which is what makes it worth its own method: the game
    /// gives the stars a clock of their own, and it is not symmetric — they take an hour to begin
    /// after sunset and are gone a full two hours before sunrise, so a night has more starlight in
    /// its first half than its second.
    pub fn starlight(self) -> f32 {
        // Hours since the sun went down, which is the frame the whole schedule is written in.
        let since = (self.hour() - SUNSET).rem_euclid(DAY);
        let night = DAY - (SUNSET - SUNRISE);
        let rising = ((since - STARS_AFTER_SUNSET) / STARS_FADE).clamp(0.0, 1.0);
        let setting = ((night - STARS_BEFORE_SUNRISE - since) / STARS_FADE).clamp(0.0, 1.0);
        rising.min(setting)
    }
}

impl std::fmt::Display for WorldTime {
    /// As the hour of the day, which is the form the command line reads back.
    ///
    /// The date is dropped rather than printed: `--time` takes an hour and nothing else, so a
    /// running total would not round-trip through the argument it is the default for.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.hour())
    }
}

impl Default for WorldTime {
    /// Nine in the morning of the first day.
    ///
    /// The hour whose orbit is 0.5, which is what the sun was fixed at before there were any others
    /// — so the default frame is the frame this project has been looking at all along.
    fn default() -> Self {
        Self::hours(9.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_orbit_runs_from_one_at_sunrise_to_minus_one_at_sunset() {
        assert_eq!(WorldTime::hours(SUNRISE).orbit(), 1.0);
        assert_eq!(WorldTime::hours(SUNSET).orbit(), -1.0);
        // And the middle of the day is the hour the clock calls noon, which it is only because the
        // game's own sunrise and sunset happen to straddle it evenly.
        assert_eq!(WorldTime::hours(12.0).orbit(), 0.0);

        // The default is the sun this renderer was built against.
        assert_eq!(WorldTime::default().orbit(), 0.5);

        // **And a later day is the same sun.** The clock counts on past midnight so the moons can
        // tell one day from the next; everything the sun does has to be blind to that.
        for day in [1.0, 7.0, 365.0] {
            let later = WorldTime::hours(9.0 + day * DAY);
            assert_eq!(later.orbit(), WorldTime::default().orbit(), "day {day}");
            assert_eq!(later.hour(), 9.0);
            assert_eq!(later.starlight(), WorldTime::default().starlight());
        }
    }

    #[test]
    fn the_date_advances_with_the_hours_and_the_hour_wraps_under_it() {
        // Midnight of the first day is day zero, and the clock face reads zero with it.
        assert_eq!(WorldTime::hours(0.0).day(), 0.0);
        assert_eq!(WorldTime::hours(0.0).hour(), 0.0);
        // Noon is half a day in, which is the fraction the moons' terminator crawls on.
        assert_eq!(WorldTime::hours(12.0).day(), 0.5);
        // And twenty-four hours later the face repeats while the date does not.
        assert_eq!(WorldTime::hours(24.0).hour(), 0.0);
        assert_eq!(WorldTime::hours(24.0).day(), 1.0);
        assert_eq!(WorldTime::hours(36.0).day(), 1.5);
        // Before the start, which is what a clock nudged backwards past its own beginning gives.
        assert_eq!(WorldTime::hours(-1.0).hour(), 23.0);
        assert!(WorldTime::hours(-1.0).day() < 0.0);

        // The command line reads back what this prints, and prints the face rather than the total —
        // which is what lets clap take the default from here rather than from a literal of its own.
        assert_eq!(WorldTime::default().to_string(), "9");
        assert_eq!(WorldTime::hours(9.0 + 5.0 * DAY).to_string(), "9");
    }

    #[test]
    fn the_stars_keep_the_game_s_own_hours() {
        // Sunset is 18 and sunrise 6, so: nothing until 19, full by 21, and gone again by 04 —
        // an hour after the sun leaves, two hours before it returns, two hours of fading either way.
        let at = |hour: f32| WorldTime::hours(hour).starlight();
        assert_eq!(at(12.0), 0.0, "no stars at noon");
        assert_eq!(at(18.0), 0.0, "none at sunset either — they wait an hour");
        assert_eq!(at(19.0), 0.0);
        assert!(
            (at(20.0) - 0.5).abs() < 1e-6,
            "half out an hour into the fade"
        );
        assert_eq!(at(21.0), 1.0);
        assert_eq!(at(25.0), 1.0, "and still out past midnight");
        assert_eq!(at(26.0), 1.0);
        assert!(
            (at(27.0) - 0.5).abs() < 1e-6,
            "half gone an hour into the other fade"
        );
        assert_eq!(at(28.0), 0.0, "two hours before sunrise, as the ini says");
        assert_eq!(at(29.0), 0.0);

        // **Off-centre on purpose.** Both fades are two hours, so the ramps are exact mirrors of
        // each other — the asymmetry is where they sit: the stars wait one hour after sunset and
        // leave two before sunrise, so the full-strength window runs 21:00 to 02:00 and its middle
        // is half an hour *before* midnight rather than at it.
        assert_eq!(at(23.5), 1.0);
        assert_eq!(at(24.0), 1.0);
        // The night runs 18:00 to 06:00, so the plateau's middle sits before the night's.
        let plateau_middle = (21.0 + (2.0 + DAY)) / 2.0;
        assert!(plateau_middle < (18.0 + (6.0 + DAY)) / 2.0);
    }

    #[test]
    fn night_is_everything_outside_the_arc() {
        // Inside the arc by day and outside it by night, which is what `Sky` reads to decide there
        // is no sun — and it reads the distance rather than the fact, so dusk can fade.
        assert!(WorldTime::hours(6.0).orbit().abs() <= 1.0);
        assert!(WorldTime::hours(18.0).orbit().abs() <= 1.0);
        assert!(WorldTime::hours(5.0).orbit().abs() > 1.0);
        assert!(WorldTime::hours(19.0).orbit().abs() > 1.0);
        assert!(WorldTime::hours(0.0).orbit().abs() > 1.0);
    }
}
