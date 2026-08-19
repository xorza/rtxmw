//! An hour of Morrowind's day.

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
const SUNRISE: f32 = 6.0;
const SUNSET: f32 = 18.0;

/// When the stars come out and when they go in, all out of the same `[Weather]` section.
///
/// `Stars Post-Sunset Start=1` — they begin an hour after the sun has gone; `Stars Fading
/// Duration=2` — over two hours; `Stars Pre-Sunrise Finish=2` — and they are gone two hours before it
/// returns. Morrowind's own schedule, and the reason this is not a curve chosen by eye.
const STARS_AFTER_SUNSET: f32 = 1.0;
const STARS_BEFORE_SUNRISE: f32 = 2.0;
const STARS_FADE: f32 = 2.0;

/// A time on the game's clock.
///
/// The engine tracks no time yet, so nothing advances this — it is set once from the command line
/// and holds. What it exists for is that everything lighting an exterior is a function of it, and
/// that function had been a constant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeOfDay {
    hours: f32,
}

impl TimeOfDay {
    /// A time in hours from midnight, wrapped into a day.
    pub fn hours(hours: f32) -> Self {
        Self {
            hours: hours.rem_euclid(24.0),
        }
    }

    /// The hour itself, wrapped into a day.
    pub fn hour(self) -> f32 {
        self.hours
    }

    /// Where the sun stands in its arc: 1 at sunrise, 0 at noon, −1 at sunset.
    ///
    /// **It runs past both ends rather than stopping there**, and that is what carries the night:
    /// midnight is 1.86 and the sky reads whatever is beyond ±1 as the sun being down. Keeping it a
    /// continuous function of the hour rather than clamping is what lets dusk fade instead of
    /// switching off.
    pub fn orbit(self) -> f32 {
        1.0 - 2.0 * (self.hours - SUNRISE) / (SUNSET - SUNRISE)
    }

    /// How much of the star field is out, from none to all of it.
    ///
    /// **Not a function of the sun's height**, which is what makes it worth its own method: the game
    /// gives the stars a clock of their own, and it is not symmetric — they take an hour to begin
    /// after sunset and are gone a full two hours before sunrise, so a night has more starlight in
    /// its first half than its second.
    pub fn starlight(self) -> f32 {
        // Hours since the sun went down, which is the frame the whole schedule is written in.
        let since = (self.hours - SUNSET).rem_euclid(24.0);
        let night = 24.0 - (SUNSET - SUNRISE);
        let rising = ((since - STARS_AFTER_SUNSET) / STARS_FADE).clamp(0.0, 1.0);
        let setting = ((night - STARS_BEFORE_SUNRISE - since) / STARS_FADE).clamp(0.0, 1.0);
        rising.min(setting)
    }
}

impl std::fmt::Display for TimeOfDay {
    /// As the decimal hour, which is the form the command line reads back.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.hours)
    }
}

impl Default for TimeOfDay {
    /// Nine in the morning.
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
        assert_eq!(TimeOfDay::hours(SUNRISE).orbit(), 1.0);
        assert_eq!(TimeOfDay::hours(SUNSET).orbit(), -1.0);
        // And the middle of the day is the hour the clock calls noon, which it is only because the
        // game's own sunrise and sunset happen to straddle it evenly.
        assert_eq!(TimeOfDay::hours(12.0).orbit(), 0.0);

        // The default is the sun this renderer was built against.
        assert_eq!(TimeOfDay::default().orbit(), 0.5);
    }

    #[test]
    fn the_stars_keep_the_game_s_own_hours() {
        // Sunset is 18 and sunrise 6, so: nothing until 19, full by 21, and gone again by 04 —
        // an hour after the sun leaves, two hours before it returns, two hours of fading either way.
        let at = |hour: f32| TimeOfDay::hours(hour).starlight();
        assert_eq!(at(12.0), 0.0, "no stars at noon");
        assert_eq!(at(18.0), 0.0, "none at sunset either — they wait an hour");
        assert_eq!(at(19.0), 0.0);
        assert!(
            (at(20.0) - 0.5).abs() < 1e-6,
            "half out an hour into the fade"
        );
        assert_eq!(at(21.0), 1.0);
        assert_eq!(at(1.0), 1.0, "and still out past midnight");
        assert_eq!(at(2.0), 1.0);
        assert!(
            (at(3.0) - 0.5).abs() < 1e-6,
            "half gone an hour into the other fade"
        );
        assert_eq!(at(4.0), 0.0, "two hours before sunrise, as the ini says");
        assert_eq!(at(5.0), 0.0);

        // **Off-centre on purpose.** Both fades are two hours, so the ramps are exact mirrors of
        // each other — the asymmetry is where they sit: the stars wait one hour after sunset and
        // leave two before sunrise, so the full-strength window runs 21:00 to 02:00 and its middle
        // is half an hour *before* midnight rather than at it.
        assert_eq!(at(23.5), 1.0);
        assert_eq!(at(0.0), 1.0);
        // The night runs 18:00 to 06:00, so the plateau's middle sits before the night's.
        let plateau_middle = (21.0 + (2.0 + 24.0)) / 2.0;
        assert!(plateau_middle < (18.0 + (6.0 + 24.0)) / 2.0);
    }

    #[test]
    fn night_is_everything_outside_the_arc_and_the_hour_wraps() {
        // Inside the arc by day and outside it by night, which is what `Sky` reads to decide there
        // is no sun — and it reads the distance rather than the fact, so dusk can fade.
        assert!(TimeOfDay::hours(6.0).orbit().abs() <= 1.0);
        assert!(TimeOfDay::hours(18.0).orbit().abs() <= 1.0);
        assert!(TimeOfDay::hours(5.0).orbit().abs() > 1.0);
        assert!(TimeOfDay::hours(19.0).orbit().abs() > 1.0);

        // Midnight from either side is the same midnight, and it is dark.
        assert_eq!(TimeOfDay::hours(24.0), TimeOfDay::hours(0.0));
        assert_eq!(TimeOfDay::hours(-1.0), TimeOfDay::hours(23.0));
        assert!(TimeOfDay::hours(0.0).orbit().abs() > 1.0);

        // The command line reads back what this prints, which is what lets clap take the default
        // from here rather than from a literal of its own.
        assert_eq!(TimeOfDay::default().to_string(), "9");
    }
}
