//! An hour of Morrowind's day.

/// When the sun clears the horizon and when it returns to it, on the game's own clock.
///
/// Morrowind's, out of the original engine's ini fallbacks (`Weather_Sunrise_Time` and
/// `Weather_Sunset_Time`) rather than anything astronomical: the day is fourteen hours long and the
/// same length all year, because the game has no seasons and no latitude.
const SUNRISE: f32 = 6.0;
const SUNSET: f32 = 20.0;

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
}

impl std::fmt::Display for TimeOfDay {
    /// As the decimal hour, which is the form the command line reads back.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.hours)
    }
}

impl Default for TimeOfDay {
    /// Half past nine in the morning.
    ///
    /// The hour whose orbit is 0.5, which is what the sun was fixed at before there were any others
    /// — so the default frame is the frame this project has been looking at all along.
    fn default() -> Self {
        Self::hours(9.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_orbit_runs_from_one_at_sunrise_to_minus_one_at_sunset() {
        assert_eq!(TimeOfDay::hours(SUNRISE).orbit(), 1.0);
        assert_eq!(TimeOfDay::hours(SUNSET).orbit(), -1.0);
        // Noon is the midpoint of *this* day, which is 13:00 rather than 12:00 — the game's day is
        // not centred on the hour its clock calls noon.
        assert_eq!(TimeOfDay::hours(13.0).orbit(), 0.0);

        // The default is the sun this renderer was built against.
        assert_eq!(TimeOfDay::default().orbit(), 0.5);
    }

    #[test]
    fn night_is_everything_outside_the_arc_and_the_hour_wraps() {
        // Inside the arc by day and outside it by night, which is what `Sky` reads to decide there
        // is no sun — and it reads the distance rather than the fact, so dusk can fade.
        assert!(TimeOfDay::hours(6.0).orbit().abs() <= 1.0);
        assert!(TimeOfDay::hours(20.0).orbit().abs() <= 1.0);
        assert!(TimeOfDay::hours(5.0).orbit().abs() > 1.0);
        assert!(TimeOfDay::hours(21.0).orbit().abs() > 1.0);

        // Midnight from either side is the same midnight, and it is dark.
        assert_eq!(TimeOfDay::hours(24.0), TimeOfDay::hours(0.0));
        assert_eq!(TimeOfDay::hours(-1.0), TimeOfDay::hours(23.0));
        assert!(TimeOfDay::hours(0.0).orbit().abs() > 1.0);

        // The command line reads back what this prints, which is what lets clap take the default
        // from here rather than from a literal of its own.
        assert_eq!(TimeOfDay::default().to_string(), "9.5");
    }
}
