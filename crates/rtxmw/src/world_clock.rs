//! The clock the world runs on, which is not the wall's.

use rtxmw_scene::WorldTime;

/// Game seconds per real second at speed 1.
///
/// Morrowind's own `timescale`, which makes its twelve-hour day take twenty-four minutes of
/// sitting still. That is the right rate to *play* at and far too slow to look at, which is what
/// everything below is for.
const TIMESCALE: f32 = 30.0;

/// What one press of the speed key is worth, and how far the presses go.
///
/// **Four rather than two, and five settings in all**: 1, 4, 16, 64, 256. The timescale is already
/// a factor of thirty, which is easy to forget and worth writing down — a real second is thirty game
/// seconds before the speed multiplies it at all.
///
/// What that buys, per press:
///
/// | speed | the twelve-hour day | a fog bank crossing its own width |
/// |---|---|---|
/// | 1 | 24 minutes | 64 s |
/// | 16 | 90 s | 4 s |
/// | 64 | 22 s | 1 s |
/// | 256 | 5.6 s | a quarter second |
///
/// 64 is where the fog reads — banks forming and pulling apart at about a bank a second — and 256 is
/// where a whole day is a few seconds. Beyond that both stop being watchable rather than starting:
/// the ceiling was 4096 until the arithmetic was done, and at 4096 the whole day passes in a tenth of
/// a second.
const SPEED_STEP: f32 = 4.0;
const SPEED_MAX: f32 = 256.0;
const SPEED_MIN: f32 = 1.0;

/// The longest real interval one call may carry, in seconds.
///
/// **A clock is not a velocity, and that is why this is here.** Every other consumer of a frame's
/// delta recovers from a stall on its own — a camera that missed a second simply did not move — but
/// a clock that missed a second at 4096x has to decide whether thirty-four game hours passed. They
/// did not: the frame was loading a cell. A twentieth of a second is a long frame and anything
/// beyond it is a stall, so time stops for it rather than lurching.
const LONGEST_STEP: f32 = 0.05;

/// What one nudge of the hour key is worth.
///
/// Half an hour, which is about the coarsest step that still lands somewhere different near sunrise
/// — the sun crosses its lowest and most interesting five degrees in under one of these.
const NUDGE: f32 = 0.5;

/// How long the world has been running and what hour it is there.
///
/// **Both come off the same clock**, which is what makes one speed control enough: the fog's drift
/// is a distance the wind has carried it and the sun's height is an hour of the day, and holding a
/// key speeds up the two together the way a day actually does.
#[derive(Debug)]
pub(crate) struct WorldClock {
    /// Seconds the world has run, which is what the fog and the water drift against.
    elapsed: f32,
    /// The same, with the speed key left out — see [`Self::weather_seconds`].
    weather: f32,
    /// What the world's clock reads, which counts the days as well as the hour — the moons' phase
    /// is the thing that needs the second half.
    time: WorldTime,
    speed: f32,
    paused: bool,
}

impl WorldClock {
    /// A clock started at `time`, running at the game's own rate.
    pub(crate) fn starting_at(time: WorldTime) -> Self {
        Self {
            elapsed: 0.0,
            weather: 0.0,
            time,
            speed: SPEED_MIN,
            paused: false,
        }
    }

    /// Carries the world forward by `dt` real seconds, ignoring whatever a stall added to it.
    pub(crate) fn advance(&mut self, dt: f32) {
        if self.paused {
            return;
        }
        let step = dt.min(LONGEST_STEP);
        let seconds = step * self.speed;
        self.elapsed += seconds;
        self.weather += step;
        self.time = self.time.advanced(seconds * TIMESCALE / 3600.0);
    }

    /// Moves the hour by `steps` nudges without moving anything else.
    ///
    /// **Works while paused, and that is the point of having it.** Speeding the clock up to reach
    /// dusk drags the fog through an hour of wind on the way; this arrives there with the banks
    /// where they were, which is what makes two hours comparable.
    pub(crate) fn nudge(&mut self, steps: f32) {
        self.time = self.time.advanced(steps * NUDGE);
    }

    /// Moves the speed by `steps` presses, in either direction.
    ///
    /// Steps rather than the factor `Camera::scale_speed` takes, because the two ends of this range
    /// are three orders apart and a caller counting presses should not have to know the base.
    pub(crate) fn step_speed(&mut self, steps: f32) {
        self.speed = (self.speed * SPEED_STEP.powf(steps)).clamp(SPEED_MIN, SPEED_MAX);
    }

    pub(crate) fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    /// Seconds since the world started, at its own rate rather than the wall's.
    pub(crate) fn seconds(&self) -> f32 {
        self.elapsed
    }

    /// The same clock with the speed key taken out of it, for what happens on its own schedule.
    ///
    /// **A quarter of a second is a quarter of a second.** [`Self::seconds`] is what the speed key
    /// multiplies, which is right for everything it was built for — a fog bank crossing its own
    /// width, a swell, a day — because those are all *rates* worth watching faster. A lightning
    /// flash is not a rate: it is an event lasting the two hundred milliseconds it lasts, and run
    /// through a clock at sixteen it is over inside one frame and at 256 inside a millisecond. Asking
    /// for a bolt did nothing at all at any speed but the slowest, which is not a tuning problem but
    /// a category one — the speed key answers "how fast does the world turn", and a flash does not
    /// turn.
    ///
    /// Still stops when the clock is paused, because a paused world holding a bolt at full brightness
    /// is exactly what pausing is for.
    pub(crate) fn weather_seconds(&self) -> f32 {
        self.weather
    }

    pub(crate) fn time(&self) -> WorldTime {
        self.time
    }
}

/// An hour of the world's day as a clock reads it, `HH:MM`.
///
/// **A type rather than a format string**, because there are two places that want one — the
/// window's title and a marked spot — and the minute is derived rather than printed: whether the
/// seconds truncate or round is a decision, and one made twice is two decisions waiting to differ.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ClockFace(pub(crate) WorldTime);

impl std::fmt::Display for ClockFace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let hour = self.0.hour();
        // Truncated, which is what a clock does: it reads 14:15 for the whole of that minute rather
        // than rounding up halfway through it.
        write!(f, "{:02}:{:02}", hour as u32, (hour.fract() * 60.0) as u32)
    }
}

impl std::fmt::Display for WorldClock {
    /// As a clock face and a rate, which is what the window's title shows.
    ///
    /// Writes into the caller's formatter rather than handing back a `String`: the title is built
    /// twice a second on the frame path, and this is the difference between one allocation there
    /// and three.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ClockFace(self.time()))?;
        match self.paused {
            true => write!(f, " paused"),
            false => write!(f, " {:.0}x", self.speed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real minute, delivered the way a window delivers one: in frames.
    fn a_minute(clock: &mut WorldClock) {
        for _ in 0..(60.0 / LONGEST_STEP) as u32 {
            clock.advance(LONGEST_STEP);
        }
    }

    #[test]
    fn the_speed_key_hurries_the_world_and_leaves_the_weather_alone() {
        // **The two clocks part company here or nowhere.** A flash lasts a fixed quarter of a second,
        // so run off `seconds` it lasted that divided by the speed — one frame at sixteen, a
        // millisecond at 256 — and asking for a bolt did nothing at any setting but the slowest. The
        // fix is a clock the speed key cannot reach, and this is the only place that can say so.
        let mut clock = WorldClock::starting_at(WorldTime::hours(9.5));
        clock.step_speed(2.0);
        a_minute(&mut clock);
        assert!(
            (clock.weather_seconds() - 60.0).abs() < 1e-2,
            "a minute of wall time is a minute of weather, not {}",
            clock.weather_seconds()
        );
        assert!(
            clock.seconds() > clock.weather_seconds() * 4.0,
            "and the world should have run well past it — {} against {}",
            clock.seconds(),
            clock.weather_seconds()
        );

        // Both stop together, because a paused world holding a bolt at full brightness is what
        // pausing is for.
        let (held, weather) = (clock.seconds(), clock.weather_seconds());
        clock.toggle_pause();
        a_minute(&mut clock);
        assert_eq!((clock.seconds(), clock.weather_seconds()), (held, weather));
    }

    #[test]
    fn the_hour_advances_with_the_clock_and_the_speed_multiplies_both() {
        let mut clock = WorldClock::starting_at(WorldTime::hours(9.5));
        // A real minute at the game's own rate is thirty game minutes, which is half an hour.
        a_minute(&mut clock);
        assert!((clock.time().hour() - 10.0).abs() < 1e-3, "{clock}");
        assert!((clock.seconds() - 60.0).abs() < 1e-3);

        // Sixteen times the speed is sixteen times both — the drift the fog reads and the hour.
        clock.step_speed(2.0);
        a_minute(&mut clock);
        assert!((clock.time().hour() - 18.0).abs() < 1e-2, "{clock}");
        // Loosely, because this is twelve hundred `f32` additions and the last of them land on a
        // number whose own spacing is 6e-5.
        assert!((clock.seconds() - (60.0 + 60.0 * 16.0)).abs() < 0.2);
    }

    #[test]
    fn a_stalled_frame_carries_one_step_rather_than_all_of_it() {
        // **The case this exists for**: a cell load stalls the better part of a second, and at the
        // top speed that would be thirty-four game hours — the sun past a whole day and back — for
        // a frame in which nothing actually happened.
        let mut fast = WorldClock::starting_at(WorldTime::hours(12.0));
        for _ in 0..6 {
            fast.step_speed(1.0);
        }
        fast.advance(0.9);
        assert_eq!(fast.seconds(), LONGEST_STEP * SPEED_MAX);
        // Under a tenth of an hour rather than the two and a half hours the unclamped stall was
        // worth: `0.05 * 256 * 30 / 3600`.
        assert!((fast.time().hour() - 12.0 - 0.1067).abs() < 1e-3, "{fast}");

        // And an ordinary frame is carried whole, so the clamp costs nothing when nothing stalled.
        let mut steady = WorldClock::starting_at(WorldTime::hours(12.0));
        steady.advance(LONGEST_STEP * 0.5);
        assert_eq!(steady.seconds(), LONGEST_STEP * 0.5);
    }

    #[test]
    fn the_speed_is_clamped_and_pausing_stops_everything_but_a_nudge() {
        let mut clock = WorldClock::starting_at(WorldTime::default());
        // However hard either end is leaned on, it stops where the constants say.
        for _ in 0..20 {
            clock.step_speed(1.0);
        }
        assert!(
            clock.to_string().ends_with(&format!("{SPEED_MAX:.0}x")),
            "{clock}"
        );
        for _ in 0..20 {
            clock.step_speed(-1.0);
        }
        assert!(
            clock.to_string().ends_with(&format!("{SPEED_MIN:.0}x")),
            "{clock}"
        );

        clock.toggle_pause();
        assert!(clock.to_string().ends_with("paused"), "{clock}");
        let (before, hour) = (clock.seconds(), clock.time().hour());
        clock.advance(3600.0);
        assert_eq!(clock.seconds(), before, "a paused clock carries nothing");
        assert_eq!(clock.time().hour(), hour);

        // **But a nudge still lands**, which is what makes it useful: it moves the sun to another
        // hour without moving the fog an inch, so the two frames differ in the light and nothing
        // else.
        clock.nudge(-1.0);
        assert!((clock.time().hour() - (hour - NUDGE)).abs() < 1e-4);
        assert_eq!(clock.seconds(), before);
    }

    #[test]
    fn the_clock_reads_as_a_face_and_a_rate_and_wraps_at_midnight() {
        let face = |hour| WorldClock::starting_at(WorldTime::hours(hour)).to_string();
        assert_eq!(face(9.5), "09:30 1x");
        assert_eq!(face(0.0), "00:00 1x");
        assert_eq!(face(13.25), "13:15 1x");
        // Past midnight rather than past twenty-four, which is `WorldTime`'s doing.
        assert_eq!(face(25.5), "01:30 1x");
    }
}
