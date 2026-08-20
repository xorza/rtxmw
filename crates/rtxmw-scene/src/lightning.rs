//! What a thunderstorm does with the sky, out of the four keys the ini gives it.

use std::ops::Range;

use glam::Vec3;

use crate::ini::Ini;

/// What the air in a channel is heated to, and therefore what colour a flash is.
///
/// **Thirty thousand kelvin**, which is five times the surface of the sun and the hottest thing in
/// this world by an order of magnitude. A blackbody there is far past white and well into blue; the
/// real spectrum is nitrogen lines rather than a blackbody, but both land in the same place, which
/// is why a photograph of lightning comes out blue-white against a warm landscape.
const CHANNEL: Vec3 = Vec3::new(0.66, 0.78, 1.0);

/// How long one return stroke keeps its light, in seconds.
///
/// A stroke's luminous phase is seventy microseconds — nothing on a frame's scale — so what this
/// stands for is the eye rather than the air: the afterimage and the camera's own integration, which
/// is what makes a stroke read as a flash instead of as a missing frame.
const STROKE: f32 = 0.035;

/// How far apart return strokes come, in seconds.
///
/// **This is the flicker, and it is measured rather than invented.** A flash is not one discharge
/// but a train of them down the same channel, forty to eighty milliseconds apart, each re-lighting a
/// path the last one ionised. That interval is why lightning stutters instead of simply appearing.
const RESTRIKE: Range<f32> = 0.04..0.08;

/// How many strokes a flash carries, from one to this.
const STROKES: u32 = 4;

/// How far a crawler runs sideways, as a multiple of the deck's own height.
///
/// **The anvil crawler, and it is the longest thing lightning does.** A discharge inside a cloud has
/// no ground pulling it down and nothing to stop it spreading, so it travels along the underside of
/// the deck instead — tens of kilometres of it, branching as it goes, crossing more sky than any
/// strike ever reaches. Four to ten times the deck's height puts one across a good part of the view
/// without leaving it.
const CRAWL: Range<f32> = 4.0..10.0;

/// How far a channel buried in the deck runs, as a fraction of the cloud's own height.
///
/// **An in-cloud discharge has somewhere to go.** It is the same event as a sheet flash and differs
/// only in whether the channel shows through the water it is in, so it needs a length — a bolt
/// hanging in the deck rather than a point. A third of the way to the ground keeps it in the cloud.
const IN_CLOUD: f32 = 0.33;

/// How far a discharge stands from the eye, in world units.
///
/// **Never near, which is the one thing a storm may not do to a camera it cannot hurt.** Eighty
/// metres is close enough to light a street and far enough that the channel is a thing in the
/// distance rather than an event in the room; four hundred is where the flash is still worth
/// drawing. Beyond that a storm is the sky brightening, which is what the sheet flashes are for.
const REACH: Range<f32> = 15_000.0..35_000.0;

/// How far up the cloud deck a discharge sits, as a fraction of the sheet's own altitude.
///
/// **The base rather than the layer, and the difference is whether a bolt can be seen at all.** The
/// sheet is drawn at five hundred metres, and a strike within a few hundred metres of the eye
/// starting from up there stands at sixty to eighty degrees of elevation — over the top of a
/// seventy-five degree frame, so every bolt was above the picture. A real storm's base is ragged and
/// far below the deck's own height, and a channel coming out from under it is what a thunderhead
/// looks like from beneath. Two fifths puts a strike at the near edge of `REACH` at forty degrees
/// and one at the far edge at twenty, which is a frame with weather in it.
const DECK: f32 = 0.4;

/// How coarse the grid a flash's position is anchored to is, in world units.
///
/// **A flash has to stand still while it burns, and nothing here remembers anything.** Everything in
/// this crate is a function of the clock, which is what lets a screenshot reproduce a frame — so a
/// position taken from the camera directly would swim as the camera walked. Snapping the camera to a
/// grid far finer than `REACH` fixes the flash in the world for as long as the eye stays in one
/// cell, and a cell is sixty metres against a flash that lasts a quarter of a second.
const ANCHOR: f32 = 4_096.0;

/// Which of the three shapes a discharge takes.
///
/// **Nature's own proportions, not a preference.** Cloud-to-ground is a fifth to a quarter of all
/// discharges; well over half happen entirely inside the cloud, and of those most are swallowed by
/// the deck they happen in — which is why a storm is mostly a sky that flickers and only sometimes a
/// bolt. Sampling any other way makes every flash an event, and a storm where everything is an event
/// has no weather in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Discharge {
    /// Inside the cloud with the channel hidden: the deck lights from within and nothing is drawn.
    Sheet,
    /// Inside the cloud with the channel showing through it.
    InCloud,
    /// Along the underside of the deck rather than down out of it, for kilometres.
    Crawler,
    /// Down to the ground, which is the one that lights a landscape.
    ToGround,
}

/// A discharge in progress: what it is, where it stands, and how bright it is now.
#[derive(Debug, Clone, Copy)]
pub struct Flash {
    /// What the flash puts into the sky, as a radiance to add. Zero between flashes.
    pub radiance: Vec3,
    /// Which shape it took.
    pub kind: Discharge,
    /// Where the discharge sits, in world units — in the cloud deck for all three kinds.
    pub source: Vec3,
    /// Where the channel ends, which for a ground strike is the ground and otherwise is `source`.
    pub ground: Vec3,
    /// What the channel's own crookedness is drawn from.
    ///
    /// **Every flash needs its own or every bolt is the same bolt.** The shape is a sum of hashed
    /// octaves, and the shader has nothing to hash but what it is handed — so seeded from a constant
    /// it drew the identical crooked line with the identical forks each time, at a different place on
    /// the compass. Which flash this is, is known here and nowhere else.
    pub seed: u32,
}

impl Flash {
    /// No flash, which is every frame of every weather but one.
    pub const NONE: Self = Self {
        radiance: Vec3::ZERO,
        kind: Discharge::Sheet,
        source: Vec3::ZERO,
        ground: Vec3::ZERO,
        seed: 0,
    };

    /// Whether anything is happening, which is what the renderer branches on.
    pub fn burning(&self) -> bool {
        self.radiance != Vec3::ZERO
    }
}

/// How a weather's lightning is scheduled, out of `Morrowind.ini`.
///
/// **Four keys is all Bethesda writes**, and only `[Weather Thunderstorm]` writes them: how often,
/// how far into the weather it has to be, how fast a flash fades, and four sounds this renderer has
/// no ears for. Everything else here — the strokes, the shapes, the colour, the distance — is
/// physics, because the file has nothing to say about it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lightning {
    /// How many flashes a second, out of `Thunder Frequency`.
    pub frequency: f32,
    /// How much of the weather has to have arrived first, out of `Thunder Threshold`.
    ///
    /// Parsed and carried; nothing reads it until weather transitions exist, at which point a storm
    /// that has only half arrived should not yet be throwing bolts.
    pub threshold: f32,
    /// How fast a flash fades, in brightness a second, out of `Flash Decrement`.
    ///
    /// Four, so a quarter of a second — which is the one place the file and the measurements agree
    /// outright: a real flash runs forty to two hundred milliseconds.
    pub decrement: f32,
}

impl Lightning {
    /// A weather that never flashes, which is nine of the ten.
    pub const NONE: Self = Self {
        frequency: 0.0,
        threshold: 0.0,
        decrement: 4.0,
    };

    /// What `section` says about its lightning, or [`Self::NONE`] where it says nothing.
    pub(crate) fn read(ini: &Ini, section: &str) -> Self {
        let number = |key: &str| ini.number(section, key);
        let Some(frequency) = number("Thunder Frequency").filter(|held| *held > 0.0) else {
            return Self::NONE;
        };
        Self {
            frequency,
            threshold: number("Thunder Threshold").unwrap_or(0.6),
            decrement: number("Flash Decrement").unwrap_or(4.0).max(0.1),
        }
    }

    /// Whether this weather has lightning at all.
    pub fn strikes(&self) -> bool {
        self.frequency > 0.0
    }

    /// How far to move the storm's clock so `now` lands at the start of a flash.
    ///
    /// **For asking a storm for a bolt without teaching it to remember being asked.** Everything
    /// here is a function of the clock, so the only way to provoke a flash that does not break that
    /// is to move the clock — back to the start of the interval `now` is already inside, which
    /// brings the flash that was coming forward rather than inventing an extra one. Nothing for a
    /// weather with no lightning, which is nine of the ten.
    fn brought_forward(&self, now: f32) -> f32 {
        match self.strikes() {
            true => -now.rem_euclid(1.0 / self.frequency),
            false => 0.0,
        }
    }

    /// How far to move the clock to bring forward a flash that would be *seen* from `eye` looking
    /// `facing`, rather than merely the next one.
    ///
    /// **For the key that asks for a bolt, where the honest answer is the wrong one.** Every flash of
    /// a storm is somewhere on the compass and most of them are not where the eye is pointed, so
    /// simply taking the next one hands back a flash that is behind you four times in
    /// five — and two of the three shapes draw no channel at all. This walks ahead through the
    /// schedule instead and stops at one that has a channel and stands in front, which chooses
    /// *which* flash to bring forward without moving any of them: the storm stays a function of its
    /// own clock and only the moment being asked for changes.
    ///
    /// Falls back to the next flash of any kind if a minute of them are all facing the wrong way.
    pub fn staged(&self, now: f32, eye: Vec3, altitude: f32, facing: Vec3) -> f32 {
        if !self.strikes() {
            return 0.0;
        }
        let interval = 1.0 / self.frequency;
        let ahead = facing.truncate().normalize_or_zero();
        // **From the interval after this one, so a new flash is new.** Starting where the clock
        // already stands looks harmless and is not: a restrike parks it at the beginning of a
        // qualifying flash, so the very next request for a fresh bolt found that one still sitting
        // there and staged it again. Whatever moment this is asked from, the answer is one the eye
        // has not just been shown.
        //
        // **And the best of what it finds rather than the first thing that passes.** This returned
        // the next flash of *any* shape when a minute of them were all facing the wrong way, which
        // meant asking for a bolt sometimes handed back a sheet flash — a discharge that draws no
        // channel at all, and so a key that did nothing. A channel behind you is worth more than no
        // channel in front of you, so it is kept and only used if nothing better turns up.
        let mut anywhere = None;
        for step in 1..=(60.0 * self.frequency) as u32 {
            let at = (now / interval).floor() * interval + step as f32 * interval;
            let flash = self.flash(at, eye, altitude);
            if flash.kind == Discharge::Sheet {
                continue;
            }
            // Within a wide half of the compass, which is a good deal more forgiving than the frame
            // is and still never behind.
            let toward = (flash.source - eye).truncate().normalize_or_zero();
            if ahead.dot(toward) > 0.5 {
                return at - now;
            }
            anywhere.get_or_insert(at - now);
        }
        anywhere.unwrap_or_else(|| self.brought_forward(now))
    }

    /// The flash at `seconds`, seen from `eye`, with the cloud deck at `altitude`.
    ///
    /// **A function of the clock and nothing else**, so two runs of the same second draw the same
    /// storm and a headless screenshot is reproducible. The flash index is the second divided by the
    /// interval, and everything about a flash — its shape, its bearing, its distance, how many times
    /// it strikes — is hashed from that one integer.
    pub fn flash(&self, seconds: f32, eye: Vec3, altitude: f32) -> Flash {
        if !self.strikes() {
            return Flash::NONE;
        }
        let interval = 1.0 / self.frequency;
        let index = (seconds / interval).floor();
        let since = seconds - index * interval;
        let seed = hash(index as i64 as u64);

        // **The envelope the ini decides and the strokes physics does.** `Flash Decrement` sets how
        // long the whole flash lasts; inside it the channel re-lights every forty to eighty
        // milliseconds, which is the stutter a flash is seen as rather than the single blink an
        // exponential decay would give.
        let envelope = 1.0 - since * self.decrement;
        if envelope <= 0.0 {
            return Flash::NONE;
        }
        let strokes = 1 + (seed >> 8) % u64::from(STROKES);
        let mut latest = 0.0;
        let mut at = 0.0;
        for stroke in 0..strokes {
            if at > since {
                break;
            }
            latest = at;
            at +=
                RESTRIKE.start + unit(hash(seed ^ (stroke + 1))) * (RESTRIKE.end - RESTRIKE.start);
        }
        let brightness = envelope * (-(since - latest) / STROKE).exp();

        // Nature's own split: a fifth to the ground, and of the rest most are swallowed by the deck.
        let roll = unit(hash(seed ^ 0x9E37_79B9));
        let kind = match roll {
            _ if roll < 0.22 => Discharge::ToGround,
            _ if roll < 0.33 => Discharge::Crawler,
            _ if roll < 0.45 => Discharge::InCloud,
            _ => Discharge::Sheet,
        };

        // Anchored to a grid so the flash stands still while it burns — see `ANCHOR`.
        let anchor = (eye / ANCHOR).floor() * ANCHOR;
        let bearing = unit(hash(seed ^ 0x85EB_CA6B)) * std::f32::consts::TAU;
        let away = REACH.start + unit(hash(seed ^ 0xC2B2_AE35)) * (REACH.end - REACH.start);
        let source =
            anchor + Vec3::new(bearing.cos() * away, bearing.sin() * away, altitude * DECK);
        Flash {
            radiance: CHANNEL * brightness,
            kind,
            seed: (seed >> 32) as u32,
            source,
            ground: match kind {
                // The eye's own level stands in for the ground under a strike hundreds of metres
                // off, which is the terrain this crate cannot sample from here.
                Discharge::ToGround => Vec3::new(source.x, source.y, anchor.z),
                // Down into the deck and no further, with a lean the hash decides — a channel in a
                // cloud is a bolt with somewhere to go, not a point that happens to glow.
                Discharge::InCloud => {
                    let drop = (source.z - anchor.z) * IN_CLOUD;
                    let lean = unit(hash(seed ^ 0x27D4_EB2F)) * std::f32::consts::TAU;
                    source + Vec3::new(lean.cos() * drop * 0.4, lean.sin() * drop * 0.4, -drop)
                }
                // **Sideways, and a long way.** Nothing under a crawler is pulling it down, so it
                // spreads along the deck it is in rather than out of it — the drop over its whole
                // length is a fraction of what it covers horizontally.
                Discharge::Crawler => {
                    let deck = source.z - anchor.z;
                    let run = deck
                        * (CRAWL.start
                            + unit(hash(seed ^ 0x1B87_3593)) * (CRAWL.end - CRAWL.start));
                    let heading = unit(hash(seed ^ 0x27D4_EB2F)) * std::f32::consts::TAU;
                    source
                        + Vec3::new(
                            heading.cos() * run,
                            heading.sin() * run,
                            -deck * IN_CLOUD * 0.5,
                        )
                }
                Discharge::Sheet => source,
            },
        }
    }
}

/// A stable scramble of one integer, for drawing a flash's shape out of its index.
///
/// **The increment is not decoration.** The mixer below is a bijection that fixes zero — feed it
/// nought and it hands nought straight back — so the first flash of every storm drew a seed of all
/// zeroes and every field taken from the raw bits of it came out the same way each time. `strokes`
/// reads the low end, so flash zero always drew a single stroke and could never stutter, which is
/// what `a_flash_stutters_rather_than_fading_evenly` tripped over before it counted across flashes.
/// Adding the golden ratio first is the standard repair and the reason splitmix carries one.
fn hash(seed: u64) -> u64 {
    let mut held = seed
        .wrapping_add(0x9E37_79B9_7F4A_7C15)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    held ^= held >> 30;
    held = held.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    held ^= held >> 27;
    held ^ (held >> 31)
}

/// A hash as a value in `0..1`.
fn unit(seed: u64) -> f32 {
    (seed >> 40) as f32 / 16_777_216.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A thunderstorm as `[Weather Thunderstorm]` describes it.
    fn storm() -> Lightning {
        Lightning {
            frequency: 0.4,
            threshold: 0.6,
            decrement: 4.0,
        }
    }

    /// How many of `count` flashes fall to each shape, walked one interval at a time.
    fn shapes(count: u32) -> [u32; 4] {
        let storm = storm();
        let mut seen = [0; 4];
        for index in 0..count {
            let flash = storm.flash(index as f32 / storm.frequency, Vec3::ZERO, 5_000.0);
            seen[match flash.kind {
                Discharge::ToGround => 0,
                Discharge::Crawler => 1,
                Discharge::InCloud => 2,
                Discharge::Sheet => 3,
            }] += 1;
        }
        seen
    }

    #[test]
    fn a_weather_the_ini_gives_no_thunder_never_flashes() {
        // Nine of the ten: only `[Weather Thunderstorm]` writes `Thunder Frequency` at all, and a
        // weather without it has no schedule to evaluate rather than a schedule that never fires.
        assert!(!Lightning::NONE.strikes());
        for second in 0..200 {
            let flash = Lightning::NONE.flash(second as f32 * 0.05, Vec3::ZERO, 5_000.0);
            assert!(!flash.burning(), "clear weather flashed at {second}");
        }
    }

    #[test]
    fn a_flash_lasts_what_the_ini_says_and_recurs_at_its_own_interval() {
        let storm = storm();
        // `Flash Decrement=4`, so a quarter of a second — which is where the file and the
        // measurements agree: a real flash runs forty to two hundred milliseconds.
        let burns = |at: f32| storm.flash(at, Vec3::ZERO, 5_000.0).burning();
        assert!(burns(0.0), "a flash should start the interval");
        assert!(
            burns(0.24),
            "and last the quarter second the decrement buys"
        );
        assert!(!burns(0.26), "and be out after it");

        // `Thunder Frequency=.4` is one every two and a half seconds, so the next one starts there
        // and not before.
        assert!(!burns(2.4), "nothing between flashes");
        assert!(burns(2.5), "and the next interval brings one");
    }

    #[test]
    fn a_flash_stutters_rather_than_fading_evenly() {
        // **Return strokes, forty to eighty milliseconds apart.** A flash is not one discharge but a
        // train of them down the same channel, and that is what makes lightning stutter instead of
        // simply appearing. An even decay falls monotonically; a stuttering one rises again.
        //
        // **Counted across flashes rather than asserted of one**, because a flash draws one to four
        // strokes and a single-stroke flash is entitled not to stutter — the first one does not,
        // which is what this test caught when it looked at that one alone.
        let storm = storm();
        let bright = |at: f32| storm.flash(at, Vec3::ZERO, 5_000.0).radiance.x;
        let stutters = |flash: u32| {
            let start = flash as f32 / storm.frequency;
            let mut last = bright(start);
            (1..100).any(|step| {
                let now = bright(start + step as f32 * 0.0025);
                let rose = now > last + 1e-4;
                last = now;
                rose
            })
        };
        let counted = (0..200).filter(|flash| stutters(*flash)).count();
        // One stroke in four draws a single, so three quarters should re-light.
        assert!(
            counted > 120,
            "most flashes should re-strike rather than fade once — {counted} of 200"
        );
    }

    #[test]
    fn the_four_shapes_come_in_the_proportions_nature_gives_them() {
        // Cloud-to-ground is a fifth to a quarter of all discharges and well over half never leave
        // the cloud at all. Over a thousand flashes the draw has to land there.
        let [ground, crawler, in_cloud, sheet] = shapes(1_000);
        assert!(
            (180..=260).contains(&ground),
            "about a fifth should reach the ground, not {ground} of a thousand"
        );
        assert!(
            sheet > ground + crawler + in_cloud,
            "most should never leave the cloud — {sheet} against {ground}, {crawler}, {in_cloud}"
        );
        // **And a crawler is the rarest thing worth drawing**, which is why one is an event: it is a
        // tenth of the flashes, against a sheet that is half of them and shows nothing at all.
        assert!(
            (70..=150).contains(&crawler),
            "about a tenth should crawl the deck, not {crawler}"
        );
    }

    #[test]
    fn a_strike_never_lands_near_the_eye() {
        // **The one thing a storm may not do to a camera it cannot hurt.** Every strike of a thousand
        // has to stand at least `REACH.start` off, whatever the hash drew.
        //
        // **Strikes, and only strikes.** A crawler runs kilometres along the underside of the deck on
        // whatever heading it drew, so one aimed back this way passes overhead — which is what an
        // anvil crawler does and the best thing in the sky when it happens. It never lands, so it
        // has nothing to do with the rule this is about; scoping to `ToGround` is what the rule
        // always meant, and the crawler is what made it say so.
        let storm = storm();
        let eye = Vec3::new(1_234.0, -5_678.0, 90.0);
        let mut nearest = f32::INFINITY;
        for index in 0..1_000 {
            let flash = storm.flash(index as f32 / storm.frequency, eye, 5_000.0);
            if flash.kind != Discharge::ToGround {
                continue;
            }
            let away = (flash.ground - eye).truncate().length();
            nearest = nearest.min(away);
        }
        // The eye is offset from the grid it anchors to, so the closest possible strike is the
        // reach less the diagonal of one anchor cell.
        let closest = REACH.start - ANCHOR * std::f32::consts::SQRT_2;
        assert!(
            nearest > closest,
            "the nearest strike should clear {closest}, not {nearest}"
        );
        assert!(
            nearest.is_finite(),
            "a thousand flashes should include a strike"
        );
    }

    #[test]
    fn the_first_flash_of_a_storm_is_no_different_from_any_other() {
        // **The mixer fixes zero**, so flash nought once drew a seed of all zeroes and every field
        // taken from its raw bits came out the same way every time — a single stroke, never a
        // stutter. See `hash`. A storm may not open with a flash that is special.
        let storm = storm();
        let bright = |at: f32| storm.flash(at, Vec3::ZERO, 5_000.0).radiance.x;
        let mut last = bright(0.0);
        let rose = (1..100).any(|step| {
            let now = bright(step as f32 * 0.0025);
            let rose = now > last + 1e-4;
            last = now;
            rose
        });
        assert!(rose, "the first flash should re-strike like any other");
    }

    #[test]
    fn asking_for_a_bolt_brings_one_you_can_see() {
        // **The next flash is the wrong answer.** Two of the three shapes draw no channel and most
        // of the compass is not where the eye is pointed, so a key that simply brought the next one
        // forward would show nothing four times in five. This walks the schedule for one that has a
        // channel and stands ahead — which chooses among the storm's own flashes rather than moving
        // any of them.
        let storm = storm();
        let eye = Vec3::new(300.0, -900.0, 60.0);
        for facing in [Vec3::X, Vec3::Y, Vec3::NEG_X, -Vec3::Y] {
            let now = 3.1 + facing.x;
            let staged = now + storm.staged(now, eye, 35_000.0, facing);
            let flash = storm.flash(staged, eye, 35_000.0);
            assert!(flash.burning(), "the staged moment should burn");
            assert_ne!(
                flash.kind,
                Discharge::Sheet,
                "and should have a channel to look at"
            );
            let toward = (flash.source - eye).truncate().normalize();
            assert!(
                facing.truncate().normalize().dot(toward) > 0.5,
                "and should stand in front of {facing}, not at {toward}"
            );
        }
        // A weather with no lightning has nothing to stage.
        assert_eq!(Lightning::NONE.staged(3.1, eye, 35_000.0, Vec3::X), 0.0);
    }

    #[test]
    fn a_storm_can_be_asked_for_a_bolt_without_being_told_to_remember_it() {
        // The offset lands the clock on a flash boundary, so the moment it is asked for burns — and
        // the storm carries on from there rather than snapping back, because nothing else changed.
        let storm = storm();
        let asked = 9.37;
        let moved = asked + storm.brought_forward(asked);
        assert!(
            storm.flash(moved, Vec3::ZERO, 5_000.0).burning(),
            "asking for a bolt should land on one"
        );
        // And it brings the one that was coming forward rather than inventing an extra: the moment
        // it lands on is the start of the interval `asked` was already inside.
        assert!(
            (moved - 7.5).abs() < 1e-4,
            "9.37 sits in the interval that opened at 7.5, not {moved}"
        );
        // A weather with no lightning has no clock to move.
        assert_eq!(Lightning::NONE.brought_forward(asked), 0.0);
    }

    #[test]
    fn no_two_flashes_are_drawn_from_the_same_shape() {
        // **The bolt the shader draws is a sum of hashed octaves and the hash is all it has.** Seeded
        // from a constant, as it was, every flash of every storm came out the identical crooked line
        // with the identical three forks — only the place on the compass changed, which is the one
        // thing lightning is never the same in.
        let storm = storm();
        let seeds: std::collections::HashSet<u32> = (0..200)
            .map(|index| {
                storm
                    .flash(index as f32 / storm.frequency, Vec3::ZERO, 35_000.0)
                    .seed
            })
            .collect();
        assert_eq!(
            seeds.len(),
            200,
            "two hundred flashes want two hundred shapes"
        );
    }

    #[test]
    fn asking_for_a_bolt_never_hands_back_one_with_no_channel() {
        // **A sheet flash draws nothing at all**, which is more than half of them and exactly what a
        // key that asks for a bolt must never return. This used to take the next flash of any shape
        // when a minute of them all faced the wrong way, so now and then pressing it did nothing —
        // and looked like the key being broken rather than the storm being obeyed.
        let storm = storm();
        let eye = Vec3::new(-400.0, 700.0, 40.0);
        for step in 0..400 {
            // Every heading, and moments between flashes as well as inside them.
            let turn = step as f32 * 0.31;
            let facing = Vec3::new(turn.cos(), turn.sin(), -0.2);
            let now = step as f32 * 0.7;
            let staged = now + storm.staged(now, eye, 35_000.0, facing);
            let flash = storm.flash(staged, eye, 35_000.0);
            assert!(flash.burning(), "the staged moment should burn, at {now}");
            assert_ne!(
                flash.kind,
                Discharge::Sheet,
                "a bolt was asked for and a sheet flash came back, at {now}"
            );
        }
    }

    #[test]
    fn the_same_second_draws_the_same_storm() {
        // Everything here is a function of the clock, which is what lets a headless screenshot
        // reproduce a frame — and what lets the two tests above walk a thousand flashes at all.
        let storm = storm();
        // Inside a flash rather than between two: at `Thunder Frequency=.4` one starts every two and
        // a half seconds and burns for a quarter of one, so most of the clock is dark.
        let once = storm.flash(37.55, Vec3::ZERO, 5_000.0);
        let again = storm.flash(37.55, Vec3::ZERO, 5_000.0);
        assert_eq!(once.kind, again.kind);
        assert_eq!(once.seed, again.seed);
        assert_eq!(once.source, again.source);
        assert_eq!(once.radiance, again.radiance);
        assert!(once.burning(), "37.55 should land inside a flash");
    }
}
