//! What falls out of the sky, out of the ini's own counts and speeds.

use crate::ini::Ini;

/// How far a drop falls in a second at `Precip Gravity`, in world units.
///
/// The general `[Weather]` section's `Precip Gravity=575`, which the entrance speeds multiply.
/// Seventy units to the metre puts it at 8.2 metres a second before a weather's own speed is
/// applied — which for rain's 7 comes to 57 metres a second, half again what a real drop reaches.
/// The number is the game's and the exaggeration is the game's too: Morrowind's rain is drawn as
/// long streaks and they have to move to read as rain.
const GRAVITY: f32 = 575.0;

/// How many real drops there are for every one the game counts.
///
/// **Morrowind's counts are a sprite-era budget, not a rain rate.** Four hundred and fifty drops in
/// a cylinder nine metres across and thirteen tall is 0.6 to the cubic metre. Marshall and Palmer's
/// distribution — the standard one, `N(D) = N0 exp(-ΛD)` with `N0 = 8000 m^-3 mm^-1` and
/// `Λ = 4.1 R^-0.21` — puts moderate rain at about **three thousand** to the cubic metre, five
/// thousand times as many. Six hundred sprites was all a 2002 renderer could draw; a ray does not
/// have that problem.
///
/// The *ratios* stay the game's: thunderstorm's 650 against rain's 450 is still the heavier rain,
/// and snow's 750 flakes still outnumber both.
const RAINFALL: f32 = 5000.0;

/// The lightest count any weather names, which the spread below is measured from.
///
/// `[Weather Rain]`'s `Max Raindrops`. Rain is the weather that stays exactly where the file puts
/// it, and every other one is a ratio to it.
const LIGHTEST: f32 = 450.0;

/// How much steeper the counts are in this renderer than the ini writes them.
///
/// **The second place the game's own ratio is not obeyed, and for the same reason as the first.**
/// How much of a ray the rain covers goes as the count, and the file puts a thunderstorm at 650
/// drops against rain's 450 — a ratio of 1.44, which is a change nobody watching a storm break
/// would call one. `DEPTH_CURVE` in `sky.rs` has the whole argument: a number Bethesda tuned against
/// a fixed sprite budget is not a rate, and no rescaling that leaves rain where it is can pull 1.44
/// apart.
///
/// So the order stays the game's and the spacing does not. Two and a half leaves rain untouched by
/// construction and puts a thunderstorm at two and a half times it, which is the difference between
/// weather and weather to get out of.
const COUNT_CURVE: f32 = 2.5;

/// What a weather drops, and how thickly.
///
/// **Two kinds and the game separates them by which keys a weather names.** Rain is `Using
/// Precip=1` with a `Rain *` block — only `[Weather Rain]` and `[Weather Thunderstorm]` carry one.
/// Snow is a `Snow *` block, which only `[Weather Snow]` writes in full; `[Weather Blizzard]` names
/// none of its own and falls back to those, which is what a fallback is for and what
/// `Schedule::read` already does for the colours.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Precipitation {
    /// How many of them are in the air at once, out of `Max Raindrops` or `Max Snowflakes`.
    ///
    /// Zero is a weather with none, which is six of the ten.
    pub count: f32,
    /// How wide the volume they fall through is, out of `Rain`/`Snow Diameter`.
    ///
    /// **A cylinder round the camera rather than the world**, which is how the original engine
    /// places them and which this keeps for a better reason than fidelity: past a few metres a drop
    /// is finer than a pixel, and what stands in for it further out is the weather's own fog.
    pub diameter: f32,
    /// How tall the volume they fall through is, between `Height Min` and `Height Max`.
    ///
    /// The two are where a drop comes into the world and where it leaves it, so what stands between
    /// them is the difference — five hundred units for rain, three hundred for snow. Adding them,
    /// which this did first, is arithmetic with nothing behind it.
    pub height: f32,
    /// How fast they fall, in world units a second: `GRAVITY` times the weather's entrance speed.
    pub fall: f32,
    /// Whether these are drops or flakes, which decides how they are lit and how they are shaped.
    pub snow: bool,
}

impl Precipitation {
    /// A weather with nothing falling out of it.
    pub const NONE: Self = Self {
        count: 0.0,
        diameter: 0.0,
        height: 0.0,
        fall: 0.0,
        snow: false,
    };

    /// How much slower a flake falls than a drop of the same entrance speed.
    ///
    /// The general section's `Snow Gravity Scale=0.1`: a flake is a tenth the drop's weight for its
    /// area, so it drifts where rain falls, which is most of what tells the two apart on sight.
    const SNOW_GRAVITY: f32 = 0.1;

    /// What `section` says falls out of it, or [`Self::NONE`] where it names nothing.
    ///
    /// **`sheet` settles blizzard**, which names no precipitation block at all — no count, no
    /// diameter, nothing. It plainly snows, and what says so in the file rather than in its name is
    /// the same thing §8.59 found: its cloud sheet is `Tx_BM_Sky_Blizzard`, one of Bloodmoon's two,
    /// where ashstorm and blight — which also carry a `Storm Threshold` and plainly do *not* snow —
    /// are Vvardenfell's. So a Bloodmoon weather with no block of its own takes `[Weather Snow]`'s,
    /// which is the same fallback the colour schedules use.
    pub(crate) fn read(ini: &Ini, section: &str, sheet: &str) -> Self {
        let read = Self::named(ini, section);
        match read.falls() || !sheet.starts_with("tx_bm_") {
            true => read,
            false => Self::named(ini, "weather snow"),
        }
    }

    /// What one section names, with no fallback to anyone else's.
    fn named(ini: &Ini, section: &str) -> Self {
        let number = |key: &str| ini.number(section, key);
        // **Rain is the flag and snow is the block**, which is how the file tells them apart: only
        // the two rain weathers set `Using Precip`, and only `[Weather Snow]` writes a full `Snow`
        // block — blizzard names none and takes its own siblings', which is the same fallback the
        // colour schedules use.
        let raining = number("Using Precip").is_some_and(|flag| flag > 0.0);
        let (kind, gravity) = match raining {
            true => ("Rain", 1.0),
            false => ("Snow", Self::SNOW_GRAVITY),
        };
        let count = number(match raining {
            true => "Max Raindrops",
            false => "Max Snowflakes",
        });
        let Some(count) = count.filter(|held| *held > 0.0) else {
            return Self::NONE;
        };
        let low = number(&format!("{kind} Height Min")).unwrap_or(200.0);
        let high = number(&format!("{kind} Height Max")).unwrap_or(700.0);
        Self {
            count,
            diameter: number(&format!("{kind} Diameter")).unwrap_or(600.0),
            height: high - low,
            fall: GRAVITY * gravity * number(&format!("{kind} Entrance Speed")).unwrap_or(6.0),
            snow: !raining,
        }
    }

    /// Whether anything falls at all, which is what the renderer branches on.
    pub fn falls(&self) -> bool {
        self.count > 0.0
    }

    /// How far apart the streaks stand, in world units.
    ///
    /// **Derived from the count rather than named**, because the count is what the game gives: so
    /// many in the air at once inside a cylinder `diameter` across and `height` tall. One drop to a
    /// cube of this side puts the same number in the same volume, and the lattice the shader tests
    /// against is that cube — a square across the fall, repeating along it. At a real rain rate — see
    /// `RAINFALL` — that comes to about five units, seven centimetres, which is what three
    /// thousand drops to the cubic metre means.
    pub fn spacing(&self) -> f32 {
        if !self.falls() {
            return 0.0;
        }
        let radius = self.diameter * 0.5;
        let volume = std::f32::consts::PI * radius * radius * self.height;
        let counted = LIGHTEST * (self.count / LIGHTEST).powf(COUNT_CURVE);
        (volume / (counted * RAINFALL)).cbrt()
    }

    /// How far from the eye they are drawn, in world units.
    ///
    /// The game's own cylinder radius. **Past it a drop is finer than a pixel and the weather's fog
    /// is what stands in for it**, which is not a compromise: distant rain really is a haze, and
    /// rain weather carries nearly twice clear's `Land Fog Depth` to be that haze with.
    pub fn reach(&self) -> f32 {
        self.diameter * 0.5
    }
}
