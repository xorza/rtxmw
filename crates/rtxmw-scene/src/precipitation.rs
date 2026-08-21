//! What falls out of the sky, out of the ini's own counts and speeds.

use glam::{Vec2, Vec3};

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
/// **Morrowind's counts are a sprite-era budget, not a rain rate**, and this is the whole of the
/// conversion — derived, not chosen. Rain's cylinder is `Rain Diameter` across and the gap between
/// its two `Height` keys tall: 600 by 500 units is 412 cubic metres, and the 450 drops
/// `Max Raindrops` allows it are **1.09 to the cubic metre**. Marshall and Palmer's distribution —
/// the standard one, `N(D) = N0 exp(-ΛD)` with `N0 = 8000 m^-3 mm^-1` and `Λ = 4.1 R^-0.21` — holds
/// `N0 / Λ` drops, which for moderate rain at ten millimetres an hour is **3,165**. The ratio is
/// 2,900 and nothing here picked it.
///
/// Six hundred sprites was all a 2002 renderer could draw; a ray does not have that problem.
///
/// **It read 0.6 and five thousand before, and both were wrong.** The density was computed while
/// `height` still added the two keys instead of subtracting them — see that field — and five
/// thousand was the ratio that followed from it. The figure then went to 2,750 by eye, tuned against
/// screenshots until rain stopped burying the scene, which landed within five percent of the number
/// the distribution gives. That agreement is the reason to take the derivation and drop the tuning.
///
/// The *ratios* stay the game's, and they show: thunderstorm's 650 against rain's 450 is the heavier
/// rain, and snow's 750 flakes outnumber both.
const RAINFALL: f32 = 2900.0;

/// What a `Wind Speed` of one blows the air along at, in world units a second.
///
/// Twenty metres a second at seventy units to the metre — `FOG_GALE`'s own figure, which is what
/// makes the rain slant with the same air that carries the fog and the ash cross on it. A
/// thunderstorm's 0.5 puts its drops over at ten metres a second against a fall of forty-one, which
/// is the lean a storm has; an ashstorm's 0.8 is sixteen, and that is the whole of what moves ash.
const GALE: f32 = 1400.0;

/// What `meshes/ashcloud.nif` puts in the air, which is the only place the game says.
///
/// **The ini gives ash no numbers at all.** Rain and snow carry a count, a diameter, two heights and
/// an entrance speed; ashstorm and blight carry colours, a fog depth, a wind speed and a `Storm
/// Threshold`, and nothing about what is blowing. What the game ships instead is a mesh — seven
/// `NiParticleSystemController` emitters attached to the camera, zero triangles and 172 vertices,
/// drawing `Tx_Ash_Cloud.tga` and `Tx_Ash_Flake.tga`.
///
/// So these are read off that mesh rather than invented: a hundred particles in the `Blizzard`
/// emitter high and ahead, twelve in each of six `Dust` emitters at eye level, spread from -1224 to
/// +1224 across and standing from 116 up to 1394. It is a sprite-era budget the same way `Max
/// Raindrops`'s 450 is, and `RAINFALL` carries it to a real density the same way.
const ASH_COUNT: f32 = 172.0;
const ASH_DIAMETER: f32 = 2448.0;
const ASH_HEIGHT: f32 = 1278.0;

/// How much of its own speed ash sinks at while the wind carries it.
///
/// **Ash does not fall, it crosses.** A mote fine enough to stay up in a storm has a terminal
/// velocity of centimetres a second against a wind of tens of metres, so what decides where it goes
/// is the air and not the ground. A twentieth keeps the drift visibly downward without turning it
/// into snow.
const ASH_SETTLES: f32 = 0.05;

/// How many real motes there are for every one the mesh counts — `RAINFALL`'s counterpart for ash.
///
/// **Derived against the real figure and then knowingly far short of it, which rain is too.** A
/// severe dust storm runs about ten milligrams of PM10 to the cubic metre; a twenty-micron grain of
/// ash at 2,500 kg/m^3 masses 1.05e-11 kg, so that is **9.6e5 motes to the cubic metre**. The mesh
/// puts 172 in seventeen and a half thousand — 0.0098 — so reality is ninety-eight million times
/// what the game draws, against rain's two thousand nine hundred.
///
/// The lattice cannot go there, and it cannot go anywhere near. One mote to a cube of `spacing` at
/// the real density is a cube 0.71 units on a side, far below the `PRECIP_CELL_MIN` the march can
/// resolve — and drawing specks finer than a pixel is what §8.64 and §8.68 are both about. Rendered
/// at 2.5 units it was not a storm, it was static across the whole frame; at 10 it was scattered
/// specks. Seven is where it reads as a field of motes with the ship still legible through it.
///
/// That leaves 981 to the cubic metre against a real 9.6e5 — a thousandth — and what stands in for
/// the rest is the weather's own fog, which for these two is the thickest of the ten after a
/// blizzard. The dust that is not drawn is the dust you are looking through.
const ASHFALL: f32 = 1.0e5;

/// Which of the three a weather puts in the air.
///
/// **The file separates them by which keys a weather names.** Rain is `Using Precip=1` with a
/// `Rain *` block — only `[Weather Rain]` and `[Weather Thunderstorm]` carry one. Snow is a `Snow *`
/// block, which only `[Weather Snow]` writes in full; `[Weather Blizzard]` names none of its own and
/// falls back to those, which is what a fallback is for and what `Schedule::read` already does for
/// the colours. Ash names neither, and is known instead by what it *does* carry: a `Storm Threshold`
/// against a sheet that is not Bloodmoon's, which is what tells an ashstorm from a blizzard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Falling {
    /// Drops: drawn as streaks, and lit as the lenses they are.
    Rain,
    /// Flakes: wider, far slower, and diffuse.
    Snow,
    /// Blown ash or blight, which the wind carries rather than gravity.
    Ash,
}

/// What a weather puts in the air, and how thickly.
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
    /// Which of the three, which decides how they move, how they are shaped and how they are lit.
    pub kind: Falling,
}

impl Precipitation {
    /// A weather with nothing in the air, which is four of the ten.
    ///
    /// The kind is arbitrary and means nothing here — an enum needs a value, and `falls` is what
    /// says whether any of the rest of this is worth reading.
    pub const NONE: Self = Self {
        count: 0.0,
        diameter: 0.0,
        height: 0.0,
        fall: 0.0,
        kind: Falling::Rain,
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
        if read.falls() {
            return read;
        }
        // **A weather with no block of its own is either a blizzard or a storm, and its sheet says
        // which.** §8.59's argument, reused: blizzard is Bloodmoon's and takes `[Weather Snow]`'s
        // block, while ashstorm and blight — which carry a `Storm Threshold` too and plainly do not
        // snow — are Vvardenfell's. So the sheet is asked first, and what is left carrying a
        // threshold is blowing rather than falling.
        if sheet.starts_with("tx_bm_") {
            return Self::named(ini, "weather snow");
        }
        match ini.number(section, "Storm Threshold").is_some() {
            true => Self::ash(ini, section),
            false => Self::NONE,
        }
    }

    /// What an ashstorm or a blight carries, out of `meshes/ashcloud.nif` and the section's wind.
    ///
    /// The volume and the count are the mesh's — see `ASH_COUNT` — and the speed is the weather's
    /// own `Wind Speed` through `GALE`, because ash goes where the air goes and the file names no
    /// speed of its own for it.
    fn ash(ini: &Ini, section: &str) -> Self {
        Self {
            count: ASH_COUNT,
            diameter: ASH_DIAMETER,
            height: ASH_HEIGHT,
            fall: GALE * ini.number(section, "Wind Speed").unwrap_or(0.0),
            kind: Falling::Ash,
        }
    }

    /// How the air carries them, in world units a second.
    ///
    /// **Held here rather than assembled by the caller**, because it is the one thing that differs
    /// between the three by more than a constant: rain and snow fall and are leaned over by the
    /// wind, and ash is *carried* by it and barely sinks at all.
    pub fn velocity(&self, wind: Vec2) -> Vec3 {
        // **Nothing in the air goes nowhere.** The wind term below does not depend on `fall`, so a
        // clear day came back with a sideways drift belonging to weather it does not have — which no
        // caller reads today, every one of them asking `precip_spacing` first, and which is exactly
        // the sort of thing a caller stops asking.
        if !self.falls() {
            return Vec3::ZERO;
        }
        match self.kind {
            // Down, slanted by the wind that carries the fog — the same air, so a storm's rain comes
            // at you rather than straight down.
            Falling::Rain | Falling::Snow => Vec3::new(wind.x * GALE, wind.y * GALE, -self.fall),
            // Along the wind, because that is what moves it; `fall` is how hard that blows.
            Falling::Ash => {
                let along = wind.normalize_or_zero() * self.fall;
                Vec3::new(along.x, along.y, -self.fall * ASH_SETTLES)
            }
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
            kind: match raining {
                true => Falling::Rain,
                false => Falling::Snow,
            },
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
    /// cube of this side puts the same number in the same volume, and the shader sizes its lattice
    /// from this — finer where the air holds more, which is the whole of how `Max Raindrops` reaches
    /// the picture.
    ///
    /// **The game's own ratios, with nothing laid over them.** A curve used to steepen the counts
    /// here, on the reasoning that thunderstorm's 650 against rain's 450 was a ratio nobody watching
    /// a storm break would call a change. The reasoning was wrong about its evidence: the two looked
    /// identical because the shader clamped them both to the same coverage, not because 1.44 is too
    /// small to see. Once the clamp could not bind, 1.44 was plainly a storm — and the same curve had
    /// meanwhile turned snow's 750 flakes into a whiteout.
    pub fn spacing(&self) -> f32 {
        if !self.falls() {
            return 0.0;
        }
        let radius = self.diameter * 0.5;
        let volume = std::f32::consts::PI * radius * radius * self.height;
        // What one of the game's own counts stands for, which is not the same substance twice.
        let real = match self.kind {
            Falling::Rain | Falling::Snow => RAINFALL,
            Falling::Ash => ASHFALL,
        };
        (volume / (self.count * real)).cbrt()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clouds::UNITS_PER_METRE;

    #[test]
    fn the_lattice_holds_the_number_of_drops_marshall_and_palmer_give_the_air() {
        // **Run out here rather than written down**, so the figure `RAINFALL` is derived from is the
        // distribution rather than a number transcribed from it. `Λ = 4.1 R^-0.21` at ten
        // millimetres an hour, and `N0 / Λ` with `N0 = 8000 m^-3 mm^-1`.
        let slope = 4.1 * 10f32.powf(-0.21);
        let expected = 8_000.0 / slope;
        assert!(
            (slope - 2.528).abs() < 1e-3 && (expected - 3_164.7).abs() < 1.0,
            "the slope should put moderate rain near 3,165 drops a cubic metre, not {expected}"
        );

        let rain = Precipitation {
            count: 450.0,
            diameter: 600.0,
            height: 500.0,
            fall: 4_025.0,
            kind: Falling::Rain,
        };
        // One drop to a cube of `spacing`, so the density is its reciprocal — carried from cubic
        // units to cubic metres, which is what the figure above is in.
        let per_cubic_metre = UNITS_PER_METRE.powi(3) / rain.spacing().powi(3);
        assert!(
            (per_cubic_metre / expected - 1.0).abs() < 0.01,
            "rain should hold Marshall-Palmer's density, not {per_cubic_metre} against {expected}"
        );

        // **And the count is what moves it**, which is the only way the ini reaches the picture: a
        // thunderstorm's 650 raindrops in the same cylinder is 650/450 of the density, so its
        // spacing is the cube root of 450/650 of rain's. Exactly, not approximately — nothing curves
        // the counts on the way through any more.
        let storm = Precipitation {
            count: 650.0,
            ..rain
        };
        let ratio = storm.spacing() / rain.spacing();
        let exact = (450.0f32 / 650.0).cbrt();
        assert!(
            (ratio - exact).abs() < 1e-5,
            "a thunderstorm should stand {exact} of rain's spacing apart, not {ratio}"
        );

        // A weather with nothing falling has no lattice at all, which is what the shader leaves on.
        assert_eq!(Precipitation::NONE.spacing(), 0.0);
    }

    #[test]
    fn an_ashstorm_is_carried_across_where_rain_falls_down() {
        let Ok(all) = crate::Weather::table() else {
            return;
        };
        if all.is_empty() {
            return;
        }
        let of = |name: &str| {
            all.iter()
                .find(|w| w.name == name)
                .expect("the ten are the game's own")
                .precipitation
        };

        // **Two of the ten blow rather than fall, and the ini never says so.** Ashstorm and blight
        // name no `Rain` or `Snow` block at all — what they carry is a `Storm Threshold` and a
        // Vvardenfell sheet, which is what tells them from a blizzard, and what the game ships for
        // them is `meshes/ashcloud.nif` rather than a count.
        assert_eq!(of("ashstorm").kind, Falling::Ash);
        assert_eq!(of("blight").kind, Falling::Ash);
        assert_eq!(
            of("blizzard").kind,
            Falling::Snow,
            "a blizzard is Bloodmoon's and snows"
        );
        assert_eq!(of("rain").kind, Falling::Rain);
        // Counted rather than named, because the number moved when ash arrived and the prose that
        // said it did not: six of the ten carried nothing until the two storms were read.
        let bare: Vec<&str> = all
            .iter()
            .filter(|w| w.precipitation == Precipitation::NONE)
            .map(|w| w.name.as_str())
            .collect();
        assert_eq!(bare, ["clear", "cloudy", "foggy", "overcast"]);

        // **What moves ash is the air, not the ground.** A wind blowing east carries it east and it
        // sinks at `ASH_SETTLES` of that; the same wind leans a raindrop over while it falls three
        // and a half times faster than it is pushed. Asked at fifteen and three rather than the
        // twenty and 3.6 they are, because a threshold sitting exactly on its constant is a test of
        // the floating point unit and nothing else.
        let east = Vec2::new(0.8, 0.0);
        let blown = of("ashstorm").velocity(east);
        assert!(
            blown.x > 15.0 * blown.z.abs() && (blown.x - 1120.0).abs() < 1.0,
            "ash should cross at the wind's own speed and barely sink — {blown}"
        );

        let wet = of("rain").velocity(east);
        assert!(
            wet.z < -3.0 * wet.x,
            "and rain should do the opposite — {wet}"
        );
    }
}
