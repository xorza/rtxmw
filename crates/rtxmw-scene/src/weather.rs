//! Morrowind's ten weathers, as `Morrowind.ini` describes them.

use glam::Vec3;
use rtxmw_esm::{Cell, CellId, RegionRecord};

use crate::game_data::GameData;
use crate::ini::Ini;
use crate::world_time::{SUNRISE, SUNSET, WorldTime};

/// When one family of colours changes over, in hours either side of sunrise and sunset.
///
/// **Every family has its own**, which is why a Morrowind dusk does not change all at once: the sky
/// begins turning an hour and a half before the sun goes down and has finished half an hour after,
/// while the ambient starts an hour before and takes until an hour and a quarter after. Out of the
/// general `[Weather]` section — `Sky Pre-Sunset Time` and its eleven siblings.
#[derive(Debug, Clone, Copy)]
struct Crossover {
    pre_sunrise: f32,
    post_sunrise: f32,
    pre_sunset: f32,
    post_sunset: f32,
}

/// Which two of a family's four values an hour falls between, and how far.
///
/// The answer to the only question a schedule ever asks, so that a schedule of *colours* and a
/// schedule of *depths* can ask it in the same words — `fog_depth` was building a `Schedule` of
/// `Vec3::splat` to borrow the arithmetic before this existed.
#[derive(Debug, Clone, Copy)]
struct Between {
    /// Which of `sunrise`, `day`, `sunset`, `night` the hour is coming from and going to.
    from: Key,
    to: Key,
    /// How far along, from nought at `from` to one at `to`.
    along: f32,
}

/// One of the four times of day a weather names a colour for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Key {
    Sunrise,
    Day,
    Sunset,
    Night,
}

impl Crossover {
    /// Where `time` sits among the four, on this family's own windows.
    ///
    /// Night until the window opens before sunrise, then night to sunrise to day across it; day
    /// until the window opens before sunset, then day to sunset to night across that. Both windows
    /// are the family's own, so the sky and the ground change at different moments — which is what
    /// the twelve `Pre-` and `Post-` figures in the general section are for.
    fn between(self, time: WorldTime) -> Between {
        let hour = time.hour();
        let ramp = |from: f32, to: f32| ((hour - from) / (to - from)).clamp(0.0, 1.0);
        let held = |key: Key| Between {
            from: key,
            to: key,
            along: 0.0,
        };
        if hour < SUNRISE - self.pre_sunrise || hour > SUNSET + self.post_sunset {
            return held(Key::Night);
        }
        if hour < SUNRISE {
            return Between {
                from: Key::Night,
                to: Key::Sunrise,
                along: ramp(SUNRISE - self.pre_sunrise, SUNRISE),
            };
        }
        if hour < SUNRISE + self.post_sunrise {
            return Between {
                from: Key::Sunrise,
                to: Key::Day,
                along: ramp(SUNRISE, SUNRISE + self.post_sunrise),
            };
        }
        if hour < SUNSET - self.pre_sunset {
            return held(Key::Day);
        }
        if hour < SUNSET {
            return Between {
                from: Key::Day,
                to: Key::Sunset,
                along: ramp(SUNSET - self.pre_sunset, SUNSET),
            };
        }
        Between {
            from: Key::Sunset,
            to: Key::Night,
            along: ramp(SUNSET, SUNSET + self.post_sunset),
        }
    }

    /// The four windows named for `family` — `Sky`, `Fog`, `Ambient` or `Sun`.
    fn read(ini: &Ini, family: &str) -> Self {
        let hours = |which: &str| {
            ini.number("Weather", &format!("{family} {which} Time"))
                .unwrap_or(1.0)
        };
        Self {
            pre_sunrise: hours("Pre-Sunrise"),
            post_sunrise: hours("Post-Sunrise"),
            pre_sunset: hours("Pre-Sunset"),
            post_sunset: hours("Post-Sunset"),
        }
    }
}

/// `[Weather Clear]`'s own four sky colours, sunrise then day then sunset then night.
///
/// **What a schedule the ini does not name falls back to, and the game's numbers rather than
/// white.** These are load-bearing now: [`crate::Sky`] solves a weather's veil out of its sky and
/// fog schedules read against each other, and a schedule of white asserts *an opaque white medium
/// fills the sky* rather than *no data*. Without the game installed the engine runs on
/// [`Weather::clear`], and clear weather is this.
///
/// sRGB bytes, exactly as the file writes them, because that is what makes them checkable against
/// it by eye. [`Schedule::read`] decodes them the same way [`crate::ini::Ini::colour`] decodes the
/// real ones.
const CLEAR_SKY: [[u8; 3]; 4] = [[117, 141, 164], [95, 135, 203], [56, 89, 129], [9, 10, 11]];
/// `[Weather Clear]`'s own fog colours. See `CLEAR_SKY`.
const CLEAR_FOG: [[u8; 3]; 4] = [
    [255, 189, 157],
    [206, 227, 255],
    [255, 189, 157],
    [9, 10, 11],
];
/// `[Weather Clear]`'s own ambient colours. See `CLEAR_SKY`.
const CLEAR_AMBIENT: [[u8; 3]; 4] = [[47, 66, 96], [137, 140, 160], [68, 75, 96], [32, 35, 42]];
/// `[Weather Clear]`'s own sun colours. See `CLEAR_SKY`.
const CLEAR_SUN: [[u8; 3]; 4] = [
    [242, 159, 119],
    [255, 252, 238],
    [255, 114, 79],
    [59, 97, 176],
];

/// One family's colour at each of the four times of day the game names, and when it moves between.
#[derive(Debug, Clone, Copy)]
pub struct Schedule {
    pub sunrise: Vec3,
    pub day: Vec3,
    pub sunset: Vec3,
    pub night: Vec3,
    crossover: Crossover,
}

impl Schedule {
    /// The colour at `time`, on this family's own crossing.
    ///
    /// Night until the window opens before sunrise, then night to sunrise to day across it; day
    /// until the window opens before sunset, then day to sunset to night across that. Both windows
    /// are the family's own, so the sky and the ground change at different moments.
    pub fn at(self, time: WorldTime) -> Vec3 {
        let between = self.crossover.between(time);
        self.key(between.from)
            .lerp(self.key(between.to), between.along)
    }

    /// Whichever of the four `key` names.
    fn key(self, key: Key) -> Vec3 {
        match key {
            Key::Sunrise => self.sunrise,
            Key::Day => self.day,
            Key::Sunset => self.sunset,
            Key::Night => self.night,
        }
    }

    /// The four colours a weather names for `family`, with that family's own windows.
    ///
    /// `fallback` is what a key the ini does not name comes out as, in the order the four keys are
    /// listed below — see `CLEAR_SKY` for why it is a weather's own colours rather than white.
    fn read(ini: &Ini, section: &str, family: &str, fallback: [[u8; 3]; 4]) -> Self {
        let colour = |which: &str, spare: [u8; 3]| {
            ini.colour(section, &format!("{family} {which} Color"))
                .unwrap_or_else(|| Vec3::from(spare.map(rtxmw_texture::channel_to_linear)))
        };
        Self {
            sunrise: colour("Sunrise", fallback[0]),
            day: colour("Day", fallback[1]),
            sunset: colour("Sunset", fallback[2]),
            night: colour("Night", fallback[3]),
            crossover: Crossover::read(ini, family),
        }
    }
}

/// One of Morrowind's weathers: its colours through the day, its clouds, and how thick its air is.
///
/// **Ten of them and every figure is the game's**, which is most of the point: the sky's own
/// constants in this crate are tuned by eye and say so, while `Sky Night Color=009,010,011` and
/// `Ambient Night Color=032,035,042` are Bethesda's answers to the same questions.
#[derive(Debug, Clone)]
pub struct Weather {
    /// What the ini calls it — `Clear`, `Cloudy`, `Blight` and so on, lower-cased.
    pub name: String,
    /// What the air scatters, which is the weather's own medium said as a colour.
    ///
    /// The fog on the ground and the veil over the sky are both this: one medium, one colour, two
    /// places it is seen. [`crate::Sky`] carries it into `fog` at the dome's own level and into the
    /// veil solved against [`Self::sky`].
    pub fog: Schedule,
    /// What the whole sky averages to, air and cloud deck together.
    ///
    /// **Read against [`Self::fog`] rather than on its own**, which is the only way it means
    /// anything: one number here is being asked to be two things — the blue air under clear, the
    /// deck under overcast — and nothing in the file splits them. What *does* split them is that
    /// six of the ten write this and their fog colour as the same number, which is the file
    /// saying the medium reaches all the way up. [`crate::Sky`] solves how much of the sky the
    /// medium has taken over out of the two, and where the answer does not explain this the
    /// renderer keeps its own Rayleigh sky — see `Veil` there.
    ///
    /// **Two earlier attempts are recorded rather than repeated.** The plan was to let each
    /// weather's sky move the dome as a *departure* from clear's. On the dome that double-counts —
    /// foggy's sky is 2.59 times clear's in the ini **because** of its deck, so dimming by that
    /// deck as well made overcast brighter than clear. On the deck instead it turns overcast
    /// orange, since the departure is (2.31, 1.13, 0.46) and dividing a grey by clear's blue is a
    /// warm ratio however it is normalised. `docs/design.md` §8.60 has both in full.
    pub sky: Schedule,
    /// What a surface receives, which this renderer derives and uses this to check.
    ///
    /// **Parsed and read by nothing, deliberately.** The ground's light here is the dome's own
    /// average dimmed by however much of it the deck hides, with no authored figure anywhere;
    /// `tests/weather_lighting.rs` is what holds that derivation against these four keys, and the
    /// places the two part are more informative than closing the gap would be.
    pub ambient: Schedule,
    /// What the beam is, which is still the physical model's.
    ///
    /// **Parsed and read by nothing.** The sun's colour falls out of the same Kasten-Young air mass
    /// and Rayleigh depth the dome does, so what this would add is the extinction the *weather's*
    /// medium puts on the beam — real, and a separate slice from the sky's own colour.
    pub sun: Schedule,
    /// Which painted sheet the cloud layer is cut out of — `tx_sky_clear` and its eight siblings.
    pub cloud_texture: String,
    /// How much sky the layer may cover, out of `Clouds Maximum Percent`.
    pub cloud_cover: f32,
    /// How fast the wind carries it, out of `Cloud Speed`.
    pub cloud_speed: f32,
    /// How hard the wind blows, out of `Wind Speed`, from nothing to nine tenths.
    ///
    /// **One number and it decides three things about the fog**, because all three are the same
    /// physics: wind carries the air past, wind lifts what is in it off the ground, and wind mixes
    /// it until the banks are gone. The ten span the range the way an eye would guess — foggy and
    /// snow at **0**, clear at .1, rain at .3, thunderstorm at .5, and ashstorm, blight and
    /// blizzard at .8 and .9. A radiation fog forms in still air and sits in it; a blight storm
    /// streams past.
    ///
    /// Distinct from `Cloud Speed`, which the game keeps separate and which scrolls the painted
    /// sheet rather than describing the air.
    pub wind: f32,
    /// How thick the air is by day and by night, out of `Land Fog Day/Night Depth`.
    pub fog_day_depth: f32,
    pub fog_night_depth: f32,
}

impl Weather {
    /// Every weather the installed game describes, named in the order the ini lists them.
    ///
    /// Empty without game data, which is a state the engine runs in: a caller with no weather falls
    /// back to what it did before there were any.
    pub fn table() -> crate::Result<Vec<Self>> {
        let Some(game) = GameData::shared()? else {
            return Ok(Vec::new());
        };
        Ok(Self::from_ini(game.ini()))
    }

    /// The same, from an ini a caller already holds — which is only ever a test's.
    pub(crate) fn from_ini(ini: &Ini) -> Vec<Self> {
        ini.sections_under("weather ")
            .into_iter()
            .map(|name| Self::read(ini, name))
            .collect()
    }

    /// How thick the air is at `time`, between the day's figure and the night's.
    ///
    /// **The same crossing the fog's *colour* takes**, so the two move together — air that thinned
    /// on one schedule and paled on another would read as two effects rather than one. The ini gives
    /// a depth only for day and for night, so the two twilights take the night's.
    pub fn fog_depth(&self, time: WorldTime) -> f32 {
        let depth = |key: Key| match key {
            Key::Day => self.fog_day_depth,
            _ => self.fog_night_depth,
        };
        let between = self.fog.crossover.between(time);
        depth(between.from) + (depth(between.to) - depth(between.from)) * between.along
    }

    /// The ones that can occur where `cell` is, out of the ten, in the game's own order.
    ///
    /// **A region is the only thing in the game that limits weather to a place**, and it does it by
    /// giving each of the ten a percentage: Vvardenfell's ash wastes are the only place a blight
    /// storm blows, and nowhere on the mainland snows. A zero is not "rare" — it is the file saying
    /// that weather does not happen here.
    ///
    /// **The whole ten wherever nothing narrows them.** An interior names no region, a handful of
    /// exteriors name none either, a region the file does not describe cannot rule anything out,
    /// and a region whose every chance is zero is bad data rather than a place with no weather. In
    /// each of those the answer is everything, because a caller offering a choice between none is
    /// worse than one offering a choice the game would not have made.
    ///
    /// Ordered by [`RegionRecord::ORDER`] rather than by name, which is what makes cycling through
    /// them read as a forecast rather than as an alphabet — clear, cloudy, foggy, overcast and on
    /// into the storms. [`Self::table`] is sorted, because ini sections are.
    pub fn in_cell(cell: &CellId) -> crate::Result<Vec<Self>> {
        let mut table = Self::table()?;
        // **Ordered before anything is dropped**, so the list reads the same however much a region
        // narrows it — and so there is one sort rather than one per way out of this function.
        table.sort_by_key(|weather| {
            RegionRecord::ORDER
                .iter()
                .position(|named| *named == weather.name)
                .unwrap_or(RegionRecord::WEATHERS)
        });
        let Some(game) = GameData::shared()? else {
            return Ok(table);
        };
        // A cell this file does not hold narrows nothing. One it does hold is *read* rather than
        // guessed at: a record the index found and the parser will not take is a broken file, and
        // reporting that beats reporting a coast with no weather.
        let Some(offsets) = game.cells().cell(cell) else {
            return Ok(table);
        };
        let region = Cell::parse(&game.reader().record_at(offsets.cell)?)?
            .region
            .and_then(|named| game.cells().region(&named));
        // Borrowed for the rest of the function rather than cloned, which the shared game allows:
        // it is `&'static`, so anything reached through it outlives every caller.
        if let Some(region) = region
            && table.iter().any(|weather| region.allows(&weather.name))
        {
            table.retain(|weather| region.allows(&weather.name));
        }
        Ok(table)
    }

    /// The weather the installed game calls `name`, or clear where it has none by that name.
    ///
    /// **The lookup rather than the table**, which is what a caller naming one on a command line
    /// wants. An unknown name is clear rather than an error: the ten are the game's own list and a
    /// caller cannot be expected to have it, so the fallback names itself in what it returns.
    pub fn named(name: &str) -> crate::Result<Self> {
        let wanted = name.trim().to_ascii_lowercase();
        Ok(Self::table()?
            .into_iter()
            .find(|weather| weather.name == wanted)
            .unwrap_or_else(Self::clear))
    }

    /// Clear weather as far as anything can know it without the ini.
    ///
    /// **Every figure here is `[Weather Clear]`'s own** — `Cloud Speed=1.25`,
    /// `Land Fog Depth=0.69`, full cloud cover, the clear sheet, and the four colours of each of
    /// the four families — because those are what the reader falls back to when the ini names
    /// nothing. The colour schedules used to come out white, which was nothing the game would ever
    /// draw and safe only while nothing read them; `CLEAR_SKY` above is what that cost once something
    /// did.
    pub fn clear() -> Self {
        Self::read(&Ini::default(), "clear")
    }

    fn read(ini: &Ini, name: &str) -> Self {
        let section = format!("weather {name}");
        let number = |key: &str, fallback: f32| ini.number(&section, key).unwrap_or(fallback);
        Self {
            name: name.to_owned(),
            sky: Schedule::read(ini, &section, "Sky", CLEAR_SKY),
            fog: Schedule::read(ini, &section, "Fog", CLEAR_FOG),
            ambient: Schedule::read(ini, &section, "Ambient", CLEAR_AMBIENT),
            sun: Schedule::read(ini, &section, "Sun", CLEAR_SUN),
            cloud_texture: ini
                .get(&section, "Cloud Texture")
                .unwrap_or("Tx_Sky_Clear.tga")
                .to_ascii_lowercase()
                .replace(".tga", ".dds"),
            cloud_cover: number("Clouds Maximum Percent", 1.0),
            cloud_speed: number("Cloud Speed", 1.25),
            wind: number("Wind Speed", 0.1),
            fog_day_depth: number("Land Fog Day Depth", 0.69),
            fog_night_depth: number("Land Fog Night Depth", 0.69),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ten as the installed game describes them, or nothing where there is no game.
    fn table() -> Vec<Weather> {
        Weather::table().expect("the ini should read where the game is configured")
    }

    #[test]
    fn a_region_is_what_says_the_bitter_coast_never_sees_an_ash_storm() {
        let coast = Weather::in_cell(&CellId::Exterior { x: -2, y: -9 })
            .expect("the region should read where the game is configured");
        if coast.is_empty() {
            return;
        }
        let named = |list: &[Weather]| -> Vec<String> {
            list.iter().map(|weather| weather.name.clone()).collect()
        };
        println!("Seyda Neen's shore has {:?}", named(&coast));

        // Seyda Neen stands in the Bitter Coast, which is swamp: it fogs and it rains, and the ash
        // and the blight belong to the wastes on the other side of the island. Nowhere on
        // Vvardenfell snows — that arrived with Bloodmoon's Solstheim.
        for wet in ["clear", "cloudy", "foggy", "rain", "thunderstorm"] {
            assert!(named(&coast).iter().any(|had| had == wet), "no {wet}");
        }
        for dry in ["ashstorm", "blight", "snow", "blizzard"] {
            assert!(
                !named(&coast).iter().any(|had| had == dry),
                "the Bitter Coast should never see {dry}"
            );
        }

        // **In the game's own order rather than the ini's.** Sections come out sorted, which would
        // open the cycle on ashstorm and put clear fourth; `RegionRecord::ORDER` is what a forecast
        // reads in.
        assert_eq!(coast[0].name, "clear");
        assert!(
            coast
                .iter()
                .position(|w| w.name == "foggy")
                .zip(coast.iter().position(|w| w.name == "rain"))
                .is_some_and(|(fog, rain)| fog < rain)
        );

        // **An interior names no region, so nothing narrows it** — all ten, which is the right
        // answer for a place the question does not apply to.
        let inside = Weather::in_cell(&CellId::Interior("Balmora, Guild of Mages".into()))
            .expect("an interior should read too");
        assert_eq!(inside.len(), 10, "{:?}", named(&inside));
    }

    #[test]
    fn every_weather_the_game_ships_is_read_with_its_own_numbers() {
        let table = table();
        if table.is_empty() {
            return;
        }
        assert_eq!(table.len(), 10, "the game ships ten");

        let of = |name: &str| {
            table
                .iter()
                .find(|w| w.name == name)
                .unwrap_or_else(|| panic!("no {name}"))
        };
        // Straight out of `[Weather Clear]` and `[Weather Overcast]`, which is the point of reading
        // the file rather than writing these down: a clear day sky is blue and an overcast one is
        // grey, and nothing here had to be told so.
        let clear = of("clear");
        assert_eq!(clear.cloud_texture, "tx_sky_clear.dds");
        assert!((clear.cloud_speed - 1.25).abs() < 1e-6);
        assert!((clear.fog_day_depth - 0.69).abs() < 1e-6);
        assert!(clear.sky.day.z > clear.sky.day.x, "{:?}", clear.sky.day);

        let overcast = of("overcast");
        assert_eq!(overcast.cloud_texture, "tx_sky_overcast.dds");
        let grey = overcast.sky.day;
        assert!(
            (grey.z - grey.x).abs() < 0.1 * grey.x,
            "an overcast sky is grey: {grey:?}"
        );
        // And it is darker than a clear one, which is what overcast means.
        assert!(grey.length() < clear.sky.day.length());

        // Blight is the one weather whose fog is red, and it is red in the file rather than here.
        let blight = of("blight");
        assert!(
            blight.fog.day.x > 2.0 * blight.fog.day.z,
            "{:?}",
            blight.fog.day
        );

        // **Every one names a sheet that is really in the archives**, which is the only thing tying
        // these strings to anything. Two of the ten are Bloodmoon's and carry its own prefix —
        // `tx_bm_sky_snow` and `tx_bm_sky_blizzard` — which is why this asks the file system rather
        // than assuming a shape.
        let vfs = rtxmw_vfs::morrowind_archives().expect("the game is available");
        for weather in &table {
            let path = format!("textures\\{}", weather.cloud_texture);
            assert!(vfs.contains(&path), "{}: no {path}", weather.name);
        }
    }

    #[test]
    fn a_colour_holds_by_day_and_crosses_over_on_its_own_family_s_schedule() {
        // A synthetic weather, so the numbers under test are the ones written here.
        let ini = Ini::parse(
            "[Weather]\n\
             Sky Pre-Sunset Time=1.5\n\
             Sky Post-Sunset Time=.5\n\
             Sky Pre-Sunrise Time=.5\n\
             Sky Post-Sunrise Time=1\n\
             Ambient Pre-Sunset Time=1\n\
             Ambient Post-Sunset Time=1.25\n\
             Ambient Pre-Sunrise Time=.5\n\
             Ambient Post-Sunrise Time=2\n\
             [Weather Test]\n\
             Sky Sunrise Color=255,0,0\n\
             Sky Day Color=0,255,0\n\
             Sky Sunset Color=0,0,255\n\
             Sky Night Color=0,0,0\n\
             Ambient Sunrise Color=255,0,0\n\
             Ambient Day Color=0,255,0\n\
             Ambient Sunset Color=0,0,255\n\
             Ambient Night Color=0,0,0\n",
        );
        let test = &Weather::from_ini(&ini)[0];
        let sky = |h: f32| test.sky.at(WorldTime::hours(h));
        let day = sky(12.0);

        // Flat through the middle of the day, right up to where the window opens.
        assert_eq!(sky(10.0), day);
        assert_eq!(sky(16.4), day, "the sky's window opens at 16:30");
        // Then day toward sunset across an hour and a half, half way at 17:15.
        let half = sky(17.25);
        assert!(
            (half - day.lerp(test.sky.sunset, 0.5)).length() < 1e-5,
            "{half:?}"
        );
        // Sunset exactly at eighteen, and out to night half an hour later.
        assert!((sky(18.0) - test.sky.sunset).length() < 1e-5);
        assert_eq!(sky(18.5), test.sky.night);
        assert_eq!(sky(23.0), test.sky.night);

        // **And the ambient is somewhere else at the same moment**, which is the whole reason each
        // family carries its own windows. At 17:15 the sky is half way to its sunset colour, having
        // set off at 16:30 with an hour and a half to go; the ambient set off at 17:00 with one
        // hour, so it is only a quarter of the way.
        let ambient = test.ambient.at(WorldTime::hours(17.25));
        let quarter = test.ambient.day.lerp(test.ambient.sunset, 0.25);
        assert!((ambient - quarter).length() < 1e-5, "{ambient:?}");
        // And at half past six the sky has finished while the ambient has not.
        assert_eq!(test.sky.at(WorldTime::hours(18.5)), test.sky.night);
        assert!(test.ambient.at(WorldTime::hours(18.5)) != test.ambient.night);
    }

    #[test]
    fn the_air_thickens_at_night_on_the_fog_s_own_schedule() {
        let table = table();
        if table.is_empty() {
            return;
        }
        // Foggy is the weather that differs between day and night — 1.0 against 1.9 — and every
        // other ships the same figure for both.
        let foggy = table.iter().find(|w| w.name == "foggy").unwrap();
        assert!((foggy.fog_depth(WorldTime::hours(12.0)) - 1.0).abs() < 1e-5);
        assert!((foggy.fog_depth(WorldTime::hours(0.0)) - 1.9).abs() < 1e-5);
        // And it crosses between them rather than stepping.
        let dusk = foggy.fog_depth(WorldTime::hours(17.5));
        assert!(dusk > 1.0 && dusk < 1.9, "{dusk}");
    }
}
