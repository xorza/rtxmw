//! Masser and Secunda: the two things in Morrowind's night sky that are worth looking at.

use glam::Vec3;
use rtxmw_texture::Texture;

use crate::error::Result;
use crate::game_data::GameData;
use crate::sky::Sky;
use crate::srgb::LUMA;
use crate::sun::Sun;
use crate::world_time::WorldTime;

/// The radius of Morrowind's sky, in the game's own units.
///
/// Read off `meshes/sky_night_01.nif`, whose star dome's vertices sit exactly 2000 from the origin.
/// It is the distance every `[Moons]` size in `Morrowind.ini` is a size *at*, which is the only way
/// those numbers become an angle.
const SKY_RADIUS: f32 = 2000.0;

/// How many phases the game draws, which is how many its daily increment counts in.
///
/// Eight, and they are shipped as eight textures apiece: `tx_masser_new` through `tx_masser_one_wan`
/// and the same for Secunda. This renderer draws none of them — the terminator is carved from where
/// the sun actually is — but the count is what `Daily Increment` is denominated in, so the cycle
/// length still comes from it.
const PHASES: f32 = 8.0;

/// What a full moon's lit face radiates, and what a full Masser delivers to a surface facing it.
///
/// **Neither is physical, and the gap between them says why.** A real full moon is some 2,500 cd/m²
/// against the sun's 1.6 billion, and delivers a quarter of a lux against a hundred thousand — the
/// two are a ratio of about four hundred thousand to one, and at that ratio a moon in this renderer
/// would be black. `DAYLIGHT` is not a physical figure either, so there is no scale to hang the real
/// one on. Real units are the fix, and they are `docs/design.md` §5.1's.
///
/// **The radiance is set by where the tone curve stops keeping colour**, which is the one thing here
/// that is not free to be anything. A moon bright enough to blow all three channels is a white disc
/// whatever colour it was given — that is what a photograph of the real moon at a night exposure
/// looks like, and it throws away the only reason to draw Masser rather than a bright dot. This
/// lands Masser's red channel at the top of the range with its blue a fifth of that, so the disc is
/// unmistakably red and unmistakably a light.
const FULL_RADIANCE: f32 = 0.18;

/// What a full Masser delivers to a surface facing it, on the same unphysical scale.
///
/// Masser's rather than any moon's, because the two do not deliver the same: what a moon is worth
/// as a light goes as the sky it covers, and that is derived from the pair of `Size`s rather than
/// written down twice — see [`Moon::at`].
const FULL_IRRADIANCE: f32 = 0.5;

/// What one moon is, which is everything about it that does not change with the hour.
///
/// A satellite of [`Moon`] rather than a type in its own right: nothing outside this file has a use
/// for one, and the two that exist are written down here. **Not `Orbit`**, which is what this was
/// called until [`WorldTime::orbit`] was noticed next to it — that is the sun's place in its arc, a
/// number rather than a body, and one name for the two would have been a false pair.
#[derive(Debug, Clone, Copy)]
struct Almanac {
    /// The moon's radius on the sky of [`SKY_RADIUS`], out of `Morrowind.ini`'s `[Moons]`.
    size: f32,
    /// How far the orbit's pole is swung around the zenith from due north, in radians.
    ///
    /// **`Axis Offset`, read as a bearing rather than a tilt, and the signs are mine.** The ini
    /// gives 35 for Masser and 50 for Secunda and does not say around which axis. Tilting the pole
    /// away from the celestial one is the other reading and it does not survive the arithmetic: a
    /// pole 35 degrees higher culminates 35 degrees lower, which leaves Masser crawling along the
    /// horizon at eighteen degrees and Secunda at three. Swinging the pole around the zenith instead
    /// leaves both moons as high as the sun gets and moves where they rise, which is the visible
    /// thing two moons want — and taking the two in opposite directions is what makes their arcs
    /// cross rather than nearly coincide.
    pole_bearing: f32,
    /// Phases the moon advances in a day, out of `Daily Increment`.
    daily_increment: f32,
    /// Which phase it is on at the start of the first day, in cycles.
    ///
    /// **Chosen, not derived.** Nothing in the game data says where a moon stands when a world
    /// begins — Morrowind reads it from a save. Zero would put both moons new on day zero, so the
    /// default hour and every screenshot taken at it would have an empty sky; these put Masser just
    /// past full and Secunda waxing, so the first night has two moons in it and they are not the
    /// same shape.
    epoch: f32,
    /// The mean of every texel its vanilla portrait's own alpha calls opaque, decoded to linear.
    ///
    /// **Measured off `tx_masser_full.dds` and `tx_secunda_full.dds`**, and decoded before the mean
    /// rather than after — averaging sRGB bytes and decoding the result is the same mistake as
    /// sampling an albedo through a UNORM view.
    ///
    /// It is two things at once, which is why it is kept raw rather than pre-divided. [`Self::tint`]
    /// scales it into the colour a lit face radiates; its luminance is what the shader divides the
    /// portrait *by*, so what multiplies the moon's colour there is each texel's ratio to this and
    /// the mottling comes out as a level rather than as a level plus an overall brightness.
    face: Vec3,
}

impl Almanac {
    /// The colour a fully lit face radiates, before any level is put on it.
    ///
    /// **Both moons over one divisor, rather than each normalised to itself**, which is a decision
    /// and not a detail: Bethesda drew Masser two and a half times darker than Secunda, and
    /// normalising each separately would throw that away and leave the red moon as bright as the
    /// white one. What is divided out is the pair's overall level, which belongs to
    /// `FULL_RADIANCE`; what survives is both their colour and their contrast, which belong to the
    /// art. Secunda is the one made unity because it is the brighter.
    fn tint(&self) -> Vec3 {
        self.face / SECUNDA.face.dot(LUMA)
    }
}

/// The larger moon, red and mottled, the one Morrowind's sky is remembered for.
const MASSER: Almanac = Almanac {
    size: 94.0,
    pole_bearing: 35.0 * (std::f32::consts::PI / 180.0),
    daily_increment: 1.0,
    epoch: 0.55,
    face: Vec3::new(0.03322, 0.00993, 0.01233),
};

/// The smaller one, pale and grey, on an arc that crosses Masser's.
const SECUNDA: Almanac = Almanac {
    size: 40.0,
    pole_bearing: -50.0 * (std::f32::consts::PI / 180.0),
    daily_increment: 1.2,
    epoch: 0.30,
    face: Vec3::new(0.04402, 0.03732, 0.02946),
};

/// The two moons' vanilla portraits, which are what a disc is drawn with.
///
/// **The `full` face of each and only that one.** The game ships eight per moon and switches
/// between them; this renderer carves the terminator from where the sun actually is, so the phases
/// it needs are the one face the sun has reached all of. Loading the other fourteen would be
/// loading fourteen pictures of a shadow this already knows how to cast.
#[derive(Debug)]
pub struct MoonFaces {
    /// `tx_masser_full.dds`, or `None` where it would not read or decode.
    pub masser: Option<Texture>,
    /// `tx_secunda_full.dds`, on the same terms.
    pub secunda: Option<Texture>,
}

impl MoonFaces {
    /// Reads both from the installed game, or `None` when none is configured.
    ///
    /// A face that fails to read or decode is `None` rather than an error, the same as any other
    /// texture: a moon without its portrait is a flat disc of the right colour, which is worse and
    /// not a reason to have no renderer.
    pub fn load() -> Result<Option<Self>> {
        let Some(game) = GameData::shared()? else {
            return Ok(None);
        };
        let read = |path: &str| {
            game.vfs()
                .read(path)
                .ok()
                .and_then(|bytes| Texture::decode(&bytes).ok())
        };
        Ok(Some(Self {
            masser: read(r"textures\tx_masser_full.dds"),
            secunda: read(r"textures\tx_secunda_full.dds"),
        }))
    }
}

/// One of Morrowind's moons at an hour of one of its days.
///
/// **A sphere lit by the same sun everything else is**, which is the whole of the phase: the
/// terminator is not a painted texture chosen from eight but the line where the sun stops reaching
/// round, and the shader carves it per pixel from the sun direction it already has. So a crescent
/// always points at the sun, a moon low in the west at dusk is a thin one, and the shape moves
/// through the night instead of stepping at midnight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Moon {
    /// Unit vector the moon's light **travels along** — from the moon toward the world, so the
    /// direction *to* it is the negation. The same sense as [`Sun::direction`], so the two can be
    /// handed to one shader function.
    pub direction: Vec3,
    /// Half the angle the disc subtends, in radians.
    pub angular_radius: f32,
    /// Radiance of the fully lit face, with the air it is seen through already taken out.
    ///
    /// **Not the phase**, which is per pixel and the shader's: this is what the lit part radiates,
    /// and the unlit part of the same disc radiates nothing. Zero once the moon has set, which is
    /// how a moon that is not up says so without a flag.
    pub colour: Vec3,
    /// Luminance of the portrait's own mean texel, which the shader divides the portrait by.
    ///
    /// **So the mottling is a level and not a level plus a brightness.** The faces were drawn to be
    /// shown rather than lit — Masser's mean texel is a linear 0.033 — so what is wanted from the
    /// picture is the ratio of each texel to its own mean, and [`Self::colour`] supplies the level
    /// that ratio multiplies. Dividing each texel by *its own* luminance instead leaves every part
    /// of the disc equally bright and turns the maria into a hue shift, which is not what a moon
    /// looks like.
    pub face_mean: f32,
    /// Irradiance the moon delivers to a surface facing it, phase and air included.
    ///
    /// Zero for a new moon and for one below the horizon, so no caller needs to ask whether there is
    /// any moonlight — a black light contributes nothing through every term it feeds.
    pub light: Vec3,
}

impl Moon {
    /// A moon that is not in this sky at all, which is what an interior has two of.
    ///
    /// Contributes nothing through every term it feeds — a black disc of zero width is drawn
    /// nowhere and lights nothing — so no caller anywhere asks whether a cell has moons.
    pub const NONE: Self = Self {
        direction: Vec3::NEG_Z,
        angular_radius: 0.0,
        colour: Vec3::ZERO,
        face_mean: 1.0,
        light: Vec3::ZERO,
    };

    /// Masser at `time`, lit by `sun`.
    pub fn masser(time: WorldTime, sun: Sun) -> Self {
        Self::at(MASSER, time, sun)
    }

    /// Secunda at `time`, lit by `sun`.
    pub fn secunda(time: WorldTime, sun: Sun) -> Self {
        Self::at(SECUNDA, time, sun)
    }

    /// What the disc radiates along `direction`, before any portrait is applied to it.
    ///
    /// **Written twice on purpose**, here and in `moon_disc` in `lighting.glsl`, and
    /// `tests/sky_dome.rs` renders a frame and checks every pixel of both moons against this. Two
    /// implementations of one equation is two chances to be wrong, and the one that was wrong the
    /// first time round was a sign on the bulge — which inverts the phase and shows a gibbous moon
    /// as a crescent, in a way nothing but a picture or this would catch.
    ///
    /// `sun_direction` is a parameter rather than a field because that is what the shader has: one
    /// sun direction in the frame constants, shared by the sky, the shadows and both moons. Carrying
    /// a second copy on each moon would be a second thing to keep in step for no gain.
    pub fn radiance(&self, direction: Vec3, sun_direction: Vec3) -> Vec3 {
        let along = direction.dot(-self.direction);
        if along <= self.angular_radius.cos() || self.colour == Vec3::ZERO {
            return Vec3::ZERO;
        }
        let offset = (direction + self.direction * along) / self.angular_radius.sin();
        let across = offset.length_squared();
        if across >= 1.0 {
            return Vec3::ZERO;
        }
        // The sphere's own normal, then Lommel-Seeliger's `mu0 / (mu0 + mu)` — doubled, because at
        // opposition the operator is a half and a full moon has to come out the colour it was given.
        let emission = (1.0 - across).sqrt();
        let normal = offset + self.direction * emission;
        let incidence = normal.dot(-sun_direction).max(0.0);
        self.colour * (2.0 * incidence / (incidence + emission).max(1e-4))
    }

    /// Where `moon` stands at `time`, and what it is worth there.
    fn at(moon: Almanac, time: WorldTime, sun: Sun) -> Self {
        // How far round its cycle the moon is: 0 new, 0.5 full. This is the only thing the date is
        // read for, and it is what makes the moon lag the sun by a little more each day.
        let phase = (moon.epoch + time.day() * moon.daily_increment / PHASES).rem_euclid(1.0);
        let position = Self::position(moon, time, phase);

        // The sine of the moon's elevation, which is what both the air it is seen through and the
        // fade across the horizon are functions of.
        let climb = position.z;
        // Set over its own diameter, the same as the sun does — a disc five degrees across takes
        // that long to go.
        let radius = (moon.size / SKY_RADIUS).atan();
        let showing = Sky::showing(climb, radius);
        // The same air the sun is reddened by, and for the same reason — a moon on the horizon is
        // seen through thirty-eight atmospheres and comes up orange.
        let air = Sky::transmittance(climb.max(0.0));

        // **The lit fraction of the visible disc**, from the angle at the moon between the sun and
        // us: 1 when the moon is opposite the sun and 0 when it is in front of it. Both bodies are
        // far enough away that the two directions are all it takes.
        let opposition = -sun.direction.dot(-position);

        // **How much sky the moon covers, against Masser's**, which is what turns a radiance into
        // the light a surface receives. Secunda is less than half Masser's width and so a fifth of
        // its area; scaling by it is what makes the big moon the one a night is lit by, and it is
        // derived from the two sizes rather than a second pair of numbers to keep in step.
        let coverage = (moon.size / MASSER.size).powi(2);
        let delivered = FULL_IRRADIANCE * coverage * showing * Self::phase_law(opposition);

        Self {
            direction: -position,
            angular_radius: radius,
            colour: moon.tint() * (FULL_RADIANCE * showing) * air,
            face_mean: moon.face.dot(LUMA),
            light: moon.tint() * delivered * air,
        }
    }

    /// Which way the moon lies from the world's centre, as a unit vector.
    ///
    /// **A great circle about a pole**, which is what every body in a sky travels on and what the
    /// sun's own arc is a near miss of. The pole's elevation is not a number of its own: Morrowind's
    /// noon sun stands 53.13 degrees up, so the pole it is turning about stands at the complement of
    /// that, 36.87 — and both fall straight out of the game's `(0, 75, -100)`, which is a 3-4-5
    /// triangle. Reading it off `Sun::at` rather than writing it down is what keeps a moon on the
    /// same sky as the sun if that constant ever moves.
    fn position(moon: Almanac, time: WorldTime, phase: f32) -> Vec3 {
        // The sun at its highest, which is the one direction that fixes the whole frame. Light
        // travels *from* it, so this points down and north — and the pole is that turned a quarter
        // circle up, which for a unit vector is a swap and a sign.
        let noon = Sun::at(0.0).direction;
        let north = Vec3::new(0.0, -noon.z, noon.y);
        // Swung around the zenith by the moon's own bearing, which is what separates the two arcs.
        let (sin, cos) = moon.pole_bearing.sin_cos();
        let pole = Vec3::new(north.y * sin, north.y * cos, north.z);

        // The arc itself: due east on the horizon at hour angle zero, culminating a quarter turn
        // later. Both are perpendicular to the pole by construction, so the two span its equator.
        let rise = Vec3::new(cos, -sin, 0.0);
        let culmination = rise.cross(pole);

        // **How far round the arc it has come.** The sun is at hour angle zero when it rises and a
        // quarter turn along at noon; the moon is the same, delayed by however far round its cycle
        // it is — a full moon is half a cycle on and so rises as the sun sets. That delay growing by
        // a fraction of a day each day is the whole of a moon's motion against the sun.
        let angle = std::f32::consts::TAU * (time.turns_since_sunrise() - phase);
        rise * angle.cos() + culmination * angle.sin()
    }

    /// How much light a moon at phase angle `opposition` sends, against a full one.
    ///
    /// **The measured law rather than the geometry**, and the difference is a factor of five. The
    /// lit fraction of the disc is `(1 + cos α) / 2`, which says a half moon is half as bright as a
    /// full one; the moon is not, and the reason is that its surface is rough enough to shadow
    /// itself everywhere but at opposition. Allen's fit to the observations —
    /// `Δm = 0.026|α| + 4·10⁻⁹ α⁴`, α in degrees — puts a half moon at 0.09 of full, which is what
    /// photometry actually finds, and the quartic term is the opposition surge that makes the last
    /// few degrees before full so much brighter than the rest.
    ///
    /// `opposition` is the cosine of that angle: 1 with the moon opposite the sun, −1 in front of it.
    fn phase_law(opposition: f32) -> f32 {
        let degrees = opposition.clamp(-1.0, 1.0).acos().to_degrees();
        let dim = 0.026 * degrees + 4.0e-9 * degrees.powi(4);
        (10.0f32).powf(-0.4 * dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// How high the moon stands, in degrees above the horizon.
    fn elevation(moon: Moon) -> f32 {
        (-moon.direction).z.asin().to_degrees()
    }

    /// How much of the disc the sun reaches, from the geometry alone: 1 full, 0 new.
    fn lit(moon: Moon, sun: Sun) -> f32 {
        (1.0 - sun.direction.dot(moon.direction)) * 0.5
    }

    /// Both moons at `hour` of the first day, Masser first.
    fn moons(hour: f32) -> [Moon; 2] {
        let time = WorldTime::hours(hour);
        let sun = Sun::at(time.orbit());
        [Moon::masser(time, sun), Moon::secunda(time, sun)]
    }

    #[test]
    fn the_discs_are_the_size_the_ini_gives_at_the_radius_the_sky_mesh_has() {
        // `Masser Size=94` on a sky of 2000: `atan(94/2000)` is 2.691 degrees of radius, so the
        // disc is 5.38 across — ten times the real moon, which is what Morrowind's sky looks like.
        let [masser, secunda] = moons(23.0);
        assert!(
            (masser.angular_radius.to_degrees() - 2.6909).abs() < 1e-3,
            "{}",
            masser.angular_radius.to_degrees()
        );
        // And `Secunda Size=40`: 1.1458 degrees, so the ratio of the two is the ratio of the sizes.
        assert!(
            (secunda.angular_radius.to_degrees() - 1.1458).abs() < 1e-3,
            "{}",
            secunda.angular_radius.to_degrees()
        );
        let ratio = masser.angular_radius.tan() / secunda.angular_radius.tan();
        assert!((ratio - 94.0 / 40.0).abs() < 1e-4, "{ratio}");

        // Both are far larger than the sun's half-degree disc, which is the whole reason a moon here
        // is a thing with a face rather than a bright dot.
        assert!(secunda.angular_radius > 4.0 * Sun::REAL_ANGULAR_RADIUS);
    }

    /// Every quarter hour of forty days, which is long enough for both cycles to close.
    ///
    /// Eight days for Masser and six and two thirds for Secunda, so forty is five laps of one and
    /// six of the other. Sampling at one fixed hour instead would only ever see eight phases of
    /// Masser, because its increment is exactly an eighth of a cycle a day.
    fn every_quarter_hour() -> impl Iterator<Item = (WorldTime, Sun)> {
        (0..40 * 24 * 4).map(|i| {
            let time = WorldTime::hours(i as f32 * 0.25);
            (time, Sun::at(time.orbit()))
        })
    }

    #[test]
    fn a_moon_opposite_the_sun_is_full_and_one_beside_it_is_new() {
        // **The phase is geometry, not a counter**, so this is the claim the whole file rests on:
        // the lit fraction has to follow where the moon stands relative to the sun, with no schedule
        // in it anywhere.
        for (name, pick) in [
            ("masser", Moon::masser as fn(WorldTime, Sun) -> Moon),
            ("secunda", Moon::secunda),
        ] {
            let (mut fullest, mut newest) = (0.0f32, 1.0f32);
            for (time, sun) in every_quarter_hour() {
                let moon = pick(time, sun);
                let fraction = lit(moon, sun);
                fullest = fullest.max(fraction);
                newest = newest.min(fraction);
                // The lit fraction *is* where the moon stands, which is what stops a phase and a
                // position ever drifting apart — there is only one of them.
                let opposition = -sun.direction.dot(moon.direction);
                assert!((fraction - (1.0 + opposition) * 0.5).abs() < 1e-6);
            }
            assert!(fullest > 0.94, "{name} never comes near full: {fullest}");
            assert!(newest < 0.06, "{name} never comes near new: {newest}");
        }
    }

    #[test]
    fn a_full_moon_is_up_at_night_and_a_new_one_is_down_with_the_sun() {
        // **The consequence of the phase being geometry**, and the thing that makes a night sky
        // read: the moon that is full is the moon opposite the sun, so it is up exactly when the sun
        // is not. Nothing schedules this and nothing could — it falls out of where the two stand.
        //
        // Taken only where the sun is well down, because a moon 90% lit still stands up to 37
        // degrees off the anti-solar point: the two arcs are inclined to each other, which is why
        // neither moon is ever quite full and why an eclipse is a rare thing rather than a monthly
        // one.
        let mut nights = 0;
        for (time, sun) in every_quarter_hour() {
            if -sun.direction.z > -0.7 {
                continue;
            }
            nights += 1;
            for moon in [Moon::masser(time, sun), Moon::secunda(time, sun)] {
                let elevation = elevation(moon);
                if lit(moon, sun) > 0.9 {
                    assert!(elevation > 0.0, "a full moon below the horizon at {time}");
                }
                if lit(moon, sun) < 0.1 {
                    assert!(
                        elevation < 0.0,
                        "a new moon up in the small hours at {time}"
                    );
                }
            }
        }
        assert!(nights > 500, "only {nights} readings were deep night");
    }

    #[test]
    fn the_two_moons_keep_different_skies() {
        // The point of giving them different bearings: at any one hour they are somewhere else, and
        // over a night they do not travel together.
        let mut apart = 0.0f32;
        for hour in [19.0, 21.0, 23.0, 1.0, 3.0] {
            let [masser, secunda] = moons(hour);
            let angle = masser
                .direction
                .dot(secunda.direction)
                .clamp(-1.0, 1.0)
                .acos();
            apart = apart.max(angle.to_degrees());
        }
        assert!(
            apart > 45.0,
            "the two arcs nearly coincide: {apart} degrees"
        );

        // And they are up at different times, because their cycles are different lengths — 8 days
        // for Masser against 6.67 for Secunda, out of `Daily Increment`.
        let cycle = |increment: f32| PHASES / increment;
        assert_eq!(cycle(MASSER.daily_increment), 8.0);
        assert!((cycle(SECUNDA.daily_increment) - 6.6667).abs() < 1e-3);
    }

    #[test]
    fn a_moon_under_the_horizon_neither_shines_nor_is_drawn() {
        // Both are one expression — a moon that has set contributes nothing through any term — so
        // there is no flag anywhere asking whether a moon is up.
        let mut set = 0;
        for hour in 0..24 {
            let [masser, secunda] = moons(hour as f32);
            for moon in [masser, secunda] {
                let elevation = elevation(moon);
                if elevation < -moon.angular_radius.to_degrees() {
                    set += 1;
                    assert_eq!(moon.colour, Vec3::ZERO, "hour {hour} draws a set moon");
                    assert_eq!(moon.light, Vec3::ZERO, "hour {hour} lights by a set moon");
                }
            }
        }
        assert!(set > 8, "only {set} of 48 readings had a moon down");

        // **And it goes out over its own width rather than at a line**, which is what stops a moon
        // blinking off against a sea horizon. Masser's disc is 5.4 degrees across, so there is a
        // band that wide where it is neither fully up nor fully gone.
        let partly = (0..2000)
            .map(|i| {
                let hour = 24.0 * i as f32 / 2000.0;
                Moon::masser(
                    WorldTime::hours(hour),
                    Sun::at(WorldTime::hours(hour).orbit()),
                )
            })
            .filter(|moon| moon.colour != Vec3::ZERO && moon.colour.length() < 0.9 * FULL_RADIANCE)
            .count();
        assert!(partly > 20, "the moon sets in a step: {partly} readings");
    }

    #[test]
    fn the_phase_law_is_the_measured_one_rather_than_the_lit_fraction() {
        // Full is full, by construction — the law is normalised at opposition.
        assert!((Moon::phase_law(1.0) - 1.0).abs() < 1e-6);

        // **A half moon is a ninth of a full one, not a half.** That is the whole reason this is
        // Allen's fit and not `(1 + cos a) / 2`: the moon's surface shadows itself away from
        // opposition, and photometry finds 0.09 where the geometry would say 0.5.
        let half = Moon::phase_law(0.0);
        assert!((half - 0.091).abs() < 0.005, "{half}");
        assert!(half < 0.2 * 0.5, "the geometric fraction would be 0.5");

        // A crescent is nearly nothing, and a new moon exactly so on this scale.
        assert!(Moon::phase_law((150.0f32).to_radians().cos()) < 0.01);
        // Monotonic from full to new, which a law with a quartic term in it has to be checked for.
        let mut previous = f32::INFINITY;
        for degrees in 0..=180 {
            let value = Moon::phase_law((degrees as f32).to_radians().cos());
            assert!(value < previous, "not monotonic at {degrees} degrees");
            previous = value;
        }
    }

    #[test]
    fn a_low_moon_is_reddened_by_the_air_it_is_seen_through() {
        // The same atmosphere that turns the sun orange at dusk, applied to the same kind of disc —
        // and Masser is red to begin with, which is the image Morrowind is remembered for.
        //
        // Sampled across a night for the highest and lowest Masser that is still fully up.
        let (mut high, mut low) = (None, None);
        for i in 0..2000 {
            let time = WorldTime::hours(24.0 * i as f32 / 2000.0);
            let moon = Moon::masser(time, Sun::at(time.orbit()));
            let elevation = (-moon.direction).z.asin().to_degrees();
            if moon.colour == Vec3::ZERO {
                continue;
            }
            if high.is_none_or(|(e, _)| elevation > e) {
                high = Some((elevation, moon.colour));
            }
            if elevation > 2.0 && low.is_none_or(|(e, _)| elevation < e) {
                low = Some((elevation, moon.colour));
            }
        }
        let (_, high) = high.unwrap();
        let (_, low) = low.unwrap();
        // Redder low down: blue is scattered out five times faster than red, so what is left of a
        // beam that crossed thirty atmospheres has lost its blue and kept its red.
        assert!(
            low.z / low.x < 0.6 * (high.z / high.x),
            "{low:?} vs {high:?}"
        );
        // And dimmer, because the air took light out rather than moving it about.
        assert!(low.length() < high.length());
        // The tint is Bethesda's, so a fully lit Masser overhead is still the red of its own face.
        assert!(high.x > 2.5 * high.y, "{high:?}");
    }
}
