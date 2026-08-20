//! What lights an exterior, at an hour.

use glam::Vec3;

use crate::clouds::{CloudSheet, Clouds, SKYLIT};
use crate::moon::Moon;
use crate::srgb::LUMA;
use crate::sun::Sun;
use crate::weather::Weather;
use crate::world_time::WorldTime;

/// Optical depth of the whole atmosphere looking straight up, per channel.
///
/// Rayleigh scattering goes as the inverse fourth power of wavelength, so blue is taken out of a
/// beam five times faster than red. These are the coefficients everyone uses — 5.8, 13.5 and
/// 33.1 per micrometre-to-the-fourth, times an eight-kilometre scale height — and they are the whole
/// reason a low sun is orange: at noon a beam crosses one atmosphere and loses a quarter of its
/// blue, and at sunset it crosses thirty and loses all of it.
const ZENITH_DEPTH: Vec3 = Vec3::new(0.0464, 0.1080, 0.2648);

/// The sky's radiance against what the arithmetic below produces.
///
/// Pure scale, set so the **default hour** comes out exactly where the model before it did — a
/// luminance of 0.518, which it is re-fitted to whenever anything upstream moves the sun; §8.58's
/// change to the elevation curve is the second time — which keeps every image made against that one comparable. It is not the
/// fixed overcast blue that preceded all of this: that was 0.414, and the first version of the sky
/// matched it at *noon* while landing a quarter above it at half past nine. A test pins the hour
/// rather than the claim now.
///
/// `ZENITH_DEPTH` is the only constant in this file that came from anywhere but an eye; this, the
/// greying, the twilight width and the night floor are all tuned, and say so.
const SKY_STRENGTH: f32 = 1.7000;

/// What one unit of the ini's `Land Fog Depth` is worth in this renderer's own fog.
///
/// Clear weather's 0.69 comes to 0.30, which is the density that was settled by eye before there was
/// a weather system — half the light from a surface surviving a hundred and thirty metres, where the
/// 0.75 before it lost the far shore of Seyda Neen's bay at fifty-three. So the absolute is still
/// this renderer's and the *ratios* between the ten weathers, and between a weather's day and its
/// night, are the game's.
const FOG_SCALE: f32 = 0.30 / 0.69;

/// What the sky radiates once the sun is properly down.
///
/// **Not moonlight**, which is [`Moon`]'s and arrives as a direction rather than as a floor. This is
/// the airglow and the starlight that are there on a moonless night too — what keeps a night
/// exterior legible rather than black, blue because scotopic vision is.
///
/// **Quartering this to darken the night sky was tried and put back.** The sky and the ground are the
/// same number here by construction — §8.49 tied the ambient to the dome's own average, which is
/// right — so lowering it takes the ground down with the sky, and the ground has an albedo on it
/// besides. It lands in the range the tone curve crushes: Khronos PBR Neutral subtracts an offset
/// that for a linear 0.01 is 0.0094, sixteen times the value, and Seyda Neen went black while the
/// sky stayed legible.
///
/// **The second light source that can separate the two now exists**, which is what the moons were
/// built for: moonlight falls on surfaces and not on the sky, so a lower floor no longer has to take
/// the ground with it. It is left where it is all the same, because the hours a moon is down are
/// still lit by this alone and `NIGHT_STOPS` was settled by eye against this number.
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

/// How far up the dome the paleness reaches, in air masses.
///
/// **A thick sky is a white one.** Looking at the horizon is looking through thirty-eight
/// atmospheres, and light that has scattered that many times has lost which wavelength it started
/// as — so the horizon is pale whatever the hour and the zenith keeps its colour. Six air masses to
/// the e-fold puts the change where the eye sees it, a couple of hand-spans above the horizon.
const PALE_MASS: f32 = 6.0;

/// What the sky away from a low sun keeps of what the sky toward it has.
///
/// At dusk the light is all on one side: the air opposite the sun is in the Earth's own shadow and
/// lit only from above. This is that, as a fraction — it does nothing at noon, where there is no
/// side for the sun to be on.
const AWAY_SHARE: f32 = 0.25;

/// The sun and the sky above an exterior cell at one moment, which together are all its light.
///
/// **The dome has a direction now, and that is most of this type.** A sky is not one colour: it is
/// deep blue overhead and pale at the horizon, and once the sun is low it is orange on one side and
/// dim blue on the other. [`Self::shape`] is the shape, [`Self::ambient`] is its average, and
/// `lighting.glsl` draws the same shape per pixel — the test in `tests/sky_dome.rs` is what keeps
/// the two from drifting apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sky {
    /// Black once the sun is down, which is how a cell with no sun says so.
    pub sun: Sun,
    /// The larger moon, red and low-contrast, the one this sky is remembered for.
    ///
    /// Lit by [`Self::sun`] wherever it stands, so its phase is never told what to be — see
    /// [`Moon`]. [`Moon::NONE`] for a cell with no sky.
    pub masser: Moon,
    /// The smaller one, pale and grey, on an arc that crosses Masser's. Otherwise as above.
    pub secunda: Moon,
    /// The cloud layer, lit by the same sun and the same dome. [`Clouds::NONE`] for a cell with no
    /// sky, which draws none without a branch to say so.
    pub clouds: Clouds,
    /// The dome's average radiance, which is what an unshadowed surface receives from all of it.
    ///
    /// Derived from [`Self::shape`] rather than beside it, so the light a surface is given and
    /// the sky behind it are the same sky.
    pub ambient: Vec3,
    /// The colour a beam has left after the air the sun's own light crossed, normalised.
    ///
    /// The warm end of every tint on the dome. White with the sun overhead, deep red on the horizon.
    pub warm: Vec3,
    /// How far the sunward sky has gone toward [`Self::warm`]: nothing at noon, nearly all at dusk.
    pub warmth: f32,
    /// What the dome's shape is multiplied by to give radiance.
    pub scale: f32,
    /// How much of the star field is out, from none to all of it — [`WorldTime::starlight`].
    pub stars: f32,
    /// What the exterior's fog scatters, and how thickly it sits.
    ///
    /// **The weather's hue on the dome's own level.** `Land Fog Day Depth` and the `Fog *Color*`
    /// schedule are the game's, and both vary by weather and by hour — blight's fog is red where
    /// clear's is a pale blue, and foggy's air nearly doubles in thickness overnight. What is *not*
    /// the game's is the scale: its depths are in the original engine's units and this renderer's
    /// fog is tuned in its own, so `FOG_SCALE` ties clear weather to the figure that was settled by
    /// eye and every other weather follows the ini's ratio to it.
    pub fog: Vec3,
    pub fog_density: f32,
    /// What the hour multiplies the metered exposure by — a bias on it, never the exposure itself.
    ///
    /// **Keyed to the sky rather than to the frame.** Metering says how bright the picture is; this
    /// says how bright the world is, and only the second knows a cave at noon from a field at
    /// midnight. Noon is one and the deepest night is `NIGHT_STOPS` below it.
    pub exposure_bias: f32,
}

/// The dome's luminance at noon, which is the bright end the curve below is fitted against.
///
/// Measured from `Sky::at` rather than derived — the dome's average has no closed form and asking
/// for it here would need a `Sky` that is still being built. A test holds it to the real thing, which
/// is what stops `SKY_STRENGTH` moving and this quietly fitting the wrong range.
const DAY_LUMINANCE: f32 = 0.755;

/// How many stops darker than noon the darkest night is allowed to render.
///
/// **Metering the frame alone gives a renderer with no night in it**, which is the oldest complaint
/// in this corner of the subject. Krawczyk, Myszkowski and Seidel fitted a key value to the scene's
/// own luminance for exactly this reason, and Narkowicz states the principle plainly: *"we want to
/// have a darker image in low light conditions and a brighter image in high light conditions. This
/// way viewer has a clue as to how bright the lighting is in the current scene."* Their curve runs
/// about four and a half stops from sunlight to starlight.
///
/// **Two rather than the four and a half the curve was measured at**, because this renderer has no
/// absolute luminance
/// scale to hang the published curve on — `DAYLIGHT` is admittedly not a physical figure, and the
/// night floor is a number picked so an exterior stays legible rather than a measured one. The
/// distance between noon and midnight here is fifty to one where the world's is a hundred million to
/// one, so the curve is fitted to the range that exists rather than the one that should. Giving the
/// renderer real units is the fix that would let the published curve be used as published, and it is
/// the same fix `docs/design.md` §5.1 has been waiting for.
///
/// **Even this much needed the tone curve to stop squaring its shadows first.** Before that fix two
/// stops already sent the ground black while the sky still read, and it was the reference operator's
/// offset doing it rather than the bias — see `SHADOW_OFFSET` in `tonemap.comp`. Two and a half was
/// tried afterwards and judged too dark by eye, which is the right way to settle a number whose whole
/// job is how a frame looks.
const NIGHT_STOPS: f32 = 2.0;

impl Sky {
    /// What every direction of the dome gets whatever the hour.
    ///
    /// Drawn as well as shaded with, so the sky a surface is lit by is the sky behind it at midnight
    /// too. An associated constant rather than a field, because it is the same for every sky there
    /// is: the moons vary a night, but they do it as two discs and two directional lights rather
    /// than by lifting the floor, so this stayed a constant when they arrived.
    pub const NIGHT_FLOOR: Vec3 = NIGHT_SKY;

    /// The light above an exterior at `time`.
    pub fn at(time: WorldTime) -> Self {
        Self::under(time, &Weather::clear(), CloudSheet::NONE)
    }

    /// The light above an exterior at `time`, under `weather`.
    ///
    /// [`Self::at`] is this under clear weather, which is what every test wants and what the engine
    /// runs in until something chooses otherwise.
    pub fn under(time: WorldTime, weather: &Weather, sheet: CloudSheet) -> Self {
        let bare = Sun::at(time.orbit());
        // The sun's direction travels downward, so its climb above the horizon is the negation —
        // and it is signed, because after sunset there is a good deal of sky left to light and how
        // far under the sun has gone is the only thing that says how much.
        let climb = -bare.direction.z;
        let transmitted = Self::transmittance(climb.max(0.0));

        // **The warm end of the dome**: what one beam has left after the air the sun's own light
        // crossed. White overhead, deep red on the horizon, and the only thing on the dome that
        // knows what hour it is.
        let survived = transmitted.x + transmitted.y + transmitted.z;
        let warmth = 1.0 - survived / 3.0;
        let warm = transmitted / survived.max(1e-6);

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
        let showing = Self::showing(climb, Sun::REAL_ANGULAR_RADIUS);
        let mut sky = Self {
            sun: Sun {
                colour: bare.colour * transmitted * showing,
                ..bare
            },
            // Lit by the bare sun rather than the attenuated one: what reaches a moon is the beam
            // before this planet's air, which is the only air in the arithmetic.
            masser: Moon::masser(time, bare),
            secunda: Moon::secunda(time, bare),
            ambient: Vec3::ZERO,
            clouds: Clouds::NONE,
            warm,
            warmth,
            scale: SKY_STRENGTH * lit,
            stars: time.starlight(),
            fog: Vec3::ZERO,
            fog_density: weather.fog_depth(time) * FOG_SCALE,
            exposure_bias: 1.0,
        };
        // **The open dome first**, which is what lights the cloud tops: a deck is lit from above by
        // the sky it is under, not by itself.
        sky.ambient = sky.dome_average();
        // **The weather's hue on the dome's own light.** Fog is lit by the sky, so its level belongs
        // to the dome — which is what made the sky an honest stand-in for it — and its colour
        // belongs to the weather, which is the half the stand-in could not give: nothing about a sky
        // says that blight's air is red.
        //
        // **Normalised by its brightest channel, not by its luminance**, because what this is is a
        // scattering albedo and an albedo cannot exceed one. Blight's `Fog Day Color` is
        // (128, 19, 19), whose luminance is a twentieth of its red — so dividing by that gave a
        // multiplier of 4.0 in red and the fog came out brighter than the light that lit it, which
        // drowned the whole landscape. Against the maximum it is a deep red that is darker than a
        // clear day's, which is what a blight storm looks like.
        let hue = weather.fog.at(time);
        sky.fog = sky.ambient * (hue / hue.max_element().max(1e-4));
        // **After the dome's average, because the layer is lit by it.** The clouds are not part of
        // the average in turn: they sit under the sky rather than being it, and a dome that counted
        // them would light the ground by its own clouds.
        // **The beam before any air at all**, because the layer crosses different air from the
        // ground: less of it, and at a different angle. `Clouds` applies its own extinction and its
        // own horizon fade — see the notes there.
        sky.clouds = Clouds::at(time, bare, sky.ambient, weather, sheet);
        // **Then the ground's, which is the dome as seen from *under* the layer.** The clouds were
        // left out of the average deliberately — a dome that counted their own light would light the
        // ground by its own clouds — but leaving out their *blocking* with it made an overcast noon
        // as bright underfoot as a clear one. A deck is a lid: what is under it is what got past.
        //
        // `SKYLIT` is what a cloud sends down of the sky that lit it, so the covered fraction of the
        // dome is worth that much of the open one. The ini agrees, and is the check rather than the
        // source — see `weather_dims_the_ground_the_way_the_game_says_it_does`.
        sky.ambient *= 1.0 - (1.0 - SKYLIT) * sky.clouds.hidden_mean;
        sky.exposure_bias = sky.bias_from_dome();
        sky
    }

    /// What the hour itself multiplies the metered exposure by.
    ///
    /// **Keyed to the sky rather than to the frame**, which is the point: metering says how bright
    /// the picture is and this says how bright the *world* is, and only the second one knows the
    /// difference between a cave at noon and a field at midnight. CryEngine animates its EV
    /// compensation over the same 24-hour curve and Infamous: Second Son shipped a manual offset per
    /// time of day; this is that, computed rather than authored, because the sky already knows.
    ///
    /// The shape is Krawczyk's — a soft S in log luminance that saturates at both ends rather than a
    /// clamp, which is what keeps dusk from stepping — fitted between the dome's own darkest and
    /// brightest so that noon is untouched and the deepest night is `NIGHT_STOPS` below it.
    fn bias_from_dome(&self) -> f32 {
        // The dome's own brightness, which is the one number that says what hour it is.
        let luminance = self.ambient.dot(LUMA);
        // Krawczyk's key value, which is flat below a hundredth and above a thousand and an S
        // between: `1.03 - 2 / (2 + log10(L + 1))`.
        let key = |level: f32| 1.03 - 2.0 / (2.0 + (level + 1.0).log10());
        // Fitted between the two ends this renderer actually has rather than the ones the curve was
        // measured over, and normalised so the brightest hour comes out at one.
        // The dark end is the floor itself: with the sun far enough under, the dome is nothing else.
        let night = NIGHT_SKY.dot(LUMA);
        let span = key(DAY_LUMINANCE) - key(night);
        let along = ((key(luminance) - key(night)) / span).clamp(0.0, 1.0);
        (2.0f32).powf(NIGHT_STOPS * (along - 1.0))
    }

    /// What the sky radiates in one direction, before the night floor.
    ///
    /// **Two mixes and never a product.** Each runs between two spectra, so the colour walks the
    /// line blue → white → red and cannot leave it. Multiplying them instead is the mistake this
    /// file has now made twice: a product of a rising and a falling spectrum peaks in the middle,
    /// and the middle is green.
    ///
    /// `lighting.glsl` draws exactly this per pixel. `tests/sky_dome.rs` renders a frame of sky and
    /// checks it against this, because two implementations of one equation is two chances to be
    /// wrong.
    pub fn shape(&self, direction: Vec3) -> Vec3 {
        // The sun's own, rather than a parameter: a caller that could pass a direction of its own
        // could pass one this sky's warmth was never computed against.
        let cosine = direction.dot(-self.sun.direction);
        let sunward = (cosine * 0.5 + 0.5).clamp(0.0, 1.0);

        let rayleigh = ZENITH_DEPTH / (ZENITH_DEPTH.x + ZENITH_DEPTH.y + ZENITH_DEPTH.z);
        // Thicker air toward the horizon, and thick air has forgotten what colour it was.
        let pale = 1.0 - (-(Self::air_mass(direction.z.max(0.0)) - 1.0) / PALE_MASS).exp();
        let tint = rayleigh
            .lerp(Vec3::splat(1.0 / 3.0), pale)
            .lerp(self.warm, sunward * self.warmth);

        // Rayleigh's own phase function, normalised to average one over the sphere: a gentle
        // brightening along the sun's line and against it, and a minimum square to it.
        let phase = 0.75 * (1.0 + cosine * cosine);
        // Brighter at the horizon, where the ray crosses more lit air — and once the sun is low,
        // dimmer on the side it is not, which is the Earth's shadow said as a fraction.
        let side = AWAY_SHARE + (1.0 - AWAY_SHARE) * (1.0 - self.warmth * (1.0 - sunward));
        tint * (phase * (1.0 + pale) * side * self.scale)
    }

    /// The mean of [`Self::shape`] over every direction, plus the night floor.
    ///
    /// **The light a surface is given has to be the sky behind it**, or a frame lit by one and
    /// drawn with the other disagrees with itself. There is no closed form, so it is a Fibonacci
    /// sphere — the cheapest quadrature that is even in every direction and has no seam at a pole,
    /// which matters because the dome's whole point is that it is not the same all the way round.
    ///
    /// **Two hundred and fifty-six of them, once a frame**, because the window rebuilds the sky
    /// every frame from a clock that moves. Measured at **7.5 us**, which is 0.045% of the 16.6 ms
    /// a frame has — worth writing down rather than leaving a reader to wonder what a per-frame
    /// quadrature costs, and far too little to be worth caching against a time that changes anyway.
    fn dome_average(&self) -> Vec3 {
        const SAMPLES: u32 = 256;
        let mut total = Vec3::ZERO;
        for i in 0..SAMPLES {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / SAMPLES as f32;
            let radius = (1.0 - z * z).max(0.0).sqrt();
            // The golden angle, which is what spreads the points evenly rather than in rings.
            let theta = 2.399_963_2 * i as f32;
            total += self.shape(Vec3::new(radius * theta.cos(), radius * theta.sin(), z));
        }
        total / SAMPLES as f32 + Self::NIGHT_FLOOR
    }

    /// How much of a disc of angular radius `radius` has cleared the horizon, `climb` being the
    /// sine of the elevation of its centre.
    ///
    /// **Over its own width rather than at a line**, which is what a body setting actually does and
    /// what stops one blinking out against a sea horizon. Smoothstepped, so the last of it goes
    /// without a corner. The sun and both moons take the same answer; they differ only in how wide
    /// they are, which is the parameter.
    pub(crate) fn showing(climb: f32, radius: f32) -> f32 {
        let above = ((climb + radius) / (2.0 * radius)).clamp(0.0, 1.0);
        above * above * (3.0 - 2.0 * above)
    }

    /// What is left of a beam that crossed the atmosphere to a sun `climb` above the horizon.
    ///
    /// `climb` is the sine of the elevation. The air mass is Kasten and Young's fit rather than the
    /// flat-earth `1/sin`, which is off by 10% at ten degrees and diverges at the horizon where all
    /// of this matters most — theirs stays finite, reaching 38 atmospheres at zero.
    pub(crate) fn transmittance(climb: f32) -> Vec3 {
        Self::transmittance_above(climb, 1.0)
    }

    /// The same, for a beam arriving somewhere with `above` of the atmosphere's mass still in front
    /// of it — one at the ground, and less for anything standing above part of the air.
    ///
    /// **What makes a sunset cloud gold rather than black.** The deck at two kilometres is above a
    /// fifth of the atmosphere and its own horizon is lower than the ground's, so the beam reaching
    /// it at the moment of sunset has crossed seventeen air masses where one reaching the ground has
    /// crossed thirty-eight. Handing the clouds the ground's figure is what left them extinguished
    /// at the hour they should be at their brightest.
    pub(crate) fn transmittance_above(climb: f32, above: f32) -> Vec3 {
        (-ZENITH_DEPTH * above * Self::air_mass(climb)).exp()
    }

    /// How many atmospheres a beam crosses to something `climb` above the horizon.
    ///
    /// Kasten and Young's fit rather than the flat-earth `1/sin`, which is off by 10% at ten degrees
    /// and diverges at the horizon where all of this matters most — theirs stays finite, reaching 38
    /// atmospheres at zero.
    fn air_mass(climb: f32) -> f32 {
        let degrees = climb.clamp(0.0, 1.0).asin().to_degrees();
        1.0 / (climb + 0.50572 * (degrees + 6.07995).powf(-1.6364))
    }
}

impl Default for Sky {
    fn default() -> Self {
        Self::at(WorldTime::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sine of the sun's elevation, which is what every claim below is really about.
    fn climb(hour: f32) -> f32 {
        -Sky::at(WorldTime::hours(hour)).sun.direction.z
    }

    /// What the eye makes of a colour, which is the only thing a "darker" claim can mean.
    fn luminance(colour: Vec3) -> f32 {
        colour.dot(LUMA)
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
    fn the_hour_carries_its_own_exposure_down_to_night() {
        let bias = |hour: f32| Sky::at(WorldTime::hours(hour)).exposure_bias;
        // **`DAY_LUMINANCE` is a measurement and this is what keeps it one.** The curve is fitted
        // between the dome's own two ends, so a `SKY_STRENGTH` that moved without this moving would
        // leave the bias anchored to a noon that no longer exists — and it would fail silently, as a
        // slightly wrong exposure at every hour rather than as anything visible.
        let noon = luminance(Sky::at(WorldTime::hours(12.0)).ambient);
        assert!(
            (noon - DAY_LUMINANCE).abs() < 0.005,
            "the dome measures {noon} at noon, not the {DAY_LUMINANCE} the bias is fitted against"
        );

        // Noon is the anchor and everything else is below it, which is the whole principle: a
        // renderer that meters the frame alone renders midnight and midday the same brightness.
        assert!((bias(12.0) - 1.0).abs() < 1e-4, "{}", bias(12.0));
        assert!(
            (bias(23.0).log2() + NIGHT_STOPS).abs() < 0.05,
            "midnight is {} stops down, not {NIGHT_STOPS}",
            bias(23.0).log2()
        );

        // **Monotonic from midnight to noon**, because a step anywhere in it is a step the eye sees
        // as the sun comes up — a clamp would give one at each end and this is a curve for that
        // reason.
        let mut hour = 0.0;
        while hour < 12.0 {
            assert!(
                bias(hour + 0.25) >= bias(hour) - 1e-6,
                "the exposure jumped backwards at {hour}: {} to {}",
                bias(hour),
                bias(hour + 0.25)
            );
            hour += 0.25;
        }

        // And the whole night sits at the bottom of it rather than only the one instant of midnight.
        for hour in [21.0, 23.0, 1.0, 3.0] {
            assert!(
                bias(hour).log2() < -NIGHT_STOPS + 0.2,
                "{hour} is {}",
                bias(hour).log2()
            );
        }
    }

    #[test]
    fn the_sky_is_never_green() {
        // **Reported from the window, and the reason the tint is a mix of two spectra.** With
        // `(1 - T) * T` each channel peaked where its own transmittance was a half, and green's half
        // lands at an elevation of nine degrees — so every evening went blue, then *green*, then red.
        // A line between two endpoints cannot do that, and blue to red passes through grey.
        let mut hour = 0.0;
        while hour < 24.0 {
            let sky = Sky::at(WorldTime::hours(hour)).ambient;
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
            let night = Sky::at(WorldTime::hours(hour)).ambient;
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
            let sky = Sky::at(WorldTime::hours(hour));
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
        let at = |hour: f32| luminance(Sky::at(WorldTime::hours(hour)).ambient);
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
        assert!(climb(18.0).abs() < 1e-6, "{}", climb(18.0));

        // Highest at the middle of the day, and symmetric either side of it.
        assert!(climb(12.0) > climb(10.0) && climb(10.0) > climb(7.0));
        assert!((climb(10.0) - climb(14.0)).abs() < 1e-6);
        // Morrowind's noon, out of its own `(0, 75, -100)`: 100/125 of the way up.
        assert!((climb(12.0) - 0.8).abs() < 1e-6, "{}", climb(12.0));
    }

    #[test]
    fn the_sun_goes_out_over_its_own_width_and_not_before() {
        // Full daylight either side of the horizon by more than the disc is wide, and nothing at all
        // below it: the disc setting is the whole of what puts the sun out, so there is no second
        // fade to disagree with where the sun actually is.
        assert!(Sky::at(WorldTime::hours(6.2)).sun.colour.x > 0.0);
        assert_eq!(Sky::at(WorldTime::hours(5.5)).sun.colour, Vec3::ZERO);
        assert_eq!(Sky::at(WorldTime::hours(18.5)).sun.colour, Vec3::ZERO);

        // Half a degree of disc against fourteen hours of day is a minute or so of setting, and the
        // beam has crossed thirty-eight atmospheres by then — so what goes out is already a deep
        // red, and the blue left it long before.
        let setting = Sky::at(WorldTime::hours(18.0)).sun.colour;
        assert!(setting.x > setting.z * 100.0, "{setting:?}");
    }

    #[test]
    fn the_sun_reddens_and_dims_as_it_sets() {
        let noon = Sky::at(WorldTime::hours(12.0)).sun.colour;
        let dusk = Sky::at(WorldTime::hours(17.5)).sun.colour;

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
        let noon = Sky::at(WorldTime::hours(12.0)).ambient;
        // Blue above green above red is what Rayleigh scattering means, and the reason the sky has
        // a colour at all.
        assert!(noon.z > noon.y && noon.y > noon.x, "{noon:?}");

        // **At the default hour**, not at noon, because that is where `SKY_STRENGTH` is set: it
        // lands where the model before it did, so every image made against that one is still
        // comparable. Noon is brighter than mid-morning now, which it should always have been —
        // the old tint fell away at low air mass and took the middle of the day down with it.
        let default = Sky::at(WorldTime::default()).ambient;
        assert!((luminance(default) - 0.518).abs() < 0.01, "{default:?}");
        assert!(
            luminance(noon) > luminance(default),
            "{noon:?} against {default:?}"
        );

        // At dusk the order inverts: the blue never got through the air to be scattered.
        let dusk = Sky::at(WorldTime::hours(17.8)).ambient;
        assert!(dusk.x > dusk.z, "{dusk:?}");
        assert!(dusk.length() < noon.length(), "{dusk:?} against {noon:?}");

        // And at night there is no sun at all, so nothing casts.
        let night = Sky::at(WorldTime::hours(1.0));
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
