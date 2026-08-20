//! The layer of cloud over an exterior, and what the light does to it.

use glam::{Vec2, Vec3};

use crate::sky::Sky;
use crate::sun::Sun;
use crate::weather::Weather;
use crate::world_time::WorldTime;

/// Morrowind world units per metre, which the cloud layer is the first thing in this crate to need.
///
/// The game uses 64 units per yard; OpenMW carries the conversion as `69.99125109`, which is beyond
/// `f32` and rounds to this.
const UNITS_PER_METRE: f32 = 69.991_25;

/// How fast the air thins with height — the atmosphere's scale height.
///
/// Eight kilometres, the same figure `ZENITH_DEPTH` in `sky.rs` is a whole atmosphere's worth of.
const SCALE_HEIGHT: f32 = 8_000.0 * UNITS_PER_METRE;

/// How high the layer sits, and how far round the world curves under it.
///
/// **A shell over a curved world rather than a dome around the eye**, which is the difference
/// between clouds that reach a horizon and clouds that pile into a ring at it. `sky_clouds_01.nif`
/// is a cap of radius 1000 rising 100 to 307 — flat enough that its own artist was drawing the same
/// picture — but a mesh centred on the viewer saturates: every direction meets it at one distance,
/// so the last few degrees above the horizon smear the whole sheet into streaks.
///
/// How high the layer sits — five hundred metres, a stratocumulus base.
///
/// **Set by the shadows rather than by the sky.** A cloud's shadow is the size of the cloud, so
/// making one small enough to read across a landscape means making the other small too — and at a
/// fixed altitude that shrinks it in the sky as well. Dropping the base in step keeps the angle a
/// cloud subtends and so keeps the sky's own look. Two kilometres with eight-kilometre tiles put a
/// cloud feature at 780 m against a visible landscape of about 300, so the whole view sat under one
/// cloud and dimmed as a body; two kilometres with two-kilometre tiles fixed the shadows and turned
/// the sky into a mackerel plaid. This is the pair that gives both.
const ALTITUDE: f32 = 500.0 * UNITS_PER_METRE;

/// How far the world curves under that layer — the Earth's own radius.
///
/// It is the whole of what gives the sky depth: a ray meets the layer at [`ALTITUDE`] overhead and
/// at `sqrt(2 R h)`, 80 km, along the horizon.
const WORLD_RADIUS: f32 = 6_371_000.0 * UNITS_PER_METRE;

/// How far one tile of the sheet spans across the layer.
///
/// **Not from the game**, which scrolls one quad's worth over its dome and states no distance. Two
/// kilometres, which puts a cloud feature at about 200 metres — small enough that a shadow of one
/// reads across a view of Seyda Neen, which is only three hundred metres wide.
const TILE: f32 = 2_000.0 * UNITS_PER_METRE;

/// How fast the wind carries the layer, in tiles per game day, against `Cloud Speed`.
///
/// **A physically correct wind under this clock is a time-lapse, and that is the whole problem.**
/// Morrowind's own `timescale` is 30, so a game hour passes in two minutes of watching; a real 19
/// km/h breeze then carries a cloud across ten degrees of sky every two seconds, which reads as a
/// conveyor belt rather than as weather. This was set at that wind and looked exactly like it.
///
/// So the rate is chosen against the clock the sky is actually watched on: at `Cloud Speed=1.25` a
/// cloud crosses about ten degrees in a minute of real time, which is what a sky does. It is a
/// departure from the physical figure by roughly the timescale itself, and it is the honest place to
/// make one — the sun's hour is chronology and has to run at the game's rate, while a cloud's drift
/// is ambience and only has to look right.
const DRIFT: f32 = 1.8;

/// Which way the wind blows, as a bearing from north.
///
/// **Nothing in the game says.** `Wind Speed` is a scalar and the direction is never given, so this
/// is chosen — and chosen off-axis so the drift is not along a texture axis, which would make the
/// repeat obvious.
///
/// **The fog takes the same one**, out of [`crate::Sky::wind`]: there is one wind over a landscape,
/// and cloud shadows crossing the ground one way while the air moves another would read as two
/// weathers at once.
pub(crate) const BEARING: f32 = 0.6;

/// What a cloud's lit face radiates against the sun that lights it.
///
/// **A cloud is not a surface and this is not an albedo.** Water droplets scatter almost all of
/// what reaches them — a thick cloud's albedo is 0.7 to 0.9 — but most of that leaves in directions
/// other than the eye's, and what a lit cloud actually shows is the forward-scattered fraction.
/// This is that fraction, chosen so an overhead sun gives a white cloud without blowing it, and it
/// is the one number here with no data behind it.
const SUNLIT: f32 = 0.55;

/// How much of the sky's own light a cloud sends back down, against the sky's own radiance.
///
/// **Below one, and that is the whole of why a cloud reads at night.** A thick cloud reflects most
/// of what lands on it *upward*: plane-parallel theory puts a deck's transmission at 0.2 to 0.3, so
/// the radiance leaving its base is about a quarter of the sky that lit it. At night, when there is
/// no sun and this is the only term, that is the difference between a deck you can see and one you
/// cannot — a cloud blots out the airglow of ninety kilometres of atmosphere above it and returns a
/// fraction of it, which is why clouds are the dark shapes in a starry sky rather than the light
/// ones.
///
/// It was 0.9 first, on the reasoning that a cloud is lit from the whole dome rather than from one
/// direction. That is true of the *irradiance* reaching it and says nothing about what leaves — and
/// at 0.9 a night deck was 90% of the sky it covered, which is invisible, and a daylit cloud base
/// was nearly as bright as the sky instead of the grey it is.
///
/// Thin cloud is not made dark by this: how much of the sky a wisp replaces at all is its own alpha,
/// so a deck at 0.2 coverage comes out at 96% of the sky and a solid one at 30%.
pub(crate) const SKYLIT: f32 = 0.3;

/// What a weather's painted sheet comes to on average, which is all the sky needs of it.
///
/// **Two numbers out of a 512-square picture**, so that building a sky needs the statistics of the
/// sheet without needing the sheet: how much of the dome it hides, and how bright its cloud is. They
/// were patched into the layer by the renderer after the fact until the ambient needed them at
/// construction — the ground under a deck is dimmed by exactly the first of them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudSheet {
    /// The mean of its alpha: a quarter for clear weather's cirrus, all of it for every overcast.
    pub covering: f32,
    /// The mean luminance of what its alpha calls cloud, which its texels are read as a ratio to.
    pub mean: f32,
}

impl CloudSheet {
    /// No sheet at all, which draws no layer and dims nothing.
    pub const NONE: Self = Self {
        covering: 0.0,
        mean: 1.0,
    };
}

/// The cloud layer over an exterior at an hour.
///
/// **A vanilla asset lit rather than shown**, the same as the moons' faces and every albedo in the
/// game: `tx_sky_*.dds` is a painted photograph of a sky with 2002's lighting in it, so what is
/// taken from it is the *shape* — the alpha mask, and the texel's own luminance against the sheet's
/// mean where the mask is flat — and the colour comes from the sun, the sky and the moons. Painting
/// it on directly would light every cloud twice, which is `docs/design.md` §5.1's whole subject.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clouds {
    /// How high the layer sits over the eye.
    pub altitude: f32,
    /// How far the world curves under it, which is what puts the layer's own horizon 80 km out.
    pub world_radius: f32,
    /// How far one tile of the sheet spans across the layer.
    pub tile: f32,
    /// How far the wind has carried the layer, in tiles.
    pub drift: Vec2,
    /// What a fully lit cloud radiates, sun and sky together already in it.
    pub lit: Vec3,
    /// What a cloud in its own shadow radiates — the sky's light alone.
    pub shadowed: Vec3,
    /// What fraction of the sky the layer hides, averaged over the whole dome.
    ///
    /// **The sheet's own mean alpha times [`Self::cover`]**, and what the ground's ambient is dimmed
    /// by: a deck is a lid, and the light under one is the light that got past it. Distinct from
    /// `cover`, which is only the weather's declared ceiling — an overcast sheet reaches it
    /// everywhere and clear weather's cirrus reaches a quarter of it.
    pub hidden_mean: f32,
    /// The sheet's mean opaque luminance, which its texels are read as a ratio to.
    ///
    /// Out of [`CloudSheet::mean`], which is the sheet's own statistics rather than the sheet.
    pub sheet_mean: f32,
    /// The ceiling on how much sky the layer may cover, from none to all of it.
    ///
    /// The weather's own `Clouds Maximum Percent`, which is 1 for most of the ten and 0.66 for rain
    /// — a ceiling the sheet's own alpha then scales, which is what [`Self::hidden_mean`] is the
    /// average of. Zero is a cell with no sky, which draws no layer without a branch to say so.
    pub cover: f32,
}

impl Clouds {
    /// No layer at all, which is what an interior has.
    pub const NONE: Self = Self {
        altitude: ALTITUDE,
        world_radius: WORLD_RADIUS,
        tile: TILE,
        drift: Vec2::ZERO,
        lit: Vec3::ZERO,
        shadowed: Vec3::ZERO,
        sheet_mean: 1.0,
        hidden_mean: 0.0,
        cover: 0.0,
    };

    /// How far the layer's own horizon dips below the ground's, in radians.
    ///
    /// **Why a cloud is still lit after the sun has set**, and the whole of why a sunset glows. A
    /// cloud at height `h` over a world of radius `R` sees the sun until it is `sqrt(2h/R)` under
    /// the horizon *below the cloud* — 0.72 degrees at five hundred metres, which is three and a
    /// half minutes of the game's own clock after the ground has gone dark, and seven before the
    /// far side of the layer loses it too.
    fn sunset_dip() -> f32 {
        (2.0 * ALTITUDE / WORLD_RADIUS).sqrt()
    }

    /// How much of the atmosphere's mass is still above the layer.
    ///
    /// Air thins by `exp(-h/H)`, so five hundred metres up leaves 94% of it overhead — and a beam
    /// reaching the deck is extinguished by that much less than one reaching the ground. Derived
    /// from the altitude beside it rather than written down as the figure it comes to, which would
    /// be a second thing to move if the layer ever rises — and it has already moved once.
    fn above_layer() -> f32 {
        (-ALTITUDE / SCALE_HEIGHT).exp()
    }

    /// The layer at `time`, lit by `sun` under a sky of average radiance `ambient`.
    ///
    /// **`sun` is the beam as the layer meets it, not as the ground does** — attenuated by the air
    /// it crossed but with no horizon fade on it, because the layer's horizon is not the ground's.
    /// Handing over [`crate::Sky`]'s own faded sun instead is what made the clouds change colour in
    /// a single step at six o'clock: they were losing the sun over its own half-degree diameter,
    /// which at sunset this world crosses in half a minute.
    ///
    /// The drift is taken off the *date* rather than the hour, because it has to accumulate: the
    /// hour wraps at midnight and a layer carried by it would snap back with it.
    pub fn at(
        time: WorldTime,
        sun: Sun,
        ambient: Vec3,
        weather: &Weather,
        sheet: CloudSheet,
    ) -> Self {
        // Carried on a steady bearing, in tiles, from the hour the world has run.
        let (sin, cos) = BEARING.sin_cos();
        let travelled = time.day() * DRIFT * weather.cloud_speed;

        // **The layer's own sunset, which is later and slower than the ground's.** Full sun while
        // the sun is up at all, then out over twice the dip: a cloud straight overhead loses it at
        // the dip itself, and the layer reaches its own horizon 80 km away whose clouds keep it
        // about that much longer again.
        //
        // **Measured in how far the sun is *under*, not in the sine of where it is.** Morrowind's
        // path is `sqrt(1 - orbit^2)`, whose derivative at the horizon is infinite — the sun drops
        // from 0.57 degrees to nothing in the last eighteen seconds of the day — so anything keyed
        // to elevation steps there however smooth its own curve is. That was a 30% jump in the
        // clouds' colour at exactly six o'clock. Below the horizon the sun descends at a steady
        // `NIGHT_DESCENT` per unit of orbit and orbit is linear in time, so this is too.
        let dip = Self::sunset_dip();
        let under = (-sun.direction.z).min(0.0).abs().asin();
        // The same shaping the sun's own disc sets by, read the other way round: `showing` fades in
        // as its argument rises through a window of half-width `dip`, and `dip - under` walks that
        // window backwards as the sun goes down. Identical to `1 - smoothstep(under / 2 dip)`,
        // because `smoothstep(1 - x)` is `1 - smoothstep(x)`.
        let showing = Sky::showing(dip - under, dip);

        // **The air the beam crossed to reach the *layer*, which is not the air it would cross to
        // reach the ground.** The deck stands above a twentieth of the atmosphere and sees the sun
        // 0.72 degrees higher than the ground does, so at the moment of sunset its beam has crossed
        // twenty-seven air masses against the ground's thirty-eight. That is the whole of why a
        // sunset cloud is gold: it is lit by light the ground has already lost.
        let elevation = (-sun.direction.z).clamp(-1.0, 1.0).asin() + dip;
        let air = Sky::transmittance_above(elevation.sin().max(0.0), Self::above_layer());
        let sunward = sun.colour * air * (SUNLIT * showing);
        let skyward = ambient * SKYLIT;
        Self {
            altitude: ALTITUDE,
            world_radius: WORLD_RADIUS,
            tile: TILE,
            drift: Vec2::new(cos, sin) * travelled,
            lit: sunward + skyward,
            shadowed: skyward,
            sheet_mean: sheet.mean,
            hidden_mean: sheet.covering * weather.cloud_cover,
            cover: weather.cloud_cover,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::srgb::LUMA;

    fn at(hour: f32) -> Clouds {
        let time = WorldTime::hours(hour);
        let sky = crate::Sky::at(time);
        Clouds::at(
            time,
            sky.sun,
            sky.ambient,
            &Weather::clear(),
            CloudSheet::NONE,
        )
    }

    #[test]
    fn a_cloud_is_brighter_where_the_sun_reaches_it_than_where_it_does_not() {
        // The whole reason the sheet's painted colour is thrown away: a cloud's light comes from
        // somewhere, and the difference between its lit face and its shadowed one *is* the sun.
        let noon = at(12.0);
        assert!(
            noon.lit.dot(LUMA) > 4.0 * noon.shadowed.dot(LUMA),
            "{:?} against {:?}",
            noon.lit,
            noon.shadowed
        );

        // At night there is no sun, so the two are the same and a cloud is a flat sheet — which is
        // what a cloud at night is. No branch says so; the sun's colour is simply black.
        let midnight = at(24.0);
        assert_eq!(midnight.lit, midnight.shadowed);
        assert!(midnight.lit.dot(LUMA) > 0.0, "the sky still lights them");

        // And the low sun's own colour carries: a cloud at dusk is the colour of the light that
        // reached it, which after thirty atmospheres is red.
        let dusk = at(17.8);
        assert!(dusk.lit.x > 2.0 * dusk.lit.z, "{:?}", dusk.lit);
    }

    #[test]
    fn the_wind_carries_the_layer_on_without_resetting_at_midnight() {
        // **Off the date rather than the hour**, which is the one thing that could go wrong here:
        // the hour wraps and a layer carried by it would snap back to where it started every night.
        let before = at(23.99).drift;
        let after = at(24.01).drift;
        // A fiftieth of an hour of drift, which is what those two readings are apart — against the
        // 57 tiles a whole day carries it, so a wrap would be a thousand times this.
        assert!(
            (after - before).length() < 0.1,
            "the sky jumped at midnight: {before:?} to {after:?}"
        );
        assert!(after.length() > before.length());

        // Two days carry it exactly twice as far as one, which is what a steady wind does.
        let one = at(24.0).drift.length();
        let two = at(48.0).drift.length();
        assert!((two - 2.0 * one).abs() < 1e-3, "{one} then {two}");

        // **And the rate is the one the sky is watched at, not the one it would have.** The world
        // clock runs at Morrowind's own timescale of 30, so a minute of watching is half a game
        // hour; over that the layer should cross about ten degrees overhead. It was set at a real
        // 19 km/h wind first, which under that clock came to ten degrees every two seconds.
        let minute = at(0.5).drift.length() * TILE;
        let overhead = (minute / ALTITUDE).atan().to_degrees();
        assert!(
            (8.0..13.0).contains(&overhead),
            "a minute of watching moves the sky {overhead} degrees"
        );

        // On a bearing rather than along an axis, so the repeat never lines up with the texture.
        let drift = at(24.0).drift;
        assert!(drift.x.abs() > 0.1 * drift.length() && drift.y.abs() > 0.1 * drift.length());
    }

    #[test]
    fn the_layer_reaches_its_own_horizon_rather_than_a_ring_round_the_eye() {
        // The geometry the shader intersects, checked here because it is the difference between a
        // sky with depth and one where the last few degrees smear the whole sheet into streaks.
        let clouds = at(12.0);
        // The stable form of the root, which is what the shader solves — `-b + sqrt(b*b + c)` is
        // the same number and loses all of it, since `b*b` is 2e17 in `f32`.
        let reach = |climb: f32| {
            let b = clouds.world_radius * climb;
            let c = clouds.altitude * (2.0 * clouds.world_radius + clouds.altitude);
            c / (b + (b * b + c).sqrt())
        };
        // Straight up, the layer is exactly its own altitude away.
        assert!((reach(1.0) - clouds.altitude).abs() < 1.0, "{}", reach(1.0));
        // Along the horizon it is `sqrt(2Rh)` — 80 km, which is what a real deck's horizon is.
        let horizon = (2.0 * clouds.world_radius * clouds.altitude).sqrt();
        assert!((reach(0.0) - horizon).abs() / horizon < 1e-3);
        assert!(
            (horizon / UNITS_PER_METRE / 1000.0 - 80.0).abs() < 2.0,
            "{} km",
            horizon / UNITS_PER_METRE / 1000.0
        );
        // And it grows without bound toward the horizon rather than saturating, which is what a
        // shell centred on the eye would do — 80 times the overhead distance by one degree up.
        assert!(reach(0.0f32.to_radians().sin()) > 70.0 * reach(1.0));
    }

    #[test]
    fn an_interior_has_no_layer_at_all() {
        assert_eq!(Clouds::NONE.cover, 0.0);
        assert_eq!(Clouds::NONE.lit, Vec3::ZERO);
        assert_eq!(Clouds::NONE.drift, Vec2::ZERO);
    }
}
