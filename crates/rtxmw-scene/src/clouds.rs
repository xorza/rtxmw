//! The layer of cloud over an exterior, and what the light does to it.

use glam::{Vec2, Vec3};

use crate::sun::Sun;
use crate::world_time::WorldTime;

/// Morrowind world units per metre, which the cloud layer is the first thing in this crate to need.
///
/// The game uses 64 units per yard; OpenMW carries the conversion as `69.99125109`, which is beyond
/// `f32` and rounds to this.
const UNITS_PER_METRE: f32 = 69.991_25;

/// How high the layer sits, and how far round the world curves under it.
///
/// **A shell over a curved world rather than a dome around the eye**, which is the difference
/// between clouds that reach a horizon and clouds that pile into a ring at it. `sky_clouds_01.nif`
/// is a cap of radius 1000 rising 100 to 307 — flat enough that its own artist was drawing the same
/// picture — but a mesh centred on the viewer saturates: every direction meets it at one distance,
/// so the last few degrees above the horizon smear the whole sheet into streaks.
///
/// Two kilometres up, which is where a fair-weather deck sits.
const ALTITUDE: f32 = 2_000.0 * UNITS_PER_METRE;

/// How far the world curves under that layer — the Earth's own radius.
///
/// It is the whole of what gives the sky depth: a ray meets the layer at [`ALTITUDE`] overhead and
/// at `sqrt(2 R h)`, 160 km, along the horizon.
const WORLD_RADIUS: f32 = 6_371_000.0 * UNITS_PER_METRE;

/// How far one tile of the sheet spans across the layer.
///
/// **Not from the game**, which scrolls one quad's worth over its dome and states no distance. Eight
/// kilometres is the width of a decent field of cumulus, which is what the sheet holds a few of —
/// so a cloud comes out a kilometre or two across, which is what a cloud is.
const TILE: f32 = 8_000.0 * UNITS_PER_METRE;

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
const BEARING: f32 = 0.6;

/// `Cloud Speed` for clear weather, out of `Morrowind.ini`'s `[Weather Clear]`.
///
/// One number of the weather system, borrowed ahead of the rest of it so the layer moves at the rate
/// the game says rather than at one invented here.
const CLEAR_SPEED: f32 = 1.25;

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
const SKYLIT: f32 = 0.3;

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
    /// How far the world curves under it, which is what puts the layer's own horizon 160 km out.
    pub world_radius: f32,
    /// How far one tile of the sheet spans across the layer.
    pub tile: f32,
    /// How far the wind has carried the layer, in tiles.
    pub drift: Vec2,
    /// What a fully lit cloud radiates, sun and sky together already in it.
    pub lit: Vec3,
    /// What a cloud in its own shadow radiates — the sky's light alone.
    pub shadowed: Vec3,
    /// The sheet's mean opaque luminance, which its texels are read as a ratio to.
    ///
    /// **Filled in by whoever uploaded the sheet**, since that is the only thing that has read it.
    /// One until then, which draws a layer scaled wrongly rather than not at all — and nothing draws
    /// a layer before a sheet exists, so the placeholder is never seen.
    pub sheet_mean: f32,
    /// How much of the sky the layer covers at all, from none to all of it.
    ///
    /// `Clouds Maximum Percent`, which is 1 for every weather but a few. Zero is a cell with no
    /// sky, which draws no layer without a branch to say so.
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
        cover: 0.0,
    };

    /// The layer at `time`, lit by `sun` under a sky of average radiance `ambient`.
    ///
    /// The drift is taken off the *date* rather than the hour, because it has to accumulate: the
    /// hour wraps at midnight and a layer carried by it would snap back with it.
    pub fn at(time: WorldTime, sun: Sun, ambient: Vec3) -> Self {
        // Carried on a steady bearing, in tiles, from the hour the world has run.
        let (sin, cos) = BEARING.sin_cos();
        let travelled = time.day() * DRIFT * CLEAR_SPEED;

        // **What reaches the top of the layer**, which is the sun's beam plus the whole dome. The
        // sun's own colour already has the air it crossed taken out of it, and a cloud sits under
        // most of that air rather than above it — near enough at this scale that the difference is
        // smaller than `SUNLIT` is uncertain.
        let sunward = sun.colour * SUNLIT;
        let skyward = ambient * SKYLIT;
        Self {
            altitude: ALTITUDE,
            world_radius: WORLD_RADIUS,
            tile: TILE,
            drift: Vec2::new(cos, sin) * travelled,
            lit: sunward + skyward,
            shadowed: skyward,
            // Stood in for until a sheet is uploaded, which is the only thing that knows it.
            sheet_mean: 1.0,
            cover: 1.0,
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
        Clouds::at(time, sky.sun, sky.ambient)
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
        // Along the horizon it is `sqrt(2Rh)` — 160 km, which in metres is what a real cloud deck's
        // horizon is.
        let horizon = (2.0 * clouds.world_radius * clouds.altitude).sqrt();
        assert!((reach(0.0) - horizon).abs() / horizon < 1e-3);
        assert!(
            (horizon / UNITS_PER_METRE / 1000.0 - 160.0).abs() < 2.0,
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
