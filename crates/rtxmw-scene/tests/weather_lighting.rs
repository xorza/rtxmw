//! That a weather dims the ground the way the game says it does, and colours it the way it says.
//!
//! **The ini's `Ambient` schedule is the check, not the source.** This renderer derives what lights
//! the ground from the dome's own average, dimmed by however much of it the cloud deck hides — no
//! authored figure anywhere. Bethesda wrote down an answer to the same question for each of the ten,
//! and the two agreeing is evidence the derivation is right; the two disagreeing would be worth
//! knowing about either way.
//!
//! The *colour* is the other half and comes the other way round: the ini's `Sky` schedule read
//! against its `Fog` is what says how far up a weather's own medium reaches, and the ground is lit
//! through whatever that comes to. The two halves must not touch — a veil that moved the level
//! would land on top of the derivation the first half checks — so the first test below is also what
//! holds the second honest.

use glam::Vec3;
use rtxmw_scene::{CloudSheet, LUMA, Sky, SkyTextures, Weather, WorldTime};

fn luminance(colour: Vec3) -> f32 {
    colour.dot(LUMA)
}

/// The hour every measurement here is taken at, so no two of them can drift apart.
///
/// A function because `WorldTime::hours` is not a `const fn`.
fn noon() -> WorldTime {
    WorldTime::hours(12.0)
}

/// Each of the ten and the sky this renderer builds under it at noon.
///
/// Empty without the game, which is what makes every test here skip rather than fail on a machine
/// with no install. One walk of the table and one read of each sheet, because a `SkyTextures::load`
/// is the decode of a 512-square.
fn measured() -> Vec<(Weather, Sky)> {
    let Ok(table) = Weather::table() else {
        return Vec::new();
    };
    table
        .into_iter()
        .map(|weather| {
            let sheet = SkyTextures::load(&weather)
                .ok()
                .flatten()
                .map_or(CloudSheet::NONE, |textures| textures.sheet());
            let sky = Sky::under(noon(), &weather, sheet);
            (weather, sky)
        })
        .collect()
}

/// Whichever of the ten is called `name`, which every test below asks for by hand.
fn named<'a>(measured: &'a [(Weather, Sky)], name: &str) -> &'a (Weather, Sky) {
    measured
        .iter()
        .find(|(weather, _)| weather.name == name)
        .unwrap_or_else(|| panic!("no {name}"))
}

#[test]
fn a_deck_dims_the_ground_and_a_wisp_barely_does() {
    let ten = measured();
    if ten.is_empty() {
        return;
    }
    // Both sides as a ratio to clear's, because the ini's units are the original engine's and this
    // renderer's are its own — what the two can be held to is how far each weather sits from a
    // clear day, which is the same question in either.
    let (clear, clear_sky) = named(&ten, "clear");
    let (base, authored_base) = (
        luminance(clear_sky.ambient),
        luminance(clear.ambient.at(noon())),
    );
    let derived = |name: &str| luminance(named(&ten, name).1.ambient) / base;
    let authored = |name: &str| luminance(named(&ten, name).0.ambient.at(noon())) / authored_base;
    for (weather, sky) in &ten {
        println!(
            "{:>13}  derived {:.2}x  the ini says {:.2}x",
            weather.name,
            luminance(sky.ambient) / base,
            luminance(weather.ambient.at(noon())) / authored_base
        );
    }

    // Clear is the reference, so it is one by construction — the assertion is that nothing has
    // quietly made it something else.
    assert!((derived("clear") - 1.0).abs() < 1e-5);

    // **Where the darkness comes from a deck, the derivation lands on the game's own answer.**
    // Foggy comes out at 0.55 against an authored 0.55; overcast, snow and blizzard at 0.36 against
    // 0.44. Nothing here was told any of those: the deck's own alpha says how much of the dome it
    // hides and `SKYLIT` says what gets through.
    for name in ["foggy", "overcast", "snow", "blizzard", "rain"] {
        let (derived, authored) = (derived(name), authored(name));
        assert!(
            (derived - authored).abs() < 0.1,
            "{name}: derived {derived:.2}x against the ini's {authored:.2}x"
        );
    }

    // **Where it does not, the gap is worth knowing rather than closing.** A dust storm is dark
    // because the air is full of ash, not because a deck is over it — the ini simply asserts 0.14
    // for ashstorm and 0.17 for blight, and this renderer has no airborne dust to derive that from.
    // What it has is their fog, thicker than clear's at 1.1 against 0.69, which darkens the picture
    // without darkening the light. Same for thunderstorm, whose 0.38 is far under the 0.66 cloud
    // cover it declares.
    for name in ["ashstorm", "blight", "thunderstorm"] {
        let (derived, authored) = (derived(name), authored(name));
        assert!(derived < 1.0, "{name} should still be dimmer than clear");
        assert!(
            authored < derived,
            "{name}: the ini wants it darker than the deck explains — {authored:.2}x against \
             {derived:.2}x"
        );
    }

    // **And cloudy is where the ini looks copied.** Its ambient is clear's to within 6% despite
    // declaring three-quarters cloud cover, which is the one weather whose authored answer this
    // does not believe.
    assert!((authored("cloudy") - 1.0).abs() < 0.1);
    assert!(derived("cloudy") < 0.7);
}

#[test]
fn a_weather_veils_the_sky_only_where_the_ini_says_a_medium_could() {
    let ten = measured();
    if ten.is_empty() {
        return;
    }
    for (weather, sky) in &ten {
        println!(
            "{:>13}  veil {:.2} of {:?}",
            weather.name, sky.veil.amount, sky.veil.hue
        );
    }
    let of = |name: &str| named(&ten, name).1;

    // **Six of the ten write their sky colour and their fog colour as the same number**, which is
    // the file saying the medium is what the sky is. Nothing here tells them apart by name: the fit
    // lands on one because there is nothing left of the sky for the dome to contribute.
    for name in [
        "overcast",
        "rain",
        "thunderstorm",
        "ashstorm",
        "snow",
        "blizzard",
    ] {
        assert!(
            (of(name).veil.amount - 1.0).abs() < 1e-3,
            "{name} writes one colour twice and should be wholly its own medium"
        );
    }

    // **Blight and foggy write two colours, and land between.** Blight's sky is a paler red than
    // its fog and foggy's is a paler blue than its own, so in both the dome still shows through.
    for name in ["blight", "foggy"] {
        let amount = of(name).veil.amount;
        assert!(
            (0.2..0.95).contains(&amount),
            "{name} should be part medium and part sky, not {amount:.2}"
        );
    }

    // **Clear and cloudy are refused, and that is the whole point of the fit answering for
    // itself.** Their skies are *bluer* than either this renderer's dome or their own fog, so no
    // amount of their medium reaches them — an unconstrained fit runs past one and a clamped one
    // would assert a full veil for exactly the two weathers that should have none, which would put
    // a flat authored swatch over a Rayleigh dome.
    for name in ["clear", "cloudy"] {
        assert_eq!(
            of(name).veil.amount,
            0.0,
            "{name}'s sky is not one a medium explains"
        );
    }
}

#[test]
fn the_ground_takes_the_colour_of_the_weather_over_it() {
    let ten = measured();
    if ten.is_empty() {
        return;
    }
    let of = |name: &str| named(&ten, name).1.ambient;
    // How far the brightest channel stands over the dimmest, which is all "is this coloured" means.
    let spread = |colour: Vec3| colour.max_element() / colour.min_element().max(1e-6);

    // **A dust storm lights the ground the colour of its dust.** Both of these were a pale blue
    // before the veil, under a red fog they had nothing to do with — which is the defect this is
    // the fix for.
    for name in ["blight", "ashstorm"] {
        let ambient = of(name);
        assert!(
            ambient.x > ambient.y * 2.0 && ambient.x > ambient.z * 2.0,
            "{name} should light the ground red, not {ambient:?}"
        );
    }

    // **And an overcast day is grey, which is the failure mode this replaced.** Letting each
    // weather's sky move the deck as a departure from clear's turned overcast orange — the
    // departure is (2.31, 1.13, 0.46), because dividing a grey by clear's blue is warm however it
    // is normalised. Blight is 5.7 here and that ratio was about 5, so a wide bound still catches
    // it.
    for name in ["overcast", "snow", "rain"] {
        let ambient = of(name);
        assert!(
            spread(ambient) < 1.5,
            "{name} should stay grey, not {ambient:?} at {:.1} to one",
            spread(ambient)
        );
    }

    // Clear keeps the physical model's blue sky and the blue light under it, untouched by any of
    // this — its veil is refused, so this is the same number it was before there was one.
    let clear = of("clear");
    assert!(clear.z > clear.x, "a clear day is lit blue, not {clear:?}");
}
