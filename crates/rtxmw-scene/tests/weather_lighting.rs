//! That a weather dims the ground the way the game says it does.
//!
//! **The ini's `Ambient` schedule is the check, not the source.** This renderer derives what lights
//! the ground from the dome's own average, dimmed by however much of it the cloud deck hides — no
//! authored figure anywhere. Bethesda wrote down an answer to the same question for each of the ten,
//! and the two agreeing is evidence the derivation is right; the two disagreeing would be worth
//! knowing about either way.

use rtxmw_scene::{CloudSheet, Sky, SkyTextures, Weather, WorldTime};

/// What lights the ground under `weather` at noon, and what the ini says it should be against clear.
struct Measured {
    name: String,
    derived: f32,
    authored: f32,
}

fn luminance(colour: glam::Vec3) -> f32 {
    colour.x * 0.2126 + colour.y * 0.7152 + colour.z * 0.0722
}

fn measure() -> Vec<Measured> {
    let Ok(table) = Weather::table() else {
        return Vec::new();
    };
    let noon = WorldTime::hours(12.0);
    let sheet = |weather: &Weather| {
        SkyTextures::load(weather)
            .ok()
            .flatten()
            .map_or(CloudSheet::NONE, |t| t.sheet())
    };
    let ambient = |weather: &Weather| luminance(Sky::under(noon, weather, sheet(weather)).ambient);
    let Some(clear) = table.iter().find(|w| w.name == "clear") else {
        return Vec::new();
    };
    let (base, authored_base) = (ambient(clear), luminance(clear.ambient.at(noon)));
    table
        .iter()
        .map(|weather| Measured {
            name: weather.name.clone(),
            derived: ambient(weather) / base,
            authored: luminance(weather.ambient.at(noon)) / authored_base,
        })
        .collect()
}

#[test]
fn a_deck_dims_the_ground_and_a_wisp_barely_does() {
    let measured = measure();
    if measured.is_empty() {
        return;
    }
    for m in &measured {
        println!(
            "{:>13}  derived {:.2}x  the ini says {:.2}x",
            m.name, m.derived, m.authored
        );
    }

    let of = |name: &str| {
        measured
            .iter()
            .find(|m| m.name == name)
            .unwrap_or_else(|| panic!("no {name}"))
    };
    // Clear is the reference, so it is one by construction — the assertion is that nothing has
    // quietly made it something else.
    assert!((of("clear").derived - 1.0).abs() < 1e-5);

    // **Where the darkness comes from a deck, the derivation lands on the game's own answer.**
    // Foggy comes out at 0.55 against an authored 0.55; overcast, snow and blizzard at 0.36 against
    // 0.44. Nothing here was told any of those: the deck's own alpha says how much of the dome it
    // hides and `SKYLIT` says what gets through.
    for name in ["foggy", "overcast", "snow", "blizzard", "rain"] {
        let m = of(name);
        assert!(
            (m.derived - m.authored).abs() < 0.1,
            "{name}: derived {:.2}x against the ini's {:.2}x",
            m.derived,
            m.authored
        );
    }

    // **Where it does not, the gap is worth knowing rather than closing.** A dust storm is dark
    // because the air is full of ash, not because a deck is over it — the ini simply asserts 0.14
    // for ashstorm and 0.17 for blight, and this renderer has no airborne dust to derive that from.
    // What it has is their fog, thicker than clear's at 1.1 against 0.69, which darkens the picture
    // without darkening the light. Same for thunderstorm, whose 0.38 is far under the 0.66 cloud
    // cover it declares.
    for name in ["ashstorm", "blight", "thunderstorm"] {
        let m = of(name);
        assert!(m.derived < 1.0, "{name} should still be dimmer than clear");
        assert!(
            m.authored < m.derived,
            "{name}: the ini used to want it darker than the deck explains — {:.2}x against {:.2}x",
            m.authored,
            m.derived
        );
    }

    // **And cloudy is where the ini looks copied.** Its ambient is clear's to within 6% despite
    // declaring three-quarters cloud cover, which is the one weather whose authored answer this
    // does not believe.
    assert!((of("cloudy").authored - 1.0).abs() < 0.1);
    assert!(of("cloudy").derived < 0.7);
}
