use super::*;

#[test]
fn the_shelf_carries_only_what_it_is_deep_enough_for() {
    // Kitaigorodskii's term, at its three parts. A wave far too long for the depth is cut to
    // nothing, one short enough not to feel the bottom passes untouched, and between them the
    // knee is continuous — a step there would draw a line across the water.
    let shelf = SeaState {
        depth: 300.0,
        ..SeaState::default()
    };
    let scaled = |s: f32| s / (shelf.depth / GRAVITY).sqrt();

    assert!(
        shelf.depth_factor(scaled(0.2)) < 0.03,
        "a long swell is stopped"
    );
    assert_eq!(
        shelf.depth_factor(scaled(2.5)),
        1.0,
        "short waves pass whole"
    );
    // Both halves agree where they meet, and the curve only ever climbs.
    let knee = shelf.depth_factor(scaled(1.0));
    assert!(
        (knee - 0.5).abs() < 1e-5,
        "the knee is at a half, got {knee}"
    );
    let mut last = 0.0;
    for step in 0..=40 {
        let here = shelf.depth_factor(scaled(step as f32 * 0.075));
        assert!(here >= last - 1e-6, "the attenuation fell at {step}");
        last = here;
    }
}

#[test]
fn the_spread_is_narrow_at_the_swell_and_broad_above_it() {
    // The property §7.15 arrived at from a symptom, in its empirical form: a swell arrives as
    // parallel trains and the chop riding on it does not. Large is narrow.
    let at_peak = spread_of(1.0);
    let chop = spread_of(6.0);
    assert!(
        at_peak > 2.0,
        "the swell should be a narrow fan, got {at_peak}"
    );
    assert!(chop < 0.6, "the chop should be a wide one, got {chop}");
    assert!(at_peak > 3.0 * chop);

    // The quantiles of that spread put half the energy either side of the wind, and put more of
    // it off-axis when the fan is wider.
    assert!(
        turn_of(0.5, at_peak).abs() < 1e-6,
        "the median is the wind itself"
    );
    assert!(turn_of(0.9, chop).abs() > turn_of(0.9, at_peak).abs());
}

#[test]
fn the_waves_carry_the_height_and_the_dispersion_they_were_asked_for() {
    let sea = SeaState::default();
    let waves = sea.waves();

    // Significant height is four times the root of the variance, which is the sum of `A^2 / 2`
    // over independent sinusoids. Building the spectrum and then scaling to this is the whole
    // reason `alpha` never appears.
    let variance: f32 = waves.iter().map(|w| w.amplitude * w.amplitude / 2.0).sum();
    let height = 4.0 * variance.sqrt();
    assert!(
        (height - sea.significant_height).abs() < 1e-3,
        "asked for {} and built {height}",
        sea.significant_height
    );

    for wave in &waves {
        assert!(wave.amplitude > 0.0, "a component carries nothing");
        let direction = Vec2::from(wave.direction);
        assert!((direction.length() - 1.0).abs() < 1e-5);
        // Every component obeys the relation its own speed was taken from.
        let expected = (GRAVITY * wave.wavenumber * (wave.wavenumber * sea.depth).tanh()).sqrt();
        assert!(
            (wave.speed - expected).abs() < 1e-3 * expected,
            "speed {} against {expected} for k {}",
            wave.speed,
            wave.wavenumber
        );
    }
}

#[test]
fn shallow_water_slows_a_wave_and_deep_water_does_not() {
    // The reason this is TMA rather than JONSWAP. A wave whose orbit reaches the bottom is held
    // back; one that never feels it travels at the deep-water speed whatever the chart says.
    let long = std::f32::consts::TAU / 420.0;
    let short = std::f32::consts::TAU / 8.0;
    let deep = SeaState {
        depth: 4000.0,
        ..SeaState::default()
    };
    // Half a metre of water under a six-metre swell, which is a fourteenth of its own length.
    let shallow = SeaState {
        depth: 30.0,
        ..SeaState::default()
    };

    let open = (GRAVITY * long).sqrt();
    assert!((deep.frequency_of(long) - open).abs() < 1e-3 * open);
    // A swell in water a fourteenth of its length deep travels at about two thirds of its open
    // speed; a tenth of its length costs it a sixth, and half a length costs it nothing at all.
    let dragged = shallow.frequency_of(long) / open;
    assert!(
        (dragged - 0.649).abs() < 0.01,
        "expected about two thirds of the open speed, got {dragged}"
    );
    // A short wave cannot tell the two apart.
    assert!(
        (shallow.frequency_of(short) - deep.frequency_of(short)).abs()
            < 1e-3 * deep.frequency_of(short)
    );
}

#[test]
fn the_table_matches_the_one_the_shader_walks() {
    // The shader loops to a bound of its own and reads a struct laid out by hand. A count that
    // disagreed would walk off the end of the block; a layout that disagreed would read a
    // direction out of an amplitude, and neither shows up as anything but wrong-looking water.
    assert_eq!(WAVE_COUNT, 32);
    assert_eq!(size_of::<GpuWave>(), 20);
    assert_eq!(WAVE_COUNT % BANDS, 0, "every band takes the same share");
}

#[test]
fn the_swell_arrives_in_a_narrow_fan_and_the_chop_in_a_wide_one() {
    // The lattice §7.15 was about comes from a handful of components sharing directions, and
    // what prevents it is the spread being *wide where it matters*: curvature is dominated by
    // the shortest waves, so those are the ones that have to disagree about which way they are
    // going. Donelan-Banner says they do, and this is that end to end.
    let waves = SeaState::default().waves();
    let fan_of = |band: usize| {
        let slice = &waves[band * PER_BAND..(band + 1) * PER_BAND];
        let mut widest = 0.0f32;
        for (index, wave) in slice.iter().enumerate() {
            for other in &slice[index + 1..] {
                let cosine = Vec2::from(wave.direction)
                    .dot(Vec2::from(other.direction))
                    .clamp(-1.0, 1.0);
                widest = widest.max(cosine.acos().to_degrees());
            }
        }
        widest
    };

    let swell = fan_of(0);
    let chop = fan_of(BANDS - 1);
    // Sixty-eight degrees across four samples of the lowest band, which sits below the peak
    // where the spread has already begun to open; the chop above it fans past a half turn.
    assert!(
        swell < 80.0,
        "the swell should arrive in a narrow fan, got {swell} degrees"
    );
    assert!(
        chop > 120.0,
        "the chop should be a wide one, got {chop} degrees"
    );

    // And nothing is a duplicate of anything, in any band.
    for (index, wave) in waves.iter().enumerate() {
        for other in &waves[index + 1..] {
            assert_ne!(wave.direction, other.direction);
        }
    }
}
