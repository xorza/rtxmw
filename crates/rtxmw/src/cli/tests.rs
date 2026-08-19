//! What each command accepts, and what it says when it will not.

use ash::vk;
use clap::{CommandFactory, Parser};
use glam::Vec3;
use rtxmw_render::dlss::Preset;
use rtxmw_scene::CellId;

use crate::cli::{
    SCREENSHOT_FLAG, SCREENSHOT_SIZE, SHEET_FLAG, ScreenshotOptions, UPSCALING_DEFAULT, Upscaling,
    WindowOptions, names, names_help, size, upscaling,
};
use crate::scene_loader;
use crate::scene_loader::ViewpointOverride;
use std::path::PathBuf;

/// Parses windowed arguments as the shell would hand them over, program name and all.
///
/// The failure is rendered to its message here because `clap::Error` is not comparable, and
/// what a test has to say about a rejection is what it told the reader anyway.
fn parse(arguments: &[&str]) -> Result<WindowOptions, String> {
    WindowOptions::try_parse_from(std::iter::once("rtxmw").chain(arguments.iter().copied()))
        .map(|mut options| {
            options.dlss = Upscaling(None);
            options
        })
        .map_err(|failed| failed.to_string())
}

fn screenshot_options(arguments: &[&str]) -> Result<ScreenshotOptions, String> {
    ScreenshotOptions::try_parse_from(
        ["rtxmw", "--screenshot"]
            .into_iter()
            .chain(arguments.iter().copied()),
    )
    .map(|mut options| {
        options.dlss = Upscaling(None);
        options
    })
    .map_err(|failed| failed.to_string())
}

/// **`dlss` is flattened away by both helpers above**, because it reads the process environment
/// through clap: pinning it would fail every assertion here on a machine that has `RTXMW_DLSS` set,
/// and none of them is about that setting. It has a test of its own at the end.
fn interior(name: &str) -> CellId {
    CellId::Interior(name.to_owned())
}

/// Arguments as the shell hands them over, program name and all.
fn argv(arguments: &[&str]) -> Vec<String> {
    std::iter::once("rtxmw")
        .chain(arguments.iter().copied())
        .map(str::to_owned)
        .collect()
}

#[test]
fn the_offscreen_command_is_named_wherever_the_flag_falls() {
    // Leading is how it has always been written and stays valid; the rest is what stopped being a
    // rule when clap took over reading the arguments.
    for arguments in [
        vec!["--screenshot", "out.png"],
        vec!["--frames", "8", "--screenshot", "out.png"],
        vec!["--screenshot=out.png"],
        vec!["--samples", "1", "--screenshot=out.png", "1920x1080"],
    ] {
        assert!(
            names(&argv(&arguments), SCREENSHOT_FLAG),
            "{arguments:?} names the offscreen command"
        );
    }
    for arguments in [
        vec![],
        vec!["-2,-9"],
        vec!["--frames", "8"],
        // A near miss, which must not count: this is a cell whose name begins the same way.
        vec!["--screenshots"],
        // **Past a bare `--` the word is a value, not a flag.** Dispatching on it there would send
        // an argument the user explicitly quoted to the wrong command.
        vec!["--", "--screenshot"],
    ] {
        assert!(
            !names(&argv(&arguments), SCREENSHOT_FLAG),
            "{arguments:?} does not name the offscreen command"
        );
    }
}

#[test]
fn the_sheet_and_the_screenshot_are_told_apart() {
    // Two commands selected the same way, so neither may answer for the other — and the sheet is
    // checked first, since asking for both is a contradiction rather than a render.
    assert!(names(&argv(&["--textures", "sheet.png"]), SHEET_FLAG));
    assert!(!names(&argv(&["--textures", "sheet.png"]), SCREENSHOT_FLAG));
    assert!(!names(&argv(&["--screenshot", "out.png"]), SHEET_FLAG));
    // A near miss must not count: this is a cell whose name begins the same way.
    assert!(!names(&argv(&["--texturesque"]), SHEET_FLAG));
    assert!(names(&argv(&["--textures=sheet.png"]), SHEET_FLAG));
}

#[test]
fn help_is_recognised_in_both_spellings() {
    // `--screenshot` takes a value, so clap meets `--screenshot --help` by reporting the value
    // missing. Catching the request first is the only way that answers what was asked.
    for arguments in [vec!["--help"], vec!["-h"], vec!["--screenshot", "--help"]] {
        assert!(names_help(&argv(&arguments)), "{arguments:?} asks for help");
    }
    for arguments in [
        vec!["--screenshot", "out.png"],
        vec!["--helpful"],
        vec!["--", "--help"],
    ] {
        assert!(!names_help(&argv(&arguments)), "{arguments:?} does not");
    }
}

#[test]
fn both_commands_are_internally_consistent() {
    // clap's own audit of the declarations: duplicate names, a `default_value_t` that does not
    // survive its own `value_parser`, an argument index that cannot be reached. It panics on
    // any of them, and none of it is visible until something asks.
    WindowOptions::command().debug_assert();
    ScreenshotOptions::command().debug_assert();
}

#[test]
fn the_cell_and_the_frame_limit_go_in_either_order() {
    let default = interior(scene_loader::DEFAULT_CELL);
    assert_eq!(
        parse(&[]),
        Ok(WindowOptions {
            cell: default.clone(),
            exit_after: None,
            delight: 1.0,
            fog: 1.0,
            dlss: Upscaling(None),
        })
    );
    // A flag on its own must not be mistaken for a cell name, which is exactly what happened
    // while the first argument was parsed apart from the rest.
    assert_eq!(
        parse(&["--frames", "3"]),
        Ok(WindowOptions {
            cell: default,
            exit_after: Some(3),
            delight: 1.0,
            fog: 1.0,
            dlss: Upscaling(None),
        })
    );

    let outdoors = WindowOptions {
        cell: CellId::Exterior { x: -2, y: -9 },
        exit_after: Some(3),
        delight: 1.0,
        fog: 1.0,
        dlss: Upscaling(None),
    };
    assert_eq!(parse(&["-2,-9", "--frames", "3"]), Ok(outdoors.clone()));
    assert_eq!(parse(&["--frames", "3", "-2,-9"]), Ok(outdoors));

    // **A negative coordinate is a value, not a cluster of short flags.** Half the exteriors
    // have one, and a parser reading `-2,-9` as options rejects most of the world.
    assert_eq!(
        parse(&["-2,-9"]).expect("should parse").cell,
        CellId::Exterior { x: -2, y: -9 }
    );
    assert!(parse(&["-2,-9", "extra"]).is_err(), "one cell, not two");
    // **A typo is not an interior called `--nonsense`.** Letting a cell start with a hyphen is
    // what makes `-2,-9` reachable, and it would otherwise make every mistyped flag a cell.
    let typo = parse(&["--nonsense"]).expect_err("a mistyped flag is not a cell");
    assert!(
        typo.contains("--nonsense"),
        "a rejected argument should name itself, and said {typo:?}"
    );
}

#[test]
fn a_screenshot_can_be_told_where_to_stand_and_which_way_to_look() {
    // Flagged rather than positional, and so allowed anywhere among the positional arguments:
    // an interior's name can be any string, so a third positional could not be told from it.
    assert_eq!(
        screenshot_options(&["out.png", "--at", "10,20,30", "640x480", "--look", "0,-1,0"]),
        Ok(ScreenshotOptions {
            path: PathBuf::from("out.png"),
            size: vk::Extent2D {
                width: 640,
                height: 480
            },
            cell: interior(scene_loader::DEFAULT_CELL),
            viewpoint: ViewpointOverride {
                position: Some(Vec3::new(10.0, 20.0, 30.0)),
                forward: Some(Vec3::NEG_Y),
            },
            frames: 1,
            samples: None,
            denoise: None,
            delight: 1.0,
            fog: 1.0,
            dlss: Upscaling(None),
        })
    );
    // Either half on its own: what is not said is left to the cell rather than made up here,
    // so asking only to turn on the spot is a thing that can be asked. Negative coordinates
    // reach the parser rather than being read as flags.
    assert_eq!(
        screenshot_options(&["out.png", "--at", "-1,2,-3"])
            .expect("should parse")
            .viewpoint,
        ViewpointOverride {
            position: Some(Vec3::new(-1.0, 2.0, -3.0)),
            forward: None,
        }
    );
    assert_eq!(
        screenshot_options(&["out.png", "--look", "0,0,-1"])
            .expect("should parse")
            .viewpoint,
        ViewpointOverride {
            position: None,
            forward: Some(Vec3::NEG_Z),
        }
    );
    let short = screenshot_options(&["out.png", "--at", "1,2"]).expect_err("two is not three");
    assert!(
        short.contains(r#"expected X,Y,Z, got "1,2""#),
        "a malformed vector should say what it wanted, and said {short:?}"
    );
    assert!(screenshot_options(&["out.png", "--at"]).is_err());
}

#[test]
fn a_screenshot_takes_a_path_then_a_size_then_a_cell() {
    assert_eq!(
        screenshot_options(&["out.png"]),
        Ok(ScreenshotOptions {
            path: PathBuf::from("out.png"),
            // Through the parser, which is also how clap reaches it: a default that this rejects
            // is a default that never renders.
            size: size(SCREENSHOT_SIZE).expect("the default is a size"),
            cell: interior(scene_loader::DEFAULT_CELL),
            viewpoint: ViewpointOverride::default(),
            // One frame and the renderer's own sample count, so the short form is what it
            // always was.
            frames: 1,
            samples: None,
            denoise: None,
            delight: 1.0,
            fog: 1.0,
            dlss: Upscaling(None),
        })
    );
    assert_eq!(
        screenshot_options(&["out.png", "1920x1080", "-2,-9"]),
        Ok(ScreenshotOptions {
            path: PathBuf::from("out.png"),
            size: vk::Extent2D {
                width: 1920,
                height: 1080
            },
            cell: CellId::Exterior { x: -2, y: -9 },
            viewpoint: ViewpointOverride::default(),
            frames: 1,
            samples: None,
            denoise: None,
            delight: 1.0,
            fog: 1.0,
            dlss: Upscaling(None),
        })
    );
    // Flagged like the viewpoint, and for the same reason.
    assert_eq!(
        screenshot_options(&[
            "out.png",
            "--frames",
            "64",
            "3840x2160",
            "--samples",
            "1024",
            "--denoise",
            "0",
            "-2,-9",
        ]),
        Ok(ScreenshotOptions {
            path: PathBuf::from("out.png"),
            size: vk::Extent2D {
                width: 3840,
                height: 2160
            },
            cell: CellId::Exterior { x: -2, y: -9 },
            viewpoint: ViewpointOverride::default(),
            frames: 64,
            samples: Some(1024),
            denoise: Some(0),
            delight: 1.0,
            fog: 1.0,
            dlss: Upscaling(None),
        })
    );

    assert!(
        screenshot_options(&[]).is_err(),
        "the path is what selects this mode, so there is no rendering without one"
    );
    // **Zero frames is rejected, not clamped.** It can only be a mistake, and rendering one
    // frame for it would answer a question nobody asked. Zero à-trous passes is a real request
    // and stays one.
    for (arguments, wanted) in [
        (vec!["out.png", "--frames", "0"], "one or more"),
        (vec!["out.png", "--frames", "soon"], "one or more"),
        (vec!["out.png", "1920"], "a size like 1920x1080"),
        (vec!["out.png", "1920x"], "a size like 1920x1080"),
        (vec!["out.png", "widexhigh"], "a size like 1920x1080"),
    ] {
        let Err(failed) = screenshot_options(&arguments) else {
            panic!("{arguments:?} should not parse");
        };
        assert!(
            failed.contains(wanted),
            "{arguments:?} failed without mentioning {wanted:?}: {failed}"
        );
    }
}

#[test]
fn the_upscaler_defaults_to_on_and_can_be_turned_off() {
    // **A plain run is the engine with everything switched on**, so this setting exists to switch
    // one thing off — which is what an A/B against the unupscaled path needs, and the only reason
    // it is a setting rather than a constant.
    assert_eq!(
        upscaling(UPSCALING_DEFAULT).expect("the built-in default has to parse"),
        Upscaling(Some(Preset::Quality))
    );
    for on in ["on", "1"] {
        assert_eq!(upscaling(on).unwrap(), Upscaling(Some(Preset::Quality)));
    }
    for off in ["off", "0"] {
        assert_eq!(
            upscaling(off).unwrap(),
            Upscaling(None),
            "{off:?} should turn the upscaler off"
        );
    }
    for (name, expected) in [
        ("performance", Preset::Performance),
        ("balanced", Preset::Balanced),
        ("quality", Preset::Quality),
        ("dlaa", Preset::Dlaa),
    ] {
        assert_eq!(upscaling(name).unwrap(), Upscaling(Some(expected)));
    }

    // **Loudly, not silently.** Rendering at a mode nobody asked for is how a typo becomes a
    // measurement of the wrong thing, and the message has to say what would have worked.
    let failed = upscaling("ultra").expect_err("`ultra` is not a mode");
    assert!(
        failed.contains("ultra"),
        "it should name the value: {failed}"
    );
    assert!(
        failed.contains("off") && failed.contains("quality"),
        "it should list what does work: {failed}"
    );
}

#[test]
fn de_lighting_strength_is_a_fraction_or_it_is_refused() {
    // **On by default**: leaving the painted lighting in is the wrong answer for a renderer that
    // lights things itself, so the switch exists to turn the correction *off* for an A/B.
    assert_eq!(screenshot_options(&["out.png"]).unwrap().delight, 1.0);
    for (given, expected) in [("0", 0.0), ("0.5", 0.5), ("1", 1.0)] {
        assert_eq!(
            screenshot_options(&["out.png", "--delight", given])
                .unwrap()
                .delight,
            expected
        );
    }
    // **Refused rather than clamped.** Two is not a stronger wish, it is a misreading of what the
    // number is — dividing by the square of an estimate is not on the scale at all.
    for wrong in ["2", "-0.5", "lots"] {
        let failed = screenshot_options(&["out.png", "--delight", wrong])
            .expect_err("{wrong} is not a strength");
        assert!(
            failed.contains("0 to 1"),
            "{wrong:?} was refused without saying the range: {failed}"
        );
    }
}
