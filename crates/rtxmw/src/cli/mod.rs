//! The command line: two commands, which of them was asked for, and the parsers that turn their
//! words into values.
//!
//! Apart from `main.rs`, which is the entry point and nothing else — this is the bulk of the
//! binary's own code and half of it is tests, and neither has to be read past to find the other.

use std::path::PathBuf;

use ash::vk;
use clap::{CommandFactory, Parser};
use rtxmw_render::dlss::Preset;
use rtxmw_scene::{CellId, WorldTime};

use crate::scene_loader;
use crate::scene_loader::ViewpointOverride;

#[cfg(test)]
mod tests;

/// What the arguments ask the binary to do.
#[derive(Debug)]
pub(crate) enum Command {
    Window(WindowOptions),
    Screenshot(ScreenshotOptions),
    TextureSheet(TextureSheetOptions),
}

impl Command {
    /// Reads the arguments, program name and all.
    ///
    /// **Which command was asked for is decided here rather than by clap**, because the two take
    /// different positionals — a cell, against a size and then a cell — and no argument grammar can
    /// say that one slot means a different thing depending on a flag elsewhere.
    ///
    /// Exits the process on a bad argument or on `--help`, which is what clap's own `parse_from`
    /// does and what a command line is expected to do.
    pub(crate) fn parse_from(arguments: &[String]) -> Self {
        if names(arguments, SHEET_FLAG) {
            return Self::TextureSheet(command_or_help::<TextureSheetOptions>(arguments));
        }
        if !names(arguments, SCREENSHOT_FLAG) {
            return Self::Window(WindowOptions::parse_from(arguments));
        }
        Self::Screenshot(command_or_help::<ScreenshotOptions>(arguments))
    }
}

/// Parses `arguments` as `T`, printing `T`'s help where that is what they asked for.
///
/// **`--screenshot` and `--textures` take a value, so clap meets a `--help` beside one by reporting
/// the value missing.** Technically correct and useless: there is nothing else the two together
/// could mean. Behind a failure rather than in front of the parse, so that this crate never
/// adjudicates an argument clap was willing to read.
fn command_or_help<T: Parser + CommandFactory>(arguments: &[String]) -> T {
    match T::try_parse_from(arguments) {
        Ok(options) => options,
        Err(_) if names_help(arguments) => {
            T::command()
                .print_long_help()
                .expect("the help text should reach stdout");
            std::process::exit(0);
        }
        Err(failed) => failed.exit(),
    }
}

/// The flag that selects the offscreen render, and the one that selects the texture sheet.
const SCREENSHOT_FLAG: &str = "--screenshot";
const SHEET_FLAG: &str = "--textures";

/// Whether the arguments name `flag`, in either the spaced or the `=` spelling.
///
/// **Anywhere among them, not first.** Requiring a command's flag to lead was a rule with nothing
/// behind it once a parser read the rest, so `--frames 8 --screenshot out.png` works.
///
/// A *positional* before it still binds by that command's own grammar, where the first one is the
/// size — `-2,-9 --screenshot out.png` is a size that will not parse, not a cell. Nothing can fix
/// that: the commands disagree about what the first positional means, which is why they are
/// separate.
///
/// Scanning stops at a bare `--`, which ends option parsing: past it the same word is a value.
fn names(arguments: &[String], flag: &str) -> bool {
    let equals = format!("{flag}=");
    options(arguments).any(|argument| argument == flag || argument.starts_with(&equals))
}

/// Whether the arguments ask for help, in either of the two spellings clap accepts.
fn names_help(arguments: &[String]) -> bool {
    options(arguments).any(|argument| argument == "--help" || argument == "-h")
}

/// The arguments that are still options, for a caller reading them before a parser does.
///
/// A bare `--` ends option parsing, so everything past it is a value however it is spelled — and a
/// value someone went to the trouble of writing `--` for is the last thing to mistake for a flag.
fn options(arguments: &[String]) -> impl Iterator<Item = &String> {
    arguments.iter().take_while(|argument| *argument != "--")
}

/// Resolution of a `--screenshot` render when none is given, independent of any window.
///
/// A string because that is what a default on the command line is: clap reads it back through
/// [`size`], so a value this parser would reject cannot reach the renderer — `debug_assert` on the
/// command catches it before any argument is looked at.
///
/// Not the internal resolution the engine targets — see `docs/design.md` §5.3, which is 1920x1080 —
/// because a screenshot is usually being looked at rather than measured, and 720p renders faster
/// and reads the same. Pass `1920x1080` to measure against the budget.
const SCREENSHOT_SIZE: &str = "1280x720";

/// Reads `WIDTHxHEIGHT`.
///
/// **`vk::Extent2D` rather than a size of this module's own**, which is what the whole binary means
/// by a width and a height — the alternative was a second such type that every caller converted out
/// of within two lines.
///
/// One message for a missing separator and for an unparsable number alike: on its own `parse` would
/// report "invalid digit found in string" without saying what it was reading.
pub(super) fn size(value: &str) -> Result<vk::Extent2D, String> {
    let malformed = || format!("expected a size like 1920x1080, got {value:?}");
    let (width, height) = value.split_once('x').ok_or_else(malformed)?;
    Ok(vk::Extent2D {
        width: width.parse().map_err(|_| malformed())?,
        height: height.parse().map_err(|_| malformed())?,
    })
}

/// Reads a cell: a pair of integers is an exterior, anything else an interior's name.
///
/// **A leading hyphen has to be allowed and a leading double hyphen refused.** Half the exteriors
/// have a negative coordinate, so `-2,-9` must reach here rather than be read as short options —
/// but the flag that buys that makes every mistyped `--flag` a perfectly good interior name, and
/// silently opening the default cell is the worst answer to a typo.
fn cell(value: &str) -> Result<CellId, String> {
    if value.starts_with("--") {
        return Err(format!("unknown argument {value:?}"));
    }
    Ok(scene_loader::cell_named(value))
}

/// What the upscaler should do, including nothing.
///
/// A type of its own rather than `Option<Preset>` because clap reads an `Option` field as "this
/// argument may be absent", and absent is exactly what this must not mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Upscaling(pub(crate) Option<Preset>);

/// Environment variable, and `.env` key, choosing what DLSS runs at.
const UPSCALING_VAR: &str = "RTXMW_DLSS";

/// What DLSS runs at when nothing says otherwise.
///
/// **Quality rather than the Performance mode §5.3 budgets for.** The default build is the one to
/// look at rather than the one to measure; the budget is still reachable by name.
const UPSCALING_DEFAULT: &str = "quality";

/// Reads an upscaling mode: a preset's name, or `off` for none at all.
fn upscaling(value: &str) -> Result<Upscaling, String> {
    match value {
        "off" | "0" => Ok(Upscaling(None)),
        // Not by recursing through this function: a default that named `on` would then recurse
        // forever, and the test that pins the default would hang instead of failing. A default that
        // names no preset is this crate contradicting itself, so it says so and stops.
        "on" | "1" => Ok(Upscaling(Some(
            Preset::named(UPSCALING_DEFAULT).expect("the built-in default names a preset"),
        ))),
        name => Preset::named(name)
            .map(|preset| Upscaling(Some(preset)))
            .ok_or_else(|| {
                format!("expected off, performance, balanced, quality or dlaa, got {value:?}")
            }),
    }
}

/// What `.env` says the default should be, or the built-in one.
///
/// **The one place a file is consulted for a setting.** clap covers the flag and the variable
/// between them, so the order ends up flag, then variable, then `.env`, then this.
fn upscaling_default() -> &'static str {
    static CHOSEN: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CHOSEN.get_or_init(|| {
        rtxmw_vfs::from_dotenv(UPSCALING_VAR).unwrap_or_else(|| UPSCALING_DEFAULT.to_owned())
    })
}

/// Reads a de-lighting strength on `0..=1`.
///
/// Refused outside that rather than clamped: 2 is not a stronger wish, it is a misunderstanding of
/// what the number means, and dividing a texture by the square of an estimate is not on the scale.
fn strength(value: &str) -> Result<f32, String> {
    match value.parse::<f32>() {
        Ok(strength) if (0.0..=1.0).contains(&strength) => Ok(strength),
        _ => Err(format!("expected a strength from 0 to 1, got {value:?}")),
    }
}

/// Reads a frame count, rejecting zero.
///
/// Zero can only be a mistake, and rendering one frame for it would answer a question nobody asked.
fn at_least_one(value: &str) -> Result<u32, String> {
    match value.parse() {
        Ok(0) | Err(_) => Err(format!("expected a count of one or more, got {value:?}")),
        Ok(count) => Ok(count),
    }
}

/// Reads an hour of the game's day, as `9.5` or as `9:30`.
///
/// The colon form is the one anybody would reach for and the decimal one is what the clock actually
/// is, so both are taken. Any hour is legal, including the small ones nobody can see anything in.
fn hour(value: &str) -> Result<WorldTime, String> {
    let hours = match value.split_once(':') {
        Some((hours, minutes)) => match (hours.parse::<f32>(), minutes.parse::<f32>()) {
            (Ok(hours), Ok(minutes)) if (0.0..60.0).contains(&minutes) => {
                Ok(hours + minutes / 60.0)
            }
            _ => Err(()),
        },
        None => value.parse::<f32>().map_err(|_| ()),
    };
    match hours {
        Ok(hours) if hours.is_finite() => Ok(WorldTime::hours(hours)),
        _ => Err(format!(
            "expected an hour of the day like 9.5 or 9:30, got {value:?}"
        )),
    }
}

/// What the windowed mode was asked for.
#[derive(Parser, Debug, Clone, PartialEq)]
#[command(
    about = "A raytraced Morrowind engine.",
    after_help = concat!(
        "Add --screenshot <PATH> to render offscreen and exit, opening no window,\n",
        "or --textures <PATH> to write a contact sheet of a cell's textures.\n",
        "Their own arguments are at `rtxmw --screenshot --help` and `rtxmw --textures --help`.",
    )
)]
pub(crate) struct WindowOptions {
    /// Where to start: a coordinate pair like -2,-9 is an exterior, anything else names an
    /// interior.
    // `allow_hyphen_values` is what lets a negative coordinate through — half the exteriors have
    // one — and `cell` is what stops that from making every mistyped flag an interior's name.
    #[arg(
        value_name = "CELL",
        default_value = scene_loader::DEFAULT_CELL,
        value_parser = cell,
        allow_hyphen_values = true,
    )]
    pub(crate) cell: CellId,
    /// Draw this many frames, then exit through the ordinary shutdown path.
    #[arg(long = "frames", value_name = "N", value_parser = at_least_one)]
    pub(crate) exit_after: Option<u32>,
    /// How much of the lighting painted into each texture to divide back out, from 0 to 1.
    // A negative strength is nonsense, but it should be refused for saying so rather than for
    // looking like a flag — which is what clap does with a leading minus unless told otherwise.
    #[arg(
        long,
        value_name = "STRENGTH",
        default_value_t = 1.0,
        value_parser = strength,
        allow_negative_numbers = true,
    )]
    pub(crate) delight: f32,
    /// How much of the cell's own fog to apply, from 0 to 1.
    #[arg(
        long,
        value_name = "STRENGTH",
        default_value_t = 1.0,
        value_parser = strength,
        allow_negative_numbers = true,
    )]
    pub(crate) fog: f32,
    /// The hour to light an exterior at: 6 is sunrise, 12 noon, 18 sunset, anything else night,
    /// and past 24 the next day — which is how a still reaches a moon phase other than the first's.
    #[arg(long = "time", value_name = "HOUR", default_value_t, value_parser = hour)]
    pub(crate) time: WorldTime,
    /// What DLSS runs at: off, performance, balanced, quality or dlaa.
    #[arg(
        long = "dlss",
        env = UPSCALING_VAR,
        value_name = "MODE",
        default_value = upscaling_default(),
        value_parser = upscaling,
    )]
    pub(crate) dlss: Upscaling,
    /// Which of the game's ten weathers to stand under: clear, cloudy, foggy, overcast, rain,
    /// thunderstorm, ashstorm, blight, snow or blizzard
    #[arg(long = "weather", value_name = "NAME", default_value = "clear")]
    pub(crate) weather: String,
}

/// What an offscreen render was asked for.
#[derive(Parser, Debug, Clone, PartialEq)]
#[command(about = "Renders offscreen and exits.")]
pub(crate) struct ScreenshotOptions {
    /// Render offscreen to this PNG and exit, opening no window.
    #[arg(long = "screenshot", value_name = "PATH")]
    pub(crate) path: PathBuf,
    /// The size to render, as WIDTHxHEIGHT. Give it before the cell.
    // Ordered, because nothing else could tell the two apart: an interior's name can be any string
    // at all, `1920x1080` included.
    #[arg(value_name = "WIDTHxHEIGHT", default_value = SCREENSHOT_SIZE, value_parser = size)]
    pub(crate) size: vk::Extent2D,
    /// What to render: a coordinate pair like -2,-9 is an exterior, anything else names an
    /// interior.
    #[arg(
        value_name = "CELL",
        default_value = scene_loader::DEFAULT_CELL,
        value_parser = cell,
        allow_hyphen_values = true,
    )]
    pub(crate) cell: CellId,
    #[command(flatten)]
    pub(crate) viewpoint: ViewpointOverride,
    /// Render this many frames, holding the camera still, and write the last.
    ///
    /// One frame is a temporal upscaler's worst case rather than its output: it resolves detail
    /// across frames and is told to reset on the first. Nothing else changes with the count.
    #[arg(long, value_name = "N", default_value_t = 1, value_parser = at_least_one)]
    pub(crate) frames: u32,
    /// Indirect samples per pixel, instead of the renderer's default.
    ///
    /// A reference render turns this up far enough that the noise it is a reference for is gone.
    #[arg(long, value_name = "N")]
    pub(crate) samples: Option<u32>,
    /// À-trous denoising passes, instead of the renderer's default. Zero leaves the light as
    /// traced.
    ///
    /// A filtered reference is not one: the filter trades bias for variance, which is the right
    /// trade at four samples and the wrong one at a thousand.
    #[arg(long, value_name = "N")]
    pub(crate) denoise: Option<u32>,
    /// How much of the lighting painted into each texture to divide back out, from 0 to 1.
    // A negative strength is nonsense, but it should be refused for saying so rather than for
    // looking like a flag — which is what clap does with a leading minus unless told otherwise.
    #[arg(
        long,
        value_name = "STRENGTH",
        default_value_t = 1.0,
        value_parser = strength,
        allow_negative_numbers = true,
    )]
    pub(crate) delight: f32,
    /// How much of the cell's own fog to apply, from 0 to 1.
    #[arg(
        long,
        value_name = "STRENGTH",
        default_value_t = 1.0,
        value_parser = strength,
        allow_negative_numbers = true,
    )]
    pub(crate) fog: f32,
    /// The hour to light an exterior at: 6 is sunrise, 12 noon, 18 sunset, anything else night,
    /// and past 24 the next day — which is how a still reaches a moon phase other than the first's.
    #[arg(long = "time", value_name = "HOUR", default_value_t, value_parser = hour)]
    pub(crate) time: WorldTime,
    /// What DLSS runs at: off, performance, balanced, quality or dlaa.
    #[arg(
        long = "dlss",
        env = UPSCALING_VAR,
        value_name = "MODE",
        default_value = upscaling_default(),
        value_parser = upscaling,
    )]
    pub(crate) dlss: Upscaling,
    /// Which of the game's ten weathers to stand under: clear, cloudy, foggy, overcast, rain,
    /// thunderstorm, ashstorm, blight, snow or blizzard
    #[arg(long = "weather", value_name = "NAME", default_value = "clear")]
    pub(crate) weather: String,
}

/// What a texture sheet was asked for.
#[derive(Parser, Debug, Clone, PartialEq)]
#[command(about = "Writes a contact sheet of a cell's textures and exits.")]
pub(crate) struct TextureSheetOptions {
    /// Write the sheet to this PNG: every texture the cell uses, vanilla left, de-lit right.
    #[arg(long = "textures", value_name = "PATH")]
    pub(crate) path: PathBuf,
    /// Whose textures: a coordinate pair like -2,-9 is an exterior, anything else names an
    /// interior.
    #[arg(
        value_name = "CELL",
        default_value = scene_loader::DEFAULT_CELL,
        value_parser = cell,
        allow_hyphen_values = true,
    )]
    pub(crate) cell: CellId,
}
