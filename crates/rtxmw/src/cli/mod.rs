//! The command line: two commands, which of them was asked for, and the parsers that turn their
//! words into values.
//!
//! Apart from `main.rs`, which is the entry point and nothing else — this is the bulk of the
//! binary's own code and half of it is tests, and neither has to be read past to find the other.

use std::path::PathBuf;

use ash::vk;
use clap::{CommandFactory, Parser};
use rtxmw_scene::CellId;

use crate::scene_loader;
use crate::scene_loader::ViewpointOverride;

#[cfg(test)]
mod tests;

/// What the arguments ask the binary to do.
#[derive(Debug)]
pub(crate) enum Command {
    Window(WindowOptions),
    Screenshot(ScreenshotOptions),
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
        if !names_screenshot(arguments) {
            return Self::Window(WindowOptions::parse_from(arguments));
        }
        match ScreenshotOptions::try_parse_from(arguments) {
            Ok(options) => Self::Screenshot(options),
            // **`--screenshot` takes a value, so clap meets a `--help` beside it by reporting the
            // value missing.** Technically correct and useless: there is nothing else the two
            // together could mean.
            //
            // Behind a failure rather than in front of the parse, so that this crate never
            // adjudicates an argument clap was willing to read. The two happen to agree on what
            // `-h` is today; only one of them is the authority on it.
            Err(_) if names_help(arguments) => {
                ScreenshotOptions::command()
                    .print_long_help()
                    .expect("the help text should reach stdout");
                std::process::exit(0);
            }
            Err(failed) => failed.exit(),
        }
    }
}

/// Whether the arguments name the offscreen command.
///
/// **Anywhere among them, not first.** Requiring this one flag to lead was a rule with nothing
/// behind it once a parser read the rest, so `--frames 8 --screenshot out.png` now works.
///
/// A *positional* before it still binds by the offscreen grammar, where the first one is the size —
/// `-2,-9 --screenshot out.png` is a size that will not parse, not a cell. Nothing can fix that: the
/// two commands disagree about what the first positional means, which is why they are two.
///
fn names_screenshot(arguments: &[String]) -> bool {
    options(arguments)
        .any(|argument| argument == "--screenshot" || argument.starts_with("--screenshot="))
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

/// Reads a frame count, rejecting zero.
///
/// Zero can only be a mistake, and rendering one frame for it would answer a question nobody asked.
fn at_least_one(value: &str) -> Result<u32, String> {
    match value.parse() {
        Ok(0) | Err(_) => Err(format!("expected a count of one or more, got {value:?}")),
        Ok(count) => Ok(count),
    }
}

/// What the windowed mode was asked for.
#[derive(Parser, Debug, Clone, PartialEq)]
#[command(
    about = "A raytraced Morrowind engine.",
    after_help = concat!(
        "Add --screenshot <PATH> to render offscreen and exit, opening no window.\n",
        "Its own arguments are at `rtxmw --screenshot --help`.",
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
}
