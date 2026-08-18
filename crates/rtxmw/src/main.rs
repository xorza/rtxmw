//! Entry point for the rtxmw engine.

mod app;
mod camera;
mod headless;
mod renderer;
mod scene_loader;

use std::path::PathBuf;

use glam::Vec3;
use rtxmw_scene::CellId;
use winit::event_loop::{ControlFlow, EventLoop};

use crate::app::App;
use crate::scene_loader::ViewpointOverride;

/// Resolution of a `--screenshot` render when none is given, independent of any window.
///
/// Not the internal resolution the engine targets — see `docs/design.md` §5.3, which is 1920x1080 —
/// because a screenshot is usually being looked at rather than measured, and 720p renders faster
/// and reads the same. Pass `1920x1080` to measure against the budget.
const SCREENSHOT_SIZE: (u32, u32) = (1280, 720);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Both modes take the same optional cell: a pair of integers is an exterior and anything else
    // is an interior's name.
    //
    //     rtxmw [CELL] [--frames N]
    //     rtxmw --screenshot <path> [WIDTHxHEIGHT] [CELL] [--at X,Y,Z] [--look X,Y,Z]
    //
    // `--screenshot` renders one frame offscreen and exits, opening no window. The device it brings
    // up has no surface at all, so it works over ssh and in a script.
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments
        .first()
        .is_some_and(|first| first == "--screenshot")
    {
        return screenshot(&arguments[1..]);
    }

    let event_loop = EventLoop::new()?;
    // Redraw continuously rather than only on OS-driven repaint.
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::opening_in(WindowOptions::parse(&arguments)?))?;
    Ok(())
}

/// What the windowed mode was asked for.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WindowOptions {
    pub(crate) cell: CellId,
    /// Frames to draw before exiting, if a count was given.
    pub(crate) exit_after: Option<u32>,
}

impl WindowOptions {
    /// Reads the arguments after the program name.
    ///
    /// The cell is positional and `--frames N` may go either side of it, so neither has to come
    /// first — which is the shape that broke when the first argument was parsed separately from the
    /// rest and skipped the flag checks entirely, taking `--nonsense` for a cell name.
    fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut cell = None;
        let mut exit_after = None;
        let mut rest = arguments.iter();
        while let Some(argument) = rest.next() {
            match argument.as_str() {
                "--frames" => {
                    exit_after = Some(
                        rest.next()
                            .ok_or("--frames needs a count")?
                            .parse()
                            .map_err(|_| "--frames needs a whole number".to_owned())?,
                    );
                }
                flag if flag.starts_with("--") => {
                    return Err(format!("unknown argument {flag:?}"));
                }
                _ if cell.is_some() => {
                    return Err(format!("more than one cell given: {argument:?}"));
                }
                named => cell = Some(named),
            }
        }
        Ok(Self {
            cell: scene_loader::cell_argument(cell),
            exit_after,
        })
    }
}

/// What an offscreen render was asked for.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScreenshotOptions {
    pub(crate) path: PathBuf,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) cell: CellId,
    /// Where to stand and which way to face, in as much as the caller said.
    ///
    /// A screenshot's whole value is being able to look at what someone reported, and what they
    /// report is rarely visible from the doorway — a canopy artefact needs the camera inside the
    /// tree. Without this the only way there is the windowed binary, by hand, unrepeatably.
    pub(crate) viewpoint: ViewpointOverride,
}

impl ScreenshotOptions {
    /// Reads the arguments after `--screenshot`.
    ///
    /// Positional throughout, unlike the windowed mode's: a path, then a size, then a cell. The
    /// size has to precede the cell because an interior's name can be anything at all, so there is
    /// nothing to tell one from the other by.
    ///
    /// `--at X,Y,Z` and `--look X,Y,Z` may follow, and are the one exception — an interior's name
    /// could be anything, so a viewpoint has to be flagged rather than counted.
    fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut rest = arguments.iter();
        let path = PathBuf::from(
            rest.next()
                .ok_or("--screenshot needs a path to write the image to")?,
        );
        let mut positional: Vec<&String> = Vec::new();
        let mut at = None;
        let mut look = None;
        while let Some(argument) = rest.next() {
            let mut vector = |what: &str| -> Result<Vec3, String> {
                let value = rest
                    .next()
                    .ok_or_else(|| format!("{what} needs an X,Y,Z"))?;
                let parts: Vec<&str> = value.split(',').collect();
                let malformed = || format!("expected {what} X,Y,Z, got {value:?}");
                let [x, y, z] = parts[..] else {
                    return Err(malformed());
                };
                Ok(Vec3::new(
                    x.parse().map_err(|_| malformed())?,
                    y.parse().map_err(|_| malformed())?,
                    z.parse().map_err(|_| malformed())?,
                ))
            };
            match argument.as_str() {
                "--at" => at = Some(vector("--at")?),
                "--look" => look = Some(vector("--look")?),
                _ => positional.push(argument),
            }
        }
        let mut positional = positional.into_iter();
        let (width, height) = match positional.next() {
            None => SCREENSHOT_SIZE,
            Some(size) => {
                let malformed = || format!("expected a size like 1920x1080, got {size:?}");
                let (width, height) = size.split_once('x').ok_or_else(malformed)?;
                // The same message for a missing separator and an unparsable number: on its own
                // `parse` would report "invalid digit found in string" without saying of what.
                (
                    width.parse().map_err(|_| malformed())?,
                    height.parse().map_err(|_| malformed())?,
                )
            }
        };
        Ok(Self {
            path,
            width,
            height,
            cell: scene_loader::cell_argument(positional.next().map(String::as_str)),
            viewpoint: ViewpointOverride {
                position: at,
                forward: look,
            },
        })
    }
}

/// Renders one frame offscreen from the arguments after `--screenshot`.
fn screenshot(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let options = ScreenshotOptions::parse(arguments)?;
    let hit_fraction = headless::screenshot(
        &options.path,
        options.width,
        options.height,
        options.cell,
        options.viewpoint,
    )?;
    println!(
        "wrote {} ({:.0}% of rays hit geometry)",
        options.path.display(),
        hit_fraction * 100.0
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(arguments: &[&str]) -> Result<WindowOptions, String> {
        let owned: Vec<String> = arguments.iter().map(|a| (*a).to_string()).collect();
        WindowOptions::parse(&owned)
    }

    fn interior(name: &str) -> CellId {
        CellId::Interior(name.to_owned())
    }

    #[test]
    fn the_cell_and_the_frame_limit_go_in_either_order() {
        let default = interior(scene_loader::DEFAULT_CELL);
        assert_eq!(
            parse(&[]),
            Ok(WindowOptions {
                cell: default.clone(),
                exit_after: None
            })
        );
        // A flag on its own must not be mistaken for a cell name, which is exactly what happened
        // while the first argument was parsed apart from the rest.
        assert_eq!(
            parse(&["--frames", "3"]),
            Ok(WindowOptions {
                cell: default,
                exit_after: Some(3)
            })
        );

        let outdoors = WindowOptions {
            cell: CellId::Exterior { x: -2, y: -9 },
            exit_after: Some(3),
        };
        assert_eq!(parse(&["-2,-9", "--frames", "3"]), Ok(outdoors.clone()));
        assert_eq!(parse(&["--frames", "3", "-2,-9"]), Ok(outdoors));
    }

    fn screenshot_options(arguments: &[&str]) -> Result<ScreenshotOptions, String> {
        let owned: Vec<String> = arguments.iter().map(|a| (*a).to_string()).collect();
        ScreenshotOptions::parse(&owned)
    }

    #[test]
    fn a_screenshot_can_be_told_where_to_stand_and_which_way_to_look() {
        // Flagged rather than positional, and so allowed anywhere among the positional arguments:
        // an interior's name can be any string, so a fourth positional could not be told from it.
        assert_eq!(
            screenshot_options(&["out.png", "--at", "10,20,30", "640x480", "--look", "0,-1,0"]),
            Ok(ScreenshotOptions {
                path: PathBuf::from("out.png"),
                width: 640,
                height: 480,
                cell: interior(scene_loader::DEFAULT_CELL),
                viewpoint: ViewpointOverride {
                    position: Some(Vec3::new(10.0, 20.0, 30.0)),
                    forward: Some(Vec3::NEG_Y),
                },
            })
        );
        // Either half on its own: what is not said is left to the cell rather than made up here,
        // so asking only to turn on the spot is a thing that can be asked.
        assert_eq!(
            screenshot_options(&["out.png", "--at", "1,2,3"])
                .expect("should parse")
                .viewpoint,
            ViewpointOverride {
                position: Some(Vec3::new(1.0, 2.0, 3.0)),
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
        assert!(screenshot_options(&["out.png", "--at", "1,2"]).is_err());
        assert!(screenshot_options(&["out.png", "--at"]).is_err());
    }

    #[test]
    fn a_screenshot_takes_a_path_then_a_size_then_a_cell() {
        assert_eq!(
            screenshot_options(&["out.png"]),
            Ok(ScreenshotOptions {
                path: PathBuf::from("out.png"),
                width: SCREENSHOT_SIZE.0,
                height: SCREENSHOT_SIZE.1,
                cell: interior(scene_loader::DEFAULT_CELL),
                viewpoint: ViewpointOverride::default(),
            })
        );
        assert_eq!(
            screenshot_options(&["out.png", "1920x1080", "-2,-9"]),
            Ok(ScreenshotOptions {
                path: PathBuf::from("out.png"),
                width: 1920,
                height: 1080,
                cell: CellId::Exterior { x: -2, y: -9 },
                viewpoint: ViewpointOverride::default(),
            })
        );

        assert_eq!(
            screenshot_options(&[]),
            Err("--screenshot needs a path to write the image to".to_owned())
        );
        // One message for both ways a size can be wrong, since `parse` alone would say only
        // "invalid digit found in string" without naming what it was reading.
        for bad in ["1920", "1920x", "widexhigh"] {
            assert_eq!(
                screenshot_options(&["out.png", bad]),
                Err(format!("expected a size like 1920x1080, got {bad:?}"))
            );
        }
    }

    #[test]
    fn a_malformed_command_line_says_what_was_wrong_with_it() {
        assert_eq!(
            parse(&["--nonsense"]),
            Err("unknown argument \"--nonsense\"".to_owned())
        );
        assert_eq!(
            parse(&["-2,-9", "-3,-9"]),
            Err("more than one cell given: \"-3,-9\"".to_owned())
        );
        assert_eq!(
            parse(&["--frames"]),
            Err("--frames needs a count".to_owned())
        );
        assert_eq!(
            parse(&["--frames", "soon"]),
            Err("--frames needs a whole number".to_owned())
        );

        // An interior whose name begins with a dash is still a name, not a flag: only a leading
        // double dash marks one.
        assert_eq!(
            parse(&["-2,-9"]).map(|o| o.cell),
            Ok(CellId::Exterior { x: -2, y: -9 })
        );
    }
}
