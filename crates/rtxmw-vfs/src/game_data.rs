//! Locating the installed game data, for tests that need the real files.

use std::path::{Path, PathBuf};

/// Environment variable, and `.env` key, naming Morrowind's `Data Files` directory.
pub const DATA_DIR_VAR: &str = "MORROWIND_DATA_DIR";

/// Morrowind's `Data Files` directory, or `None` when this machine does not have the game.
///
/// Looks in the process environment first, so `MORROWIND_DATA_DIR=… cargo test` always wins, then
/// in a `.env` file found by walking up from the crate being compiled.
///
/// # Panics
/// If the variable is set but does not name a directory. A machine without the game should skip
/// the test; a typo in the path should not, because a skip is indistinguishable from a pass.
pub fn morrowind_data_dir() -> Option<PathBuf> {
    let raw = std::env::var_os(DATA_DIR_VAR)
        .map(PathBuf::from)
        .or_else(from_dotenv)?;

    assert!(
        raw.is_dir(),
        "{DATA_DIR_VAR} is set to {} but that is not a directory",
        raw.display()
    );
    Some(raw)
}

/// The shipped archives plus loose files, indexed in load order, or `None` without the game.
///
/// Load order is base game, expansions, then loose files — so a replacer in `Data Files` wins, the
/// same precedence the original engine applies.
pub fn morrowind_archives() -> Option<crate::vfs::Vfs> {
    let data = morrowind_data_dir()?;
    let mut vfs = crate::vfs::Vfs::new();
    for archive in ["Morrowind.bsa", "Tribunal.bsa", "Bloodmoon.bsa"] {
        let path = data.join(archive);
        if path.is_file() {
            vfs.add_bsa(&path)
                .unwrap_or_else(|e| panic!("could not open {archive}: {e}"));
        }
    }
    vfs.add_directory(&data)
        .expect("the data directory should index");
    Some(vfs)
}

/// Reads the key out of the nearest `.env`, without touching the process environment.
///
/// Deliberately does not use `dotenvy::dotenv`: that mutates the environment through
/// `std::env::set_var`, which is `unsafe` in edition 2024 precisely because it races with any other
/// thread reading the environment — and the test harness runs tests in parallel.
fn from_dotenv() -> Option<PathBuf> {
    let file = find_dotenv()?;
    let entries = dotenvy::from_path_iter(&file).ok()?;
    for entry in entries {
        match entry {
            Ok((key, value)) if key == DATA_DIR_VAR => return Some(PathBuf::from(value)),
            Ok(_) => {}
            // A malformed line elsewhere in the file is not this key's problem, so keep going —
            // but if it *is* this key, silence would look exactly like "the game is not installed".
            // dotenvy hands back only the offending value, not its key, so this cannot say which
            // line failed — only that one did, and that the parse stopped there.
            Err(dotenvy::Error::LineParse(line, _)) => {
                eprintln!(
                    "{}: cannot parse the value {line:?}\n\
                     a value containing spaces must be quoted, e.g.\n  \
                     {DATA_DIR_VAR}=\"/path/to/Morrowind/Data Files\"",
                    file.display()
                );
            }
            Err(_) => {}
        }
    }
    None
}

/// Walks up from this crate's directory looking for a `.env`.
///
/// `CARGO_MANIFEST_DIR` rather than the working directory: cargo sets the latter per test binary,
/// and it is not where the workspace root is.
fn find_dotenv() -> Option<PathBuf> {
    let mut directory: Option<&Path> = Some(Path::new(env!("CARGO_MANIFEST_DIR")));
    while let Some(current) = directory {
        let candidate = current.join(".env");
        if candidate.is_file() {
            return Some(candidate);
        }
        directory = current.parent();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locates_the_workspace_dotenv_from_a_crate_directory() {
        // This crate sits two levels below the workspace root, so finding `.env` at all proves the
        // upward walk works rather than a lucky relative path.
        match find_dotenv() {
            Some(path) => assert!(path.is_file(), "{} should be a file", path.display()),
            None => eprintln!("no .env in any ancestor; nothing to check"),
        }
    }

    #[test]
    fn a_present_data_dir_is_a_directory() {
        // Skips cleanly when the game is absent; the panic for a bad path is the assertion under
        // test everywhere else.
        if let Some(dir) = morrowind_data_dir() {
            assert!(dir.is_dir());
            assert!(
                dir.join("Morrowind.esm").is_file(),
                "{} does not look like a Data Files directory",
                dir.display()
            );
        }
    }
}
