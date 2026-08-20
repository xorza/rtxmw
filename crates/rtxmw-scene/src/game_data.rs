//! The installed game, opened once and shared by everything that reads it.

use std::sync::OnceLock;

use rtxmw_esm::{CellIndex, EsmReader};
use rtxmw_vfs::{Vfs, morrowind_data_dir};

use crate::error::{Result, SceneError};
use crate::ini::Ini;
use crate::static_scene::ModelIndex;

/// Everything the installed game is: the master file, the archives, and the two indices over it.
///
/// **One of these per process, and every reader of game content goes through it.** Opening the game
/// costs 46 ms warm — 25 to read `Morrowind.esm`'s 79 megabytes, 10 to index the archives, 11 to
/// walk the file for its cells and its models — and it was being paid three times over: once by the
/// cell the camera opens in, once again by the streaming thread, and the archives a third time for
/// the moons' two portraits. All three now find this already built.
///
/// **What made a shared instance look impossible was not true.** The note this replaces claimed the
/// cell index and the model table borrow the reader, so that whatever owns the bytes must outlive
/// all three and only a stack of locals could hold them. They do not: [`CellIndex`] and
/// [`ModelIndex`] take a reader to `build` and own everything they keep. The one borrower is
/// [`EsmReader`] itself, over the bytes alone, and constructing one is a header parse that measures
/// zero — so it is handed out on demand by [`Self::reader`] rather than stored, and nothing here is
/// self-referential.
///
/// Shared across threads by `&'static`, which the archives were already built for: `BsaArchive`
/// reads at an offset rather than seeking, precisely so two threads can read one file at once.
#[derive(Debug)]
pub(crate) struct GameData {
    /// `Morrowind.esm` entire. Kept as bytes because [`EsmReader`] borrows what it reads.
    esm: Vec<u8>,
    vfs: Vfs,
    cells: CellIndex,
    models: ModelIndex,
    ini: Ini,
}

impl GameData {
    /// The process's one instance, opening the game on the first call.
    ///
    /// `None` where no game data is configured, which is a normal state the engine comes up in and
    /// says so. An error where a directory *is* configured and its archives will not open, because
    /// that means the path is wrong and silently drawing nothing would hide it.
    ///
    /// **A failure is not remembered**, so a caller that reports one and carries on does not poison
    /// every later attempt with a cached error — which also keeps this returning the crate's own
    /// `Result` rather than something clonable enough to store. Two threads racing the first call
    /// would both open the game and one would drop its copy; nothing in either front end does that,
    /// since the main thread has the moons' portraits before the streamer is spawned.
    pub(crate) fn shared() -> Result<Option<&'static Self>> {
        static SHARED: OnceLock<Option<GameData>> = OnceLock::new();
        if let Some(opened) = SHARED.get() {
            return Ok(opened.as_ref());
        }
        let opened = Self::open()?;
        Ok(SHARED.get_or_init(|| opened).as_ref())
    }

    /// A reader over the master file.
    ///
    /// Borrows rather than being stored, which is what keeps this type from having to refer to
    /// itself — and costs nothing to take: the bytes were parsed once already at [`Self::open`], so
    /// this is a header re-read that measured 0.0 ms and cannot fail on data that failed nowhere.
    pub(crate) fn reader(&self) -> EsmReader<'_> {
        EsmReader::new(&self.esm).expect("the master file parsed when it was opened")
    }

    /// Where every cell and its terrain sit in the master file.
    pub(crate) fn cells(&self) -> &CellIndex {
        &self.cells
    }

    /// What model each placeable object names, and the lights among them.
    pub(crate) fn models(&self) -> &ModelIndex {
        &self.models
    }

    /// The archives every mesh and texture is read out of.
    pub(crate) fn vfs(&self) -> &Vfs {
        &self.vfs
    }

    /// `Morrowind.ini`, which is where the game keeps its weathers, its moons and its day length.
    pub(crate) fn ini(&self) -> &Ini {
        &self.ini
    }

    /// Reads and indexes the whole installation.
    fn open() -> Result<Option<Self>> {
        let Some(directory) = morrowind_data_dir() else {
            return Ok(None);
        };
        let vfs = rtxmw_vfs::morrowind_archives().ok_or_else(|| {
            SceneError::Io(std::io::Error::other(
                "the game directory is configured but its archives could not be opened",
            ))
        })?;
        let esm = std::fs::read(directory.join("Morrowind.esm")).map_err(SceneError::Io)?;
        // **Beside `Data Files` rather than in it**, which is where an install puts it. Missing is
        // not an error: the ini is the game's tuning and every reader of it has a fallback, so an
        // install without one renders with the figures written down in this crate.
        let ini = directory
            .parent()
            .map(|game| game.join("Morrowind.ini"))
            .and_then(|path| std::fs::read_to_string(path).ok())
            .map_or_else(Ini::default, |text| Ini::parse(&text));
        // Built here rather than by each caller, which is most of the point: walking the file for
        // its cells and its models is 11 ms that two callers were each paying. Scoped, because the
        // reader borrows `esm` and `esm` is moved into the struct below.
        let (cells, models) = {
            let reader = EsmReader::new(&esm)?;
            (CellIndex::build(&reader)?, ModelIndex::build(&reader)?)
        };
        Ok(Some(Self {
            esm,
            vfs,
            cells,
            models,
            ini,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_game_is_opened_once_and_every_caller_gets_the_very_same_one() {
        // **The whole point of the type, and the only thing that cannot be measured in a benchmark
        // afterwards.** Three callers used to open the game for themselves — the startup cell, the
        // streaming thread and the moons' portraits — and the way that comes back is silently, as a
        // slower session, so it is asserted here rather than watched for.
        let Some(first) = GameData::shared().expect("the game should open") else {
            // No game data configured, which is a state the engine runs in. Nothing to assert.
            return;
        };
        let second = GameData::shared()
            .expect("a second call should not fail where the first did not")
            .expect("nor should it stop finding the game");
        assert!(
            std::ptr::eq(first, second),
            "two callers were handed two different games"
        );

        // And it arrives *indexed*, which is the half that used to be rebuilt per caller: 11 ms of
        // walking the whole file, paid by the startup cell and again by the streaming thread.
        assert!(first.cells().len() > 100, "{}", first.cells().len());
        assert!(first.models().len() > 100, "{}", first.models().len());
        assert!(!first.vfs().is_empty());

        // And the ini came with it — the day length every other reading here depends on.
        assert_eq!(first.ini().number("Weather", "Sunset Time"), Some(18.0));
        assert_eq!(first.ini().sections_under("weather ").len(), 10);

        // The reader is handed out rather than stored, so taking two is legal and neither borrows
        // the other — which is the property that let the indices be shared at all.
        let (one, two) = (first.reader(), first.reader());
        assert_eq!(one.header().version, two.header().version);
    }
}
