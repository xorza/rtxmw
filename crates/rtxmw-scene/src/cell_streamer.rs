//! Loading cells off the main thread, so crossing into one costs the frame nothing.

use std::sync::mpsc::{Receiver, Sender, channel};

use crate::error::Result;
use crate::game_files::GameFiles;
use crate::loaded_cell::LoadedCell;
use crate::static_scene::ModelIndex;
use rtxmw_esm::{CellId, CellIndex, EsmReader};

/// One cell the worker finished with, however it went.
///
/// The id rides along with the outcome because a failure has to be attributable: a request for a
/// grid square that is open sea fails, and a caller that could not tell which one would either
/// retry it forever or stop asking for anything.
#[derive(Debug)]
pub struct StreamedCell {
    pub id: CellId,
    pub loaded: Result<LoadedCell>,
}

/// A worker thread that loads whichever cells it is asked for.
///
/// Two things make this worth a thread rather than a function. The content file is opened, read
/// and indexed **once** for the life of the streamer — 79 MB and two passes over it, which is most
/// of what loading a cell used to cost — and the reading, decoding and mesh building that remains
/// happens while the renderer draws, rather than between two frames.
///
/// Requests and results are separate channels rather than a call and a return, because the two
/// have nothing to say to each other in order: the camera asks for cells faster than they load,
/// and what comes back is whatever finished.
#[derive(Debug)]
pub struct CellStreamer {
    requests: Sender<CellId>,
    results: Receiver<StreamedCell>,
}

impl CellStreamer {
    /// Starts the worker.
    ///
    /// The files are opened when the first request arrives, not here, so a session that never
    /// leaves an interior never reads them — and the cost lands on the thread that can afford it
    /// rather than on startup.
    pub fn spawn() -> Self {
        let (requests, inbox) = channel();
        let (outbox, results) = channel();
        // Detached rather than joined on drop. It holds no device or file handle that outlives the
        // process, and joining would make quitting wait out whichever cell was mid-load.
        std::thread::Builder::new()
            .name("cell streamer".to_owned())
            .spawn(move || serve(&inbox, &outbox))
            .expect("the operating system should give us a thread");
        Self { requests, results }
    }

    /// Asks for a cell. Loaded in the order asked for, and delivered by [`Self::take_ready`].
    ///
    /// Nothing deduplicates here: a caller asking twice gets it twice, because only the caller
    /// knows what it already holds.
    pub fn request(&self, id: CellId) {
        // The worker only stops when the game's files could not be opened, which it has already
        // reported through the result channel.
        let _ = self.requests.send(id);
    }

    /// Waits for the next cell to finish, or `None` once the worker has stopped.
    ///
    /// For a caller with no frame to keep running — a screenshot filling its window before it
    /// draws — where polling would be a spin loop.
    pub fn wait_ready(&self) -> Option<StreamedCell> {
        self.results.recv().ok()
    }

    /// One cell that has finished loading, or `None` if none has since the last call.
    ///
    /// Never blocks: this is called from the frame loop, where waiting for a cell would be the
    /// stall the thread exists to avoid.
    pub fn take_ready(&self) -> Option<StreamedCell> {
        self.results.try_recv().ok()
    }
}

/// Waits for something to be wanted, opens the game's files, and answers until the channel closes.
///
/// Split from [`answer`] so that a failure to open the files is reported *against the request that
/// was waiting on it*. A caller that asked for a cell and got back an error naming a different one,
/// or naming none, would have no way to stop waiting for the one it asked for.
fn serve(requests: &Receiver<CellId>, results: &Sender<StreamedCell>) {
    let Ok(first) = requests.recv() else {
        return;
    };
    let files = match GameFiles::open() {
        Ok(Some(files)) => files,
        // No game data configured at all. The engine has said so already, and there is nothing to
        // answer any request with.
        Ok(None) => return,
        Err(e) => {
            let _ = results.send(StreamedCell {
                id: first,
                loaded: Err(e),
            });
            return;
        }
    };
    if let Err(e) = answer(&files, requests, results, first.clone()) {
        let _ = results.send(StreamedCell {
            id: first,
            loaded: Err(e),
        });
    }
}

/// Loads `first`, and everything asked for after it, until the channel closes.
///
/// The reader borrows the bytes it reads, and the index and model table borrow the reader, so all
/// three are locals here — which is the whole reason this is a loop in a thread rather than a
/// struct holding them. Only those three setup steps can return an error: once the loop is running,
/// a cell that fails is reported as itself and the next one is taken.
fn answer(
    files: &GameFiles,
    requests: &Receiver<CellId>,
    results: &Sender<StreamedCell>,
    first: CellId,
) -> Result<()> {
    let esm = EsmReader::new(&files.esm)?;
    let index = CellIndex::build(&esm)?;
    let models = ModelIndex::build(&esm)?;

    let mut wanted = first;
    loop {
        let loaded = LoadedCell::streamed(&esm, &index, &models, &files.vfs, wanted.clone());
        if results.send(StreamedCell { id: wanted, loaded }).is_err() {
            return Ok(());
        }
        let Ok(next) = requests.recv() else {
            return Ok(());
        };
        wanted = next;
    }
}
