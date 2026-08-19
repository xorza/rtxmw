//! Loading cells off the main thread, so crossing into one costs the frame nothing.

use std::sync::mpsc::{Receiver, Sender, channel};

use crate::error::Result;
use crate::game_data::GameData;
use crate::loaded_cell::LoadedCell;
use crate::static_scene::CellDetail;
use rtxmw_esm::CellId;

/// One cell the worker finished with, however it went.
///
/// The id rides along with the outcome because a failure has to be attributable: a request for a
/// grid square that is open sea fails, and a caller that could not tell which one would either
/// retry it forever or stop asking for anything.
#[derive(Debug)]
pub struct StreamedCell {
    pub id: CellId,
    /// What it was asked for at, so a caller holding both tiers knows which one arrived.
    pub detail: CellDetail,
    pub loaded: Result<LoadedCell>,
}

/// One cell asked for, and how much of it.
#[derive(Debug, Clone)]
struct Request {
    id: CellId,
    detail: CellDetail,
}

/// A worker thread that loads whichever cells it is asked for.
///
/// **What makes it worth a thread is where the work happens, not what it caches.** The reading,
/// decoding and mesh building happen while the renderer draws rather than between two frames. It
/// used to be worth one for a second reason — it opened and indexed the 79 MB content file once
/// for the life of the streamer — and that reason has moved to `GameData`, which does it once for
/// the life of the *process* and hands the same indices to the startup cell as well.
///
/// Requests and results are separate channels rather than a call and a return, because the two
/// have nothing to say to each other in order: the camera asks for cells faster than they load,
/// and what comes back is whatever finished.
#[derive(Debug)]
pub struct CellStreamer {
    requests: Sender<Request>,
    results: Receiver<StreamedCell>,
}

impl CellStreamer {
    /// Starts the worker.
    ///
    /// The game is reached when the first request arrives, not here, so a session that never leaves
    /// an interior never asks for it — and where it has not been opened yet, the cost lands on this
    /// thread rather than on startup.
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
    pub fn request(&self, id: CellId, detail: CellDetail) {
        // The worker only stops when the game's files could not be opened, which it has already
        // reported through the result channel.
        let _ = self.requests.send(Request { id, detail });
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

/// Waits for something to be wanted, reaches the game, and answers until the channel closes.
///
/// **The first request is taken before the game is**, which is the whole shape of this function: a
/// failure to reach it has to be reported against the request that was waiting on it, and a caller
/// that asked for a cell and got back an error naming a different one — or naming none — would have
/// no way to stop waiting for the one it asked for.
///
/// Nothing after that point can fail. A cell that will not load is reported as itself and the next
/// one is taken, so the loop has no error path of its own; it had one when the reader and both
/// indices were built here, and `GameData` builds them now.
fn serve(requests: &Receiver<Request>, results: &Sender<StreamedCell>) {
    let Ok(mut wanted) = requests.recv() else {
        return;
    };
    let game = match GameData::shared() {
        Ok(Some(game)) => game,
        // No game data configured at all. The engine has said so already, and there is nothing to
        // answer any request with.
        Ok(None) => return,
        Err(e) => {
            let _ = results.send(StreamedCell {
                id: wanted.id,
                detail: wanted.detail,
                loaded: Err(e),
            });
            return;
        }
    };

    loop {
        let loaded = LoadedCell::streamed(game, wanted.id.clone(), wanted.detail);
        let sent = results.send(StreamedCell {
            id: wanted.id,
            detail: wanted.detail,
            loaded,
        });
        if sent.is_err() {
            return;
        }
        let Ok(next) = requests.recv() else {
            return;
        };
        wanted = next;
    }
}
