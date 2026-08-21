//! What every controller carries, and which of them a renderer can act on.

use crate::cursor::{Cursor, Link};
use crate::error::Result;

/// Which channel a controller drives.
///
/// The six share a header and differ only in what their data block means, so they are one block
/// type here with a tag rather than six that parse identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControllerKind {
    /// Moves the node it is attached to — the only one anything reads today.
    Keyframe,
    Visibility,
    Alpha,
    /// Cycles a material through a list of textures.
    Flip,
    MaterialColour,
    /// Spins the node about its own forward axis.
    Roll,
}

/// The header every controller begins with, and the block it reads its values from.
///
/// **The node it drives is not `target`.** A controller is reached from the node's own `controller`
/// link and the target points back at it; following the target instead would work for these and
/// break for a `.kf` file, where the chain hangs off a sequence helper and the targets are the only
/// thing naming anything.
#[derive(Debug, Clone)]
pub struct TimeController {
    pub kind: ControllerKind,
    /// The next controller on the same node, or none.
    pub next: Link,
    pub flags: u16,
    /// How fast the clip runs and where in it this controller starts.
    pub frequency: f32,
    pub phase: f32,
    /// The span of the clip this controller covers, in seconds.
    pub start: f32,
    pub stop: f32,
    pub target: Link,
    pub data: Link,
}

impl TimeController {
    pub(crate) fn read(cursor: &mut Cursor<'_>, kind: ControllerKind) -> Result<Self> {
        Ok(Self {
            kind,
            next: cursor.link()?,
            flags: cursor.u16()?,
            frequency: cursor.f32()?,
            phase: cursor.f32()?,
            start: cursor.f32()?,
            stop: cursor.f32()?,
            target: cursor.link()?,
            data: cursor.link()?,
        })
    }

    /// Whether the clip repeats rather than holding its last pose.
    ///
    /// Bit one of the flags, against bit two for ping-pong and bit zero for whether it is running
    /// at all. Nothing here reads the other two: everything animated in the shipped content that
    /// this drives loops.
    pub fn loops(&self) -> bool {
        self.flags & 0b10 != 0
    }
}
