//! What a `NiParticleSystemController` emits, and how fast it lets go of it.

use crate::cursor::{Cursor, Link};
use crate::error::Result;

/// The emitter behind every flame, steam vent and puff of smoke the game ships.
///
/// **Sized before it was read.** The block was already consumed field by field so the ones after it
/// would land — `NiBSPArrayController` is the same record under another name — and this keeps the
/// values instead of skipping them. What it does not keep is the *live* particle array: 40 bytes
/// per particle of position, velocity, age and generation, which is the emitter's state at the
/// moment the artist saved the file and says nothing about how it runs.
#[derive(Debug, Clone, Default)]
pub struct ParticleSystem {
    /// Bit zero is whether it is running at all; see [`crate::TimeController::loops`] for the rest.
    pub flags: u16,
    /// How fast the clip runs and where in it this emitter starts.
    pub frequency: f32,
    pub phase: f32,
    /// The span of the clip this emitter covers, in seconds.
    pub start: f32,
    pub stop: f32,
    /// How fast a particle leaves, in world units a second, and by how much that varies.
    pub speed: f32,
    pub speed_variation: f32,
    /// Where it leaves towards, as a polar angle off the emitter's `+Z` and an azimuth about it,
    /// both in radians, each with the half-width it varies by.
    ///
    /// Zero declination is straight up, which is what a flame wants and why so many of these are
    /// near it.
    pub declination: f32,
    pub declination_variation: f32,
    pub azimuth: f32,
    pub azimuth_variation: f32,
    /// The colour every particle starts at, before any modifier moves it.
    pub colour: [f32; 4],
    /// How wide it is drawn at birth, in world units.
    pub size: f32,
    /// The window within the clip that the emitter is actually emitting over.
    pub emit_start: f32,
    pub emit_stop: f32,
    /// Particles born a second.
    pub birth_rate: f32,
    /// How long one lives, and by how much that varies.
    pub lifetime: f32,
    pub lifetime_variation: f32,
    /// Which of the emitter node's own extents a particle is born across, in its local space.
    ///
    /// A box rather than a point: a chimney's smoke leaves across the whole flue.
    pub spread: [f32; 3],
    /// The node particles are born at, or none where that is the controller's own target.
    pub emitter: Link,
    /// How many may be alive at once, which is the budget the file was authored against.
    pub capacity: u16,
    /// The head of the modifier chain — gravity, the size ramp, the colour ramp.
    pub modifier: Link,
}

impl ParticleSystem {
    /// Reads one, consuming exactly its bytes.
    pub(crate) fn read(cursor: &mut Cursor<'_>) -> Result<Self> {
        let mut out = Self {
            // The shared controller header, minus the data link these do not have: an emitter
            // carries its parameters inline rather than pointing at a key list.
            flags: {
                cursor.skip(4)?; // Next controller in the chain.
                cursor.u16()?
            },
            frequency: cursor.f32()?,
            phase: cursor.f32()?,
            start: cursor.f32()?,
            stop: cursor.f32()?,
            ..Default::default()
        };
        cursor.skip(4)?; // Target node.

        out.speed = cursor.f32()?;
        out.speed_variation = cursor.f32()?;
        out.declination = cursor.f32()?;
        out.declination_variation = cursor.f32()?;
        out.azimuth = cursor.f32()?;
        out.azimuth_variation = cursor.f32()?;
        cursor.skip(12)?; // Initial normal, which nothing at this version writes.
        out.colour = cursor.color4()?;
        out.size = cursor.f32()?;
        out.emit_start = cursor.f32()?;
        out.emit_stop = cursor.f32()?;
        cursor.skip(1)?; // Reset-on-loop flag.
        out.birth_rate = cursor.f32()?;
        out.lifetime = cursor.f32()?;
        out.lifetime_variation = cursor.f32()?;
        cursor.skip(2)?; // Emit flags.
        out.spread = cursor.vec3()?;
        out.emitter = cursor.link()?;
        cursor.skip(2 + 4 + 2 + 4 + 4)?; // Spawn generation, percentage, multiplier, chaos.
        out.capacity = cursor.u16()?;
        cursor.skip(2)?; // How many were live when the file was saved.
        // Velocity, rotation axis, age, lifespan, last update, generation, code — the emitter's
        // saved state, which a run of it replaces on the first frame.
        cursor.skip(out.capacity as usize * (12 + 12 + 4 + 4 + 4 + 2 + 2))?;
        cursor.skip(4)?; // Emitter modifier, which nothing at this version writes.
        out.modifier = cursor.link()?;
        cursor.skip(4)?; // Collider chain; one shipped mesh has one.
        cursor.skip(1)?; // Static target bound flag.
        Ok(out)
    }
}
