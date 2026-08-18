//! Measuring how long the device spent on each part of a frame.

use ash::vk;

use crate::device::Device;
use crate::error::Result;
use crate::physical_device::{PhysicalDevice, TimestampSupport};

/// A pool of timestamps written between passes, read back as durations.
///
/// **These are device timings, not wall clock.** A timestamp is written by the queue when
/// everything recorded before it has completed, so the gap between two of them is what the GPU
/// spent — which is the number a frame budget is about, and not the same as how long a submission
/// took to come back.
///
/// It only means anything because the passes are separated by full barriers. Without them the
/// device would overlap adjacent work and a per-pass figure would be a fiction; with them each
/// stage genuinely finishes before the next begins.
pub struct Timestamps {
    /// A handle copy, not an owner: the real `Device` outlives this by construction.
    device: ash::Device,
    pool: vk::QueryPool,
    support: TimestampSupport,
    capacity: u32,
}

impl std::fmt::Debug for Timestamps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Timestamps")
            .field("capacity", &self.capacity)
            .field("period_ns", &self.support.period_ns)
            .finish_non_exhaustive()
    }
}

impl Timestamps {
    /// Most timestamps a pool may hold.
    ///
    /// A ceiling rather than a limitation: it is what lets a read work in a stack array, so
    /// measuring a frame never allocates on the frame's own path.
    pub const MAX: usize = 16;

    /// Creates a pool holding `capacity` timestamps.
    pub fn new(device: &Device, physical: &PhysicalDevice, capacity: u32) -> Result<Self> {
        assert!(capacity >= 2, "a duration needs two timestamps");
        assert!(
            capacity as usize <= Self::MAX,
            "{capacity} timestamps is past the {} a stack read is sized for",
            Self::MAX
        );
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(capacity);
        // SAFETY: `info` is fully initialised and the device is alive.
        let pool = unsafe { device.raw().create_query_pool(&info, None)? };
        Ok(Self {
            device: device.raw().clone(),
            pool,
            support: physical.timestamps(),
            capacity,
        })
    }

    /// Whether the queue can time anything at all.
    ///
    /// A device is allowed to offer no timestamps on a queue, in which case every duration below
    /// would be a made-up number and the caller should report nothing rather than zeroes.
    pub fn available(&self) -> bool {
        self.support.period_ns > 0.0 && self.support.valid_bits > 0
    }

    /// Clears every query, which must be recorded before any of them is written.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state.
    pub unsafe fn reset(&self, command_buffer: vk::CommandBuffer) {
        // SAFETY: the caller guarantees the command buffer is recording.
        unsafe {
            self.device
                .cmd_reset_query_pool(command_buffer, self.pool, 0, self.capacity)
        };
    }

    /// Writes timestamp `index` once everything recorded before it has finished.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state and [`Timestamps::reset`] must have been
    /// recorded into it first.
    pub unsafe fn write(&self, command_buffer: vk::CommandBuffer, index: u32) {
        // On the frame path, and the indices are constants at every call site.
        debug_assert!(index < self.capacity, "timestamp {index} is past the pool");
        // `ALL_COMMANDS` because these bracket whole passes: the point is "everything so far is
        // done", not "this stage of the pipeline was reached".
        // SAFETY: the caller guarantees the command buffer is recording.
        unsafe {
            self.device.cmd_write_timestamp2(
                command_buffer,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.pool,
                index,
            )
        };
    }

    /// Milliseconds between each consecutive pair of timestamps, into `out`.
    ///
    /// Blocks until the queries have resolved, so the submission that wrote them must have been
    /// made. `out` must hold exactly one fewer value than the pool has queries — a duration per
    /// gap — and is left untouched on a queue that cannot time anything, which
    /// [`Timestamps::available`] is how a caller tells apart from a frame that took no time.
    pub fn read(&self, out: &mut [f32]) -> Result<()> {
        assert_eq!(
            out.len() + 1,
            self.capacity as usize,
            "a pool of {} queries measures {} gaps",
            self.capacity,
            self.capacity - 1
        );
        if !self.available() {
            return Ok(());
        }

        let mut raw = [0u64; Self::MAX];
        let raw = &mut raw[..self.capacity as usize];
        // SAFETY: `raw` is sized to the pool and the device is alive.
        unsafe {
            self.device.get_query_pool_results(
                self.pool,
                0,
                raw,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?
        };

        for (gap, pair) in out.iter_mut().zip(raw.windows(2)) {
            let ticks = elapsed_ticks(pair[0], pair[1], self.support.valid_bits);
            *gap = ticks as f32 * self.support.period_ns / 1.0e6;
        }
        Ok(())
    }
}

/// Ticks between two timestamps, given how many of their bits mean anything.
///
/// Only the low `valid_bits` carry a count; everything above them is undefined rather than zero, so
/// a plain subtraction differences that noise as well. Masking after the subtraction as well as
/// before is what makes a counter that wrapped come out as a small positive number instead of an
/// enormous one — which is the whole reason a queue advertising fewer than 64 bits is worth
/// handling rather than assuming.
fn elapsed_ticks(start: u64, end: u64, valid_bits: u32) -> u64 {
    let mask = if valid_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    };
    (end & mask).wrapping_sub(start & mask) & mask
}

impl Drop for Timestamps {
    fn drop(&mut self) {
        // SAFETY: the caller waits for device idle before dropping the renderer, and every test
        // submission blocks on a fence.
        unsafe { self.device.destroy_query_pool(self.pool, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_duration_ignores_the_bits_the_queue_does_not_promise() {
        // The whole counter is meaningful, which is what this machine's graphics queue reports.
        assert_eq!(elapsed_ticks(100, 350, 64), 250);

        // Only 32 bits are. The high half is undefined, and differencing it unmasked would give a
        // duration of some four billion ticks rather than 250.
        let noise = 0xDEAD_BEEF_0000_0000u64;
        assert_eq!(
            elapsed_ticks(noise | 100, 0x1234_5678_0000_0000 | 350, 32),
            250
        );

        // And a counter that wrapped inside those 32 bits: 20 ticks before the top, 30 after.
        assert_eq!(elapsed_ticks(0xFFFF_FFEC, 0x0000_001E, 32), 50);
    }
}
