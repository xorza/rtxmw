//! The glow a bright thing leaves around itself, which is a property of the optic and not of it.

use ash::vk;
use rtxmw_gpu::{Binding, ComputePipeline, Device, Image, Memory, memory_barrier};

use crate::shaders;

/// Matches `local_size_x`/`local_size_y` in the three bloom shaders.
const WORKGROUP: u32 = 8;

/// Format of every level of the pyramid.
///
/// The frame's own, and it has to be: a candle flame arrives here at sixty times the mean radiance
/// of the room, which is the whole reason there is anything to spread, and eight bits would have
/// clipped it to white before the first halving.
const FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Smallest a level may be on its short side before the pyramid stops.
///
/// **Where the glow stops widening.** Each level doubles the reach of the blur, so the coarsest one
/// sets how far a highlight throws light — at four pixels across a 1080-line frame that is nine
/// halvings and a skirt spanning the whole picture, which is what a real optic does with a bright
/// point and what a fixed-radius blur cannot.
const FINEST: u32 = 4;

/// Most levels a pyramid can have, which fixes how many passes are created up front.
///
/// **Because the pipelines have to outlive a resize.** A window that changes size, or an upscaler
/// arriving and moving the frame to its output resolution, replaces every image — and the renderer's
/// contract for that is that the pipelines and their descriptor sets survive it. Halving an 8192-line
/// frame reaches four pixels in eleven steps, so twelve is a ceiling nothing can reach and the assert
/// in [`Bloom::build`] is what says so.
const MAX_LEVELS: usize = 12;

/// How much of the light is scattered instead of focused.
///
/// **The eye's figure rather than a lens's, and the reason is the display.** Veiling glare in a
/// clean photographic lens is a couple of percent by the ISO 9358 index, and if this render were
/// standing in for a camera that is what belongs here. It is not: it is standing in for *being in
/// the room*, and the scatter that matters then happens inside the viewer's own eye — cornea, lens
/// and vitreous together put something near a tenth of the light where it was not aimed, which is
/// why a candle in a dark room has a halo whether or not there is a camera anywhere near it.
///
/// A monitor cannot provoke that. A flame arrives here at sixty times the mean radiance of the room
/// and leaves the tone curve at rather less than twice it, so the luminance that would have made a
/// real eye scatter never reaches the real eye looking at the screen. The render has to supply what
/// the display cannot, which is the whole reason this sits at the eye's end of the range and not
/// the lens's. Eight percent is just under a young eye's own.
///
/// It is a *fraction taken out of the frame*, not an amount added to it — see `bloom_apply.comp` —
/// so raising it cannot make the picture brighter, only hazier.
const SCATTERED: f32 = 0.08;

/// The pyramid, and the passes that climb it.
///
/// **Down and back up rather than one wide blur.** Reaching several hundred pixels directly would
/// cost hundreds of taps a pixel; halving six or seven times, then climbing back with a tent at
/// every level, reaches the same distance for a fraction of the work — and the sum over the levels
/// is a *mixture* of widths, a bright core with a long faint skirt, which is much closer to a real
/// point spread function than any single Gaussian. `docs/design.md` §8.100.
#[derive(Debug)]
pub(crate) struct Bloom {
    /// What fraction of the light is scattered — [`SCATTERED`] unless a caller said otherwise.
    strength: f32,
    /// Level zero is half the frame; each after it is half again.
    levels: Vec<Image>,
    /// One per step down, and one per step back up. A pipeline apiece because each step names a
    /// different pair of images and a `ComputePipeline` owns one descriptor set — cheap at startup,
    /// and it keeps a pyramid out of the set-management the rest of the renderer does not need.
    down: Vec<ComputePipeline>,
    up: Vec<ComputePipeline>,
    apply: ComputePipeline,
}

impl Bloom {
    /// Creates the pyramid for a frame of `extent`, and the passes over it.
    pub(crate) fn new(
        device: &Device,
        memory: &Memory,
        extent: vk::Extent2D,
    ) -> rtxmw_gpu::Result<Self> {
        let pair = || [Binding::storage_image(0), Binding::storage_image(1)];
        let passes = |shader, count| {
            (0..count)
                .map(|_| ComputePipeline::new(device, &pair(), 0, shader))
                .collect::<rtxmw_gpu::Result<Vec<_>>>()
        };
        let mut bloom = Self {
            strength: SCATTERED,
            levels: Vec::new(),
            down: passes(shaders::bloom_down(), MAX_LEVELS)?,
            up: passes(shaders::bloom_up(), MAX_LEVELS - 1)?,
            apply: ComputePipeline::new(
                device,
                &pair(),
                size_of::<Glare>() as u32,
                shaders::bloom_apply(),
            )?,
        };
        bloom.build(memory, extent)?;
        Ok(bloom)
    }

    /// Replaces the pyramid's images at a new size. The caller rebinds.
    ///
    /// The passes are untouched: there are [`MAX_LEVELS`] of them however few the frame needs, so a
    /// resize is image allocations and a round of descriptor writes rather than a rebuild.
    pub(crate) fn resize(
        &mut self,
        memory: &Memory,
        extent: vk::Extent2D,
    ) -> rtxmw_gpu::Result<()> {
        self.levels.clear();
        self.build(memory, extent)
    }

    /// Sizes the levels, halving until one would be finer than the glow needs to reach.
    fn build(&mut self, memory: &Memory, extent: vk::Extent2D) -> rtxmw_gpu::Result<()> {
        let mut size = vk::Extent2D {
            width: (extent.width / 2).max(1),
            height: (extent.height / 2).max(1),
        };
        // At least one level however small the frame is: a test fixture renders at sixty-four
        // pixels and still has to go through the same code the window does.
        while self.levels.is_empty() || size.width.min(size.height) >= FINEST {
            self.levels.push(Image::new(
                memory,
                "bloom level",
                size,
                FORMAT,
                vk::ImageUsageFlags::STORAGE,
            )?);
            size = vk::Extent2D {
                width: (size.width / 2).max(1),
                height: (size.height / 2).max(1),
            };
        }

        assert!(
            self.levels.len() <= MAX_LEVELS,
            "a {}x{} frame wants {} bloom levels against {MAX_LEVELS} built",
            extent.width,
            extent.height,
            self.levels.len()
        );
        Ok(())
    }

    /// Sets what fraction of the light is scattered, for a caller that needs none of it.
    ///
    /// **Zero is what a measurement asks for**, and for the same reason `set_fog(0.0)` exists: a
    /// glow moves light across the whole frame, which is what it is *for* and exactly what a test
    /// counting the pixels one localised effect changed must not have underneath it.
    pub(crate) fn set_strength(&mut self, strength: f32) {
        assert!(
            (0.0..=1.0).contains(&strength),
            "a scattered fraction runs from none to all of it, not {strength}"
        );
        self.strength = strength;
    }

    /// Every level, for the caller that has to put them in the layout a dispatch reads.
    ///
    /// **A frame starts them from `UNDEFINED` like everything else it writes**: the first halving
    /// covers level zero in full and each level after it covers the next, so nothing here is ever
    /// read before it is written and preserving what the last frame left would cost a transition
    /// and buy nothing.
    pub(crate) fn levels(&self) -> impl Iterator<Item = &Image> {
        self.levels.iter()
    }

    /// Points the pyramid at the frame it spreads.
    pub(crate) fn bind(&mut self, frame: &Image) {
        // The first halving reads the frame; every one after reads the level above it.
        self.down[0].bind_storage_images(0, &[frame, &self.levels[0]]);
        for level in 1..self.levels.len() {
            self.down[level]
                .bind_storage_images(0, &[&self.levels[level - 1], &self.levels[level]]);
        }
        // Climbing back: the coarser level is spread into the finer one, ending at level zero.
        for step in 0..self.steps_up() {
            let coarse = self.levels.len() - 1 - step;
            self.up[step].bind_storage_images(0, &[&self.levels[coarse], &self.levels[coarse - 1]]);
        }
        self.apply.bind_storage_images(0, &[&self.levels[0], frame]);
    }

    /// Records the whole pyramid: down, back up, and blended into the frame.
    ///
    /// # Safety
    /// `command_buffer` must be recording, [`Bloom::bind`] must have run, and every image involved
    /// must be in `GENERAL`.
    pub(crate) unsafe fn record(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        if self.strength == 0.0 {
            return;
        }
        // SAFETY: the caller guarantees the command buffer is recording and the sets are written.
        unsafe {
            for level in 0..self.levels.len() {
                self.down[level].dispatch(command_buffer, groups(self.levels[level].extent()), &[]);
                // Every step reads what the one before it wrote, so none of them may overlap.
                memory_barrier::full(device, command_buffer);
            }
            for step in 0..self.steps_up() {
                let target = self.levels.len() - 2 - step;
                self.up[step].dispatch(command_buffer, groups(self.levels[target].extent()), &[]);
                memory_barrier::full(device, command_buffer);
            }
            let glare = Glare {
                strength: self.strength,
                normalise: 1.0 / self.levels.len() as f32,
            };
            self.apply.dispatch(
                command_buffer,
                groups(self.frame_extent()),
                bytemuck::bytes_of(&glare),
            );
        }
    }

    /// How many steps the climb takes, which is one fewer than there are levels — and none at all
    /// for a frame small enough to have only one.
    fn steps_up(&self) -> usize {
        self.levels.len() - 1
    }

    /// The frame's own size, which is twice the finest level's.
    fn frame_extent(&self) -> vk::Extent2D {
        let finest = self.levels[0].extent();
        vk::Extent2D {
            width: finest.width * 2,
            height: finest.height * 2,
        }
    }
}

/// What the blend at the end is told — see `bloom_apply.comp`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Glare {
    strength: f32,
    normalise: f32,
}

/// Workgroups covering `extent`.
fn groups(extent: vk::Extent2D) -> [u32; 3] {
    [
        extent.width.div_ceil(WORKGROUP),
        extent.height.div_ceil(WORKGROUP),
        1,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// How many levels a frame of `height` lines gets, by the same halving [`Bloom::build`] does.
    fn levels_for(width: u32, height: u32) -> usize {
        let (mut w, mut h, mut n) = ((width / 2).max(1), (height / 2).max(1), 0);
        while n == 0 || w.min(h) >= FINEST {
            n += 1;
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
        n
    }

    #[test]
    fn the_pyramid_reaches_across_the_whole_frame() {
        // **How far the glow throws, counted rather than assumed.** 1080 lines halve to 540, 270,
        // 135, 67, 33, 16, 8 and 4 — eight levels, the last of which is four pixels tall, so one
        // texel of it covers 135 lines of the frame. That is the skirt; the levels above it are the
        // core.
        assert_eq!(levels_for(1920, 1080), 8);
        assert_eq!(levels_for(3840, 2160), 9);

        // A fixture renders at sixty-four pixels and must still climb something rather than
        // dividing by an empty pyramid.
        assert_eq!(levels_for(64, 64), 4);
        assert_eq!(levels_for(8, 8), 1);
        assert_eq!(levels_for(1, 1), 1);
    }

    #[test]
    fn the_blend_takes_light_out_of_the_frame_rather_than_adding_it() {
        // A flat field blooms to itself: every level of the pyramid holds the same value a box
        // average of a constant does, the climb sums `levels` copies of it, and `normalise` divides
        // exactly that many back out — so the mix has the same colour on both sides and the frame
        // is untouched whatever `SCATTERED` is. That is the property that makes the constant a
        // fraction of the light rather than a gain on it.
        let flat = 0.25_f32;
        for levels in 1..=9 {
            let accumulated = flat * levels as f32;
            let bloomed = accumulated * (1.0 / levels as f32);
            let mixed = flat * (1.0 - SCATTERED) + bloomed * SCATTERED;
            assert!(
                (mixed - flat).abs() < 1.0e-6,
                "{levels} levels moved a flat field"
            );
        }
    }
}
