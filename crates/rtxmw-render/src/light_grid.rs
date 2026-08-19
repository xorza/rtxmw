//! A uniform grid over the lights, so a shading point walks the ones that reach it.

use glam::{IVec3, UVec3, Vec3};
use rtxmw_gpu::{Buffer, Uploader};

use crate::gpu_light::GpuLight;

/// Cell size the grid tries first, in world units — one terrain tile.
///
/// A starting point rather than the answer: [`LightGrid::build`] doubles it until the grid fits the
/// budgets below, so a town's lights get fine cells and a world's worth get coarse ones without a
/// caller choosing.
const FINEST_CELL: f32 = 512.0;

/// Most cells the grid may have, which is what its offset table costs — 256 KB here.
const MAX_CELLS: usize = 1 << 16;

/// Most entries the index list may have, across every cell.
///
/// A separate budget because the two run out for different reasons: a wide world overruns the cell
/// count, and one light with an enormous reach overruns this while the grid is still small.
const MAX_ENTRIES: usize = 1 << 18;

/// Where a light grid sits in the world, as the shader addresses it.
///
/// Zero dimensions is the empty grid — a cell with no lights at all — and the shader's bounds test
/// rejects every lookup against it without a flag of its own.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct LightGridExtent {
    pub(crate) origin: Vec3,
    /// Reciprocal of the cell size, so a lookup multiplies rather than divides.
    pub(crate) scale: f32,
    pub(crate) dimensions: [u32; 3],
}

impl LightGridExtent {
    /// How many cells the grid holds, saturating rather than wrapping.
    ///
    /// The saturation is load-bearing: this is asked of grids that have not been accepted yet, and
    /// the first size tried for a world-sized spread overflows. Coming back as `usize::MAX` fails
    /// the budget and the cell size doubles, where a wrapped product could pass it.
    fn cells(&self) -> usize {
        self.dimensions
            .iter()
            .try_fold(1usize, |total, &side| total.checked_mul(side as usize))
            .unwrap_or(usize::MAX)
    }

    /// The half-open box of cells a light's bounding box covers, clamped to the grid.
    fn covered(&self, light: &GpuLight) -> CellRange {
        let dimensions = UVec3::from_array(self.dimensions);
        let corner = |at: Vec3| {
            ((at - self.origin) * self.scale)
                .floor()
                .as_ivec3()
                .clamp(IVec3::ZERO, dimensions.as_ivec3() - IVec3::ONE)
                .as_uvec3()
        };
        let centre = Vec3::from_array(light.position);
        let reach = Vec3::splat(light.radius);
        CellRange {
            low: corner(centre - reach),
            high: corner(centre + reach) + UVec3::ONE,
        }
    }

    /// Where a cell's entry sits in the offset table.
    fn slot(&self, at: UVec3) -> usize {
        let [width, height, _] = self.dimensions;
        ((at.z * height + at.y) * width + at.x) as usize
    }

    /// How many cells that box is, which is what the light costs the index list.
    fn span(&self, light: &GpuLight) -> usize {
        let CellRange { low, high } = self.covered(light);
        let side = high - low;
        side.x as usize * side.y as usize * side.z as usize
    }
}

/// The half-open box of grid cells one light's reach covers.
#[derive(Debug, Clone, Copy)]
struct CellRange {
    low: UVec3,
    high: UVec3,
}

/// The grid's two flat arrays, before they reach the device.
#[derive(Debug)]
struct Bins {
    /// Prefix offsets, one per cell plus a trailing sentinel.
    offsets: Vec<u32>,
    /// Light indices, grouped by cell.
    indices: Vec<u32>,
}

/// Which lights reach each cell of a grid over the world, as two flat buffers.
///
/// **The point is what a pixel does not read.** The shader walked every light in the scene for every
/// shading point, primary and bounce alike, and measured at 1920x1080 that costs 0.031 ms per light
/// per frame whether or not the light is anywhere near — Balmora's 53 were 1.6 ms of a 7.3 ms trace
/// spent rejecting them. A grid turns that into the handful whose reach actually covers the point.
///
/// Offsets and indices rather than a list per cell: the offsets are a prefix sum with a trailing
/// sentinel, so cell `i` owns `indices[offsets[i]..offsets[i + 1]]` and the whole structure is two
/// allocations however many cells it has.
#[derive(Debug)]
pub(crate) struct LightGrid {
    offsets: Buffer,
    indices: Buffer,
    extent: LightGridExtent,
}

impl LightGrid {
    /// Bins `lights` into a grid and uploads it.
    pub(crate) fn build(uploader: &mut Uploader, lights: &[GpuLight]) -> rtxmw_gpu::Result<Self> {
        let extent = Self::extent_of(lights);
        let bins = Self::bin(&extent, lights);

        Ok(Self {
            offsets: Buffer::storage_of(
                uploader,
                "light grid offsets",
                bytemuck::cast_slice(&bins.offsets),
            )?,
            indices: Buffer::storage_of(
                uploader,
                "light grid indices",
                bytemuck::cast_slice(&bins.indices),
            )?,
            extent,
        })
    }

    /// The coarsest cell size that fits both budgets, and the grid it implies.
    ///
    /// Doubling rather than solving for a size: each doubling divides the cell count by eight, so
    /// even a light a thousand cells away from the rest is absorbed in a handful of rounds, and the
    /// result is a power-of-two multiple of a size chosen for the content.
    fn extent_of(lights: &[GpuLight]) -> LightGridExtent {
        if lights.is_empty() {
            return LightGridExtent::default();
        }
        let mut low = Vec3::INFINITY;
        let mut high = Vec3::NEG_INFINITY;
        for light in lights {
            let centre = Vec3::from_array(light.position);
            low = low.min(centre - light.radius);
            high = high.max(centre + light.radius);
        }

        let mut cell = FINEST_CELL;
        loop {
            let scale = 1.0 / cell;
            let sides = ((high - low) * scale).floor().as_uvec3() + UVec3::ONE;
            let extent = LightGridExtent {
                origin: low,
                scale,
                dimensions: sides.to_array(),
            };
            let entries: usize = lights
                .iter()
                .map(|light| extent.span(light))
                .fold(0usize, usize::saturating_add);
            if extent.cells() <= MAX_CELLS && entries <= MAX_ENTRIES {
                return extent;
            }
            cell *= 2.0;
        }
    }

    /// A counting sort of the lights into cells: count, prefix-sum, then place.
    fn bin(extent: &LightGridExtent, lights: &[GpuLight]) -> Bins {
        // Walked twice — once to count and once to place — rather than built into a list per cell,
        // which would be an allocation for each of sixty-odd thousand of them.
        let each_cell = |light: &GpuLight, visit: &mut dyn FnMut(usize)| {
            let CellRange { low, high } = extent.covered(light);
            for z in low.z..high.z {
                for y in low.y..high.y {
                    for x in low.x..high.x {
                        visit(extent.slot(UVec3::new(x, y, z)));
                    }
                }
            }
        };

        let mut offsets = vec![0u32; extent.cells() + 1];
        for light in lights {
            each_cell(light, &mut |cell| offsets[cell + 1] += 1);
        }
        for index in 1..offsets.len() {
            offsets[index] += offsets[index - 1];
        }

        let mut filled = offsets.clone();
        let mut indices = vec![0u32; *offsets.last().unwrap_or(&0) as usize];
        for (index, light) in lights.iter().enumerate() {
            each_cell(light, &mut |cell| {
                indices[filled[cell] as usize] = index as u32;
                filled[cell] += 1;
            });
        }
        Bins { offsets, indices }
    }

    /// Prefix offsets, one per cell plus a trailing sentinel.
    pub(crate) fn offsets(&self) -> &Buffer {
        &self.offsets
    }

    /// Light indices, grouped by cell.
    pub(crate) fn indices(&self) -> &Buffer {
        &self.indices
    }

    /// Where the grid sits, for the frame the shader reads.
    pub(crate) fn extent(&self) -> LightGridExtent {
        self.extent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rtxmw_scene::Light;

    impl LightGridExtent {
        /// Which cell `at` falls in, or `None` for a point outside the grid.
        ///
        /// **Only a test asks this** — in the engine it is `lights_reaching` in `lighting.glsl`
        /// that looks a point up, and this is here so the assertions below can ask what that
        /// function would be handed. The indexing itself is [`LightGridExtent::slot`], shared with
        /// the binning, so the two cannot disagree about where a cell's entry sits.
        fn slot_at(&self, at: Vec3) -> Option<usize> {
            let dimensions = IVec3::from_array(self.dimensions.map(|side| side as i32));
            let cell = ((at - self.origin) * self.scale).floor().as_ivec3();
            if cell.cmplt(IVec3::ZERO).any() || cell.cmpge(dimensions).any() {
                return None;
            }
            Some(self.slot(cell.as_uvec3()))
        }
    }

    fn light(position: Vec3, radius: f32) -> GpuLight {
        GpuLight::new(Light {
            position,
            colour: Vec3::ONE,
            radius,
        })
    }

    /// The light indices the shader would be handed at `at`.
    fn reaching(extent: &LightGridExtent, bins: &Bins, at: Vec3) -> Vec<u32> {
        let Some(slot) = extent.slot_at(at) else {
            return Vec::new();
        };
        let range = bins.offsets[slot] as usize..bins.offsets[slot + 1] as usize;
        bins.indices[range].to_vec()
    }

    #[test]
    fn a_cell_offers_every_light_that_reaches_it_and_few_that_do_not() {
        // **The contract is one-sided**: the grid may offer a light that turns out not to reach —
        // it bins by bounding box, and the shader's distance test is what settles it — but it must
        // never withhold one that does, because nothing downstream would notice a light quietly
        // going missing. Checked against the brute-force answer, which is what the shader used to
        // compute for itself at every pixel.
        //
        // Two clusters far apart with a gap between them, which is the arrangement a grid is for:
        // a point in one must never be handed the other's.
        let mut lights = Vec::new();
        for i in 0..24u32 {
            let step = i as f32;
            lights.push(light(Vec3::new(step * 90.0, step * 40.0, 0.0), 300.0));
            lights.push(light(
                Vec3::new(20_000.0 + step * 90.0, step * 40.0, 100.0),
                300.0,
            ));
        }
        let extent = LightGrid::extent_of(&lights);
        let bins = LightGrid::bin(&extent, &lights);

        let mut offered = 0usize;
        let mut reaching_total = 0usize;
        let mut probes = 0usize;
        for i in 0..40u32 {
            for j in 0..40u32 {
                let at = Vec3::new(
                    -500.0 + i as f32 * 550.0,
                    -500.0 + j as f32 * 60.0,
                    (i % 5) as f32 * 60.0,
                );
                let found = reaching(&extent, &bins, at);
                for (index, light) in lights.iter().enumerate() {
                    if (Vec3::from_array(light.position) - at).length() < light.radius {
                        reaching_total += 1;
                        assert!(
                            found.contains(&(index as u32)),
                            "light {index} reaches {at} and the grid did not offer it"
                        );
                    }
                }
                offered += found.len();
                probes += 1;
            }
        }
        // The fixture has to *have* lights reaching its probes, or the assertion above is vacuous.
        assert!(
            reaching_total > 100,
            "the probes were lit only {reaching_total} times between them"
        );
        // And the grid has to be doing something: walking all 48 at every probe would satisfy the
        // contract above perfectly and save nothing at all.
        let average = offered as f32 / probes as f32;
        println!(
            "grid offers {average:.1} of {} lights per probe",
            lights.len()
        );
        assert!(
            average < lights.len() as f32 / 4.0,
            "the grid offered {average} of {} lights per probe, which is barely a reduction",
            lights.len()
        );
    }

    #[test]
    fn the_cell_size_grows_until_the_grid_fits_its_budgets() {
        // One light against another a long way off — the case that decides the cell size, because
        // the grid has to span both. At the finest cell size that is 20,000 cells a side.
        let far = [light(Vec3::ZERO, 256.0), light(Vec3::splat(1.0e7), 256.0)];
        let extent = LightGrid::extent_of(&far);
        assert!(
            extent.cells() <= MAX_CELLS,
            "{} cells is past the budget",
            extent.cells()
        );
        // Doubling from 512, so the size is always 512 times a power of two.
        let cell = 1.0 / extent.scale;
        assert_eq!(cell / FINEST_CELL, (cell / FINEST_CELL).round());
        assert!(cell > FINEST_CELL, "the cell size never grew");

        // And one light so wide it fills whatever grid it is given, which overruns the *entry*
        // budget while the cell count is still comfortable.
        let vast = [light(Vec3::ZERO, 200_000.0)];
        let extent = LightGrid::extent_of(&vast);
        let bins = LightGrid::bin(&extent, &vast);
        assert!(bins.indices.len() <= MAX_ENTRIES, "{}", bins.indices.len());
        // It still has to be findable at its own centre, whatever the grid coarsened to.
        assert_eq!(reaching(&extent, &bins, Vec3::ZERO), vec![0]);
    }

    #[test]
    fn a_scene_with_no_lights_offers_none_anywhere() {
        // The empty grid needs no flag of its own: zero dimensions makes every lookup fall outside,
        // which is the same path a point standing outside a populated grid takes.
        let extent = LightGrid::extent_of(&[]);
        let bins = LightGrid::bin(&extent, &[]);
        assert_eq!(extent, LightGridExtent::default());
        assert_eq!(extent.slot_at(Vec3::ZERO), None);
        assert!(reaching(&extent, &bins, Vec3::ZERO).is_empty());
        assert!(reaching(&extent, &bins, Vec3::splat(1.0e6)).is_empty());
    }
}
