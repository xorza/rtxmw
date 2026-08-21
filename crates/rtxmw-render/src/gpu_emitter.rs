//! One particle emitter, as the shader reads it.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use rtxmw_scene::ParticleEmitter;

/// How many of an emitter's slots a ray actually walks.
///
/// **The budget the file was authored against is not always one worth drawing.** Half the shipped
/// emitters ask for thirty or fewer and nine in ten for two hundred; what is above that is
/// `ashcloud.nif` at 7,412, which is drawn analytically as weather instead (§8.87), and Akulakhan's
/// heart. Capping the walk thins those two rather than distorting them: the slots carry independent
/// phases, so the first two hundred and fifty-six are spread evenly through the emitter's life
/// exactly as the whole set would be.
pub(crate) const PARTICLE_LIMIT: u32 = 256;

/// One emitter, laid out for `scalar` block layout — see `struct Emitter` in `particles.glsl`.
///
/// **Every particle is derived from this and the clock, and nothing else is stored.** There is no
/// per-particle state anywhere: slot `i` of an emitter is a closed form in its own hash and the
/// time, so there is nothing to simulate, nothing to step, and nothing to allocate on a frame. That
/// is the same bargain `precipitation.glsl` makes with its lattice and for the same reason — see
/// `docs/design.md` §8.99.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuEmitter {
    /// Where particles are born, in world space.
    pub(crate) origin: [f32; 3],
    /// How far one can get from that origin, which is the sphere a ray rejects the emitter by.
    pub(crate) reach: f32,
    /// The emitter's own axes in world space: declination is measured off `axis_z`, azimuth about
    /// it, and the birth box is stated in all three.
    pub(crate) axis_x: [f32; 3],
    /// Half the width of the box a particle is born across, along `axis_x`, and so on.
    pub(crate) spread_x: f32,
    pub(crate) axis_y: [f32; 3],
    pub(crate) spread_y: f32,
    pub(crate) axis_z: [f32; 3],
    pub(crate) spread_z: f32,
    /// Constant acceleration, world units a second squared.
    pub(crate) gravity: [f32; 3],
    /// How fast one leaves, and the half-width that varies by.
    pub(crate) speed: f32,
    pub(crate) speed_variation: f32,
    pub(crate) declination: f32,
    pub(crate) declination_variation: f32,
    pub(crate) azimuth: f32,
    pub(crate) azimuth_variation: f32,
    /// The tint over the texture, and how much of it there is.
    pub(crate) colour: [f32; 4],
    /// The tint keyed over a life — at birth, at `ramp_mid`, and at death. Three keys because every
    /// one of the game's is three, and linear because every one of them is linear.
    pub(crate) ramp: [[f32; 4]; 3],
    pub(crate) ramp_mid: f32,
    /// How wide one is drawn at its fullest, in world units.
    pub(crate) size: f32,
    /// How long one lives, and the half-width that varies by.
    pub(crate) lifetime: f32,
    pub(crate) lifetime_variation: f32,
    /// Seconds to reach full size, and seconds to vanish, as fractions of a life.
    pub(crate) grow: f32,
    pub(crate) fade: f32,
    /// How fast the sprite turns, in radians a second.
    pub(crate) spin: f32,
    /// How many slots to walk — the emitter's own capacity, held under [`PARTICLE_LIMIT`].
    pub(crate) count: u32,
    /// Which material it draws with, which is where the shader finds its texture.
    pub(crate) material: u32,
    /// One where it adds to the frame rather than covering it. A float rather than a flag because
    /// every use of it is a blend between the two branches.
    pub(crate) additive: f32,
    /// Distinguishes one emitter's hash stream from the next, so two candles do not flicker
    /// together. Its index at upload, which is stable for as long as the cell is resident.
    pub(crate) seed: u32,
}

impl GpuEmitter {
    /// Flattens a placed emitter into the form a ray reads.
    ///
    /// **The model's scale rides into the particles rather than beside them.** A half-size brazier
    /// is a half-size flame moving half as fast over half the distance, and folding that in here
    /// means the shader never has to know a placement had a scale at all.
    pub(crate) fn new(emitter: &ParticleEmitter, seed: u32) -> Self {
        let scale = emitter.scale();
        let axis = |column: Vec3| (column / scale.max(1e-6)).to_array();
        let basis = emitter.placement.matrix3;
        Self {
            origin: emitter.placement.translation.into(),
            reach: emitter.reach(),
            axis_x: axis(basis.x_axis.into()),
            spread_x: 0.5 * scale * emitter.spread.x,
            axis_y: axis(basis.y_axis.into()),
            spread_y: 0.5 * scale * emitter.spread.y,
            axis_z: axis(basis.z_axis.into()),
            spread_z: 0.5 * scale * emitter.spread.z,
            gravity: emitter.gravity.to_array(),
            speed: scale * emitter.speed,
            speed_variation: scale * emitter.speed_variation,
            declination: emitter.declination,
            declination_variation: emitter.declination_variation,
            azimuth: emitter.azimuth,
            azimuth_variation: emitter.azimuth_variation,
            colour: emitter.colour.to_array(),
            ramp: emitter.ramp.map(|stop| stop.to_array()),
            ramp_mid: emitter.ramp_mid,
            size: scale * emitter.size,
            lifetime: emitter.lifetime,
            lifetime_variation: emitter.lifetime_variation.min(emitter.lifetime),
            grow: emitter.grow,
            fade: emitter.fade,
            spin: emitter.spin,
            count: emitter.capacity.min(PARTICLE_LIMIT),
            material: emitter.material,
            additive: f32::from(u8::from(emitter.additive)),
            seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::offset_of;

    use glam::{Affine3A, Vec3, Vec4};
    use rtxmw_scene::ParticleEmitter;

    use super::*;

    /// A flame two units wide, born at the origin, leaving straight up at ten units a second for
    /// one second — so it reaches 10 + 1 = 11 units.
    fn flame() -> ParticleEmitter {
        ParticleEmitter {
            placement: Affine3A::IDENTITY,
            spread: Vec3::ZERO,
            speed: 10.0,
            speed_variation: 0.0,
            declination: 0.0,
            declination_variation: 0.0,
            azimuth: 0.0,
            azimuth_variation: 0.0,
            colour: Vec4::ONE,
            ramp: [Vec4::ONE; 3],
            ramp_mid: 0.5,
            size: 2.0,
            birth_rate: 10.0,
            lifetime: 1.0,
            lifetime_variation: 0.0,
            capacity: 10,
            gravity: Vec3::ZERO,
            grow: 0.0,
            fade: 0.0,
            spin: 0.0,
            material: 0,
            additive: true,
        }
    }

    #[test]
    fn the_emitter_matches_the_layout_the_shader_declares() {
        // Fifty-two four-byte fields with nothing between them, which is what `scalar` block
        // layout gives the shader — see `struct Emitter` in `bindings.glsl`. Every offset below is
        // a place the two could drift apart without either side failing to compile.
        assert_eq!(size_of::<GpuEmitter>(), 52 * 4);
        assert_eq!(offset_of!(GpuEmitter, reach), 12);
        assert_eq!(offset_of!(GpuEmitter, axis_x), 16);
        assert_eq!(offset_of!(GpuEmitter, gravity), 64);
        assert_eq!(offset_of!(GpuEmitter, colour), 100);
        assert_eq!(offset_of!(GpuEmitter, ramp), 116);
        assert_eq!(offset_of!(GpuEmitter, ramp_mid), 164);
        assert_eq!(offset_of!(GpuEmitter, count), 192);
        assert_eq!(offset_of!(GpuEmitter, seed), 204);
    }

    #[test]
    fn a_scaled_placement_scales_the_flame_and_leaves_its_axes_alone() {
        let plain = GpuEmitter::new(&flame(), 0);
        assert_eq!(plain.reach, 11.0);
        assert_eq!(plain.axis_z, [0.0, 0.0, 1.0]);
        assert_eq!(plain.additive, 1.0);

        // **A half-size brazier is a half-size flame**, moving half as fast over half the distance
        // — so every length halves together and the reach with them, while the axes stay unit
        // vectors because a direction has no size.
        let mut small = flame();
        small.placement = Affine3A::from_scale(Vec3::splat(0.5));
        let scaled = GpuEmitter::new(&small, 1);
        assert_eq!(scaled.speed, 5.0);
        assert_eq!(scaled.size, 1.0);
        assert_eq!(scaled.reach, 5.5);
        assert_eq!(scaled.axis_z, [0.0, 0.0, 1.0]);
        assert_eq!(scaled.seed, 1);

        // A cap rather than a distortion: the slots carry independent phases, so drawing the first
        // 256 of a larger set thins the emitter evenly instead of truncating its life.
        let mut crowded = flame();
        crowded.capacity = 7_412;
        assert_eq!(GpuEmitter::new(&crowded, 0).count, PARTICLE_LIMIT);
    }
}
