//! One plume, as the shader marches it.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use rtxmw_scene::ParticleEmitter;

/// How far a plume opens over its own travel, at the least.
///
/// A vent's declination variation is a tenth of a radian, which over three units is a third of one:
/// narrower than the plume is at its foot. The floor is what gives smoke the shape of a plume rather
/// than of a column.
///
/// **A sine and not a tangent, which is not a detail.** The half-angle is the file's own, and six of
/// the game's emitters write `pi/2` for it — `ex_waterfall_mist_01` sprays into a whole hemisphere.
/// A tangent of that is twelve hundred, so a plume built on one opened to a couple of thousand units
/// across, was clipped away entirely by its own bounding sphere, and the mist under Vivec's drains
/// simply did not appear. A sine runs to one and the widest cone the format can state comes out at
/// forty-five degrees of opening, which is a spray.
const LEAST_FLARE: f32 = 0.22;

/// One emitter reduced to the plume it describes, laid out for `scalar` block layout — see
/// `struct Emitter` in `bindings.glsl`.
///
/// **The file's numbers, not the file's drawing.** Everything here comes out of a
/// `NiParticleSystemController`: where it is, which way it lets go, how fast, how long a parcel
/// lives, how wide it opens and what colour it is. What it does *not* carry is the sprite the game
/// drew with, the count of them, the rate they were born at or the texture — a volume needs none of
/// those, and the texture is a photograph of fire at an unrecorded exposure that had no business
/// deciding what a flame looks like. `docs/design.md` §8.103.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuEmitter {
    /// Where the plume starts, in world space.
    pub(crate) origin: [f32; 3],
    /// How far anything can get from that origin, which is the sphere a ray rejects it by.
    pub(crate) reach: f32,
    /// The direction parcels actually leave in, unit length.
    ///
    /// **The emission cone's own axis and not the node's.** Declination and azimuth turn the
    /// emitter's `+Z` into where the gas goes, and six of the game's vents point straight down —
    /// `xact_sotha_steam00` is `pi` off its axis — so a plume built along the node's `+Z` would
    /// send their steam up through the ceiling.
    pub(crate) axis: [f32; 3],
    /// How wide the plume is where it leaves, in world units.
    pub(crate) foot: f32,
    /// Where gravity has carried a parcel by the end of its life, in world units.
    ///
    /// Folded into a displacement here rather than left as an acceleration, because the shader
    /// knows how far up the plume it is and not how long that took.
    pub(crate) drop: [f32; 3],
    /// How far a parcel gets before its life runs out, in world units.
    pub(crate) height: f32,
    /// How long that takes, in seconds — the file's own lifetime.
    ///
    /// **The clock the noise is advected on.** Height over lifetime is the speed the gas rises at,
    /// and reading the field at a position that falls back at exactly that rate is the whole of the
    /// animation: the eddies climb as fast as the file says the parcels do, and nothing is kept
    /// between frames.
    pub(crate) lifetime: f32,
    /// What every parcel is tinted by, over the ramp below.
    pub(crate) colour: [f32; 4],
    /// The tint keyed over the rise — at the foot, at `ramp_mid`, and at the top. For smoke that is
    /// the file's own ramp, which is an albedo; for fire it is the blackbody one the host built out
    /// of the temperature of the gas.
    pub(crate) ramp: [[f32; 4]; 3],
    pub(crate) ramp_mid: f32,
    /// How far the plume opens over its own travel, as the sine of the cone's half-angle.
    pub(crate) flare: f32,
    /// One where it burns rather than being lit — which decides both the shape it takes and whether
    /// it emits or scatters.
    pub(crate) additive: f32,
    /// What separates one emitter's noise from the next, so two candles do not flicker as one.
    pub(crate) seed: u32,
}

impl GpuEmitter {
    /// Flattens a placed emitter into the plume a ray marches.
    ///
    /// **The model's scale rides into the plume rather than beside it.** A half-size brazier is a
    /// half-size flame rising half as far, and folding that in here means the shader never has to
    /// know a placement had a scale at all.
    pub(crate) fn new(emitter: &ParticleEmitter, seed: u32) -> Self {
        let scale = emitter.scale();
        let basis = emitter.placement.matrix3 / scale.max(1e-6);
        // Declination off the emitter's own `+Z`, azimuth about it — the format's own convention.
        let (declination, azimuth) = (emitter.declination, emitter.azimuth);
        let local = Vec3::new(
            declination.sin() * azimuth.cos(),
            declination.sin() * azimuth.sin(),
            declination.cos(),
        );
        let axis = (basis * local).normalize_or(Vec3::Z);
        let lifetime = emitter.lifetime.max(1e-3);
        Self {
            origin: emitter.placement.translation.into(),
            reach: emitter.reach(),
            axis: axis.to_array(),
            // As wide as the box parcels are born across, or as one parcel, whichever is more.
            foot: scale * (0.5 * emitter.spread.length()).max(0.5 * emitter.size),
            drop: (0.5 * emitter.gravity * lifetime * lifetime).to_array(),
            height: (scale * emitter.speed * lifetime).max(scale * emitter.size),
            lifetime,
            colour: emitter.colour.to_array(),
            ramp: emitter.ramp.map(|stop| stop.to_array()),
            ramp_mid: emitter.ramp_mid,
            flare: emitter.declination_variation.sin().abs().max(LEAST_FLARE),
            additive: f32::from(u8::from(emitter.additive)),
            seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::offset_of;

    use glam::{Affine3A, Vec4};
    use rtxmw_scene::ParticleEmitter;

    use super::*;

    /// A flame two units wide leaving straight up at ten units a second for one second.
    fn flame() -> ParticleEmitter {
        ParticleEmitter {
            placement: Affine3A::IDENTITY,
            spread: Vec3::ZERO,
            speed: 10.0,
            speed_variation: 0.0,
            declination: 0.0,
            declination_variation: 0.0,
            azimuth: 0.0,
            colour: Vec4::ONE,
            ramp: [Vec4::ONE; 3],
            ramp_mid: 0.5,
            size: 2.0,
            lifetime: 1.0,
            lifetime_variation: 0.0,
            gravity: Vec3::ZERO,
            additive: true,
        }
    }

    #[test]
    fn the_emitter_matches_the_layout_the_shader_declares() {
        // Thirty-three four-byte fields with nothing between them, which is what `scalar` block
        // layout gives the shader. Every offset below is a place the two could drift apart without
        // either side failing to compile.
        assert_eq!(size_of::<GpuEmitter>(), 33 * 4);
        assert_eq!(offset_of!(GpuEmitter, reach), 12);
        assert_eq!(offset_of!(GpuEmitter, axis), 16);
        assert_eq!(offset_of!(GpuEmitter, foot), 28);
        assert_eq!(offset_of!(GpuEmitter, drop), 32);
        assert_eq!(offset_of!(GpuEmitter, height), 44);
        assert_eq!(offset_of!(GpuEmitter, lifetime), 48);
        assert_eq!(offset_of!(GpuEmitter, colour), 52);
        assert_eq!(offset_of!(GpuEmitter, ramp), 68);
        assert_eq!(offset_of!(GpuEmitter, ramp_mid), 116);
        assert_eq!(offset_of!(GpuEmitter, seed), 128);
    }

    #[test]
    fn a_plume_leaves_along_the_cone_the_file_names_and_scales_with_its_model() {
        // Ten units a second for one second, and a two-unit sprite: forty units of rise with the
        // plume one unit wide where it leaves.
        let plain = GpuEmitter::new(&flame(), 0);
        assert_eq!(plain.height, 10.0);
        assert_eq!(plain.foot, 1.0);
        assert_eq!(plain.axis, [0.0, 0.0, 1.0]);
        assert_eq!(plain.additive, 1.0);

        // **A vent pointing down sends its steam down.** Six of the game's do — declination is `pi`
        // off the node's own axis — and a plume built along that axis instead would put their steam
        // through the ceiling.
        let mut inverted = flame();
        inverted.declination = std::f32::consts::PI;
        let down = GpuEmitter::new(&inverted, 0);
        assert!(
            (Vec3::from_array(down.axis) - Vec3::NEG_Z).length() < 1.0e-6,
            "a vent at pi should point down; it points {:?}",
            down.axis
        );

        // **A half-size brazier is a half-size flame**, rising half as far from a foot half as
        // wide — every length halving together, while the axis stays a unit vector because a
        // direction has no size.
        let mut small = flame();
        small.placement = Affine3A::from_scale(Vec3::splat(0.5));
        let scaled = GpuEmitter::new(&small, 1);
        assert_eq!(scaled.height, 5.0);
        assert_eq!(scaled.foot, 0.5);
        assert_eq!(scaled.axis, [0.0, 0.0, 1.0]);
        assert_eq!(scaled.seed, 1);
    }
}
