//! One point light, as the shader reads it.

use bytemuck::{Pod, Zeroable};
use rtxmw_scene::Light;

/// One point light.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuLight {
    pub(crate) position: [f32; 3],
    /// Reach in world units. Nothing beyond this receives any of the light.
    pub(crate) radius: f32,
    /// Linear RGB, already scaled by the intensity the record does not carry.
    pub(crate) colour: [f32; 3],
    /// How large the emitter itself is, which is what gives its shadows a penumbra.
    pub(crate) source_radius: f32,
}

/// Converts Morrowind's radius into radiant intensity.
///
/// The record gives a colour and a reach and no brightness at all — the original renderer's fixed
/// attenuation curve supplied that, so there is no value here to be faithful to. Scaling by radius
/// squared is what makes a large lamp and a small candle differ by their reach rather than by an
/// arbitrary per-light number, and the constant sets how bright a light is at half its radius.
///
/// Tuned by eye, and provisional: vanilla albedo already has light painted into it, so every one of
/// these is fighting illumination that is already in the texture. See `docs/design.md` §5.1 — the
/// de-lighting spike is what makes this number mean anything.
///
/// The pi is the Lambertian BRDF's `1/pi`, which the shader now applies where it belongs. It lived
/// here while direct light was the only term and a single scale on it was unobservable; with an
/// indirect term integrating over the hemisphere the ratio between the two became real, so the
/// factor moved and this compensates. The lights look exactly as they did.
const INTENSITY: f32 = 0.25 * std::f32::consts::PI;

/// The emitter's own size, as a fraction of its reach.
///
/// Morrowind records no such thing — a light is a point with a falloff curve, and its shadows in
/// the original engine were whatever the shadow-blob decal looked like. A real emitter has area,
/// and that area is the only reason a shadow has a soft edge, so one has to be invented. A fraction
/// of reach rather than a constant keeps a lantern's penumbra wider than a candle's, which is the
/// relationship the sizes would have had.
const SOURCE_FRACTION: f32 = 0.08;

/// Floor on the emitter size, in world units.
///
/// About 14 cm at Morrowind's scale — roughly a candle flame. Without it the smallest lights would
/// come out as points again and their shadows would snap back to hard edges.
const MIN_SOURCE_RADIUS: f32 = 10.0;

impl GpuLight {
    /// Flattens a scene light, folding its intensity into the colour.
    pub(crate) fn new(light: Light) -> Self {
        let scale = light.radius * light.radius * INTENSITY;
        Self {
            position: light.position.to_array(),
            radius: light.radius,
            colour: (light.colour * scale).to_array(),
            source_radius: (light.radius * SOURCE_FRACTION).max(MIN_SOURCE_RADIUS),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn the_light_matches_the_layout_the_shader_declares() {
        assert_eq!(size_of::<GpuLight>(), 32);
    }

    #[test]
    fn a_wider_light_is_brighter_in_proportion_to_its_reach() {
        let small = GpuLight::new(Light {
            position: Vec3::ZERO,
            colour: Vec3::ONE,
            radius: 64.0,
        });
        let large = GpuLight::new(Light {
            position: Vec3::ZERO,
            colour: Vec3::ONE,
            radius: 128.0,
        });

        // Twice the radius is four times the intensity, so the illumination at the *same* fraction
        // of each light's reach comes out equal — which is what makes radius the only control the
        // data needs to give.
        assert_eq!(large.colour[0] / small.colour[0], 4.0);
        assert_eq!(small.colour[0], 64.0 * 64.0 * INTENSITY);

        // Colour survives the scaling as a ratio.
        let warm = GpuLight::new(Light {
            position: Vec3::ZERO,
            colour: Vec3::new(1.0, 0.5, 0.25),
            radius: 64.0,
        });
        assert_eq!(warm.colour[1] / warm.colour[0], 0.5);
        assert_eq!(warm.radius, 64.0);
    }

    #[test]
    fn the_emitter_size_follows_reach_but_never_reaches_zero() {
        let of = |radius: f32| {
            GpuLight::new(Light {
                position: Vec3::ZERO,
                colour: Vec3::ONE,
                radius,
            })
            .source_radius
        };

        // A lantern's emitter is wider than a candle's, so its penumbra is softer.
        assert!(of(512.0) > of(256.0));
        assert_eq!(of(512.0), 512.0 * SOURCE_FRACTION);

        // Below the floor the fraction would shrink to a point and the shadows would snap hard.
        assert_eq!(of(64.0), MIN_SOURCE_RADIUS);
        assert!(of(1.0) > 0.0);
    }
}
