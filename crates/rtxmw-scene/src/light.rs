//! A light placed in a cell.

use glam::Vec3;

/// One point light, in world space.
///
/// Morrowind stores a colour and a radius but no intensity: the original renderer had a fixed
/// attenuation curve and no physical units, so brightness fell out of the curve. The radius is
/// therefore the only control the data gives, and whatever turns this into radiance has to supply
/// the scale itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    pub position: Vec3,
    /// Linear-space RGB in `0..1`.
    pub colour: Vec3,
    /// Reach in world units, beyond which the light contributes nothing.
    pub radius: f32,
}

/// A cell's fixed lighting, which for an interior is most of its illumination.
///
/// Morrowind interiors were authored against a renderer with no global illumination, so the ambient
/// term stands in for every bounce the original engine could not compute. Applying it unchanged on
/// top of real light double-counts — the same problem the pre-lit albedo has, and recorded in
/// `docs/design.md` §5.1 alongside it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Ambient {
    /// Linear-space RGB.
    pub colour: Vec3,
    /// Directional term the original engine used for interiors faking a sun.
    pub sunlight: Vec3,
    pub fog: Vec3,
}
