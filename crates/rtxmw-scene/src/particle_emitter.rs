//! The flames, steam and smoke a model emits, taken out of its own emitters.

use glam::{Affine3A, Vec3, Vec4};
use rtxmw_nif::{Block, ColourKey, NifFile, ParticleEffect, ParticleSystem};

use crate::blackbody;
use crate::material::Properties;
use crate::rig;

/// What a flame is at its hottest and at its coldest, in kelvin.
///
/// **A flame's colour is its temperature, and the game has no say in it.** The shipped art paints
/// fire as a photograph — `tx_firealpha10` is a pinkish tan puff, which is what a camera at some
/// unrecorded exposure made of it — so there is nothing in the file to be faithful to, and the
/// thing that *is* true of a flame is how hot it is. A candle's envelope runs about 1800 K where
/// the fuel burns and falls to around 1100 K by the time the gas has risen and cooled to where it
/// stops glowing.
///
/// **The fade comes out of the same number.** A blackbody radiates as the fourth power of its
/// temperature, so the tip at 1100 K sends out `(1100/1800)^4` — a seventh — of what the base does,
/// and a flame goes out at the top without anything being asked to fade it. See
/// [`blackbody::colour`], which gives the hue at unit brightness so this can supply the level.
const FLAME_HOT: f32 = 1800.0;
const FLAME_COOL: f32 = 1100.0;

/// How deep a modifier chain is followed before it is called a cycle.
///
/// The longest the shipped content writes is three — gravity, the size ramp and a spin — so this is
/// a guard on a malformed file rather than a limit on a real one.
const MODIFIERS: usize = 8;

/// One emitter: a candle's flame, a vent's steam, a chimney's smoke.
///
/// **Not geometry, and never in the acceleration structure.** A particle system carries no
/// triangles at all — the sprites *are* the drawing — and putting a few thousand sub-pixel quads
/// into a structure that is rebuilt as cells arrive would cost more than it could ever return. They
/// go into the transparency layer instead, beside the rain, and are drawn analytically from these
/// numbers: `docs/design.md` §8.99.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleEmitter {
    /// Where particles are born, and the axes [`declination`] is measured against.
    ///
    /// The emitter node's own placement, which is a node of its own in every shipped file —
    /// `CandleFlame Emitter` sits beside the flame it feeds rather than being it.
    ///
    /// [`declination`]: Self::declination
    pub placement: Affine3A,
    /// The box they are born across, in the emitter's own axes and full width.
    pub spread: Vec3,
    /// How fast one leaves, in world units a second, and the half-width that varies by.
    pub speed: f32,
    pub speed_variation: f32,
    /// Where it leaves towards: a polar angle off the emitter's `+Z` and an azimuth about it, both
    /// in radians, each with the half-width it varies by.
    pub declination: f32,
    pub declination_variation: f32,
    pub azimuth: f32,
    /// What every particle is tinted by, over its texture.
    pub colour: Vec4,
    /// The tint again, keyed over a life: at birth, at [`ramp_mid`], and at death.
    ///
    /// **Three keys and always three**, which is what every one of the game's 85 colour ramps
    /// carries, all of them linear — so this is the ramp itself rather than a sampling of it. It is
    /// also the whole of what makes smoke smoke: an alpha-blended emitter fades out through here
    /// and an additive one does not fade at all, which is why every ramp in the game belongs to one
    /// of the former and none to one of the latter. White throughout is no ramp.
    ///
    /// [`ramp_mid`]: Self::ramp_mid
    pub ramp: [Vec4; 3],
    /// Where the middle key sits in a life. The shipped ones are 0.05, 0.10, 0.15, 0.50 and 0.91 —
    /// a spread wide enough that anything evenly spaced would have thrown the shape away.
    pub ramp_mid: f32,
    /// How wide one is drawn at its fullest, in world units.
    pub size: f32,
    /// How long a parcel lives, which with the speed is how far the plume reaches.
    pub lifetime: f32,
    pub lifetime_variation: f32,
    /// Constant acceleration, in world units a second squared.
    pub gravity: Vec3,
    /// Whether it *adds* to the frame rather than covering it — see [`rtxmw_nif::AlphaProperty`].
    ///
    /// The whole of the difference between a flame and a puff of smoke, and it decides both how the
    /// two composite and how they are lit: 474 of the game's 678 emitters add, and what adds is its
    /// own light rather than something the room shines on.
    pub additive: bool,
}

impl ParticleEmitter {
    /// Every emitter in `nif`, in the model's own space.
    ///
    /// **A walk of its own, and it costs nothing to keep separate.** The mesh walk exists to put
    /// vertices in an array and hands back where each block's landed; this needs a node's world
    /// placement and no vertices at all, so the two share no answer that could disagree. What it
    /// does need that a single walk cannot give is the *emitter* node's placement, which is
    /// somewhere else in the graph entirely and may be visited long after the system naming it.
    pub(crate) fn collect(nif: &NifFile, out: &mut Vec<Self>) {
        let mut places = vec![None; nif.blocks().len()];
        let mut found: Vec<(usize, Properties)> = Vec::new();
        for &root in nif.roots() {
            place(
                nif,
                root,
                Affine3A::IDENTITY,
                Properties::default(),
                0,
                &mut places,
                &mut found,
            );
        }
        for (block, properties) in found {
            let Some(system) = system_of(nif, block) else {
                continue;
            };
            // The emitter node where the file names one and the particle node where it does not:
            // the birth point is a place in the graph, and every shipped file gives it its own.
            let Some(placement) = system
                .emitter
                .index()
                .and_then(|at| places.get(at).copied().flatten())
                .or(places[block])
            else {
                continue;
            };
            let mut emitter = Self {
                placement,
                spread: Vec3::from_array(system.spread),
                speed: system.speed,
                speed_variation: system.speed_variation,
                declination: system.declination,
                declination_variation: system.declination_variation,
                azimuth: system.azimuth,
                colour: Vec4::from_array(system.colour),
                ramp: [Vec4::ONE; 3],
                ramp_mid: 0.5,
                size: system.size,
                lifetime: system.lifetime,
                lifetime_variation: system.lifetime_variation,
                gravity: Vec3::ZERO,
                additive: properties.adds(nif),
            };
            emitter.absorb_modifiers(nif, system);
            if emitter.additive {
                emitter.burn();
            }
            if emitter.is_drawable() {
                out.push(emitter);
            }
        }
    }

    /// Folds the modifier chain into the emitter, which is where a size ramp and a pull belong.
    fn absorb_modifiers(&mut self, nif: &NifFile, system: &ParticleSystem) {
        let mut link = system.modifier;
        for _ in 0..MODIFIERS {
            let Some(Block::ParticleModifier(modifier)) = nif.resolve(link) else {
                return;
            };
            match modifier.effect {
                // **The direction is the model's, not the world's**, which is the only reading that
                // survives a torch bracketed to a wall: what the artist wrote is where the smoke
                // goes relative to the thing that emits it.
                ParticleEffect::Gravity(gravity) if !gravity.towards_point => {
                    self.gravity = self.placement.matrix3
                        * Vec3::from_array(gravity.direction).normalize_or_zero()
                        * gravity.force;
                }
                // A pull towards a point, which nothing shipped uses.
                ParticleEffect::Gravity(_) => {}
                // The size ramp and the spin shaped a *sprite*; a plume is shaped by its own
                // profile and roiled by noise, so neither has anything to say here.
                ParticleEffect::GrowFade { .. } => {}
                ParticleEffect::Colour { keys } => {
                    if let Some(Block::Colour(track)) = nif.resolve(keys)
                        && !track.keys.is_empty()
                    {
                        // The middle key's own time, so a three-key ramp is reproduced exactly
                        // rather than resampled — see [`Self::ramp_mid`].
                        self.ramp_mid = track.keys[track.keys.len() / 2].time.clamp(0.0, 1.0);
                        self.ramp = [0.0, self.ramp_mid, 1.0].map(|at| sample(&track.keys, at));
                    }
                }
                ParticleEffect::Rotation { .. } => {}
            }
            link = modifier.next;
        }
    }

    /// Replaces the ramp with what a cooling flame actually radiates.
    ///
    /// **Only for what adds to the frame**, which is the classification that already separates fire
    /// from smoke: 474 of the game's 678 emitters blend `SRC_ALPHA, ONE`, and a thing that adds its
    /// own light to a room is a thing that is burning. A puff of smoke keeps the file's own ramp,
    /// which for it is an albedo rather than an emission — see [`Self::ramp`].
    fn burn(&mut self) {
        self.ramp = [0.0, 0.5, 1.0].map(|through| {
            let kelvin = FLAME_HOT + (FLAME_COOL - FLAME_HOT) * through;
            // **Stefan-Boltzmann against the base, so the ramp carries the fade as well as the
            // hue.** `blackbody::colour` comes back at unit luminance, so the base is exactly as
            // bright as an unramped emitter was and only its colour has changed — what the fourth
            // power adds is how much less the cooler gas at the top is radiating.
            let power = (kelvin / FLAME_HOT).powi(4);
            (blackbody::colour(kelvin) * power).extend(1.0)
        });
        self.ramp_mid = 0.5;
    }

    /// Whether it would put anything on the screen at all.
    ///
    /// A dead emitter is not an error — a file may carry one switched off, and one with no lifetime
    /// divides by zero downstream — so it is dropped here rather than uploaded and skipped per
    /// pixel by every ray in the frame.
    fn is_drawable(&self) -> bool {
        self.lifetime > 0.0 && self.size > 0.0 && self.colour.w > 0.0
    }

    /// The same emitter placed into the world by a cell reference.
    ///
    /// The gravity turns with it for the reason it was rotated in the first place: it is written
    /// against the model, and a brazier stood on its side takes its smoke with it.
    pub(crate) fn placed(&self, transform: Affine3A) -> Self {
        Self {
            placement: transform * self.placement,
            gravity: transform.matrix3 * self.gravity,
            ..*self
        }
    }

    /// How far from [`placement`]'s origin a particle can get, in world units.
    ///
    /// **What a ray tests before it tests anything inside.** A candle's is five units and a steam
    /// jet's a thousand, so one sphere rejects the emitter for almost every pixel of the frame —
    /// which is the whole reason this can be a flat list rather than a structure.
    ///
    /// [`placement`]: Self::placement
    pub fn reach(&self) -> f32 {
        let life = self.lifetime + self.lifetime_variation;
        let speed = self.speed + self.speed_variation;
        let scale = self.scale();
        0.5 * (self.placement.matrix3 * self.spread).length()
            + scale * (speed * life + 0.5 * self.gravity.length() * life * life)
            + 0.5 * scale * self.size
    }

    /// The model's own scale, which a flame is as subject to as the candle under it.
    pub fn scale(&self) -> f32 {
        // Uniform in every shipped placement, so one axis answers for all three.
        self.placement.matrix3.x_axis.length()
    }
}

/// A linear colour track read at `at`, which for the shipped three-key ramps is one of its keys.
fn sample(keys: &[ColourKey], at: f32) -> Vec4 {
    let value = |key: &ColourKey| Vec4::from_array(key.value);
    let after = keys.iter().position(|key| key.time >= at);
    match after {
        None => value(&keys[keys.len() - 1]),
        Some(0) => value(&keys[0]),
        Some(index) => {
            let (before, here) = (&keys[index - 1], &keys[index]);
            let span = here.time - before.time;
            let across = if span > 0.0 {
                (at - before.time) / span
            } else {
                0.0
            };
            value(before).lerp(value(here), across)
        }
    }
}

/// Records where each block sits, and which of them draw particles.
///
/// Stops at a hidden node the way the mesh walk does: a flame the artist switched off is switched
/// off for the same reason its geometry is.
fn place(
    nif: &NifFile,
    link: rtxmw_nif::Link,
    parent: Affine3A,
    inherited: Properties,
    depth: u32,
    places: &mut [Option<Affine3A>],
    found: &mut Vec<(usize, Properties)>,
) {
    const MAX_DEPTH: u32 = 64;
    let Some(at) = link.index() else { return };
    if depth > MAX_DEPTH {
        return;
    }
    match nif.resolve(link) {
        Some(Block::Node(node)) => {
            if node.av.is_hidden() {
                return;
            }
            let here = parent * rig::affine_of(&node.av.transform);
            places[at] = Some(here);
            let properties = inherited.overridden_by(nif, &node.av.properties);
            for &child in &node.children {
                place(nif, child, here, properties, depth + 1, places, found);
            }
        }
        Some(Block::Particles(particles)) => {
            if particles.av.is_hidden() {
                return;
            }
            places[at] = Some(parent * rig::affine_of(&particles.av.transform));
            found.push((at, inherited.overridden_by(nif, &particles.av.properties)));
        }
        _ => {}
    }
}

/// The particle system on `block`'s controller chain, of which every shipped file has exactly one.
fn system_of(nif: &NifFile, block: usize) -> Option<&ParticleSystem> {
    let Some(Block::Particles(particles)) = nif.blocks().get(block) else {
        return None;
    };
    let mut link = particles.av.net.controller;
    for _ in 0..MODIFIERS {
        match nif.resolve(link)? {
            Block::ParticleSystem(system) => return Some(system),
            Block::Controller(controller) => link = controller.next,
            _ => return None,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `light_de_candle_25.nif` writes, read off the file itself.
    ///
    /// One emitter: 22 slots at 34.02 a second living two thirds of one apiece — and 34.02 × 2/3
    /// is 22.7, which is where the capacity came from — leaving at 5.25 units a second straight up,
    /// drawn 2.5 units wide, and shrinking to nothing over a fade it never reaches the end of.
    const CANDLE: &str = "meshes/l/light_de_candle_25.nif";

    #[test]
    fn a_candle_emits_one_flame_that_goes_straight_up_and_reaches_no_further_than_it_should() {
        let Some(vfs) = rtxmw_vfs::morrowind_archives() else {
            eprintln!("skipping: the game is not installed");
            return;
        };
        let bytes = vfs.read(CANDLE).expect("the candle should read");
        let nif = NifFile::parse(&bytes).expect("it should parse");
        let mut found = Vec::new();
        ParticleEmitter::collect(&nif, &mut found);

        assert_eq!(found.len(), 1, "a candle has one flame");
        let flame = found[0];
        assert_eq!(flame.size, 2.5);
        assert!(
            flame.additive,
            "a flame adds its own light rather than covering the room"
        );
        assert_eq!(flame.declination, 0.0, "it goes straight up");

        // **How far a parcel gets, computed rather than guessed**: nothing to be born across, and
        // 5.25 units a second for two thirds of one is 3.5 — plus half of a 2.5-wide parcel, so
        // 4.75, and nothing outside that sphere can belong to this candle.
        assert_eq!(flame.speed, 5.25);
        assert_eq!(flame.lifetime, 2.0 / 3.0);
        assert_eq!(flame.spread, Vec3::ZERO);
        assert_eq!(flame.gravity, Vec3::ZERO);
        assert!(
            (flame.reach() - 4.75).abs() < 1.0e-5,
            "a candle flame reaches {} units",
            flame.reach()
        );

        // **And its colour is its temperature.** The file's own ramp is replaced for anything that
        // burns — see `ParticleEmitter::burn` — so what is here is a blackbody cooling from 1800 K
        // at the wick to 1100 K where it stops glowing. Every stop is red over green over nothing,
        // because sRGB has no red deep enough for a flame and clamps the blue away entirely.
        for stop in flame.ramp {
            assert!(
                stop.x > stop.y && stop.z == 0.0,
                "a flame stop came out {stop}"
            );
        }
        // Stefan-Boltzmann is what fades it: `(1100/1800)^4` is 0.1395, so the tip radiates a
        // seventh of what the wick does without anything being told to fade.
        let luminance = |stop: Vec4| stop.truncate().dot(Vec3::new(0.2126, 0.7152, 0.0722));
        assert!((luminance(flame.ramp[0]) - 1.0).abs() < 0.01);
        assert!((luminance(flame.ramp[2]) - 0.1395).abs() < 0.01);

        // **Placed, it is the same flame somewhere else.** A cell reference is a rotation and a
        // translation, and what has to survive both is that the flame still leaves along the
        // emitter's own axis rather than along the world's.
        let turned = Affine3A::from_rotation_x(std::f32::consts::FRAC_PI_2);
        let placed = flame.placed(turned);
        assert!((placed.reach() - flame.reach()).abs() < 1.0e-3);
        let axis = |emitter: &ParticleEmitter| emitter.placement.matrix3.z_axis.normalize();
        assert!(
            (axis(&placed) - turned.matrix3 * axis(&flame)).length() < 1.0e-5,
            "a flame on its side leaves along the candle, not along the world"
        );
    }
}
