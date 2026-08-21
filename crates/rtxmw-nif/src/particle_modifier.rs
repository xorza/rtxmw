//! What acts on a particle between its birth and its death.

use crate::cursor::{Cursor, Link};
use crate::error::Result;

/// One link in a particle system's modifier chain.
///
/// **A chain rather than a set**, so the order is the file's: a system names its head and each
/// modifier the next. Nothing shipped depends on the order — no two of these touch the same
/// quantity — but following `next` is how the list is read at all.
#[derive(Debug, Clone)]
pub struct ParticleModifier {
    pub next: Link,
    pub effect: ParticleEffect,
}

/// What one modifier does.
///
/// Four of the six the format defines, which is every one the shipped content uses more than once:
/// there is a single `NiPlanarCollider` in the whole game and no `NiParticleBomb` at all.
#[derive(Debug, Clone, Copy)]
pub enum ParticleEffect {
    Gravity(Gravity),
    /// The size ramp: in over `grow` seconds, out over `fade`. On 245 of the 272 files that emit
    /// anything, which makes it the one a particle is actually shaped by.
    GrowFade {
        grow: f32,
        fade: f32,
    },
    /// Colour over life, keyed in a `NiColorData`.
    Colour {
        keys: Link,
    },
    /// Spin about a fixed axis, in radians a second.
    Rotation {
        speed: f32,
    },
}

/// A constant pull, or a pull towards a point.
#[derive(Debug, Clone, Copy)]
pub struct Gravity {
    /// How fast the force falls off with distance. Zero is no falloff, which is every shipped use.
    pub decay: f32,
    /// Acceleration, in world units a second squared.
    pub force: f32,
    /// Whether `direction` is a direction to pull along, or a point to pull towards.
    pub towards_point: bool,
    pub position: [f32; 3],
    pub direction: [f32; 3],
}

impl ParticleModifier {
    /// Reads a `NiGravity`, consuming exactly its bytes.
    pub(crate) fn read_gravity(cursor: &mut Cursor<'_>) -> Result<Self> {
        let next = Self::read_header(cursor)?;
        Ok(Self {
            next,
            effect: ParticleEffect::Gravity(Gravity {
                decay: cursor.f32()?,
                force: cursor.f32()?,
                // Zero is a direction and one is a point; the format writes it as a word.
                towards_point: cursor.u32()? != 0,
                position: cursor.vec3()?,
                direction: cursor.vec3()?,
            }),
        })
    }

    /// Reads a `NiParticleGrowFade`.
    pub(crate) fn read_grow_fade(cursor: &mut Cursor<'_>) -> Result<Self> {
        let next = Self::read_header(cursor)?;
        Ok(Self {
            next,
            effect: ParticleEffect::GrowFade {
                grow: cursor.f32()?,
                fade: cursor.f32()?,
            },
        })
    }

    /// Reads a `NiParticleColorModifier`.
    pub(crate) fn read_colour(cursor: &mut Cursor<'_>) -> Result<Self> {
        let next = Self::read_header(cursor)?;
        Ok(Self {
            next,
            effect: ParticleEffect::Colour {
                keys: cursor.link()?,
            },
        })
    }

    /// Reads a `NiParticleRotation`.
    pub(crate) fn read_rotation(cursor: &mut Cursor<'_>) -> Result<Self> {
        let next = Self::read_header(cursor)?;
        cursor.skip(1)?; // Random-initial-axis flag.
        cursor.skip(12)?; // The axis itself, which a screen-facing sprite has no use for.
        Ok(Self {
            next,
            effect: ParticleEffect::Rotation {
                speed: cursor.f32()?,
            },
        })
    }

    /// The two links every modifier begins with, returning the one that continues the chain.
    fn read_header(cursor: &mut Cursor<'_>) -> Result<Link> {
        let next = cursor.link()?;
        cursor.skip(4)?; // A controller on the modifier itself; nothing shipped has one.
        Ok(next)
    }
}
