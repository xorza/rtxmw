//! What a surface is made of, as far as vanilla Morrowind describes it.

use glam::Vec3;
use rtxmw_nif::{Block, Link, NifFile};

use crate::material_table::MaterialTable;

/// Index into a scene's texture path list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub u32);

/// How a surface's alpha is interpreted.
///
/// The distinction matters to the acceleration structure, not just to shading: an opaque geometry
/// can be built with `OPAQUE` and skip the any-hit path entirely, while a masked one must run the
/// candidate loop on every intersection. Getting it wrong makes foliage into solid rectangles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlphaMode {
    /// Every texel is fully present.
    Opaque,
    /// Texels below the threshold are absent, as a fraction of full alpha.
    Mask(f32),
    /// Alpha weights the surface against what is behind it.
    Blend,
}

/// One surface description, resolved from a NIF's property stack.
///
/// Vanilla Morrowind has no normal or roughness maps — the NIF format at this version cannot carry
/// them — so this is a base colour texture plus the fixed-function colours the original renderer
/// used. See `docs/design.md` §5.1: the albedo is also pre-lit, which is a content problem rather
/// than one more slot in this struct.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Base colour texture, or `None` for geometry drawn with vertex colour alone.
    pub base_colour: Option<TextureId>,
    pub diffuse: Vec3,
    pub emissive: Vec3,
    /// Constant opacity from the material property, before any texture alpha.
    pub opacity: f32,
    pub alpha: AlphaMode,
}

impl Default for Material {
    /// An untextured, fully opaque white surface — what a geometry with no properties at all draws
    /// as, rather than black.
    fn default() -> Self {
        Self {
            base_colour: None,
            diffuse: Vec3::ONE,
            emissive: Vec3::ZERO,
            opacity: 1.0,
            alpha: AlphaMode::Opaque,
        }
    }
}

/// The property links in effect at a point in the node graph.
///
/// NIF properties are inherited: one set on a `NiNode` applies to everything beneath it until a
/// descendant sets its own of the same kind. Resolving them at the geometry rather than where they
/// are declared is what makes a whole building share one texture property.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Properties {
    texturing: Link,
    material: Link,
    alpha: Link,
}

impl Properties {
    /// This set with any property in `links` replacing the inherited one of its kind.
    pub(crate) fn overridden_by(&self, nif: &NifFile, links: &[Link]) -> Self {
        let mut merged = *self;
        for &link in links {
            match nif.resolve(link) {
                Some(Block::Texturing(_)) => merged.texturing = link,
                Some(Block::Material(_)) => merged.material = link,
                Some(Block::Alpha(_)) => merged.alpha = link,
                _ => {}
            }
        }
        merged
    }

    /// Turns the links into a material, interning any texture it names.
    pub(crate) fn resolve(&self, nif: &NifFile, table: &mut MaterialTable) -> Material {
        let mut material = Material::default();

        if let Some(Block::Material(properties)) = nif.resolve(self.material) {
            material.diffuse = Vec3::from_array(properties.diffuse);
            material.emissive = Vec3::from_array(properties.emissive);
            material.opacity = properties.alpha;
        }

        if let Some(Block::Texturing(texturing)) = nif.resolve(self.texturing)
            && let Some(slot) = &texturing.base
            && let Some(Block::SourceTexture(source)) = nif.resolve(slot.source)
            && source.external
            && !source.file_name.is_empty()
        {
            material.base_colour = Some(table.intern_texture(&texture_path(&source.file_name)));
        }

        if let Some(Block::Alpha(alpha)) = nif.resolve(self.alpha) {
            // Testing is checked first because it is the mode the acceleration structure cares
            // about: a surface doing both still has to run the any-hit path for the cutout, and
            // treating it as blended would lose the threshold.
            material.alpha = if alpha.tests() {
                AlphaMode::Mask(f32::from(alpha.threshold) / 255.0)
            } else if alpha.blends() {
                AlphaMode::Blend
            } else {
                AlphaMode::Opaque
            };
        }

        material
    }
}

/// Turns a NIF's texture name into a virtual file system path.
///
/// Two fixups the original data needs, both quirks rather than conventions: the name is relative to
/// `textures/`, and it routinely claims an extension the shipped file does not have — the game
/// converted its art to DDS and never updated the references.
fn texture_path(name: &str) -> String {
    let slashed = name.replace('\\', "/");
    let stem = slashed
        .rsplit_once('.')
        .map_or(slashed.as_str(), |(s, _)| s);
    if stem.to_ascii_lowercase().starts_with("textures/") {
        format!("{stem}.dds")
    } else {
        format!("textures/{stem}.dds")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_texture_name_gains_its_directory_and_loses_its_extension() {
        // The shipped art is DDS but the references were never updated, so a `.tga` name has to be
        // rewritten rather than looked up as written.
        assert_eq!(texture_path("tx_stone.tga"), "textures/tx_stone.dds");
        assert_eq!(texture_path("tx_stone.dds"), "textures/tx_stone.dds");
        // Backslashes are how the original tools wrote subdirectories.
        assert_eq!(
            texture_path("bookart\\book.tga"),
            "textures/bookart/book.dds"
        );
        // A name that already carries the directory must not gain a second one.
        assert_eq!(
            texture_path("textures/tx_stone.tga"),
            "textures/tx_stone.dds"
        );
        assert_eq!(
            texture_path("Textures\\tx_Stone.TGA"),
            "Textures/tx_Stone.dds"
        );
        // No extension at all is still a DDS lookup.
        assert_eq!(texture_path("tx_stone"), "textures/tx_stone.dds");
    }
}
