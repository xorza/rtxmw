//! What a surface is made of, as far as vanilla Morrowind describes it.

use glam::{Vec2, Vec3};
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

/// Which shading model a surface is rendered with.
///
/// Not a parameter of one model but a choice between models: water is not a diffuse surface with
/// unusual settings, it reflects and refracts and absorbs along the path behind it, and none of
/// that is expressible as a colour. Morrowind's lava and slime will want entries of their own for
/// the same reason.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MaterialKind {
    /// Everything a NIF describes: a base colour lit by what reaches it.
    #[default]
    Diffuse,
    /// A water surface. Its `diffuse` and texture are ignored — the shader owns its appearance.
    Water,
    /// Ground, whose albedo is blended across the four terrain textures it carries.
    ///
    /// A cell names one texture per 512-unit tile and nothing in between, so a surface that simply
    /// sampled the tile it stood on would meet its neighbours along a straight edge — which is what
    /// the ground did.
    ///
    /// The four ride in the variant rather than beside it because ground without them is not a
    /// thing: a shader handed the terrain model with no textures to blend would sample slot zero
    /// four times and draw the fallback, silently.
    Terrain(TerrainLayers),
}

/// The four terrain textures a point on the ground is blended from.
///
/// In the order the blend wants them: the tile below-left of the point, then across, then the row
/// above. A cell's tiles are laid out on a 512-unit grid and the blend is bilinear between their
/// *centres*, so the four are the corners of the square of centres the point falls inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerrainLayers(pub [TextureId; 4]);

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
    /// How far the texture slides across the surface each second, in texture coordinates.
    ///
    /// **What makes Vivec's water run and Red Mountain's lava crawl.** Neither is animated
    /// geometry: both are a flat sheet with a `NiUVController` walking the texture over it, which
    /// is the whole of how the original engine drew a moving fluid. Zero for everything else.
    ///
    /// **It belongs to the material and not beside it**, because materials are interned by value —
    /// two sheets of the same lava scrolling at different rates have to stay two entries, and a
    /// field here is what says so without anything else being asked.
    pub scroll: Vec2,
    /// Which shading model this surface uses, and whatever that model needs of its own. Everything
    /// loaded from a NIF is [`Diffuse`].
    ///
    /// The ground's four textures live in here rather than in a table of their own because they are
    /// what makes one patch of ground differ from another: two patches blending the same four tiles
    /// are the same material and intern to one entry, which is what keeps a cell's thirty-two
    /// squared quadrants down to the seventy-odd distinct blends it actually has.
    ///
    /// [`Diffuse`]: MaterialKind::Diffuse
    pub kind: MaterialKind,
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
            scroll: Vec2::ZERO,
            kind: MaterialKind::Diffuse,
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

    /// Whether what is drawn here adds to the frame rather than covering it.
    ///
    /// Not part of [`Material`], because nothing else in the scene needs it: a NIF's *geometry* is
    /// sorted and blended the same way whatever its destination factor, and the one thing that
    /// turns on it is whether a particle is a flame or a puff of smoke.
    pub(crate) fn adds(&self, nif: &NifFile) -> bool {
        matches!(nif.resolve(self.alpha), Some(Block::Alpha(alpha)) if alpha.adds())
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
pub(crate) fn texture_path(name: &str) -> String {
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
