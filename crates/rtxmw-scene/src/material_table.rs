//! The scene-wide set of distinct materials and the textures they name.

use std::collections::HashMap;

use crate::material::{Material, TextureId};

/// Materials and texture paths, deduplicated across every mesh in a scene.
///
/// Scene-wide rather than per-mesh because that is the granularity the GPU wants: one bindless
/// texture array and one material buffer for the whole cell, indexed at a hit. A per-mesh table
/// would have to be merged and remapped later, so meshes intern into this as they are built.
#[derive(Debug, Default)]
pub struct MaterialTable {
    materials: Vec<Material>,
    /// Virtual file system paths, indexed by [`TextureId`].
    textures: Vec<String>,
    texture_ids: HashMap<String, TextureId>,
}

impl MaterialTable {
    /// The index of `material`, adding it if it is new.
    ///
    /// Linear because the comparison is a handful of floats and the counts are small: one interior
    /// resolves around 120 materials, and interning the entire shipped mesh library — 4,593
    /// materials, so roughly ten million comparisons — still finishes in a fraction of a second.
    /// `Material` holds `f32`s, so it is not `Hash`, and quantising them into a key would trade a
    /// real cost for an imaginary one.
    pub fn intern(&mut self, material: Material) -> u32 {
        if let Some(index) = self.materials.iter().position(|m| *m == material) {
            return index as u32;
        }
        self.materials.push(material);
        (self.materials.len() - 1) as u32
    }

    /// The id of `path`, adding it if it is new.
    pub fn intern_texture(&mut self, path: &str) -> TextureId {
        if let Some(&id) = self.texture_ids.get(path) {
            return id;
        }
        let id = TextureId(self.textures.len() as u32);
        self.textures.push(path.to_owned());
        self.texture_ids.insert(path.to_owned(), id);
        id
    }

    /// Every distinct material, indexed by what [`MaterialTable::intern`] returned.
    pub fn materials(&self) -> &[Material] {
        &self.materials
    }

    /// Every distinct texture path, indexed by [`TextureId`].
    pub fn textures(&self) -> &[String] {
        &self.textures
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::AlphaMode;
    use glam::Vec3;

    #[test]
    fn identical_materials_collapse_and_different_ones_do_not() {
        let mut table = MaterialTable::default();
        let stone = table.intern_texture("textures/tx_stone.dds");
        assert_eq!(stone, table.intern_texture("textures/tx_stone.dds"));
        assert_eq!(table.textures().len(), 1);

        let opaque = Material {
            base_colour: Some(stone),
            ..Material::default()
        };
        assert_eq!(table.intern(opaque), 0);
        assert_eq!(table.intern(opaque), 0, "the same material was added twice");

        // The alpha mode alone must separate two otherwise identical materials, because it decides
        // whether the geometry is built opaque.
        let masked = Material {
            alpha: AlphaMode::Mask(0.5),
            ..opaque
        };
        assert_eq!(table.intern(masked), 1);
        assert_eq!(table.materials().len(), 2);

        // So must the emissive colour, which is the only light some interiors have.
        let glowing = Material {
            emissive: Vec3::splat(0.5),
            ..opaque
        };
        assert_eq!(table.intern(glowing), 2);
    }

    #[test]
    fn texture_ids_index_the_path_list_in_insertion_order() {
        let mut table = MaterialTable::default();
        let first = table.intern_texture("a.dds");
        let second = table.intern_texture("b.dds");
        assert_eq!((first, second), (TextureId(0), TextureId(1)));
        assert_eq!(table.textures(), ["a.dds", "b.dds"]);
    }
}
