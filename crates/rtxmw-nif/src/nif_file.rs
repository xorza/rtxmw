//! A parsed NIF: its blocks and its roots.

use crate::block::{Block, Transform};
use crate::cursor::{Cursor, Link};
use crate::error::{NifError, Result};

/// Packs a version into the BCD form the header stores.
pub const fn version(major: u32, minor: u32, patch: u32, revision: u32) -> u32 {
    (major << 24) | (minor << 16) | (patch << 8) | revision
}

/// The version Morrowind ships.
pub const VER_MORROWIND: u32 = version(4, 0, 0, 2);

/// A NIF's blocks, in file order, plus the indices of its roots.
#[derive(Debug)]
pub struct NifFile {
    blocks: Vec<Block>,
    roots: Vec<Link>,
}

impl NifFile {
    /// Parses `data` in full.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);

        let header = cursor.line()?;
        if !header.starts_with("NetImmerse File Format")
            && !header.starts_with("Gamebryo File Format")
        {
            return Err(NifError::NotANif);
        }

        let ver = cursor.u32()?;
        if ver != VER_MORROWIND {
            return Err(NifError::UnsupportedVersion { version: ver });
        }

        let block_count = cursor.u32()? as usize;
        let mut blocks = Vec::with_capacity(block_count);

        for index in 0..block_count {
            let offset = cursor.position();
            let kind = cursor.string()?;
            if kind.is_empty() {
                return Err(NifError::Malformed {
                    index,
                    kind: String::new(),
                    reason: "blank block type".into(),
                });
            }
            let parsed = Block::read(&kind, &mut cursor, ver).map_err(|e| match e {
                NifError::UnknownBlock { .. } | NifError::InBlock { .. } => e,
                other => NifError::InBlock {
                    index,
                    kind: kind.clone(),
                    source: Box::new(other),
                },
            })?;
            match parsed {
                Some(block) => blocks.push(block),
                None => {
                    return Err(NifError::UnknownBlock {
                        index,
                        kind,
                        offset,
                    });
                }
            }
        }

        let root_count = cursor.u32()? as usize;
        let roots = cursor.links(root_count)?;
        for root in &roots {
            if let Some(index) = root.index()
                && index >= blocks.len()
            {
                return Err(NifError::BadRoot { link: root.0 });
            }
        }

        let mut file = Self { blocks, roots };
        file.discard_root_transform();
        Ok(file)
    }

    /// Resets the transform on block zero, when that block is a node.
    ///
    /// **A model's outermost node carries a transform the original engine ignores.** Roughly one
    /// Morrowind mesh in sixteen has a non-identity one, and honouring it turns those models
    /// against everything around them — Seyda Neen's fireplace is built with a half turn there, and
    /// applied it presents its back to the room while the fire the cell places burns behind it.
    ///
    /// Only block zero, and only a node: a mesh whose outermost block is geometry keeps its
    /// transform, as does a rig's `bip01`, which is an animation root that the skeleton is defined
    /// against rather than a stray placement.
    ///
    /// This is the same rule, and the same exception, that OpenMW applies while reading
    /// (`components/nif/node.cpp:182`), where the comment calls it a stopgap and the wrong layer to
    /// do it in. It is kept here for the same reason it is there: it is a property of how the
    /// content must be read, so every caller should get it rather than each remembering to.
    fn discard_root_transform(&mut self) {
        let Some(Block::Node(node)) = self.blocks.first_mut() else {
            return;
        };
        if node.av.net.name.eq_ignore_ascii_case("bip01") {
            return;
        }
        node.av.transform = Transform::default();
    }

    /// Every block, in file order.
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    /// The blocks the scene graph starts from.
    ///
    /// Unused until the scene layer walks the graph, but a caller cannot find anything without it.
    pub fn roots(&self) -> &[Link] {
        &self.roots
    }

    /// Resolves a link, or `None` when it points nowhere.
    ///
    /// Pairs with [`Self::roots`]; the two are how the block table is traversed at all.
    pub fn resolve(&self, link: Link) -> Option<&Block> {
        self.blocks.get(link.index()?)
    }

    /// Total triangles across every geometry block.
    pub fn triangle_count(&self) -> usize {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                Block::GeometryData(data) => Some(data.triangle_count()),
                _ => None,
            })
            .sum()
    }

    /// Total vertices across every geometry block.
    pub fn vertex_count(&self) -> usize {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                Block::GeometryData(data) => Some(data.vertices.len()),
                _ => None,
            })
            .sum()
    }
}
