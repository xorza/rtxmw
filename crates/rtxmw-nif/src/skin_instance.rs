//! Which bones move a skinned mesh, and where its weights live.

use crate::cursor::{Cursor, Link};
use crate::error::Result;

/// The link between a geometry and the skeleton that moves it.
///
/// Separate from [`crate::SkinData`] because the format separates them: this names the bone *nodes*
/// in the scene graph, and the data block holds their bind poses and the weights. The two are
/// parallel — bone `n` here is bone `n` there — and neither is meaningful alone.
#[derive(Debug, Clone, Default)]
pub struct SkinInstance {
    pub data: Link,
    /// The node every bone's pose is measured against.
    pub skeleton_root: Link,
    /// One node link per bone, in the order [`crate::SkinData::bones`] uses.
    pub bones: Vec<Link>,
}

impl SkinInstance {
    pub(crate) fn read(cursor: &mut Cursor<'_>) -> Result<Self> {
        let data = cursor.link()?;
        let skeleton_root = cursor.link()?;
        let count = cursor.u32()? as usize;
        Ok(Self {
            data,
            skeleton_root,
            bones: cursor.links(count)?,
        })
    }
}
