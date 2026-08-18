//! Everything a cell needs, assembled from the installed game.

use rtxmw_esm::{CellId, EsmReader};
use rtxmw_texture::Texture;
use rtxmw_vfs::{Vfs, morrowind_data_dir};

use crate::door::Door;
use crate::error::{Result, SceneError};
use crate::static_scene::{ModelIndex, StaticScene};

/// A cell's geometry and the textures its materials name.
///
/// The two arrive together because they come from the same content file and the same archives, and
/// separating them only moves the decode loop to whoever asked. Every caller wanting real data
/// wants both.
#[derive(Debug)]
pub struct LoadedCell {
    /// Which cell this is. Carried rather than remembered by the caller, so a report about it
    /// cannot name a different one than was loaded.
    pub id: CellId,
    pub scene: StaticScene,
    /// One entry per path in the scene's texture list, `None` where the file could not be decoded.
    ///
    /// A miss is normal rather than an error: the shipped meshes carry a tail of references to art
    /// that was removed, and a renderer substitutes a fallback for them.
    pub textures: Vec<Option<Texture>>,
    /// Every door elsewhere in the file that leads into this cell.
    ///
    /// Each one names a place the game itself puts the player, which is what makes any of them a
    /// defensible spawn point — see [`Door`]. Empty for a cell nothing leads to, which a cell
    /// reached only by script or by coordinates legitimately is.
    pub entrances: Vec<Door>,
}

impl LoadedCell {
    /// Loads whichever cell `id` names, or `None` when no game data is configured.
    ///
    /// The dispatch a caller holding a `CellId` wants: an interior is found by name and an exterior
    /// by grid position, and nothing above here should have to know which of those it is holding.
    ///
    /// `radius` widens an exterior into the block of cells around it and is ignored for an
    /// interior, which has no neighbours to stream — a door is the only way out of one.
    pub fn load_at(id: CellId, radius: i32) -> Result<Option<Self>> {
        match id {
            CellId::Interior(name) => Self::load_interior(&name),
            CellId::Exterior { x, y } => Self::load_exterior_grid(x, y, radius),
        }
    }

    /// Loads the block of exterior cells within `radius` of `(x, y)`.
    ///
    /// Its entrances are the doors leading into the *centre* cell, since that is where a traveller
    /// stepping outside arrives; the surrounding cells are there to be seen and walked into.
    pub fn load_exterior_grid(x: i32, y: i32, radius: i32) -> Result<Option<Self>> {
        Self::load(CellId::Exterior { x, y }, |esm, models, vfs| {
            StaticScene::load_exterior_grid(esm, models, vfs, x, y, radius)
        })
    }

    /// Loads the named interior from the installed game, or `None` when none is configured.
    ///
    /// Reads the whole of `Morrowind.esm` each call, so loading several cells this way rereads a
    /// 79 MB file each time. That is fine for opening one cell at startup and wrong for streaming;
    /// when cells start streaming, the reader wants hoisting above this.
    pub fn load_interior(cell_name: &str) -> Result<Option<Self>> {
        Self::load(
            CellId::Interior(cell_name.to_owned()),
            |esm, models, vfs| StaticScene::load_interior(esm, models, vfs, cell_name),
        )
    }

    /// Brings up the game's files, assembles one cell with `build`, and decodes its textures.
    ///
    /// The part every cell shares, whichever way it is addressed: the reader, the model index, the
    /// door search and the texture decode differ only in which scene comes out of the middle.
    fn load(
        destination: CellId,
        build: impl FnOnce(&EsmReader<'_>, &ModelIndex, &Vfs) -> Result<StaticScene>,
    ) -> Result<Option<Self>> {
        let Some(data) = morrowind_data_dir() else {
            return Ok(None);
        };
        let vfs = rtxmw_vfs::morrowind_archives().ok_or_else(|| {
            SceneError::Io(std::io::Error::other(
                "the game directory is configured but its archives could not be opened",
            ))
        })?;
        let bytes = std::fs::read(data.join("Morrowind.esm")).map_err(SceneError::Io)?;
        let esm = EsmReader::new(&bytes)?;
        let models = ModelIndex::build(&esm)?;
        let scene = build(&esm, &models, &vfs)?;
        let entrances = Door::leading_to(&esm, &models, &destination)?;
        // `destination` is moved into the result below, so the door search borrows it first.

        // A texture that fails to read or decode becomes `None` rather than aborting the load: one
        // dangling reference in a cell of hundreds is not a reason to have no cell.
        let textures = scene
            .materials
            .textures()
            .iter()
            .map(|path| {
                vfs.read(path)
                    .ok()
                    .and_then(|bytes| Texture::decode(&bytes).ok())
            })
            .collect();

        Ok(Some(Self {
            id: destination,
            scene,
            textures,
            entrances,
        }))
    }

    /// Loads the exterior cell at grid position `(x, y)`, or `None` when none is configured.
    ///
    /// Its entrances come from the same search an interior's do: a door leading outside names no
    /// cell, so the grid square its arrival point falls in identifies one — which
    /// [`Door::leading_to`] already resolves. Standing where the game would put someone stepping
    /// out of a building is as good a viewpoint outdoors as in.
    pub fn load_exterior(x: i32, y: i32) -> Result<Option<Self>> {
        Self::load_exterior_grid(x, y, 0)
    }

    /// How many of the cell's texture references resolved to nothing.
    pub fn missing_textures(&self) -> usize {
        self.textures.iter().filter(|t| t.is_none()).count()
    }
}
