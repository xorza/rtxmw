//! `LAND` records: the height, shading and texturing of one exterior cell.

use crate::cell::CellId;
use crate::error::Result;
use crate::esm_reader::Record;

/// Vertices along one side of a cell's terrain.
///
/// The last row and column are shared with the neighbouring cell rather than being its own, which
/// is what makes adjacent terrain meet without a seam.
pub const GRID: usize = 65;

/// Vertices in a cell's terrain.
pub const VERTICES: usize = GRID * GRID;

/// Texture tiles along one side. Each covers four vertex spacings, or 512 world units.
pub const TEXTURE_GRID: usize = 16;

/// World units between neighbouring vertices — a cell's 8,192 divided by its 64 spans.
pub const SPACING: f32 = 128.0;

/// Height a cell with no `LAND` record sits at.
pub const DEFAULT_HEIGHT: f32 = -2048.0;

/// Units one step of the stored height gradient represents.
const HEIGHT_SCALE: f32 = 8.0;

/// One exterior cell's terrain.
#[derive(Debug, Clone)]
pub struct LandRecord {
    pub grid_x: i32,
    pub grid_y: i32,
    /// Row-major heights in world units, `GRID` by `GRID`, south row first.
    pub heights: Vec<f32>,
    /// Row-major vertex normals, parallel to `heights`. Empty when the record carries none.
    pub normals: Vec<[f32; 3]>,
    /// Row-major texture indices, `TEXTURE_GRID` by `TEXTURE_GRID`, already untransposed.
    ///
    /// Zero means the region's default texture; anything else is one *past* its index in the
    /// plugin's `LTEX` palette. Empty when the record carries none.
    pub textures: Vec<u16>,
}

impl LandRecord {
    /// Parses a `LAND` record, or `None` if it carries no heightmap or no coordinates.
    ///
    /// A record without `VHGT` is legal and appears in the shipped files — it is a cell that exists
    /// for its objects, and the terrain under it is flat at [`DEFAULT_HEIGHT`].
    ///
    /// One without `INTV` is not legal, and the reason it comes back as nothing rather than
    /// defaulting is that a caller looks terrain up *by* its coordinates: a record with none would
    /// claim to be cell `(0, 0)`, which is open sea south-west of Vvardenfell, and be handed out
    /// to whoever asked for it.
    pub fn parse(record: &Record<'_>) -> Result<Option<Self>> {
        let mut grid = None;
        let mut heights = Vec::new();
        let mut normals = Vec::new();
        let mut textures = Vec::new();

        for sub in record.subrecords() {
            let sub = sub?;
            let data = sub.data();
            match &sub.name().0 {
                b"INTV" if data.len() >= 8 => {
                    let int = |at: usize| i32::from_le_bytes(data[at..at + 4].try_into().unwrap());
                    grid = Some((int(0), int(4)));
                }
                b"VHGT" => heights = decode_heights(data),
                b"VNML" => normals = decode_normals(data),
                b"VTEX" => textures = decode_textures(data),
                _ => {}
            }
        }

        let Some((grid_x, grid_y)) = grid.filter(|_| !heights.is_empty()) else {
            return Ok(None);
        };
        Ok(Some(Self {
            grid_x,
            grid_y,
            heights,
            normals,
            textures,
        }))
    }

    /// Which cell a `LAND` record belongs to, reading its coordinates and nothing else.
    ///
    /// Indexing a file means touching every `LAND` in it, and [`LandRecord::parse`] would undo a
    /// delta-coded heightmap for each one to answer a question the first eight bytes settle.
    pub fn grid_of(record: &Record<'_>) -> Result<Option<CellId>> {
        for sub in record.subrecords() {
            let sub = sub?;
            let data = sub.data();
            if &sub.name().0 == b"INTV" && data.len() >= 8 {
                let int = |at: usize| i32::from_le_bytes(data[at..at + 4].try_into().unwrap());
                return Ok(Some(CellId::Exterior {
                    x: int(0),
                    y: int(4),
                }));
            }
        }
        Ok(None)
    }

    /// The height at a vertex, indexed from the cell's south-west corner.
    pub fn height(&self, x: usize, y: usize) -> f32 {
        self.heights[y * GRID + x]
    }
}

/// Undoes the gradient encoding a `VHGT` subrecord stores heights in.
///
/// **Every value is a difference, not a height.** The first is a delta from the record's own float
/// offset; each row's first column steps from the row above, and every other column steps from its
/// left-hand neighbour. So a cell is one running total down its western edge and 65 running totals
/// across. Reading them as absolute heights gives terrain that looks plausible and is wrong
/// everywhere but the corner.
fn decode_heights(data: &[u8]) -> Vec<f32> {
    // A float offset, one signed byte per vertex, and three bytes of padding.
    if data.len() < 4 + VERTICES {
        return Vec::new();
    }
    let offset = f32::from_le_bytes(data[0..4].try_into().unwrap());
    let deltas = &data[4..4 + VERTICES];

    let mut heights = Vec::with_capacity(VERTICES);
    let mut row = offset;
    for y in 0..GRID {
        row += deltas[y * GRID] as i8 as f32;
        let mut column = row;
        heights.push(column * HEIGHT_SCALE);
        for x in 1..GRID {
            column += deltas[y * GRID + x] as i8 as f32;
            heights.push(column * HEIGHT_SCALE);
        }
    }
    heights
}

/// Signed bytes to unit vectors.
///
/// Stored per vertex rather than per face, so terrain is smooth-shaded without the loader having to
/// average anything.
fn decode_normals(data: &[u8]) -> Vec<[f32; 3]> {
    if data.len() < VERTICES * 3 {
        return Vec::new();
    }
    data[..VERTICES * 3]
        .chunks_exact(3)
        .map(|n| {
            let v = [n[0] as i8 as f32, n[1] as i8 as f32, n[2] as i8 as f32];
            let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            // A zero normal appears where a cell's data is incomplete; up is the only answer that
            // does not put a hole in the lighting.
            if length > 0.0 {
                [v[0] / length, v[1] / length, v[2] / length]
            } else {
                [0.0, 0.0, 1.0]
            }
        })
        .collect()
}

/// Untransposes the texture indices.
///
/// **`VTEX` is stored as sixteen 4×4 blocks, not as one 16×16 grid** — the file walks a block at a
/// time, and within it a row at a time. Reading it as a plain grid scrambles the terrain's
/// texturing into a recognisable checkerboard of the right textures in the wrong places.
fn decode_textures(data: &[u8]) -> Vec<u16> {
    const TILES: usize = TEXTURE_GRID * TEXTURE_GRID;
    if data.len() < TILES * 2 {
        return Vec::new();
    }
    let mut out = vec![0u16; TILES];
    let mut read = 0;
    for block_y in 0..4 {
        for block_x in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let index =
                        u16::from_le_bytes(data[read * 2..read * 2 + 2].try_into().unwrap());
                    out[(block_y * 4 + y) * TEXTURE_GRID + (block_x * 4 + x)] = index;
                    read += 1;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esm_reader::EsmReader;
    use crate::esm_reader::internals::{SubrecordSpec, push_header, push_record};

    /// A `VHGT` payload from a grid of absolute heights, in the record's own units.
    ///
    /// The inverse of what the decoder does, so the two disagreeing shows up as a round trip that
    /// does not close rather than as terrain that merely looks odd.
    fn encode_heights(absolute: &[f32; VERTICES], offset: f32) -> Vec<u8> {
        let mut out = offset.to_le_bytes().to_vec();
        let mut deltas = vec![0i8; VERTICES];
        let mut row = offset;
        for y in 0..GRID {
            let first = absolute[y * GRID];
            deltas[y * GRID] = (first - row) as i8;
            row = first;
            let mut column = row;
            for x in 1..GRID {
                let here = absolute[y * GRID + x];
                deltas[y * GRID + x] = (here - column) as i8;
                column = here;
            }
        }
        out.extend(deltas.iter().map(|d| *d as u8));
        out.extend_from_slice(&[0; 3]);
        out
    }

    #[test]
    fn heights_are_a_running_total_in_both_directions() {
        // A ramp that rises east and north at different rates, so reading the deltas as absolute
        // heights — or accumulating along only one axis — gives a different surface rather than a
        // shifted one.
        let mut absolute = [0.0f32; VERTICES];
        for y in 0..GRID {
            for x in 0..GRID {
                absolute[y * GRID + x] = (x as f32) + 3.0 * (y as f32);
            }
        }
        let payload = encode_heights(&absolute, -10.0);

        let decoded = decode_heights(&payload);
        assert_eq!(decoded.len(), VERTICES);
        for y in 0..GRID {
            for x in 0..GRID {
                // The stored gradient is in units of eight.
                let expected = absolute[y * GRID + x] * HEIGHT_SCALE;
                assert!(
                    (decoded[y * GRID + x] - expected).abs() < 1e-3,
                    "vertex ({x}, {y}) came back {} rather than {expected}",
                    decoded[y * GRID + x]
                );
            }
        }
    }

    #[test]
    fn a_truncated_heightmap_is_no_heightmap() {
        // Untrusted data: a short subrecord is a broken file, not a reason to index past the end.
        assert!(decode_heights(&[0; 100]).is_empty());
        assert!(decode_normals(&[0; 100]).is_empty());
        assert!(decode_textures(&[0; 100]).is_empty());
    }

    #[test]
    fn texture_indices_come_out_of_their_four_by_four_blocks() {
        // Numbered in the order the file stores them, so the decode is correct exactly when the
        // result reads 0..255 in *blocks*: the first sixteen values belong in the bottom-left 4x4
        // of the grid, not in its first row.
        let payload: Vec<u8> = (0..256u16).flat_map(|i| i.to_le_bytes()).collect();
        let grid = decode_textures(&payload);

        // Value 0 is the first tile of the first block, so the grid's origin.
        assert_eq!(grid[0], 0);
        // Values 0..3 fill the first row of that block — the grid's first four columns.
        assert_eq!(&grid[0..4], &[0, 1, 2, 3]);
        // Value 4 starts the block's second row, which is the grid's *second* row, not its fifth
        // column. Reading the payload as a plain 16x16 grid would put 4 at column four.
        assert_eq!(grid[TEXTURE_GRID], 4);
        // The second block begins at column four of the grid's first row.
        assert_eq!(grid[4], 16);
        // And the last block's last tile is the far corner.
        assert_eq!(grid[TEXTURE_GRID * TEXTURE_GRID - 1], 255);
    }

    #[test]
    fn normals_come_back_as_unit_vectors_pointing_the_right_way() {
        let mut payload = vec![0i8; VERTICES * 3];
        // Straight up.
        payload[2] = 127;
        // Leaning east: a slope whose normal tilts toward +X.
        payload[3] = 90;
        payload[5] = 90;
        let bytes: Vec<u8> = payload.iter().map(|v| *v as u8).collect();

        let normals = decode_normals(&bytes);
        assert_eq!(normals.len(), VERTICES);
        assert!((normals[0][2] - 1.0).abs() < 1e-5, "{:?}", normals[0]);
        let tilted = normals[1];
        assert!(
            (tilted[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3,
            "{tilted:?}"
        );
        // A vertex the file left at zero cannot be normalised, and a zero normal would leave a hole
        // in the lighting rather than a flat patch.
        assert_eq!(normals[2], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_record_without_coordinates_yields_nothing_rather_than_claiming_the_origin() {
        // Terrain is looked up *by* its grid position, so a record that defaulted to zero would be
        // handed to whoever asked for cell (0, 0) — open sea, south-west of the island.
        let mut out = Vec::new();
        push_header(&mut out);
        let heights = encode_heights(&[2.0; VERTICES], 0.0);
        push_record(
            &mut out,
            b"LAND",
            0,
            &[SubrecordSpec {
                name: b"VHGT",
                data: &heights,
            }],
        );
        let esm = EsmReader::new(&out).unwrap();
        let record = esm.records().next().unwrap().unwrap();
        assert!(LandRecord::parse(&record).unwrap().is_none());
    }

    #[test]
    fn a_record_without_a_heightmap_yields_nothing() {
        let mut out = Vec::new();
        push_header(&mut out);
        let coords: Vec<u8> = [-3i32, 7].iter().flat_map(|v| v.to_le_bytes()).collect();
        push_record(
            &mut out,
            b"LAND",
            0,
            &[SubrecordSpec {
                name: b"INTV",
                data: &coords,
            }],
        );
        let esm = EsmReader::new(&out).unwrap();
        let record = esm.records().next().unwrap().unwrap();
        // Legal and present in the shipped files: a cell that exists for its objects, over terrain
        // that is flat at the default height.
        assert!(LandRecord::parse(&record).unwrap().is_none());
    }

    #[test]
    fn a_land_record_carries_its_grid_position() {
        let mut out = Vec::new();
        push_header(&mut out);
        let coords: Vec<u8> = [-3i32, 7].iter().flat_map(|v| v.to_le_bytes()).collect();
        let heights = encode_heights(&[2.0; VERTICES], 0.0);
        push_record(
            &mut out,
            b"LAND",
            0,
            &[
                SubrecordSpec {
                    name: b"INTV",
                    data: &coords,
                },
                SubrecordSpec {
                    name: b"VHGT",
                    data: &heights,
                },
            ],
        );
        let esm = EsmReader::new(&out).unwrap();
        let record = esm.records().next().unwrap().unwrap();
        let land = LandRecord::parse(&record)
            .unwrap()
            .expect("has a heightmap");

        assert_eq!((land.grid_x, land.grid_y), (-3, 7));
        assert_eq!(land.height(0, 0), 16.0);
        assert_eq!(land.height(64, 64), 16.0);
        assert!(land.normals.is_empty() && land.textures.is_empty());
    }
}
