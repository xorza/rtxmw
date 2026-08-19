//! A contact sheet of a cell's textures, vanilla beside de-lit.
//!
//! **What `docs/design.md` §5.1 asks for.** The correction it describes is judged by eye against the
//! same surfaces, and its failure mode is over-correction — flat, washed-out output where the
//! estimate removed painted detail rather than painted light. A number cannot see that; a sheet of
//! every texture a cell uses, each one twice, can be scanned in seconds.

use rtxmw_gpu::readback;
use rtxmw_scene::LoadedCell;
use rtxmw_texture::Texture;

use crate::cli::TextureSheetOptions;

/// Side of one thumbnail, in pixels.
const THUMB: u32 = 96;

/// Gap between tiles, and the border around the sheet.
const GAP: u32 = 4;

/// Writes the sheet named by `options`.
pub(crate) fn write(options: &TextureSheetOptions) -> Result<(), Box<dyn std::error::Error>> {
    let cell = LoadedCell::load_at(options.cell.clone())?
        .ok_or("no game data configured — set MORROWIND_DATA_DIR, or put it in .env")?;
    let paths = cell.scene.materials.textures();
    let pairs: Vec<(&str, &Texture)> = paths
        .iter()
        .zip(&cell.textures)
        .filter_map(|(path, texture)| texture.as_ref().map(|t| (path.as_str(), t)))
        .collect();
    if pairs.is_empty() {
        return Err(format!("{:?} names no textures", options.cell).into());
    }

    // A pair is two thumbnails side by side, so the columns count pairs and the rows follow.
    let across = (pairs.len() as f64).sqrt().ceil() as u32;
    let down = (pairs.len() as u32).div_ceil(across);
    let pair_width = THUMB * 2 + GAP;
    let width = GAP + across * (pair_width + GAP);
    let height = GAP + down * (THUMB + GAP);

    // Mid grey, so that a thumbnail going pale from over-correction reads against the ground rather
    // than into it.
    let mut sheet = vec![0x40u8; (width * height * 4) as usize];
    for pixel in sheet.chunks_exact_mut(4) {
        pixel[3] = 255;
    }

    for (index, (_, texture)) in pairs.iter().enumerate() {
        let column = index as u32 % across;
        let row = index as u32 / across;
        let x = GAP + column * (pair_width + GAP);
        let y = GAP + row * (THUMB + GAP);
        let vanilla = texture.to_rgba8();
        let corrected = delit(texture, &vanilla);
        blit(
            &mut sheet,
            width,
            x,
            y,
            &vanilla,
            texture.width(),
            texture.height(),
        );
        blit(
            &mut sheet,
            width,
            x + THUMB + GAP,
            y,
            &corrected,
            texture.width(),
            texture.height(),
        );
    }

    readback::write_png(&options.path, &sheet, width, height);
    println!(
        "wrote {} — {} textures of {:?}, vanilla left, de-lit right",
        options.path.display(),
        pairs.len(),
        options.cell
    );
    Ok(())
}

/// `rgba` with the texture's own estimated shading divided out.
///
/// The same arithmetic the shader does, at the same strength, so what the sheet shows is what a
/// frame would draw rather than a second opinion about it.
fn delit(texture: &Texture, rgba: &[u8]) -> Vec<u8> {
    let map = texture.shading_map();
    let side = map.width();
    let shading = map.to_rgba8();
    let (width, height) = (texture.width().max(1), texture.height().max(1));

    let mut out = rgba.to_vec();
    for (index, texel) in out.chunks_exact_mut(4).enumerate() {
        let (x, y) = (index as u32 % width, index as u32 / width);
        // Nearest rather than bilinear: the map is smooth enough that the difference is invisible,
        // and a sheet is not where filtering deserves its own implementation.
        let at = ((y * side / height).min(side - 1) * side + (x * side / width).min(side - 1)) * 4;
        // Asked for rather than worked out here: the transfer function and the scale belong to the
        // map, and a second copy of them would let this sheet drift from what a frame draws.
        let divisor = Texture::shading_multiplier(shading[at as usize]).max(1e-3);
        for channel in &mut texel[..3] {
            *channel = (f32::from(*channel) / divisor).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Draws `rgba` into `sheet` at `x, y`, scaled to fit a [`THUMB`] square.
fn blit(sheet: &mut [u8], pitch: u32, x: u32, y: u32, rgba: &[u8], width: u32, height: u32) {
    let (width, height) = (width.max(1), height.max(1));
    for row in 0..THUMB {
        for column in 0..THUMB {
            // Nearest-neighbour, which for a thumbnail of a 128-square texture is what a reader
            // wants anyway: a smoothed one would hide exactly the loss of detail this is looking for.
            let sx = (column * width / THUMB).min(width - 1);
            let sy = (row * height / THUMB).min(height - 1);
            let from = ((sy * width + sx) * 4) as usize;
            let to = (((y + row) * pitch + x + column) * 4) as usize;
            sheet[to..to + 3].copy_from_slice(&rgba[from..from + 3]);
        }
    }
}
