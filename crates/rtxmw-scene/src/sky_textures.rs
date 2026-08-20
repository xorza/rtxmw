//! The pictures the sky itself is drawn with, as against anything a cell places.

use rtxmw_texture::Texture;

use crate::clouds::CloudSheet;
use crate::error::Result;
use crate::game_data::GameData;
use crate::srgb::{LUMA, channel_to_linear};
use crate::weather::Weather;

/// Every vanilla picture the sky needs: the two moons' portraits and the weather's cloud sheet.
///
/// **One type because the renderer already treats them as one.** They are reserved together at the
/// front of the bindless array — `SKY_SLOTS` in `surface.glsl` and `scene_residency.rs` — ahead of
/// anything a cell names, because a material's id addresses a fixed pair of slots and a picture
/// inserted later would displace every texture behind it. They were two types with two loaders, two
/// renderer entry points and two calls in each front end before that grouping was noticed.
///
/// A picture that fails to read or decode is `None` rather than an error, the same as any other
/// texture: a moon without its portrait is a flat disc and a sky without its sheet has no clouds,
/// both of which are worse and neither of which is a reason to have no renderer.
#[derive(Debug)]
pub struct SkyTextures {
    /// `tx_masser_full.dds` — the larger moon's face.
    pub masser: Option<Texture>,
    /// `tx_secunda_full.dds` — the smaller one's.
    pub secunda: Option<Texture>,
    /// The painted sheet the cloud layer is cut out of — the weather's own.
    ///
    /// `Weather::cloud_texture` names it, which for eight of the ten is `tx_sky_*` and for
    /// Bloodmoon's snow and blizzard is `tx_bm_sky_*`.
    pub clouds: Option<Texture>,
    /// Mean alpha of the cloud sheet — how much of the sky the layer hides, on average.
    ///
    /// A quarter for clear weather's cirrus and all of it for every overcast one, which is the
    /// difference between a sky the ground is lit by and a lid over it. Nought where there is no
    /// sheet, which draws no layer anyway.
    cloud_cover_mean: f32,
    /// Mean luminance of the cloud sheet, weighted by its own alpha and decoded to linear.
    ///
    /// What its texels are read as a ratio to, so the painting supplies structure and not a level —
    /// see [`crate::Clouds`]. One where there is no sheet, which draws no layer anyway.
    cloud_mean: f32,
}

impl SkyTextures {
    /// What the cloud sheet comes to on average, which is what building a sky needs of it.
    pub fn sheet(&self) -> CloudSheet {
        CloudSheet {
            covering: self.cloud_cover_mean,
            mean: self.cloud_mean,
        }
    }

    /// Reads all of them from the installed game, with `weather`'s cloud sheet.
    ///
    /// `None` where no game data is configured. A sheet that will not read leaves the layer undrawn
    /// rather than failing, the same as a moon without its portrait is a flat disc.
    pub fn load(weather: &Weather) -> Result<Option<Self>> {
        let Some(game) = GameData::shared()? else {
            return Ok(None);
        };
        let read = |path: &str| {
            game.vfs()
                .read(path)
                .ok()
                .and_then(|bytes| Texture::decode(&bytes).ok())
        };
        // **Mipped here, because the file is not.** The sheets ship one level apiece, and the
        // layer repeats its tile some forty times across the last degrees above the horizon — a
        // minifying lookup with nothing to fall back to. The moons' portraits ship their own chains
        // and are magnified rather than minified, so they are left as they are.
        let clouds = read(&format!(r"textures\{}", weather.cloud_texture)).map(Texture::with_mips);
        Ok(Some(Self {
            masser: read(r"textures\tx_masser_full.dds"),
            secunda: read(r"textures\tx_secunda_full.dds"),
            cloud_mean: clouds.as_ref().map_or(1.0, Self::mean_of),
            cloud_cover_mean: clouds.as_ref().map_or(0.0, Self::cover_of),
            clouds,
        }))
    }

    /// The mean luminance of everything the sheet's alpha calls cloud, decoded to linear first.
    ///
    /// Weighted by that alpha rather than cut at a threshold: a cloud's edge is half a cloud, and
    /// the clear sheet is wisps whose alpha is nowhere near either end. Decoded before the mean
    /// rather than after, which is the same mistake as sampling an albedo through a UNORM view.
    /// The mean of the sheet's alpha, which is the fraction of sky it hides.
    fn cover_of(texture: &Texture) -> f32 {
        let rgba = texture.to_rgba8();
        let total: f32 = rgba.chunks_exact(4).map(|t| t[3] as f32 / 255.0).sum();
        total / (rgba.len() / 4).max(1) as f32
    }

    fn mean_of(texture: &Texture) -> f32 {
        let rgba = texture.to_rgba8();
        let (mut total, mut weight) = (0.0f32, 0.0f32);
        for texel in rgba.chunks_exact(4) {
            let alpha = texel[3] as f32 / 255.0;
            let colour = glam::Vec3::new(
                channel_to_linear(texel[0]),
                channel_to_linear(texel[1]),
                channel_to_linear(texel[2]),
            );
            total += colour.dot(LUMA) * alpha;
            weight += alpha;
        }
        match weight > 0.0 {
            true => total / weight,
            false => 1.0,
        }
    }
}
