//! Errors raised while decoding a texture.

/// Anything that can go wrong turning file bytes into a texture.
#[derive(Debug)]
pub enum TextureError {
    /// The file ends before the header or pixel data it declares.
    Truncated { wanted: usize, available: usize },
    /// A DDS compression scheme this decoder does not handle.
    UnsupportedFourCc([u8; 4]),
    /// An uncompressed DDS layout other than `A8R8G8B8`, with its red, green, blue and alpha masks.
    UnsupportedPixelFormat { bit_count: u32, masks: [u32; 4] },
    /// A TGA image type or depth this decoder does not handle.
    UnsupportedTga { image_type: u8, bits: u8 },
    /// A header declaring no pixels at all.
    EmptyImage { width: u32, height: u32 },
}

impl std::fmt::Display for TextureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated { wanted, available } => {
                write!(
                    f,
                    "file ends early: wanted {wanted} bytes, have {available}"
                )
            }
            Self::UnsupportedFourCc(code) => {
                write!(
                    f,
                    "unsupported dds compression {:?}",
                    String::from_utf8_lossy(code)
                )
            }
            Self::UnsupportedPixelFormat { bit_count, masks } => write!(
                f,
                "unsupported uncompressed dds layout: {bit_count} bits, masks \
                 r{:#010x} g{:#010x} b{:#010x} a{:#010x}",
                masks[0], masks[1], masks[2], masks[3]
            ),
            Self::UnsupportedTga { image_type, bits } => {
                write!(f, "unsupported tga: image type {image_type}, {bits} bits")
            }
            Self::EmptyImage { width, height } => {
                write!(f, "image declares no pixels: {width}x{height}")
            }
        }
    }
}

impl std::error::Error for TextureError {}

/// Result alias for texture decoding.
pub type Result<T> = std::result::Result<T, TextureError>;
