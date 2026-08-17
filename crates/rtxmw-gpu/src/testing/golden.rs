//! Screenshot writing and golden-image comparison.

use std::path::{Path, PathBuf};

/// Set this environment variable to rewrite baselines instead of failing.
const UPDATE_VAR: &str = "UPDATE_GOLDEN";

/// Writes `pixels` (8-bit RGBA, row-major) to `path` as a PNG, creating parent directories.
pub fn write_png(path: &Path, pixels: &[u8], width: u32, height: u32) {
    assert_eq!(
        pixels.len(),
        (width * height * 4) as usize,
        "pixel buffer is {} bytes, expected {} for {width}x{height} RGBA",
        pixels.len(),
        width * height * 4
    );

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("could not create output directory");
    }
    let file = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("could not create {}: {e}", path.display()));

    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .expect("could not write PNG header")
        .write_image_data(pixels)
        .expect("could not write PNG data");
}

/// Reads an 8-bit RGBA PNG back into a pixel buffer.
fn read_png(path: &Path) -> ImageData {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("could not open {}: {e}", path.display()));
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().expect("could not read PNG header");
    let mut pixels = vec![0; reader.output_buffer_size().expect("PNG too large")];
    let info = reader.next_frame(&mut pixels).expect("could not read PNG");
    pixels.truncate(info.buffer_size());
    ImageData {
        pixels,
        width: info.width,
        height: info.height,
    }
}

/// A decoded image and its dimensions.
#[derive(Debug)]
struct ImageData {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

/// How far two images differ.
#[derive(Debug, Clone, Copy)]
pub struct Difference {
    /// Count of pixels with any channel differing.
    pub differing_pixels: usize,
    /// Largest absolute difference across all channels, 0-255.
    pub max_channel_delta: u8,
    /// Mean absolute channel difference across the whole image.
    pub mean_channel_delta: f64,
}

impl Difference {
    /// Whether the two images are byte-identical.
    pub fn is_identical(&self) -> bool {
        self.differing_pixels == 0
    }
}

/// Compares `pixels` against the baseline for `name`, failing the test on mismatch.
///
/// Baselines live in `tests/golden/<name>.png` relative to the crate root. On mismatch the actual
/// image is written next to the baseline as `<name>.actual.png` so it can be inspected, and the
/// failure reports numbers rather than just "images differ".
///
/// Run with `UPDATE_GOLDEN=1` to write baselines instead of comparing — review the diff before
/// committing, since this will happily bless a regression.
#[track_caller]
pub fn assert_matches(name: &str, pixels: &[u8], width: u32, height: u32) {
    let baseline = golden_dir().join(format!("{name}.png"));

    if std::env::var_os(UPDATE_VAR).is_some() {
        write_png(&baseline, pixels, width, height);
        eprintln!("{UPDATE_VAR} set: wrote {}", baseline.display());
        return;
    }

    if !baseline.exists() {
        let actual = golden_dir().join(format!("{name}.actual.png"));
        write_png(&actual, pixels, width, height);
        panic!(
            "no baseline at {}\nwrote the current render to {}\nre-run with {UPDATE_VAR}=1 to \
             accept it",
            baseline.display(),
            actual.display()
        );
    }

    let expected = read_png(&baseline);
    assert_eq!(
        (expected.width, expected.height),
        (width, height),
        "baseline {name} is {}x{} but the render is {width}x{height}",
        expected.width,
        expected.height
    );

    let difference = compare(&expected.pixels, pixels);
    if difference.is_identical() {
        return;
    }

    let actual = golden_dir().join(format!("{name}.actual.png"));
    write_png(&actual, pixels, width, height);
    panic!(
        "render does not match baseline {name}\n  \
         differing pixels   {} of {}\n  \
         max channel delta  {}\n  \
         mean channel delta {:.4}\n  \
         actual written to  {}\n  \
         re-run with {UPDATE_VAR}=1 to accept",
        difference.differing_pixels,
        width * height,
        difference.max_channel_delta,
        difference.mean_channel_delta,
        actual.display()
    );
}

/// Per-channel comparison of two equally sized 8-bit RGBA buffers.
pub fn compare(expected: &[u8], actual: &[u8]) -> Difference {
    assert_eq!(
        expected.len(),
        actual.len(),
        "cannot compare buffers of {} and {} bytes",
        expected.len(),
        actual.len()
    );

    let mut differing_pixels = 0;
    let mut max_channel_delta = 0u8;
    let mut total_delta = 0u64;

    for (a, b) in expected.chunks_exact(4).zip(actual.chunks_exact(4)) {
        let mut differs = false;
        for (x, y) in a.iter().zip(b) {
            let delta = x.abs_diff(*y);
            if delta != 0 {
                differs = true;
            }
            max_channel_delta = max_channel_delta.max(delta);
            total_delta += u64::from(delta);
        }
        if differs {
            differing_pixels += 1;
        }
    }

    let channels = expected.len().max(1) as f64;
    Difference {
        differing_pixels,
        max_channel_delta,
        mean_channel_delta: total_delta as f64 / channels,
    }
}

/// Where baselines live: `<crate>/tests/golden`.
fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_buffers_compare_equal() {
        let pixels = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let difference = compare(&pixels, &pixels);
        assert!(difference.is_identical());
        assert_eq!(difference.max_channel_delta, 0);
        assert_eq!(difference.mean_channel_delta, 0.0);
    }

    #[test]
    fn difference_reports_counts_deltas_and_mean() {
        // Two pixels; the second differs in one channel by 10.
        let expected = vec![0u8, 0, 0, 255, 100, 0, 0, 255];
        let actual = vec![0u8, 0, 0, 255, 110, 0, 0, 255];
        let difference = compare(&expected, &actual);

        assert_eq!(difference.differing_pixels, 1);
        assert_eq!(difference.max_channel_delta, 10);
        // One channel differs by 10 across 8 channels total: 10 / 8 = 1.25.
        assert!((difference.mean_channel_delta - 1.25).abs() < 1e-9);
        assert!(!difference.is_identical());
    }

    #[test]
    fn png_round_trips_through_disk() {
        let width = 3;
        let height = 2;
        let pixels: Vec<u8> = (0..width * height * 4).map(|i| i as u8).collect();

        let path = std::env::temp_dir().join("rtxmw-golden-roundtrip.png");
        write_png(&path, &pixels, width, height);
        let read = read_png(&path);
        std::fs::remove_file(&path).ok();

        assert_eq!(read.width, width);
        assert_eq!(read.height, height);
        assert_eq!(read.pixels, pixels);
    }
}
