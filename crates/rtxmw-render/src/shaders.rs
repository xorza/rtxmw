//! Compiled shader modules, embedded at build time.

/// Wraps bytes so they land on a four-byte boundary.
///
/// SPIR-V is a stream of `u32`, and Vulkan takes it as one. `include_bytes!` only promises a `u8`
/// array, whose alignment is one — reinterpreting that as `u32` would be undefined behaviour on
/// every architecture that cares.
#[repr(C, align(4))]
struct Aligned<T: ?Sized>(T);

/// Embeds a module the build script produced, as the `u32` slice Vulkan wants.
macro_rules! include_spirv {
    ($name:literal) => {{
        static ALIGNED: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(
            env!("OUT_DIR"),
            "/",
            $name,
            ".spv"
        )));
        // SAFETY: `Aligned` forces four-byte alignment, and the build script only emits modules
        // `spirv-val` accepted, so the length is a whole number of words.
        unsafe { std::slice::from_raw_parts(ALIGNED.0.as_ptr().cast::<u32>(), ALIGNED.0.len() / 4) }
    }};
}

/// Primary visibility: one ray query per pixel, writing false colour to a storage image.
pub fn primary_visibility() -> &'static [u32] {
    include_spirv!("primary_visibility.comp")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SPIR-V magic number, little-endian as the format defines it.
    const MAGIC: u32 = 0x0723_0203;

    #[test]
    fn the_embedded_module_is_spirv_and_correctly_aligned() {
        let module = primary_visibility();
        assert!(!module.is_empty());
        assert_eq!(module[0], MAGIC, "not a SPIR-V module");
        // A misaligned cast would be undefined behaviour rather than a wrong value, so assert the
        // property directly instead of trusting that it happened to work.
        assert_eq!(module.as_ptr().addr() % align_of::<u32>(), 0);
    }
}
