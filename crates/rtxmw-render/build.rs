//! Compiles every GLSL shader to SPIR-V and validates it.
//!
//! Shaders are built here rather than at runtime so a syntax error fails the build instead of the
//! frame, and so a release binary carries no compiler. Validation runs as part of the build for the
//! same reason: `spirv-val` catches things `glslc` accepts, and a module that only fails inside the
//! driver is far harder to diagnose.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Vulkan version the modules target. ash 0.38 ships 1.3 headers, so 1.3 is the ceiling.
const TARGET_ENV: &str = "vulkan1.3";

fn main() {
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));

    // Any change under `shaders/` rebuilds, which covers `#include`d fragments so long as they
    // live here too — an include reached from outside this tree would not retrigger.
    println!("cargo:rerun-if-changed={}", shader_dir.display());
    println!("cargo:rerun-if-changed=build.rs");

    let mut built = 0;
    for entry in std::fs::read_dir(&shader_dir).expect("shaders directory should exist") {
        let source = entry.expect("readable directory entry").path();
        if !is_shader(&source) {
            continue;
        }
        let name = source
            .file_name()
            .and_then(|n| n.to_str())
            .expect("shader names are UTF-8");
        let output = out_dir.join(format!("{name}.spv"));

        compile(&source, &output);
        validate(&output);
        built += 1;
    }
    assert!(built > 0, "no shaders found in {}", shader_dir.display());
}

/// Whether the path is a GLSL source file rather than an include or stray file.
///
/// `.glsl` is deliberately excluded: those are shared fragments meant to be `#include`d, not
/// compiled on their own.
fn is_shader(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("comp" | "vert" | "frag" | "rgen" | "rchit" | "rahit" | "rmiss" | "rint" | "rcall")
    )
}

fn compile(source: &Path, output: &Path) {
    let result = Command::new("glslc")
        .arg(format!("--target-env={TARGET_ENV}"))
        // Keep names and line numbers so a validation failure or a debugger points at GLSL.
        .arg("-g")
        .arg("-O")
        .arg("-o")
        .arg(output)
        .arg(source)
        .output();

    let result = result.unwrap_or_else(|e| {
        panic!("could not run glslc ({e}); it is required to build shaders — install shaderc")
    });
    if !result.status.success() {
        panic!(
            "glslc failed for {}:\n{}",
            source.display(),
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

fn validate(module: &Path) {
    let result = Command::new("spirv-val")
        .arg("--target-env")
        .arg(TARGET_ENV)
        // The shaders declare `scalar` block layout and the device enables the matching feature, so
        // the validator has to be told the same. Without it a `vec3`-carrying struct array is
        // rejected for a stride that is correct under the rules actually in force.
        .arg("--scalar-block-layout")
        .arg(module)
        .output();

    let result = result.unwrap_or_else(|e| {
        panic!(
            "could not run spirv-val ({e}); it is required to build shaders — install spirv-tools"
        )
    });
    if !result.status.success() {
        panic!(
            "spirv-val rejected {}:\n{}",
            module.display(),
            String::from_utf8_lossy(&result.stderr)
        );
    }
}
