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

    #[cfg(feature = "dlss")]
    link_ngx();

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

/// Points the linker at NVIDIA's NGX SDK, or explains where it should be.
///
/// **The SDK is not in this repository and cannot be.** It is NVIDIA's, under the RTX SDK licence,
/// so it is fetched into `.refs/` like the OpenMW checkout and the game data — see `docs/design.md`
/// §6. `DLSS_SDK_DIR` overrides the location for a machine that keeps it elsewhere.
///
/// Only called under the `dlss` feature, so a build without it neither needs the SDK nor mentions
/// it.
#[cfg(feature = "dlss")]
fn link_ngx() {
    println!("cargo:rustc-check-cfg=cfg(ngx)");
    println!("cargo:rerun-if-env-changed=DLSS_SDK_DIR");
    let named = std::env::var_os("DLSS_SDK_DIR").map(PathBuf::from);
    let sdk = named.clone().unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".refs/dlss")
    });
    let lib = sdk.join("lib/Linux_x86_64");

    if !lib.join("libnvsdk_ngx.a").is_file() {
        // **Missing is not an error unless it was asked for by name.** `--all-features` is the
        // documented verification chain, so a fresh clone has to build with it; the SDK is NVIDIA's
        // and cannot be in this repository, and a build that fails without it would make the chain
        // undocumentable. Absent, the feature compiles to nothing and its tests are not there to
        // skip — the same shape as a machine without the game.
        assert!(
            named.is_none(),
            "DLSS_SDK_DIR names {}, and there is no libnvsdk_ngx.a under it",
            sdk.display()
        );
        println!(
            "cargo:warning=no NGX SDK at {}, so DLSS is compiled out. Fetch it with `git clone \
             --depth 1 https://github.com/NVIDIA/DLSS.git .refs/dlss`, or set DLSS_SDK_DIR.",
            sdk.display()
        );
        return;
    }

    println!("cargo:rustc-cfg=ngx");
    // Where NGX must be told to look for its feature libraries: the default search is the
    // application folder alone, and these are not beside the binary. Handed down rather than
    // rebuilt, so `DLSS_SDK_DIR` is honoured everywhere the SDK is named and not only here.
    println!(
        "cargo:rustc-env=NGX_FEATURE_DIR={}",
        lib.join("rel").display()
    );
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=nvsdk_ngx");
    // The SDK is C++ and loads the driver's NGX core at runtime, so it wants both.
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=dl");
}
