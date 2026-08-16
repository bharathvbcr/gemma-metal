//! AOT-compile `kernels/*.metal` → gemma-metal metallib (Phase 2 hot path).
//!
//! Set `GEMMA_METAL_SKIP_AOT=1` to skip (CI / offline). Encode uses
//! `metal-runtime`'s Metal 4 path; these kernels load as an overlay via
//! [`GpuRuntime::add_metallib`].

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=GEMMA_METAL_SKIP_AOT");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");

    if env::var_os("GEMMA_METAL_SKIP_AOT").is_some() {
        println!("cargo:warning=GEMMA_METAL_SKIP_AOT set; skipping gemma metallib AOT");
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let crate_lib = manifest_dir.join("default.metallib");
        println!(
            "cargo:rustc-env=GEMMA_METAL_METALLIB={}",
            crate_lib.display()
        );
        return;
    }

    ensure_developer_dir();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels_dir = manifest_dir.join("kernels");
    if !kernels_dir.is_dir() {
        println!("cargo:warning=no kernels/ dir; skipping metallib");
        println!("cargo:rustc-env=GEMMA_METAL_METALLIB=");
        return;
    }

    let sdk = xcrun_stdout(&["--sdk", "macosx", "--show-sdk-path"]);
    let metal = resolve_metal();
    let metallib = PathBuf::from(xcrun_stdout(&["-f", "metallib"]));

    let mut sources: Vec<PathBuf> = fs::read_dir(&kernels_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("metal"))
        .collect();
    sources.sort();

    let mut air_files = Vec::new();
    for src in &sources {
        let stem = src.file_stem().unwrap().to_string_lossy();
        let air = out_dir.join(format!("{stem}.air"));
        let ok = try_compile(&metal, &sdk, src, &air, "metal4.0")
            || try_compile(&metal, &sdk, src, &air, "metal3.2");
        if !ok {
            panic!("failed to compile {}", src.display());
        }
        air_files.push(air);
    }

    if air_files.is_empty() {
        println!("cargo:rustc-env=GEMMA_METAL_METALLIB=");
        return;
    }

    let metallib_out = out_dir.join("default.metallib");
    let mut link = Command::new(&metallib);
    for air in &air_files {
        link.arg(air);
    }
    link.arg("-o").arg(&metallib_out);
    let status = link.status().expect("metallib spawn");
    if !status.success() {
        panic!("metallib link failed");
    }
    let _ = fs::copy(&metallib_out, manifest_dir.join("default.metallib"));
    println!(
        "cargo:rustc-env=GEMMA_METAL_METALLIB={}",
        metallib_out.display()
    );
}

fn try_compile(metal: &Path, sdk: &str, src: &Path, air: &Path, metal_std: &str) -> bool {
    let std_flag = format!("-std={metal_std}");
    Command::new(metal)
        .args([
            std_flag.as_str(),
            "-O2",
            "-isysroot",
            sdk,
            "-mmacosx-version-min=26.0",
            "-c",
        ])
        .arg(src)
        .arg("-o")
        .arg(air)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn ensure_developer_dir() {
    if env::var_os("DEVELOPER_DIR").is_some() {
        return;
    }
    let xcode = Path::new("/Applications/Xcode.app/Contents/Developer");
    if xcode.is_dir() {
        unsafe { env::set_var("DEVELOPER_DIR", xcode) };
    }
}

fn resolve_metal() -> PathBuf {
    PathBuf::from(xcrun_stdout(&["-f", "metal"]))
}

fn xcrun_stdout(args: &[&str]) -> String {
    let out = Command::new("xcrun")
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("xcrun {:?} spawn: {e}", args));
    if !out.status.success() {
        panic!(
            "xcrun {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}
