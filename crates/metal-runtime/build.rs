//! AOT-compile `kernels/*.metal` → `default.metallib`.
//!
//! Targets Metal 4 / macOS 26 for TensorOps (`matmul2d`) with bf16 enabled in
//! the language dialect. The portable `simdgroup_matrix` kernel is always
//! included for A/B.
//!
//! Requires the Xcode Metal Toolchain component:
//!   `xcodebuild -downloadComponent MetalToolchain`
//!
//! Important: do **not** invoke `xcrun -sdk macosx metal` — the `-sdk` switch
//! breaks cryptex Metal Toolchain resolution on Xcode 26+. Use `xcrun metal`
//! plus an explicit `-isysroot`.
//!
//! `METAL_RUNTIME_SKIP_AOT` hazard: when set, this script skips compilation and
//! points `METAL_RUNTIME_METALLIB` at the crate-root `default.metallib`. That
//! file may be stale or missing — only use for intentional offline/CI skips
//! after a known-good metallib is already present at the crate root.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=METAL_RUNTIME_SKIP_AOT");

    // CoreGraphics is required for MTLCreateSystemDefaultDevice.
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");

    if env::var_os("METAL_RUNTIME_SKIP_AOT").is_some() {
        println!("cargo:warning=METAL_RUNTIME_SKIP_AOT set; skipping metallib AOT");
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let crate_lib = manifest_dir.join("default.metallib");
        println!(
            "cargo:rustc-env=METAL_RUNTIME_METALLIB={}",
            crate_lib.display()
        );
        return;
    }

    ensure_developer_dir();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels_dir = manifest_dir.join("kernels");

    let sdk = xcrun_stdout(&["--sdk", "macosx", "--show-sdk-path"]);
    let metal = resolve_metal();
    let metallib = resolve_metallib();

    let mut air_files: Vec<PathBuf> = Vec::new();

    // TensorOps kernels — Metal 4 dialect (macOS 26+ / MPP). Hard-fail: NAX GEMM
    // is the hot path; a simdgroup-only metallib is not acceptable.
    let tensorops_sources = ["matmul_tensorops.metal"];
    for name in &tensorops_sources {
        let src = kernels_dir.join(name);
        if !src.exists() {
            panic!(
                "required TensorOps source missing: {}; Metal 4 / macOS 26 toolchain required",
                src.display()
            );
        }
        let air = out_dir.join(format!("{}.air", src.file_stem().unwrap().to_string_lossy()));
        let status = Command::new(&metal)
            .args([
                "-std=metal4.0",
                "-O2",
                "-isysroot",
                &sdk,
                "-mmacosx-version-min=26.0",
                "-c",
            ])
            .arg(&src)
            .arg("-o")
            .arg(&air)
            .status()
            .unwrap_or_else(|e| panic!("failed to spawn metal for {name}: {e}"));
        if !status.success() {
            panic!(
                "{name} failed to compile (need Metal 4 / macOS 26 SDK + MetalToolchain); \
                 refusing simdgroup-only metallib"
            );
        }
        air_files.push(air);
    }
    println!("cargo:rustc-cfg=metal_runtime_tensorops");

    // All other .metal sources (simdgroup GEMM + util kernels).
    let skip: &[&str] = &["matmul_tensorops.metal"];
    let mut others: Vec<PathBuf> = fs::read_dir(&kernels_dir)
        .unwrap_or_else(|e| panic!("read kernels/: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("metal")
                && !skip
                    .iter()
                    .any(|s| p.file_name().and_then(|n| n.to_str()) == Some(*s))
        })
        .collect();
    others.sort();

    for src in &others {
        let stem = src.file_stem().unwrap().to_string_lossy();
        let air = out_dir.join(format!("{stem}.air"));
        let ok_m4 = try_metal_compile(&metal, &sdk, src, &air, "metal4.0");
        if !ok_m4 {
            println!(
                "cargo:warning={} failed under -std=metal4.0; falling back to -std=metal3.2 \
                 (shader dialect only; encode remains Metal 4)",
                src.file_name().and_then(|n| n.to_str()).unwrap_or(&stem)
            );
            run(
                Command::new(&metal)
                    .args([
                        "-std=metal3.2",
                        "-O2",
                        "-isysroot",
                        &sdk,
                        "-mmacosx-version-min=26.0",
                        "-c",
                    ])
                    .arg(src)
                    .arg("-o")
                    .arg(&air),
                &format!("metal compile {} (metal3.2 fallback)", src.display()),
            );
        }
        air_files.push(air);
    }

    let metallib_out = out_dir.join("default.metallib");
    let mut link = Command::new(&metallib);
    for air in &air_files {
        link.arg(air);
    }
    link.arg("-o").arg(&metallib_out);
    run(&mut link, "metallib link");

    let crate_copy = manifest_dir.join("default.metallib");
    let _ = std::fs::copy(&metallib_out, &crate_copy);

    println!(
        "cargo:rustc-env=METAL_RUNTIME_METALLIB={}",
        metallib_out.display()
    );
}

fn try_metal_compile(metal: &Path, sdk: &str, src: &Path, air: &Path, metal_std: &str) -> bool {
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
    if let Ok(p) = xcrun_try(&["-f", "metal"]) {
        return PathBuf::from(p);
    }
    panic!(
        "metal compiler not found. Install Xcode and run:\n  \
         sudo xcode-select -s /Applications/Xcode.app/Contents/Developer\n  \
         xcodebuild -downloadComponent MetalToolchain"
    );
}

fn resolve_metallib() -> PathBuf {
    PathBuf::from(xcrun_stdout(&["-f", "metallib"]))
}

fn xcrun_stdout(args: &[&str]) -> String {
    let out = Command::new("xcrun")
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("xcrun {:?} failed to spawn: {e}", args));
    if !out.status.success() {
        panic!(
            "xcrun {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn xcrun_try(args: &[&str]) -> Result<String, ()> {
    let out = Command::new("xcrun").args(args).output().map_err(|_| ())?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err(())
    }
}

fn run(cmd: &mut Command, label: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("{label}: failed to spawn: {e}"));
    if !status.success() {
        panic!("{label}: exited with {status}");
    }
}
