//! GEMM / encode A/B flags (env, read once, then overridable for inference).
//!
//! Preserves metal-native Audit 4/6 lessons that affect the shared runtime:
//! - always-on Device barrier after each dispatch (golden-safe)
//! - f32 GEMM interior offset tiles off by default
//! - TensorOps multiply_accumulate off by default
//!
//! Env names accept `METAL_RUNTIME_*` first, then fall back to `METAL_NATIVE_*`
//! so existing A/B scripts still work.

use std::sync::atomic::{AtomicI8, Ordering};
use std::sync::OnceLock;

pub(crate) fn env_truthy(names: &[&str]) -> Option<bool> {
    for name in names {
        match std::env::var(name).ok().as_deref() {
            Some("1") | Some("true") | Some("TRUE") | Some("yes") => return Some(true),
            Some("0") | Some("false") | Some("FALSE") | Some("no") => return Some(false),
            _ => {}
        }
    }
    None
}

fn flags() -> &'static (bool, bool) {
    static FLAGS: OnceLock<(bool, bool)> = OnceLock::new();
    FLAGS.get_or_init(|| {
        let gemm_interior = env_truthy(&[
            "METAL_RUNTIME_GEMM_INTERIOR",
            "METAL_NATIVE_GEMM_INTERIOR",
        ])
        .unwrap_or(false);
        let gemm_accum = env_truthy(&["METAL_RUNTIME_GEMM_ACCUM", "METAL_NATIVE_GEMM_ACCUM"])
            .unwrap_or(false);
        (gemm_interior, gemm_accum)
    })
}

/// -1 = unset (read env), 0 = always-on barriers, 1 = skip auto (hazard mode).
static HAZARD_BARRIERS: AtomicI8 = AtomicI8::new(-1);

/// Force skip/enable always-on Device barrier. Call before first encode for decode.
/// **Must** insert explicit [`crate::dispatch::Binder::barrier`] at RAW edges when
/// enabling (skip_auto=true).
pub fn set_hazard_barriers(skip_auto: bool) {
    HAZARD_BARRIERS.store(if skip_auto { 1 } else { 0 }, Ordering::Relaxed);
}

/// True once [`set_hazard_barriers`] has been called (or env was read by
/// [`hazard_barriers`]). Used by GPU init to avoid clobbering an explicit choice.
pub fn hazard_barriers_explicitly_set() -> bool {
    HAZARD_BARRIERS.load(Ordering::Relaxed) >= 0
}

/// Use interior offset tile extents in f32 GEMM. Default: false.
pub fn gemm_interior_offsets() -> bool {
    flags().0
}

/// Skip always-on Dispatch→Dispatch Device barrier after every `Binder::dispatch`.
/// Packed multi-dispatch ops still insert explicit barriers. Default: false
/// (golden-safe). **Do not enable as default** without RAW-edge barriers.
pub fn hazard_barriers() -> bool {
    let v = HAZARD_BARRIERS.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let from_env = env_truthy(&[
        "METAL_RUNTIME_HAZARD_BARRIERS",
        "METAL_NATIVE_HAZARD_BARRIERS",
    ])
    .unwrap_or(false);
    HAZARD_BARRIERS.store(if from_env { 1 } else { 0 }, Ordering::Relaxed);
    from_env
}

/// Use TensorOps `multiply_accumulate` for accumulate GEMMs. Default: false.
pub fn gemm_accum() -> bool {
    flags().1
}

/// -1 unset, 0 fine-grained RAW barriers, 1 phase-coarsened (default when hazard).
static COARSE_BARRIERS: AtomicI8 = AtomicI8::new(-1);

/// Coarsen decode RAW barriers to phase edges (fewer Device drains).
/// Default: on when [`hazard_barriers`] is true. Override with
/// `GEMMA_METAL_COARSE_BARRIERS=0|1` / `METAL_RUNTIME_COARSE_BARRIERS`.
pub fn coarse_barriers() -> bool {
    let v = COARSE_BARRIERS.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let from_env = env_truthy(&[
        "GEMMA_METAL_COARSE_BARRIERS",
        "METAL_RUNTIME_COARSE_BARRIERS",
    ]);
    let on = from_env.unwrap_or_else(hazard_barriers);
    COARSE_BARRIERS.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

/// Explicit RAW barrier needed for a phase edge. In fine mode: always when hazard.
/// In coarse mode: only when `phase_edge` is true (major producer→consumer).
pub fn need_barrier(phase_edge: bool) -> bool {
    if !hazard_barriers() {
        return false;
    }
    if coarse_barriers() {
        phase_edge
    } else {
        true
    }
}
