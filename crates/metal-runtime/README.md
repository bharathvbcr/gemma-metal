# Metal Runtime

Shared **Metal 4-only** encode substrate extracted from
`arch_02_value_resid/metal-native` for Gemma inference (`gemma-metal`) **without**
dragging training bwd / Muon / XSA / VE.

**Training stays in metal-native.** Do not train models in this crate. Optional
future: metal-native depends on metal-runtime for encode; today this is a
**duplicate extract** (metal-native left intact).

## What was extracted

| Module | Role |
|--------|------|
| `runtime` | Device, MTL4 CB / compute encoder, residency set, Hot/Cold/Bump pool, packed binder, const arena (~1 MiB), SharedEvent sync |
| `dispatch` | Argument-table binds + helpers (`set_tensor`, scalars, 1D dispatch) |
| `tensor` | `GpuBuffer` / `Tensor` views, dtypes |
| `gemm` | TensorOps `matmul2d` (preferred) + simdgroup fallback — **prefill shapes** |
| `ops` | `softcap_f32` and small util launches |
| `mtl_tensor` | WWDC26-330 quantized MTLTensor **prep** (Int8 maps today; Int4/FP8 Err until SDK) |
| `ab_flags` | A/B toggles carried from native extract |

Kernels (AOT via `build.rs`): `matmul_tensorops.metal`, `matmul_simdgroup.metal`,
`utils.metal`.

## Doctrine

- Metal **4-only** encode — no classic M3 command-buffer fallback
- Residency / packed encoder / **no host-zero mid-CB** (Audit 4/6)
- Prefill: TensorOps GEMM (± native quant MTLTensor later)
- Decode GEMV / dual FA / PLE / KV: **live in `gemma-metal`**, not here
- Do **not** drive M=1 decode through TensorOps GEMM tiles

## MTLTensor hooks

- `QuantDType::{Int8, Int4, Fp8E8M0}` — Int8 → `MTLTensorDataType::Int8`; others error until bindings exist
- `probe_tensor_support` / `alloc_device_tensor` / `tensor_from_buffer` — **experimental**; comments note objc2 SIGSEGV risk on some layouts — prefer descriptor-only unit tests
- `try_quant_tensorops_prefill_gemm` — still returns not-wired (quant MTLTensor prefill). Phase 2 landed hand **GEMV / dual FA / PLE / MLP** in `gemma-metal`; decode stays GEMV

## How to test

```bash
cd Rust_MLKit/crates/metal-runtime
cargo test
cargo test --release
```

GPU tests need a Metal 4 capable Mac (macOS 26+). Metallib path:
`metal_runtime::metallib_path()`.

## Consumers

- **`gemma-metal`** — primary (inference)
- **metal-native** — unchanged training stack; not wired to this crate yet
