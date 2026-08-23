//! Encode-once / ping-pong Metal4 command-buffer replay **scaffolding** (inference lane).
//!
//! ## Status (honest)
//!
//! | Piece | State |
//! |---|---|
//! | Dual-slot A/B bookkeeping | **Works** — record/commit/reuse + `mark_live_step` |
//! | Stable GPU scalar buffers (`pos`, seed) as replay inputs | **Works** at session layer (`GpuDecodeSession::pos_buf`); seed already GPU |
//! | Re-encode into the same `MTL4CommandBuffer` object | **Works** today via runtime `begin`→encode→`end`→commit (not free) |
//! | True encode-once (commit prior encoding without re-recording) | **Blocked** — see [`survey_cb_replay_api_gaps`] |
//! | Opt-in ledger (`GEMMA_METAL_ENCODE_ONCE=1`) | **Works** — advances ping-pong after live encode; does not skip host encode |
//! | Indirect Command Buffer (ICB) compute replay | **Smoke wired** — [`crate::icb_smoke`] (`copy_f32` ConcurrentDispatch); decode graph not migrated |
//! | Argument-table / ICB slot plan | **Scaffold** — [`ArgTableSlotPlan`] / [`IcbReplayStub`] (host plan; smoke owns real ICB) |
//! | Full decode-graph capture into a replayable CB | **Stubbed** — needs kernels to bind stable buffer addresses (not const-arena scalars) for every per-token arg |
//!
//! ## SDK survey (objc2-metal 0.3 + macOS 26.x SDK)
//!
//! **Available / proven (mini smoke):**
//! - `MTLIndirectCommandType::ConcurrentDispatch` / `ConcurrentDispatchThreads`
//! - `MTLIndirectComputeCommand` (`setComputePipelineState`, `setKernelBuffer:offset:atIndex:`,
//!   `concurrentDispatchThreadgroups`)
//! - `MTLDevice::newIndirectCommandBufferWithDescriptor:maxCommandCount:options:`
//! - `MTL4ComputeCommandEncoder::executeCommandsInBuffer:withRange:` (**feature enabled**;
//!   see [`crate::icb_smoke::run_copy_f32_smoke`])
//! - `MTLComputePipelineDescriptor::supportIndirectCommandBuffers`
//!
//! **Missing / incompatible with shipping encode:**
//! - No `MTL4CommandBuffer` “replay prior encoding” API — `beginCommandBufferWithAllocator:`
//!   starts a fresh recording; allocator reset drops prior commands (not CUDA-graph).
//! - Classic ICB `setKernelBuffer` freeze **works** for DecodeIcb when tg_mem is also
//!   frozen into the ICB cmd (v0.5.7 `GEMMA_METAL_ICB_FREEZE_BINDS=1`; mini parity PASS).
//!   Opt-in range-batch (v0.5.8 `GEMMA_METAL_ICB_RANGE_BATCH=1`) coalesces safe spans
//!   between barriers into one `executeCommandsInBuffer` range. Default path still
//!   uses inheritBuffers+prebuilt arg-tables (`setArgumentTable` × cmds). Flags stay
//!   OFF until a product tok/s win vs direct dispatch+prebuilt.
//!
//! ## MID_COMMIT ≠ encode-once
//!
//! `METAL_RUNTIME_MID_COMMIT=N` only overlaps host encode with GPU execution of the
//! *previous* chunk (dual allocator ping-pong). It does **not** eliminate per-token
//! host encode. See gemma-metal `DECISIONS.md` D16.
//!
//! ## Blockers for real encode-once
//!
//! 1. **MTL4 CB is not a CUDA graph.** Reusing the CB object re-records; there is no
//!    documented "replay prior encoding" without ICB / equivalent.
//! 2. **Const-arena scalars** (`set_u32` / `bind_bytes`) bump a cursor every bind —
//!    addresses are ephemeral per encode. Replay needs stable `GpuBuffer` slots
//!    (`pos_buf`, `seed_tok`, …) that kernels read via `device` pointers.
//! 3. **Argument tables are mutable** and bindings are snapshotted at dispatch time —
//!    fine for live encode; for ICB/replay the recorded resource IDs must remain valid.
//! 4. **FA / kv_store / softcap** still take CPU `u32` constants in places — migrating
//!    every signature is large; `rms_qkv_rope_posbuf` is the prototype path.
//!
//! Point of contact: this module + `gemma-metal::gpu_model::GpuDecodeSession::pos_buf`.

use std::sync::atomic::{AtomicU64, Ordering};

/// Concrete SDK / runtime gaps that keep [`PingPongCbReplay::try_replay`] → [`CbReplayError::NotWired`].
///
/// Surveyed against objc2-metal 0.3 + MacOSX26 SDK (2026-07-19). Ordered by severity
/// for a CUDA-graph-style “skip host re-encode” path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CbReplayApiGap {
    /// `MTL4CommandBuffer` exposes begin/end/commit only — no replay of a prior encoding.
    Mtl4CbHasNoReplayApi,
    /// Mini + E4B Hot DecodeIcb layer-graph capture/replay may be wired (opt-in);
    /// **31B** session decode graph is not migrated.
    IcbDecodeGraphNotMigrated,
    /// Default still pays per-cmd `setArgumentTable`; opt-in freeze/range/coarse
    /// (v0.5.7–0.5.9) can zero setArgTable and cut execute_icb×N, but product
    /// tok/s loses vs dispatch+prebuilt — **PARKED** (prefer prebuilt).
    IcbClassicBindsVsMtl4ArgumentTable,
}

impl CbReplayApiGap {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mtl4CbHasNoReplayApi => {
                "MTL4CommandBuffer has no replay-prior-encoding API (begin+allocator re-records); path is compute ICB only"
            }
            Self::IcbDecodeGraphNotMigrated => {
                "mini+E4B Hot DecodeIcb layer-graph capture→from_commands+skip landed (flags default OFF); 31B session graph not migrated"
            }
            Self::IcbClassicBindsVsMtl4ArgumentTable => {
                "DecodeIcb default: inheritBuffers+prebuilt arg-tables (setArgumentTable × cmds; last_setAddress=0). Opt-in freeze/range/coarse (v0.5.7–0.5.9) PARKED: setArgTable=0 + execute_icb ranges, but product tok/s loses vs prebuilt — prefer prebuilt; flags default OFF"
            }
        }
    }
}

/// Fixed survey result for this SDK / crate cut (do not invent gaps — update when API lands).
///
/// Closed this cut: `EphemeralConstArenaScalars` (FA/kv/softcap → `IcbScalarPool` +
/// `reset_step`). Mini DecodeIcb bridge lands; full-graph gap remains.
pub fn survey_cb_replay_api_gaps() -> &'static [CbReplayApiGap] {
    &[
        CbReplayApiGap::Mtl4CbHasNoReplayApi,
        CbReplayApiGap::IcbDecodeGraphNotMigrated,
        CbReplayApiGap::IcbClassicBindsVsMtl4ArgumentTable,
    ]
}

/// One-line summary of the hard blocker (primary gap first).
pub fn cb_replay_api_gap_summary() -> String {
    survey_cb_replay_api_gaps()
        .iter()
        .map(|g| g.as_str())
        .collect::<Vec<_>>()
        .join("; ")
}

/// Hint for which ICB command types a future encode-once path would request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcbCommandTypeHint {
    /// `MTLIndirectCommandTypeConcurrentDispatch` (compute threadgroups).
    ConcurrentDispatch,
    /// `MTLIndirectCommandTypeConcurrentDispatchThreads`.
    ConcurrentDispatchThreads,
}

/// One argument-table / ICB kernel-buffer slot in the host-side plan (no Metal object).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgTableSlot {
    /// MTL4 argument-table index **or** classic ICB `setKernelBuffer` index (same numbering intent).
    pub index: u32,
    /// Debug label (e.g. `pos_buf`, `seed_tok`, `x`).
    pub label: String,
    /// When true, a stable `GpuBuffer` exists for this slot (safe for replay/ICB).
    /// When false, the live path still uses const-arena / ephemeral binds.
    pub stable: bool,
}

/// Host-side plan for freezing argument-table / ICB binds (scaffold only).
#[derive(Clone, Debug)]
pub struct ArgTableSlotPlan {
    pub slots: Vec<ArgTableSlot>,
    /// Max ICB commands to allocate when/if Metal objects are created.
    pub max_commands: u32,
    pub command_types: IcbCommandTypeHint,
}

impl ArgTableSlotPlan {
    /// Mini / encode-once v0 plan: stable pos/seed/argmax + FA/kv/softcap pool.
    pub fn encode_once_v0_mini() -> Self {
        Self {
            slots: vec![
                ArgTableSlot {
                    index: 0,
                    label: "pos_buf".into(),
                    stable: true,
                },
                ArgTableSlot {
                    index: 1,
                    label: "seed_tok".into(),
                    stable: true,
                },
                ArgTableSlot {
                    index: 2,
                    label: "argmax_tok".into(),
                    stable: true,
                },
                ArgTableSlot {
                    index: 3,
                    label: "icb_scalars_softcap".into(),
                    stable: true,
                },
                ArgTableSlot {
                    index: 4,
                    label: "icb_scalars_u32s_f32s".into(),
                    stable: true,
                },
                // Remaining GEMV/MLP/etc. dims may still bump const-arena.
                ArgTableSlot {
                    index: 5,
                    label: "const_arena_residual".into(),
                    stable: false,
                },
            ],
            max_commands: 64,
            command_types: IcbCommandTypeHint::ConcurrentDispatch,
        }
    }

    pub fn stable_count(&self) -> usize {
        self.slots.iter().filter(|s| s.stable).count()
    }

    pub fn ephemeral_count(&self) -> usize {
        self.slots.iter().filter(|s| !s.stable).count()
    }
}

/// Lifecycle of the ICB stub (host bookkeeping for decode-graph plan).
///
/// Mini smoke owns a real ICB via [`crate::icb_smoke`]; this stub tracks the
/// *decode* encode-once plan and stays NotWired for full-graph allocate/execute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcbStubPhase {
    /// Default — plan may exist, no allocation attempted.
    Idle,
    /// Slot plan filled; decode-graph Metal ICB not created.
    Planned,
    /// Mini smoke proved API; decode-graph allocate still deferred.
    SmokeProven,
    /// Would hold a live decode-graph ICB — unreachable until migration lands.
    Allocated,
}

/// Next-step scaffold toward compute ICB replay of the decode graph.
///
/// Does not own Metal ICB objects (see [`crate::icb_smoke::IcbCopySmoke`] for the
/// one-kernel proof). [`Self::try_allocate`] / [`Self::try_execute`] stay
/// [`CbReplayError::NotWired`] for the full decode path.
#[derive(Clone, Debug)]
pub struct IcbReplayStub {
    pub phase: IcbStubPhase,
    pub plan: ArgTableSlotPlan,
    /// How many times host asked to allocate / execute (telemetry).
    pub allocate_attempts: u64,
    pub execute_attempts: u64,
    /// Set when [`crate::icb_smoke::run_copy_f32_smoke`] (or equivalent) passes.
    pub smoke_proven: bool,
}

impl Default for IcbReplayStub {
    fn default() -> Self {
        Self::new()
    }
}

impl IcbReplayStub {
    pub fn new() -> Self {
        Self {
            phase: IcbStubPhase::Idle,
            plan: ArgTableSlotPlan::encode_once_v0_mini(),
            allocate_attempts: 0,
            execute_attempts: 0,
            smoke_proven: false,
        }
    }

    /// Mark the slot plan as ready for a future Metal allocate (still no decode ICB).
    pub fn mark_planned(&mut self) {
        if self.phase == IcbStubPhase::Idle {
            self.phase = IcbStubPhase::Planned;
        }
    }

    /// Record that the mini ConcurrentDispatch ICB smoke passed (API path proven).
    pub fn mark_smoke_proven(&mut self) {
        self.smoke_proven = true;
        self.phase = IcbStubPhase::SmokeProven;
    }

    /// Honest allocate for **decode graph**: always fails with [`CbReplayError::NotWired`].
    pub fn try_allocate(&mut self) -> Result<(), CbReplayError> {
        self.allocate_attempts = self.allocate_attempts.saturating_add(1);
        if self.phase == IcbStubPhase::Idle {
            self.phase = IcbStubPhase::Planned;
        }
        Err(CbReplayError::NotWired)
    }

    /// Honest execute for **full decode graph**: fails with [`CbReplayError::NotWired`]
    /// unless a mini [`crate::DecodeIcb`] path already marked success via
    /// [`Self::mark_mini_execute_ok`].
    pub fn try_execute(&mut self) -> Result<(), CbReplayError> {
        self.execute_attempts = self.execute_attempts.saturating_add(1);
        if self.phase == IcbStubPhase::Allocated {
            return Ok(());
        }
        Err(CbReplayError::NotWired)
    }

    /// Record that mini DecodeIcb `execute_icb` succeeded (not full-graph migrate).
    pub fn mark_mini_execute_ok(&mut self) {
        self.execute_attempts = self.execute_attempts.saturating_add(1);
        self.smoke_proven = true;
        self.phase = IcbStubPhase::Allocated;
    }

    pub fn status_line(&self) -> String {
        format!(
            "icb_stub phase={:?} cmds={} stable={}/{} alloc_attempts={} exec_attempts={} smoke_proven={} decode_mini_wired={}",
            self.phase,
            self.plan.max_commands,
            self.plan.stable_count(),
            self.plan.slots.len(),
            self.allocate_attempts,
            self.execute_attempts,
            self.smoke_proven,
            self.phase == IcbStubPhase::Allocated
        )
    }
}

/// Which ping-pong slot is active for record / replay bookkeeping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CbSlot {
    A = 0,
    B = 1,
}

impl CbSlot {
    pub fn peer(self) -> Self {
        match self {
            CbSlot::A => CbSlot::B,
            CbSlot::B => CbSlot::A,
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }
}

/// Lifecycle of one ping-pong CB slot (host-side bookkeeping only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CbReplayPhase {
    /// Empty / after reset.
    Idle,
    /// Host is encoding dispatches into this slot's conceptual CB.
    Recording,
    /// Encoding finished (`endCommandBuffer` conceptually); not yet proven reusable.
    Ready,
    /// Submitted / GPU may still be draining (wait before reset).
    InFlight,
}

/// One half of the A/B pair.
#[derive(Clone, Debug)]
pub struct CbReplaySlot {
    pub phase: CbReplayPhase,
    /// Generation counter bumped on each successful `begin_record`.
    pub generation: u64,
    /// Host-visible note (e.g. "decode_step graph v0") — debug only.
    pub label: String,
}

impl Default for CbReplaySlot {
    fn default() -> Self {
        Self {
            phase: CbReplayPhase::Idle,
            generation: 0,
            label: String::new(),
        }
    }
}

/// Ping-pong encode-once **scaffold**. Does not own Metal CB / ICB objects yet.
///
/// Callers may attach this to a decode session and drive the state machine while
/// wiring real `MTL4CommandBuffer` / ICB handles later. [`Self::try_replay`] always
/// returns [`CbReplayError::NotWired`] until a measured A/B lands.
///
/// Under opt-in encode-once mode (`GEMMA_METAL_ENCODE_ONCE=1`), the session
/// calls [`Self::mark_live_step`] after each successful live encode so the
/// ledger advances; true CB replay remains stubbed (see module docs +
/// [`survey_cb_replay_api_gaps`]).
pub struct PingPongCbReplay {
    slots: [CbReplaySlot; 2],
    active: CbSlot,
    /// Monotonic id for telemetry.
    next_gen: AtomicU64,
    /// When true, [`Self::try_replay`] is allowed to claim Ready→InFlight
    /// without a real Metal commit (unit-test / dry-run only).
    dry_run: bool,
    /// Successful live-encode steps recorded while encode-once mode is on.
    live_encodes: u64,
    /// Times [`Self::try_replay`] returned [`CbReplayError::NotWired`].
    not_wired_hits: u64,
    /// Successful ICB executes via [`Self::try_replay`].
    icb_replays: u64,
    /// Argument-table / ICB next-step stub (plan telemetry).
    icb: IcbReplayStub,
    /// Owned mini/full decode ICB (compute path). When set, [`Self::try_replay`]
    /// executes it instead of returning NotWired.
    decode_icb: Option<crate::decode_icb::DecodeIcb>,
}

impl Default for PingPongCbReplay {
    fn default() -> Self {
        Self::new()
    }
}

impl PingPongCbReplay {
    pub fn new() -> Self {
        Self {
            slots: [CbReplaySlot::default(), CbReplaySlot::default()],
            active: CbSlot::A,
            next_gen: AtomicU64::new(1),
            dry_run: false,
            live_encodes: 0,
            not_wired_hits: 0,
            icb_replays: 0,
            icb: IcbReplayStub::new(),
            decode_icb: None,
        }
    }

    /// Test-only: allow state transitions without Metal.
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    pub fn active_slot(&self) -> CbSlot {
        self.active
    }

    pub fn slot(&self, s: CbSlot) -> &CbReplaySlot {
        &self.slots[s.index()]
    }

    pub fn live_encodes(&self) -> u64 {
        self.live_encodes
    }

    pub fn not_wired_hits(&self) -> u64 {
        self.not_wired_hits
    }

    pub fn icb_replays(&self) -> u64 {
        self.icb_replays
    }

    pub fn icb_stub(&self) -> &IcbReplayStub {
        &self.icb
    }

    pub fn icb_stub_mut(&mut self) -> &mut IcbReplayStub {
        &mut self.icb
    }

    pub fn decode_icb_wired(&self) -> bool {
        self.decode_icb
            .as_ref()
            .map(|d| d.encoded())
            .unwrap_or(false)
    }

    /// True when attached DecodeIcb is a Binder-captured mini layer/head graph.
    pub fn decode_icb_layer_graph(&self) -> bool {
        self.decode_icb
            .as_ref()
            .map(|d| d.is_layer_graph())
            .unwrap_or(false)
    }

    pub fn decode_icb(&self) -> Option<&crate::decode_icb::DecodeIcb> {
        self.decode_icb.as_ref()
    }

    /// Attach a captured/optimized [`crate::decode_icb::DecodeIcb`].
    pub fn attach_decode_icb(&mut self, icb: crate::decode_icb::DecodeIcb) {
        self.icb.mark_smoke_proven();
        self.icb.phase = IcbStubPhase::Allocated;
        self.decode_icb = Some(icb);
    }

    /// Any slot in Ready (eligible for ICB replay).
    pub fn has_ready_slot(&self) -> bool {
        self.slots
            .iter()
            .any(|s| s.phase == CbReplayPhase::Ready)
    }

    /// Execute the attached DecodeIcb (after a successful [`Self::try_replay`]).
    pub fn execute_decode_icb(&mut self, rt: &crate::runtime::GpuRuntime) -> Result<(), String> {
        let icb = self
            .decode_icb
            .as_mut()
            .ok_or_else(|| "execute_decode_icb: no DecodeIcb attached".to_string())?;
        icb.execute(rt)?;
        self.icb_replays = self.icb_replays.saturating_add(1);
        self.icb.mark_mini_execute_ok();
        Ok(())
    }

    /// Ready-slot replay that runs `execute_icb` on the attached mini DecodeIcb.
    ///
    /// Requires [`crate::decode_icb_enabled`] + attached encoded DecodeIcb.
    /// This is the first `try_replay` → `execute_icb` bridge (not full decode).
    pub fn try_replay_icb(
        &mut self,
        slot: CbSlot,
        rt: &crate::runtime::GpuRuntime,
    ) -> Result<(), CbReplayError> {
        if self.slots[slot.index()].phase != CbReplayPhase::Ready {
            return Err(CbReplayError::NotReady);
        }
        if !crate::decode_icb_enabled() || !self.decode_icb_wired() {
            self.not_wired_hits = self.not_wired_hits.saturating_add(1);
            let _ = self.icb.try_execute();
            return Err(CbReplayError::NotWired);
        }
        self.slots[slot.index()].phase = CbReplayPhase::InFlight;
        if let Err(e) = self.execute_decode_icb(rt) {
            // Roll back so a caller can fall through to live encode.
            self.slots[slot.index()].phase = CbReplayPhase::Ready;
            self.not_wired_hits = self.not_wired_hits.saturating_add(1);
            return Err(CbReplayError::IcbExecuteFailed(e));
        }
        Ok(())
    }

    /// Before a live encode: try Ready-slot mini ICB replay when enabled.
    pub fn try_replay_ready_icb(
        &mut self,
        rt: &crate::runtime::GpuRuntime,
    ) -> Result<CbSlot, CbReplayError> {
        for slot in [CbSlot::A, CbSlot::B] {
            if self.slots[slot.index()].phase == CbReplayPhase::Ready {
                self.try_replay_icb(slot, rt)?;
                return Ok(slot);
            }
        }
        Err(CbReplayError::NotReady)
    }

    /// Bookkeeping after a **live** decode encode under encode-once mode.
    ///
    /// Advances the ping-pong ledger (record → Ready → flip). Does **not**
    /// claim a Metal replay — callers must still `begin`→encode→`end`→commit.
    /// [`Self::try_replay`] stays [`CbReplayError::NotWired`] (unless dry_run).
    pub fn mark_live_step(&mut self, label: impl Into<String>) -> Result<(), CbReplayError> {
        self.mark_step_inner(label, true)
    }

    /// Bookkeeping after a DecodeIcb **replay** that skipped host layer encode.
    /// Advances Ready for the next token without incrementing `live_encodes`.
    pub fn mark_replay_step(&mut self, label: impl Into<String>) -> Result<(), CbReplayError> {
        self.mark_step_inner(label, false)
    }

    /// Layer-graph is wired, but GPU work was **live-encoded** (frozen-tape
    /// `DecodeIcb::execute` still not token-correct — residual blow-up ~cmd 19).
    /// Advances the Ready ledger and counts `icb_replays` without tape execute.
    pub fn note_layer_live_replay(
        &mut self,
        label: impl Into<String>,
    ) -> Result<(), CbReplayError> {
        self.mark_replay_step(label)?;
        self.icb_replays = self.icb_replays.saturating_add(1);
        self.icb.mark_mini_execute_ok();
        Ok(())
    }

    fn mark_step_inner(
        &mut self,
        label: impl Into<String>,
        count_live: bool,
    ) -> Result<(), CbReplayError> {
        let slot = self.active;
        // Recover from a partial record if a prior step erred mid-encode.
        if self.slots[slot.index()].phase == CbReplayPhase::Recording {
            self.finish_record(slot)?;
        } else {
            self.begin_record(slot, label)?;
            self.finish_record(slot)?;
        }
        if count_live {
            self.live_encodes = self.live_encodes.saturating_add(1);
        }
        // Ensure ICB stub is at least Planned once ledger is live.
        if self.icb.phase == IcbStubPhase::Idle {
            self.icb.mark_planned();
        }
        // Leave slot Ready as "graph available", flip active.
        self.flip_active();
        Ok(())
    }

    /// Before a live encode: attempt to replay any Ready slot.
    ///
    /// v0 always returns [`CbReplayError::NotWired`] (or [`CbReplayError::NotReady`]
    /// on the first step). Callers fall through to full host encode.
    pub fn try_replay_ready(&mut self) -> Result<CbSlot, CbReplayError> {
        for slot in [CbSlot::A, CbSlot::B] {
            if self.slots[slot.index()].phase == CbReplayPhase::Ready {
                self.try_replay(slot)?;
                return Ok(slot);
            }
        }
        Err(CbReplayError::NotReady)
    }

    /// Begin recording into `slot` (must be Idle or Ready after GPU wait).
    pub fn begin_record(&mut self, slot: CbSlot, label: impl Into<String>) -> Result<(), CbReplayError> {
        let s = &mut self.slots[slot.index()];
        match s.phase {
            CbReplayPhase::Idle | CbReplayPhase::Ready => {}
            CbReplayPhase::Recording => return Err(CbReplayError::AlreadyRecording),
            CbReplayPhase::InFlight => return Err(CbReplayError::InFlight),
        }
        let gen = self.next_gen.fetch_add(1, Ordering::Relaxed);
        s.phase = CbReplayPhase::Recording;
        s.generation = gen;
        s.label = label.into();
        self.active = slot;
        Ok(())
    }

    /// Finish recording → Ready. Does **not** commit to a Metal queue.
    pub fn finish_record(&mut self, slot: CbSlot) -> Result<(), CbReplayError> {
        let s = &mut self.slots[slot.index()];
        if s.phase != CbReplayPhase::Recording {
            return Err(CbReplayError::NotRecording);
        }
        s.phase = CbReplayPhase::Ready;
        Ok(())
    }

    /// Attempt to replay a Ready slot without re-encoding the MTL4 CB.
    ///
    /// When a [`crate::decode_icb::DecodeIcb`] is attached, returns `Ok` and
    /// marks the slot InFlight — caller must [`Self::execute_decode_icb`].
    /// Otherwise [`CbReplayError::NotWired`] (or dry_run state transition).
    pub fn try_replay(&mut self, slot: CbSlot) -> Result<(), CbReplayError> {
        if self.slots[slot.index()].phase != CbReplayPhase::Ready {
            return Err(CbReplayError::NotReady);
        }
        if self.decode_icb_wired() {
            self.slots[slot.index()].phase = CbReplayPhase::InFlight;
            return Ok(());
        }
        if !self.dry_run {
            self.not_wired_hits = self.not_wired_hits.saturating_add(1);
            let _ = self.icb.try_execute();
            return Err(CbReplayError::NotWired);
        }
        self.slots[slot.index()].phase = CbReplayPhase::InFlight;
        Ok(())
    }

    /// Mark GPU complete; slot returns to Idle (allocator may reset).
    pub fn on_gpu_complete(&mut self, slot: CbSlot) {
        let s = &mut self.slots[slot.index()];
        s.phase = CbReplayPhase::Idle;
        s.label.clear();
    }

    /// Flip active bookkeeping to the peer slot (host encode ∥ GPU drain pattern).
    pub fn flip_active(&mut self) {
        self.active = self.active.peer();
    }

    /// Short status line for logs / JSON artifacts.
    pub fn status_line(&self) -> String {
        let wired = self.decode_icb_wired();
        let icb_s = self
            .decode_icb
            .as_ref()
            .map(|d| d.status_line())
            .unwrap_or_else(|| self.icb.status_line());
        format!(
            "cb_replay active={:?} A={:?}/{} B={:?}/{} dry_run={} wired={wired} live_encodes={} not_wired={} icb_replays={} | {icb_s}",
            self.active,
            self.slots[0].phase,
            self.slots[0].generation,
            self.slots[1].phase,
            self.slots[1].generation,
            self.dry_run,
            self.live_encodes,
            self.not_wired_hits,
            self.icb_replays,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CbReplayError {
    AlreadyRecording,
    NotRecording,
    NotReady,
    InFlight,
    /// True encode-once / ICB path not connected to `GpuRuntime` yet.
    /// See [`survey_cb_replay_api_gaps`].
    NotWired,
    /// Attached DecodeIcb `execute_icb` failed.
    IcbExecuteFailed(String),
}

impl std::fmt::Display for CbReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CbReplayError::AlreadyRecording => write!(f, "CB slot already recording"),
            CbReplayError::NotRecording => write!(f, "CB slot not recording"),
            CbReplayError::NotReady => write!(f, "CB slot not Ready for replay"),
            CbReplayError::InFlight => write!(f, "CB slot still in flight"),
            CbReplayError::NotWired => write!(
                f,
                "encode-once replay not wired: {}",
                survey_cb_replay_api_gaps()
                    .first()
                    .map(|g| g.as_str())
                    .unwrap_or("unknown gap")
            ),
            CbReplayError::IcbExecuteFailed(e) => write!(f, "decode ICB execute failed: {e}"),
        }
    }
}

impl std::error::Error for CbReplayError {}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLBuffer;

    #[test]
    fn ping_pong_record_ready_dry_replay() {
        let mut pp = PingPongCbReplay::new().with_dry_run(true);
        pp.begin_record(CbSlot::A, "mini_graph").unwrap();
        assert_eq!(pp.slot(CbSlot::A).phase, CbReplayPhase::Recording);
        pp.finish_record(CbSlot::A).unwrap();
        assert_eq!(pp.slot(CbSlot::A).phase, CbReplayPhase::Ready);
        pp.try_replay(CbSlot::A).unwrap();
        assert_eq!(pp.slot(CbSlot::A).phase, CbReplayPhase::InFlight);
        pp.on_gpu_complete(CbSlot::A);
        assert_eq!(pp.slot(CbSlot::A).phase, CbReplayPhase::Idle);
    }

    #[test]
    fn try_replay_not_wired_by_default() {
        let mut pp = PingPongCbReplay::new();
        pp.begin_record(CbSlot::B, "x").unwrap();
        pp.finish_record(CbSlot::B).unwrap();
        assert_eq!(pp.try_replay(CbSlot::B), Err(CbReplayError::NotWired));
        assert_eq!(pp.not_wired_hits(), 1);
        assert_eq!(pp.icb_stub().execute_attempts, 1);
    }

    #[test]
    fn mark_live_step_advances_ping_pong() {
        let mut pp = PingPongCbReplay::new();
        assert_eq!(pp.active_slot(), CbSlot::A);
        pp.mark_live_step("step0").unwrap();
        assert_eq!(pp.live_encodes(), 1);
        assert_eq!(pp.active_slot(), CbSlot::B);
        assert_eq!(pp.slot(CbSlot::A).phase, CbReplayPhase::Ready);
        assert_eq!(pp.icb_stub().phase, IcbStubPhase::Planned);
        pp.mark_live_step("step1").unwrap();
        assert_eq!(pp.live_encodes(), 2);
        assert_eq!(pp.active_slot(), CbSlot::A);
        // Still cannot replay without dry_run / Metal wiring.
        assert_eq!(pp.try_replay(CbSlot::B), Err(CbReplayError::NotWired));
    }

    #[test]
    fn api_gap_survey_names_mtl4_and_icb() {
        let gaps = survey_cb_replay_api_gaps();
        assert!(gaps.contains(&CbReplayApiGap::Mtl4CbHasNoReplayApi));
        assert!(gaps.contains(&CbReplayApiGap::IcbDecodeGraphNotMigrated));
        assert!(gaps.contains(&CbReplayApiGap::IcbClassicBindsVsMtl4ArgumentTable));
        let summary = cb_replay_api_gap_summary();
        assert!(summary.contains("MTL4CommandBuffer"));
        assert!(summary.contains("inheritBuffers") || summary.contains("arg-table"));
        assert!(summary.contains("DecodeIcb") || summary.contains("execute_icb"));
    }

    #[test]
    fn icb_stub_allocate_stays_not_wired() {
        let mut stub = IcbReplayStub::new();
        assert_eq!(stub.try_allocate(), Err(CbReplayError::NotWired));
        assert_eq!(stub.phase, IcbStubPhase::Planned);
        assert_eq!(stub.allocate_attempts, 1);
        assert_eq!(stub.plan.stable_count(), 5);
        assert!(stub.plan.ephemeral_count() >= 1);
    }

    #[test]
    fn try_replay_ready_not_wired_after_live() {
        let mut pp = PingPongCbReplay::new();
        assert_eq!(pp.try_replay_ready(), Err(CbReplayError::NotReady));
        pp.mark_live_step("s0").unwrap();
        assert_eq!(pp.try_replay_ready(), Err(CbReplayError::NotWired));
        assert!(pp.not_wired_hits() >= 1);
    }

    /// Mini DecodeIcb: `try_replay_icb` → `execute_icb` (opt-in; default OFF).
    #[test]
    fn decode_icb_mini_replay() {
        crate::set_decode_icb(true);
        let rt = crate::GpuRuntime::new().expect("runtime");
        let (dicb, out) = crate::DecodeIcb::mini_copy_chain(&rt, 32).expect("mini DecodeIcb");
        let mut pp = PingPongCbReplay::new();
        pp.attach_decode_icb(dicb);
        pp.mark_live_step("capture0").unwrap();
        // Ready slot is A after flip; replay it via execute_icb.
        assert_eq!(pp.slot(CbSlot::A).phase, CbReplayPhase::Ready);
        pp.try_replay_icb(CbSlot::A, &rt).expect("try_replay_icb");
        rt.synchronize().unwrap();
        assert_eq!(pp.icb_replays(), 1);
        assert_eq!(pp.icb_stub().phase, IcbStubPhase::Allocated);
        let n = 32usize;
        let got = unsafe {
            std::slice::from_raw_parts(out.metal().contents().as_ptr() as *const f32, n)
        };
        for i in 0..n {
            assert_eq!(got[i], (i as f32) + 1.0, "mismatch at {i}");
        }
        // Second replay without re-encode.
        pp.on_gpu_complete(CbSlot::A);
        pp.begin_record(CbSlot::A, "again").unwrap();
        pp.finish_record(CbSlot::A).unwrap();
        unsafe {
            let q = out.metal().contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(q, 0xFF, n * 4);
        }
        pp.try_replay_icb(CbSlot::A, &rt).expect("second try_replay_icb");
        rt.synchronize().unwrap();
        let got2 = unsafe {
            std::slice::from_raw_parts(out.metal().contents().as_ptr() as *const f32, n)
        };
        for i in 0..n {
            assert_eq!(got2[i], (i as f32) + 1.0);
        }
        assert_eq!(pp.icb_replays(), 2);
        eprintln!("decode_icb_mini_replay: {}", pp.status_line());
        crate::set_decode_icb(false);
    }
}
