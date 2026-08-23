//! Metal 4 bind + dispatch helpers.
//!
//! [`Binder`] targets the Metal 4 argument table + const arena. Call sites use
//! `set_*` / `bind_*` sugar; [`GpuRuntime::with_binder`] opens the command
//! buffer encoder.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTL4ArgumentTable, MTL4CommandEncoder, MTL4ComputeCommandEncoder, MTL4VisibilityOptions,
    MTLBuffer, MTLComputePipelineState, MTLIndirectCommandBuffer, MTLResourceID, MTLSize,
    MTLStages,
};

use crate::runtime::{mtl_size, GpuRuntime};
use crate::tensor::{GpuBuffer, Tensor};

/// Metal 4 compute binder (argument table + const arena).
pub struct Binder<'a> {
    enc: &'a ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    table: &'a ProtocolObject<dyn MTL4ArgumentTable>,
    const_staging: &'a ProtocolObject<dyn MTLBuffer>,
    const_cursor: &'a mut usize,
    /// Last pipeline set (Retained) for DecodeIcb capture.
    last_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    /// A2: latch `setArgumentTable` once per binder scope (table is persistent).
    arg_table_latched: bool,
    /// Pointer identity of the last adopted / latched argument table (skip redundant
    /// `setArgumentTable` when the same table is reused across tape cmds).
    last_arg_table_ptr: Option<usize>,
}

impl<'a> Binder<'a> {
    pub(crate) fn new(
        enc: &'a ProtocolObject<dyn MTL4ComputeCommandEncoder>,
        table: &'a ProtocolObject<dyn MTL4ArgumentTable>,
        const_staging: &'a ProtocolObject<dyn MTLBuffer>,
        const_cursor: &'a mut usize,
    ) -> Self {
        Self {
            enc,
            table,
            const_staging,
            const_cursor,
            last_pipeline: None,
            arg_table_latched: false,
            last_arg_table_ptr: None,
        }
    }

    /// Latch the persistent argument table onto the encoder (idempotent).
    ///
    /// No-op when any table is already latched (including a prebuilt table
    /// adopted via [`Self::adopt_argument_table`]) — do not overwrite.
    #[inline]
    pub fn latch_argument_table(&mut self) {
        if self.arg_table_latched {
            return;
        }
        self.enc.setArgumentTable(Some(self.table));
        self.arg_table_latched = true;
        self.last_arg_table_ptr = Some(self.table as *const _ as usize);
    }

    /// Switch the encoder to a prebuilt argument table (DecodeIcb tape path).
    ///
    /// Marks the binder as latched so [`Self::dispatch`] will not overwrite with
    /// the runtime's persistent table. Returns `true` when a Metal
    /// `setArgumentTable` call was issued; `false` when the encoder already
    /// held this same table (pointer identity — A2 v0.5.7 sticky adopt).
    #[inline]
    pub fn adopt_argument_table(
        &mut self,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
    ) -> bool {
        let ptr = table as *const _ as usize;
        if self.arg_table_latched && self.last_arg_table_ptr == Some(ptr) {
            return false;
        }
        self.enc.setArgumentTable(Some(table));
        self.arg_table_latched = true;
        self.last_arg_table_ptr = Some(ptr);
        true
    }

    /// Copy bytes into the const arena; return the GPU address.
    ///
    /// Does **not** call `setAddress` or capture — used when writing into a
    /// prebuilt per-command argument table (Immediate residual only).
    pub fn materialize_bytes(&mut self, bytes: &[u8]) -> u64 {
        let align = 16usize;
        let mut cursor = (*self.const_cursor + align - 1) & !(align - 1);
        let staging_len = self.const_staging.length() as usize;
        if cursor + bytes.len() > staging_len {
            panic!(
                "const arena exhausted: need {} at cursor {cursor} (cap {staging_len})",
                bytes.len()
            );
        }
        let addr = self.const_staging.gpuAddress() + cursor as u64;
        unsafe {
            let dst = (self.const_staging.contents().as_ptr() as *mut u8).add(cursor);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
        cursor += bytes.len().max(4);
        *self.const_cursor = cursor;
        addr
    }

    pub fn set_pipeline(&mut self, pipeline: &ProtocolObject<dyn MTLComputePipelineState>) {
        self.enc.setComputePipelineState(pipeline);
        if crate::decode_icb::decode_icb_capture_active() {
            // Retain via pipeline cache clone path: callers pass cache Retained refs.
            // We re-lookup is unavailable here — store a raw retain if possible.
            // SAFETY: pipeline is a live Objective-C object retained by the caller
            // for the duration of with_binder; we retain an extra ref for the tape.
            let retained = unsafe {
                Retained::retain(pipeline as *const _ as *mut _)
                    .expect("retain pipeline")
            };
            self.last_pipeline = Some(retained);
            if let Some(ref p) = self.last_pipeline {
                crate::decode_icb::capture_note_pipeline(p.clone());
            }
        }
    }

    pub fn bind_buf(
        &mut self,
        buf: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        index: usize,
    ) {
        let addr = buf.gpuAddress().wrapping_add(offset as u64);
        self.bind_addr(addr, index);
    }

    /// Bind a precomputed GPU address (DecodeIcb tape replay bind-tax cut).
    #[inline]
    pub fn bind_addr(&mut self, gpu_addr: u64, index: usize) {
        unsafe {
            self.table.setAddress_atIndex(gpu_addr, index);
        }
    }

    pub fn bind_tensor(&mut self, t: &Tensor, index: usize) {
        self.bind_buf(t.buffer.metal(), t.byte_offset, index);
        if crate::decode_icb::decode_icb_capture_active() {
            crate::decode_icb::capture_note_bind(index, &t.buffer, t.byte_offset);
        }
    }

    pub fn bind_gpu_buf(&mut self, b: &GpuBuffer, index: usize) {
        self.bind_buf(b.metal(), 0, index);
        if crate::decode_icb::decode_icb_capture_active() {
            crate::decode_icb::capture_note_bind(index, b, 0);
        }
    }

    /// Bind an `MTLResourceID` (e.g. [`crate::mtl_tensor::GpuTensor`]) at a buffer index.
    ///
    /// # Safety
    /// `index` must be within the argument table's buffer bind count.
    pub unsafe fn bind_resource_id(&mut self, resource_id: MTLResourceID, index: usize) {
        unsafe {
            self.table
                .setResource_atBufferIndex(resource_id, index as _);
        }
    }

    /// Bind raw bytes into the const arena; returns the GPU address written.
    pub fn bind_bytes(&mut self, bytes: &[u8], index: usize) -> u64 {
        let align = 16usize;
        let mut cursor = (*self.const_cursor + align - 1) & !(align - 1);
        let staging_len = self.const_staging.length() as usize;
        if cursor + bytes.len() > staging_len {
            panic!(
                "const arena exhausted: need {} at cursor {cursor} (cap {staging_len})",
                bytes.len()
            );
        }
        let addr = self.const_staging.gpuAddress() + cursor as u64;
        unsafe {
            let dst = (self.const_staging.contents().as_ptr() as *mut u8).add(cursor);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
            self.table.setAddress_atIndex(addr, index);
        }
        cursor += bytes.len().max(4);
        *self.const_cursor = cursor;
        if crate::decode_icb::decode_icb_capture_active() {
            crate::decode_icb::capture_note_immediate(index, bytes);
        }
        addr
    }

    pub fn bind_u32(&mut self, v: u32, index: usize) {
        self.bind_bytes(&v.to_ne_bytes(), index);
    }

    pub fn bind_f32(&mut self, v: f32, index: usize) {
        self.bind_bytes(&v.to_ne_bytes(), index);
    }

    /// Dynamic threadgroup memory (`threadgroup T *ptr [[threadgroup(index)]]`).
    pub fn set_threadgroup_memory(&mut self, index: usize, length: usize) {
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex(length as _, index as _);
        }
        if crate::decode_icb::decode_icb_capture_active() {
            crate::decode_icb::capture_note_tg_mem(index, length);
        }
    }

    /// Dispatch threadgroups. Optionally inserts a Dispatch→Dispatch Device
    /// barrier after the dispatch (default on; skip via
    /// `METAL_RUNTIME_HAZARD_BARRIERS=1`). Packed multi-dispatch ops that need
    /// RAW/WAR still call [`Self::barrier`] explicitly.
    pub fn dispatch(&mut self, threadgroups: MTLSize, threads_per_tg: MTLSize) {
        self.latch_argument_table();
        self.enc
            .dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        crate::infer_trace::on_dispatch();
        if crate::decode_icb::decode_icb_capture_active() {
            crate::decode_icb::capture_note_dispatch(threadgroups, threads_per_tg);
        }
        if !crate::ab_flags::hazard_barriers() {
            self.enc
                .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                    MTLStages::Dispatch,
                    MTLStages::Dispatch,
                    MTL4VisibilityOptions::Device,
                );
            crate::infer_trace::on_barrier();
            // Freeze always-on auto-barrier into the DecodeIcb tape.
            if crate::decode_icb::decode_icb_capture_active() {
                crate::decode_icb::capture_note_barrier();
            }
        }
    }

    /// Explicit producer→consumer barrier inside a packed encoder
    /// (Dispatch→Dispatch Device).
    pub fn barrier(&mut self) {
        self.enc
            .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            );
        crate::infer_trace::on_barrier();
        // Shipping hazard skip-auto: RAW edges land here — capture for tape replay.
        if crate::decode_icb::decode_icb_capture_active() {
            crate::decode_icb::capture_note_barrier();
        }
    }

    /// Execute a pre-encoded compute [`MTLIndirectCommandBuffer`] range.
    ///
    /// When `inherit_arg_table` is true, latches the current MTL4 argument table
    /// so `inheritBuffers=true` ICB cmds see host binds. Freeze-binds
    /// (`inheritBuffers=false` + classic `setKernelBuffer`) passes false — no
    /// `setArgumentTable` traffic.
    pub fn execute_icb(
        &mut self,
        icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
        start: u64,
        count: u64,
    ) {
        self.execute_icb_ex(icb, start, count, true);
    }

    /// Like [`Self::execute_icb`] with explicit inherit-table control.
    pub fn execute_icb_ex(
        &mut self,
        icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
        start: u64,
        count: u64,
        inherit_arg_table: bool,
    ) {
        let range = NSRange {
            location: start as _,
            length: count as _,
        };
        if inherit_arg_table {
            // Latch so ICB `inheritBuffers=true` sees MTL4 binds.
            self.latch_argument_table();
        }
        unsafe {
            self.enc
                .executeCommandsInBuffer_withRange(icb, range);
        }
        crate::infer_trace::on_dispatch();
        if !crate::ab_flags::hazard_barriers() {
            self.enc
                .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                    MTLStages::Dispatch,
                    MTLStages::Dispatch,
                    MTL4VisibilityOptions::Device,
                );
            crate::infer_trace::on_barrier();
            if crate::decode_icb::decode_icb_capture_active() {
                crate::decode_icb::capture_note_barrier();
            }
        }
    }

    /// Optimize an ICB range after CPU-side encode (recommended once before reuse).
    pub fn optimize_icb(
        &mut self,
        icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
        start: u64,
        count: u64,
    ) {
        let range = NSRange {
            location: start as _,
            length: count as _,
        };
        unsafe {
            self.enc
                .optimizeIndirectCommandBuffer_withRange(icb, range);
        }
    }
}

// --- Free helpers (call-site sugar) -----------------------------------------

pub fn set_tensor(bnd: &mut Binder<'_>, t: &Tensor, index: usize) {
    bnd.bind_tensor(t, index);
}

pub fn set_gpu_buf(bnd: &mut Binder<'_>, buf: &GpuBuffer, index: usize) {
    bnd.bind_gpu_buf(buf, index);
}

/// Bind `buf` at a byte offset (slice / slot views without host round-trip).
pub fn set_gpu_buf_offset(bnd: &mut Binder<'_>, buf: &GpuBuffer, byte_offset: usize, index: usize) {
    bnd.bind_buf(buf.metal(), byte_offset, index);
    if crate::decode_icb::decode_icb_capture_active() {
        crate::decode_icb::capture_note_bind(index, buf, byte_offset);
    }
}

pub fn set_u32(bnd: &mut Binder<'_>, v: u32, index: usize) {
    bnd.bind_u32(v, index);
}

pub fn set_f32(bnd: &mut Binder<'_>, v: f32, index: usize) {
    bnd.bind_f32(v, index);
}

/// Dispatch `n` threads with automatic threadgroup sizing.
pub fn dispatch_1d(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    n: usize,
    encode_bufs: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if n == 0 {
        return Ok(());
    }
    let width = pipeline.threadExecutionWidth() as usize;
    let tpt = width.min(n).max(1);
    let groups = (n + tpt - 1) / tpt;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        encode_bufs(bnd);
        bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

/// 2D grid of threadgroups with fixed threads-per-threadgroup (FA-2 tiles).
pub fn dispatch_2d_tg(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    groups_x: usize,
    groups_y: usize,
    threads_per_tg: usize,
    encode_bufs: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if groups_x == 0 || groups_y == 0 || threads_per_tg == 0 {
        return Ok(());
    }
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        encode_bufs(bnd);
        bnd.dispatch(
            mtl_size(groups_x, groups_y, 1),
            mtl_size(threads_per_tg, 1, 1),
        );
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binder_path_copy_via_dispatch_1d() {
        let rt = GpuRuntime::new().expect("runtime");
        let n = 32usize;
        let src = rt.alloc_buffer(n * 4).unwrap();
        let dst = rt.alloc_buffer(n * 4).unwrap();
        unsafe {
            let p = src.metal().contents().as_ptr() as *mut f32;
            for i in 0..n {
                *p.add(i) = (i as f32) * 2.0;
            }
        }
        let pipe = rt.pipeline("copy_f32").unwrap();
        dispatch_1d(&rt, &pipe, n, |bnd| {
            set_gpu_buf(bnd, &src, 0);
            set_gpu_buf(bnd, &dst, 1);
            set_u32(bnd, n as u32, 2);
        })
        .unwrap();
        rt.synchronize().unwrap();
        let out = unsafe {
            std::slice::from_raw_parts(dst.metal().contents().as_ptr() as *const f32, n)
        };
        for i in 0..n {
            assert_eq!(out[i], (i as f32) * 2.0);
        }
    }
}
