// alloc_and_pool.rs — Aligned allocation, memory pool, and allocation stats
//
// Demonstrates mohu-buffer's memory management layer: SIMD-aligned allocation,
// the two-tier buffer pool (thread-local + global), and allocation statistics.
//
// This is the kind of low-level memory control that NumPy doesn't expose
// directly, but is critical for high-performance scientific computing.

use mohu_buffer::{
    AllocHandle, AllocStats, Strategy,
    BufferPool, PoolStats, GLOBAL_POOL,
    SIMD_ALIGN, CACHE_LINE,
    Buffer,
};
use mohu_dtype::DType;

fn main() {
    println!("── mohu allocation constants ──");
    println!("SIMD_ALIGN:  {} bytes (AVX-512 register width)", SIMD_ALIGN);
    println!("CACHE_LINE:  {} bytes", CACHE_LINE);

    // ── Direct allocation with AllocHandle ─────────────────────────────────
    println!("\n── AllocHandle (raw aligned allocation) ──");
    let handle = AllocHandle::alloc(1024, SIMD_ALIGN).unwrap();
    println!("Allocated: len={}, align={}, strategy={:?}",
        handle.len(), handle.align(), handle.strategy());
    println!("  Pointer: {:p}", handle.as_ptr());
    println!("  Aligned to 64 bytes: {}", handle.is_aligned_to(64));
    assert!(matches!(handle.strategy(), Strategy::Heap));

    // Zeroed allocation
    let zeroed = AllocHandle::alloc_zeroed(512, SIMD_ALIGN).unwrap();
    println!("\nZeroed alloc: len={}, all zeros: {}",
        zeroed.len(),
        zeroed.as_byte_slice().iter().all(|&b| b == 0));

    // Zero-size allocation
    let empty = AllocHandle::alloc(0, SIMD_ALIGN).unwrap();
    println!("Zero-size: len={}, is_empty={}, strategy={:?}",
        empty.len(), empty.is_empty(), empty.strategy());

    // ── Global allocation stats ────────────────────────────────────────────
    println!("\n── Allocation stats (global) ──");
    let stats = AllocStats::snapshot();
    println!("  Live bytes:  {}", stats.live_bytes);
    println!("  Peak bytes:  {}", stats.peak_bytes);
    println!("  Alloc count: {}", stats.alloc_count);
    println!("  Free count:  {}", stats.free_count);
    println!("  Live count:  {}", stats.live_count());

    // Drop the handles and observe stats change
    let bytes_before = AllocStats::snapshot().live_bytes;
    drop(handle);
    drop(zeroed);
    let bytes_after = AllocStats::snapshot().live_bytes;
    println!("\n  After dropping 2 handles:");
    println!("  Freed: {} bytes", bytes_before - bytes_after);

    // ── Buffer pool (GLOBAL_POOL) ──────────────────────────────────────────
    println!("\n── Buffer pool (GLOBAL_POOL) ──");
    let pool = &*GLOBAL_POOL;

    // Acquire a handle from the pool (initially empty, so falls through to alloc)
    let h1 = pool.fast_acquire(4096).unwrap();
    println!("Acquired: {} bytes from pool", h1.len());

    // Release it back to the pool for reuse
    let h1_ptr = h1.as_ptr();
    pool.fast_release(h1);
    println!("Released back to pool.");

    // Acquire again — should reuse the same allocation
    let h2 = pool.fast_acquire(4096).unwrap();
    let reused = h2.as_ptr() == h1_ptr;
    println!("Re-acquired: reused={reused}"); // typically true
    pool.fast_release(h2);

    // ── Pool statistics ────────────────────────────────────────────────────
    let pstats = pool.stats();
    println!("\n── Pool stats ──");
    println!("  Cached bytes:  {}", pstats.cached_bytes);
    println!("  Cached blocks: {}", pstats.cached_blocks);
    println!("  Hit count:     {}", pstats.hit_count);
    println!("  Miss count:    {}", pstats.miss_count);
    println!("  Hit rate:      {:.1}%", pstats.hit_rate * 100.0);

    // ── Pool warm-up (pre-populating) ──────────────────────────────────────
    // Pre-warm with buffers of common sizes to avoid cold-start allocations.
    println!("\n── Pool warm-up ──");
    pool.warm(&[8192, 32768], 2).unwrap();
    let after_warm = pool.stats();
    println!("After warm-up: cached={} bytes, blocks={}",
        after_warm.cached_bytes, after_warm.cached_blocks);

    // Per-size-class breakdown
    for sc in pool.size_class_stats() {
        println!("  Class {:>8} bytes: {} handles ({} bytes cached)",
            sc.size_class, sc.cached_handles, sc.cached_bytes);
    }

    // ── Pool trim (reduce memory footprint) ────────────────────────────────
    pool.trim(0); // evict everything
    println!("\nAfter trim(0): cached={} bytes", pool.cached_bytes());

    // ── Practical example: buffer reuse in a computation loop ──────────────
    println!("\n── Buffer reuse pattern ──");
    for i in 0..3 {
        let buf = Buffer::zeros(DType::F32, &[1000]).unwrap();
        println!("  Iteration {i}: allocated {} elements, {} bytes",
            buf.len(), buf.nbytes());
        // buf is dropped here, its allocation returns to the pool
    }
    println!("  Pool after loop: {} cached bytes", pool.cached_bytes());

    println!("\nDone!");
}
