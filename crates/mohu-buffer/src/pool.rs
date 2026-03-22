/// Buffer pool for allocation reuse — two-tier: thread-local (zero-lock) + global.
///
/// # Architecture
///
/// ```text
/// fast_acquire(n) ──► TL cache (zero-lock) ──hit──► return handle
///                          │ miss
///                          ▼
///                    GlobalPool (Mutex) ──hit──► return handle
///                          │ miss
///                          ▼
///                    AllocHandle::alloc (syscall/mmap)
/// ```
///
/// The thread-local (TL) cache is a per-thread `Vec<(size_class, AllocHandle)>`.
/// It has zero lock overhead — ideal for tight allocation/free loops in a
/// single thread (e.g., expression evaluation).
///
/// The global pool is a `Mutex<BTreeMap<size_class, PoolBucket>>`.  It is
/// consulted only when the TL cache misses, sharing large allocations across
/// threads.
///
/// # Size classes
///
/// Requests are rounded up to the nearest power of two.
///
/// # Thread safety
///
/// `BufferPool` is `Send + Sync`.  The TL cache is `!Send` by nature of
/// `thread_local!` — it lives and dies with its owning thread.  When a
/// thread exits, its TL cache drops all handles (freeing them), **not**
/// returning them to the global pool.  Call [`drain_thread_local`] before
/// the thread exits if you want to reclaim handles.
use std::{
    cell::RefCell,
    collections::BTreeMap,
    sync::{Mutex, MutexGuard},
};

use mohu_error::MohuResult;

use crate::alloc::{AllocHandle, MmapAdvice, SIMD_ALIGN};

// ─── Thread-local cache ───────────────────────────────────────────────────────

/// Maximum number of handles stored in the TL cache.
const TL_SLOTS: usize = 32;

/// Maximum total bytes in the TL cache per thread (4 MiB).
const TL_MAX_BYTES: usize = 4 * 1024 * 1024;

thread_local! {
    static TL_CACHE: RefCell<Vec<(usize, AllocHandle)>> =
        RefCell::new(Vec::with_capacity(TL_SLOTS));
    static TL_STATS: RefCell<TlStats> =
        const { RefCell::new(TlStats { hits: 0, misses: 0, returns: 0, cached_bytes: 0 }) };
}

/// Per-thread pool cache statistics.  Accumulated locally — no atomics needed.
#[derive(Debug, Clone, Copy, Default)]
pub struct TlStats {
    /// Number of TL cache hits.
    pub hits:         u64,
    /// Number of TL cache misses (fell through to global pool).
    pub misses:       u64,
    /// Number of handles returned to the TL cache.
    pub returns:      u64,
    /// Total bytes currently cached in this thread's TL cache.
    pub cached_bytes: usize,
}

impl TlStats {
    /// Reads the calling thread's TL-cache statistics.
    pub fn current() -> Self {
        TL_STATS.with(|s| *s.borrow())
    }
}

// ─── Size class ───────────────────────────────────────────────────────────────

#[inline]
fn size_class(n: usize) -> usize {
    if n == 0 { return 1; }
    n.next_power_of_two()
}

// ─── PoolBucket ───────────────────────────────────────────────────────────────

struct PoolBucket {
    handles:      Vec<AllocHandle>,
    cached_bytes: usize,
}

impl PoolBucket {
    fn new() -> Self {
        Self { handles: Vec::new(), cached_bytes: 0 }
    }

    fn push(&mut self, handle: AllocHandle) {
        self.cached_bytes += handle.len();
        self.handles.push(handle);
    }

    fn pop(&mut self) -> Option<AllocHandle> {
        let h = self.handles.pop()?;
        self.cached_bytes -= h.len();
        Some(h)
    }

    #[allow(dead_code)]
    fn clear(&mut self) {
        self.handles.clear();
        self.cached_bytes = 0;
    }

    fn len(&self) -> usize { self.handles.len() }
}

// ─── SizeClassStats ───────────────────────────────────────────────────────────

/// Statistics for a single size class in the global pool.
#[derive(Debug, Clone, Copy)]
pub struct SizeClassStats {
    /// Size class in bytes (always a power of two).
    pub size_class:     usize,
    /// Number of cached handles.
    pub cached_handles: usize,
    /// Total cached bytes in this class.
    pub cached_bytes:   usize,
}

// ─── BufferPool ───────────────────────────────────────────────────────────────

/// A thread-safe two-tier allocation pool.
///
/// Use [`fast_acquire`](Self::fast_acquire) / [`fast_release`](Self::fast_release)
/// for the zero-lock TL-first path.
/// Use [`acquire`](Self::acquire) / [`release`](Self::release) for the global
/// pool directly (no TL cache involvement).
pub struct BufferPool {
    inner:            Mutex<PoolInner>,
    max_cached_bytes: usize,
}

struct PoolInner {
    buckets:      BTreeMap<usize, PoolBucket>,
    cached_bytes: usize,
    hit_count:    u64,
    miss_count:   u64,
    return_count: u64,
}

impl PoolInner {
    fn new() -> Self {
        Self {
            buckets:      BTreeMap::new(),
            cached_bytes: 0,
            hit_count:    0,
            miss_count:   0,
            return_count: 0,
        }
    }
}

/// Statistics snapshot for a `BufferPool`.
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    /// Total bytes currently cached in the pool.
    pub cached_bytes:  usize,
    /// Total number of cached allocation handles.
    pub cached_blocks: usize,
    /// Number of successful acquisitions from the cache.
    pub hit_count:     u64,
    /// Number of acquisitions that required a new allocation.
    pub miss_count:    u64,
    /// Number of handles returned to the pool.
    pub return_count:  u64,
    /// Cache hit rate as a fraction in `[0.0, 1.0]`.
    pub hit_rate:      f64,
    /// Number of distinct active size classes.
    pub size_classes:  usize,
}

impl BufferPool {
    /// Creates a new `BufferPool` with `max_cached_bytes` capacity.
    pub fn new(max_cached_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(PoolInner::new()),
            max_cached_bytes,
        }
    }

    // ─── Global pool (existing API) ───────────────────────────────────────────

    /// Acquires an `AllocHandle` of at least `nbytes` bytes, consulting only
    /// the **global** pool (no TL cache involvement).
    ///
    /// Prefer [`fast_acquire`](Self::fast_acquire) in performance-critical code.
    pub fn acquire(&self, nbytes: usize) -> MohuResult<AllocHandle> {
        if nbytes == 0 {
            return AllocHandle::alloc(0, SIMD_ALIGN);
        }
        let class = size_class(nbytes);
        let mut inner = self.lock();
        if let Some(bucket) = inner.buckets.get_mut(&class) {
            if let Some(handle) = bucket.pop() {
                inner.cached_bytes -= handle.len();
                inner.hit_count += 1;
                tracing::trace!(bytes = nbytes, class, "pool: hit");
                return Ok(handle);
            }
        }
        inner.miss_count += 1;
        drop(inner); // release lock before slow allocation
        tracing::trace!(bytes = nbytes, class, "pool: miss — allocating");
        AllocHandle::alloc(class, SIMD_ALIGN)
    }

    /// Returns an `AllocHandle` to the **global** pool.
    pub fn release(&self, handle: AllocHandle) {
        if handle.is_empty() { return; }
        let class = size_class(handle.len());
        let mut inner = self.lock();
        if inner.cached_bytes + handle.len() > self.max_cached_bytes {
            tracing::trace!(bytes = handle.len(), "pool: evict (at capacity)");
            inner.return_count += 1;
            return;
        }
        let handle_len = handle.len();
        inner.cached_bytes += handle_len;
        inner.return_count += 1;
        inner.buckets
            .entry(class)
            .or_insert_with(PoolBucket::new)
            .push(handle);
        tracing::trace!(bytes = handle_len, class, "pool: returned");
    }

    // ─── Two-tier fast path ───────────────────────────────────────────────────

    /// Acquires an `AllocHandle` — TL cache first, then global pool, then syscall.
    ///
    /// Zero locking overhead when the TL cache has a matching size class.
    /// This is the **recommended** API for hot paths.
    pub fn fast_acquire(&self, nbytes: usize) -> MohuResult<AllocHandle> {
        if nbytes == 0 {
            return AllocHandle::alloc(0, SIMD_ALIGN);
        }
        let class = size_class(nbytes);

        // 1. Thread-local — no lock needed.
        if class <= TL_MAX_BYTES {
            let from_tl = TL_CACHE.with(|cache| {
                let mut c = cache.borrow_mut();
                c.iter()
                    .position(|(k, _)| *k == class)
                    .map(|pos| {
                        let (_, handle) = c.swap_remove(pos);
                        handle
                    })
            });
            if let Some(handle) = from_tl {
                TL_STATS.with(|s| {
                    let mut st = s.borrow_mut();
                    st.hits += 1;
                    st.cached_bytes = st.cached_bytes.saturating_sub(handle.len());
                });
                tracing::trace!(bytes = nbytes, class, "tl-pool: hit");
                return Ok(handle);
            }
            TL_STATS.with(|s| s.borrow_mut().misses += 1);
        }

        // 2. Global pool.
        self.acquire(nbytes)
    }

    /// Returns an `AllocHandle` — TL cache first (for small handles), then global.
    ///
    /// Optionally poisons the handle before caching (debug builds only).
    pub fn fast_release(&self, mut handle: AllocHandle) {
        if handle.is_empty() { return; }
        handle.poison(); // no-op in release builds

        let class = size_class(handle.len());

        // 1. Try TL cache (zero-lock, no contention).
        if class <= TL_MAX_BYTES {
            let accepted = TL_CACHE.with(|cache| {
                let c = cache.borrow_mut();
                // Compute current TL cached bytes
                let cur_bytes: usize = c.iter().map(|(_, h)| h.len()).sum();
                if c.len() < TL_SLOTS && cur_bytes + handle.len() <= TL_MAX_BYTES {
                    true
                } else {
                    false
                }
            });
            if accepted {
                let hlen = handle.len();
                TL_CACHE.with(|cache| {
                    cache.borrow_mut().push((class, handle));
                });
                TL_STATS.with(|s| {
                    let mut st = s.borrow_mut();
                    st.returns += 1;
                    st.cached_bytes += hlen;
                });
                return;
            }
        }

        // 2. Fall back to global pool.
        self.release(handle);
    }

    /// Flushes all handles from the calling thread's TL cache to the global pool.
    ///
    /// Call before a thread exits to reclaim TL-cached memory rather than
    /// letting it leak (be freed by the thread-local destructor).
    pub fn drain_thread_local(&self) {
        let handles: Vec<(usize, AllocHandle)> =
            TL_CACHE.with(|c| std::mem::take(&mut *c.borrow_mut()));
        TL_STATS.with(|s| {
            let mut st = s.borrow_mut();
            st.cached_bytes = 0;
        });
        for (_, handle) in handles {
            self.release(handle);
        }
    }

    // ─── Pre-warming ──────────────────────────────────────────────────────────

    /// Pre-populates the pool with `count_per_size` handles of each given size.
    ///
    /// Call during application startup before the hot path begins, so that
    /// early allocations hit the pool instead of the OS allocator.
    ///
    /// ```rust,ignore
    /// // Pre-warm with 8 handles of 64 KiB and 4 handles of 4 MiB.
    /// GLOBAL_POOL.warm(&[64 * 1024, 4 * 1024 * 1024], 4)?;
    /// ```
    pub fn warm(&self, sizes: &[usize], count_per_size: usize) -> MohuResult<()> {
        for &size in sizes {
            if size == 0 { continue; }
            let class = size_class(size);
            // Allocate outside the lock.
            let mut handles = Vec::with_capacity(count_per_size);
            for _ in 0..count_per_size {
                handles.push(AllocHandle::alloc(class, SIMD_ALIGN)?);
            }
            // Return to the pool.
            let mut inner = self.lock();
            for h in handles {
                if inner.cached_bytes + h.len() > self.max_cached_bytes {
                    break; // pool full — remaining handles dropped
                }
                let hlen = h.len();
                inner.cached_bytes += hlen;
                inner.return_count += 1;
                inner.buckets
                    .entry(class)
                    .or_insert_with(PoolBucket::new)
                    .push(h);
            }
        }
        Ok(())
    }

    // ─── Trim ─────────────────────────────────────────────────────────────────

    /// Releases cached handles until the pool holds at most `keep_bytes` bytes.
    ///
    /// Handles are evicted largest-first.  Dropped handles are freed immediately.
    /// Useful for reducing memory footprint during idle periods.
    pub fn trim(&self, keep_bytes: usize) {
        let mut inner = self.lock();
        while inner.cached_bytes > keep_bytes {
            // Find the largest non-empty bucket.
            let largest_class = inner
                .buckets
                .iter()
                .rev()
                .find(|(_, b)| b.len() > 0)
                .map(|(&c, _)| c);

            if let Some(class) = largest_class {
                if let Some(bucket) = inner.buckets.get_mut(&class) {
                    if let Some(h) = bucket.pop() {
                        inner.cached_bytes -= h.len();
                        drop(h); // frees the allocation
                    }
                }
            } else {
                break; // no more handles
            }
        }
    }

    // ─── Advise all cached handles ────────────────────────────────────────────

    /// Applies a memory advice hint to all currently cached handles.
    ///
    /// For example, call with `MmapAdvice::DontNeed` during idle time to
    /// release physical pages back to the OS while keeping the virtual
    /// address space reserved.
    pub fn advise_all(&self, advice: MmapAdvice) {
        let mut inner = self.lock();
        for bucket in inner.buckets.values_mut() {
            for handle in &bucket.handles {
                handle.advise(advice).ok();
            }
        }
    }

    // ─── Stats & introspection ────────────────────────────────────────────────

    /// Returns an instantaneous statistics snapshot.
    pub fn stats(&self) -> PoolStats {
        let inner = self.lock();
        let total_calls = inner.hit_count + inner.miss_count;
        PoolStats {
            cached_bytes:  inner.cached_bytes,
            cached_blocks: inner.buckets.values().map(|b| b.len()).sum(),
            hit_count:     inner.hit_count,
            miss_count:    inner.miss_count,
            return_count:  inner.return_count,
            hit_rate: if total_calls == 0 {
                0.0
            } else {
                inner.hit_count as f64 / total_calls as f64
            },
            size_classes: inner.buckets.len(),
        }
    }

    /// Returns per-size-class breakdown of cached blocks.
    pub fn size_class_stats(&self) -> Vec<SizeClassStats> {
        let inner = self.lock();
        inner.buckets.iter().map(|(&class, b)| SizeClassStats {
            size_class:     class,
            cached_handles: b.len(),
            cached_bytes:   b.cached_bytes,
        }).collect()
    }

    /// Current number of cached bytes (global pool only).
    pub fn cached_bytes(&self) -> usize {
        self.lock().cached_bytes
    }

    /// Maximum allowed cached bytes.
    pub fn max_cached_bytes(&self) -> usize {
        self.max_cached_bytes
    }

    /// Drops all cached handles, freeing their memory immediately.
    pub fn clear(&self) {
        let mut inner = self.lock();
        inner.buckets.clear();
        inner.cached_bytes = 0;
    }

    fn lock(&self) -> MutexGuard<'_, PoolInner> {
        self.inner.lock().expect("BufferPool mutex poisoned")
    }
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("BufferPool")
            .field("cached_bytes",  &stats.cached_bytes)
            .field("cached_blocks", &stats.cached_blocks)
            .field("size_classes",  &stats.size_classes)
            .field("max_bytes",     &self.max_cached_bytes)
            .field("hit_rate",      &format_args!("{:.1}%", stats.hit_rate * 100.0))
            .finish()
    }
}

// ─── Global pool ─────────────────────────────────────────────────────────────

/// Process-wide shared two-tier buffer pool with a 256 MiB global cache cap.
///
/// # Fast path
///
/// ```rust,ignore
/// let handle = GLOBAL_POOL.fast_acquire(nbytes)?;
/// // ... use handle ...
/// GLOBAL_POOL.fast_release(handle);
/// ```
pub static GLOBAL_POOL: std::sync::LazyLock<BufferPool> =
    std::sync::LazyLock::new(|| BufferPool::new(256 * 1024 * 1024));
