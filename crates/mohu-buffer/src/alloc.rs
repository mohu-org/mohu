/// Aligned memory allocation with mmap fallback, memory advice, prefetch,
/// page-pinning, huge-page hints, debug poison, and per-strategy live stats.
///
/// # Allocation strategy
///
/// | Range       | Strategy   | Notes                                            |
/// |-------------|------------|--------------------------------------------------|
/// | 0 bytes     | zero-size  | dangling pointer — must never be dereferenced    |
/// | 1 – 1 MiB   | heap       | `std::alloc::alloc` with SIMD alignment           |
/// | > 1 MiB     | mmap       | anonymous mmap — pages committed lazily by the OS |
///
/// # Platform features
///
/// | Feature          | Platform         | API                         |
/// |------------------|------------------|-----------------------------|
/// | `advise`         | unix             | `madvise(2)`                |
/// | `prefetch_read`  | x86_64 / aarch64 | `_mm_prefetch` / `__pld`    |
/// | `mlock`/`munlock`| unix             | `mlock(2)` / `munlock(2)`   |
/// | `try_grow`       | Linux            | `mremap(2)`                 |
/// | `poison` checks  | debug builds     | 0xDE fill + verify          |
use std::{
    alloc::{self, Layout as StdLayout},
    ptr::NonNull,
    sync::atomic::{AtomicI64, AtomicU64, Ordering},
};

use mohu_error::{MohuError, MohuResult};

// ─── Constants ────────────────────────────────────────────────────────────────

/// Minimum alignment for every mohu allocation — matches AVX-512 register width.
pub const SIMD_ALIGN: usize = 64;

/// CPU cache-line size on x86-64 and ARM Cortex-A.
pub const CACHE_LINE: usize = 64;

/// Arrays larger than this go through anonymous mmap rather than the heap.
/// 1 MiB balances the kernel's per-mmap overhead against heap fragmentation.
pub const MMAP_THRESHOLD: usize = 1 << 20; // 1 MiB

/// Byte value written to free memory in debug builds.
/// 0xDE is easy to spot in a debugger and forms 0xDEAD when adjacent.
pub const POISON_BYTE: u8 = 0xDE;

// ─── Global statistics ────────────────────────────────────────────────────────

static LIVE_BYTES:      AtomicI64 = AtomicI64::new(0);
static PEAK_BYTES:      AtomicU64 = AtomicU64::new(0);
static ALLOC_COUNT:     AtomicU64 = AtomicU64::new(0);
static FREE_COUNT:      AtomicU64 = AtomicU64::new(0);
static HEAP_LIVE_BYTES: AtomicI64 = AtomicI64::new(0);
static MMAP_LIVE_BYTES: AtomicI64 = AtomicI64::new(0);

/// Instantaneous snapshot of mohu's global allocation counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocStats {
    /// Net bytes currently held by live mohu buffers.
    pub live_bytes:      i64,
    /// Maximum `live_bytes` observed since process start.
    pub peak_bytes:      u64,
    /// Total number of successful allocations performed.
    pub alloc_count:     u64,
    /// Total number of frees performed.
    pub free_count:      u64,
    /// Bytes currently held by heap (`std::alloc`) allocations.
    pub heap_live_bytes: i64,
    /// Bytes currently held by mmap allocations.
    pub mmap_live_bytes: i64,
}

impl AllocStats {
    /// Takes an atomic snapshot of all global counters.
    pub fn snapshot() -> Self {
        Self {
            live_bytes:      LIVE_BYTES.load(Ordering::Relaxed),
            peak_bytes:      PEAK_BYTES.load(Ordering::Relaxed),
            alloc_count:     ALLOC_COUNT.load(Ordering::Relaxed),
            free_count:      FREE_COUNT.load(Ordering::Relaxed),
            heap_live_bytes: HEAP_LIVE_BYTES.load(Ordering::Relaxed),
            mmap_live_bytes: MMAP_LIVE_BYTES.load(Ordering::Relaxed),
        }
    }

    /// Returns `alloc_count - free_count`.
    pub fn live_count(self) -> i64 {
        self.alloc_count as i64 - self.free_count as i64
    }
}

fn record_alloc(bytes: usize, strategy: Strategy) {
    let new = LIVE_BYTES.fetch_add(bytes as i64, Ordering::Relaxed) + bytes as i64;
    let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
    while (new as u64) > peak {
        match PEAK_BYTES.compare_exchange_weak(
            peak, new as u64, Ordering::Relaxed, Ordering::Relaxed,
        ) {
            Ok(_)        => break,
            Err(current) => peak = current,
        }
    }
    ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    match strategy {
        Strategy::Heap  => { HEAP_LIVE_BYTES.fetch_add(bytes as i64, Ordering::Relaxed); }
        Strategy::Mmap  => { MMAP_LIVE_BYTES.fetch_add(bytes as i64, Ordering::Relaxed); }
        Strategy::ZeroSize => {}
    }
}

fn record_free(bytes: usize, strategy: Strategy) {
    LIVE_BYTES.fetch_sub(bytes as i64, Ordering::Relaxed);
    FREE_COUNT.fetch_add(1, Ordering::Relaxed);
    match strategy {
        Strategy::Heap  => { HEAP_LIVE_BYTES.fetch_sub(bytes as i64, Ordering::Relaxed); }
        Strategy::Mmap  => { MMAP_LIVE_BYTES.fetch_sub(bytes as i64, Ordering::Relaxed); }
        Strategy::ZeroSize => {}
    }
}

// ─── Allocation strategy ──────────────────────────────────────────────────────

/// The mechanism used to back a particular [`AllocHandle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Strategy {
    /// Zero-size — pointer is dangling and must not be dereferenced.
    ZeroSize,
    /// Aligned heap allocation via `std::alloc`.
    Heap,
    /// Anonymous memory-mapped region — pages committed lazily by the OS.
    Mmap,
}

// ─── Memory advice ────────────────────────────────────────────────────────────

/// Kernel hints for how a memory region will be accessed.
///
/// Passed to [`AllocHandle::advise`] on Unix systems.
/// On non-Unix platforms, `advise` is a no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MmapAdvice {
    /// No special advice (OS default behaviour).
    Normal,
    /// Sequential access expected — OS should aggressively read-ahead.
    Sequential,
    /// Random access expected — disable OS prefetching.
    Random,
    /// Pages will be accessed soon — fault them in immediately.
    WillNeed,
    /// Pages are no longer needed — OS may evict them without flushing to disk.
    DontNeed,
    /// Enable Linux transparent huge pages (2 MiB) on this range.
    ///
    /// Reduces TLB pressure for arrays > 2 MiB.  No-op on macOS.
    HugePage,
    /// Disable Linux transparent huge pages on this range.
    NoHugePage,
    /// Mark pages as lazily reclaimable (`MADV_FREE`).
    ///
    /// Pages remain accessible but the OS may silently zero them under
    /// memory pressure — useful when returning buffers to the pool.
    Free,
}

#[cfg(unix)]
impl MmapAdvice {
    pub(crate) fn to_libc(self) -> libc::c_int {
        match self {
            MmapAdvice::Normal     => libc::MADV_NORMAL,
            MmapAdvice::Sequential => libc::MADV_SEQUENTIAL,
            MmapAdvice::Random     => libc::MADV_RANDOM,
            MmapAdvice::WillNeed   => libc::MADV_WILLNEED,
            MmapAdvice::DontNeed   => libc::MADV_DONTNEED,
            #[cfg(target_os = "linux")]
            MmapAdvice::HugePage   => libc::MADV_HUGEPAGE,
            #[cfg(target_os = "linux")]
            MmapAdvice::NoHugePage => libc::MADV_NOHUGEPAGE,
            #[cfg(not(target_os = "linux"))]
            MmapAdvice::HugePage   => libc::MADV_NORMAL,
            #[cfg(not(target_os = "linux"))]
            MmapAdvice::NoHugePage => libc::MADV_NORMAL,
            MmapAdvice::Free       => {
                // MADV_FREE: Linux >= 4.5 uses 8, macOS uses 5.
                // Both libc crates define MADV_FREE if available.
                #[cfg(any(target_os = "linux", target_os = "macos"))]
                { libc::MADV_FREE }
                #[cfg(not(any(target_os = "linux", target_os = "macos")))]
                { libc::MADV_NORMAL }
            }
        }
    }
}

// ─── AllocInner ───────────────────────────────────────────────────────────────

enum AllocInner {
    ZeroSize,
    Heap { ptr: NonNull<u8>, layout: StdLayout },
    #[cfg(feature = "mmap")]
    Mmap(Box<memmap2::MmapMut>),
}

// SAFETY: AllocInner exclusively owns its bytes.
unsafe impl Send for AllocInner {}
unsafe impl Sync for AllocInner {}

// ─── AllocHandle ─────────────────────────────────────────────────────────────

/// Owns a raw, SIMD-aligned byte allocation and frees it on drop.
///
/// This is the lowest-level ownership primitive in `mohu-buffer`.
/// All higher-level buffer types ultimately hold one, wrapped inside an
/// `Arc<RawBuffer>`.
///
/// # Guarantees
///
/// - `as_ptr()` is valid for [`len`](Self::len) bytes.
/// - Alignment is at least [`SIMD_ALIGN`] bytes.
/// - `Drop` frees the memory exactly once.
pub struct AllocHandle {
    inner:   AllocInner,
    len:     usize,
    align:   usize,
    #[cfg(unix)]
    mlocked: bool,
}

impl AllocHandle {
    // ─── Constructors ─────────────────────────────────────────────────────────

    /// Allocates `len` uninitialized bytes with at least `align`-byte alignment.
    ///
    /// Effective alignment = `max(align, SIMD_ALIGN)` rounded to next power of two.
    pub fn alloc(len: usize, align: usize) -> MohuResult<Self> {
        let align = align.max(SIMD_ALIGN).next_power_of_two();
        if len == 0 {
            return Ok(Self {
                inner:   AllocInner::ZeroSize,
                len:     0,
                align,
                #[cfg(unix)]
                mlocked: false,
            });
        }

        #[cfg(feature = "mmap")]
        let inner = if len >= MMAP_THRESHOLD {
            let mmap = memmap2::MmapMut::map_anon(len)
                .map_err(|_| MohuError::alloc(len))?;
            AllocInner::Mmap(Box::new(mmap))
        } else {
            heap_alloc_inner(len, align)?
        };

        #[cfg(not(feature = "mmap"))]
        let inner = heap_alloc_inner(len, align)?;

        let handle = Self {
            inner,
            len,
            align,
            #[cfg(unix)]
            mlocked: false,
        };
        record_alloc(len, handle.strategy());
        tracing::trace!(
            bytes = len, align, strategy = ?handle.strategy(),
            "mohu-buffer: alloc"
        );
        Ok(handle)
    }

    /// Allocates `len` zeroed bytes with at least `align`-byte alignment.
    pub fn alloc_zeroed(len: usize, align: usize) -> MohuResult<Self> {
        let handle = Self::alloc(len, align)?;
        if len > 0 {
            match &handle.inner {
                AllocInner::Heap { ptr, .. } => {
                    unsafe { ptr.as_ptr().write_bytes(0, len) };
                }
                #[cfg(feature = "mmap")]
                AllocInner::Mmap(_) => { /* mmap pages arrive zero-filled from the OS */ }
                AllocInner::ZeroSize => {}
            }
        }
        Ok(handle)
    }

    // ─── Raw pointer access ───────────────────────────────────────────────────

    /// Returns a const raw pointer to the start of the allocation.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        match &self.inner {
            AllocInner::ZeroSize            => NonNull::dangling().as_ptr(),
            AllocInner::Heap { ptr, .. }    => ptr.as_ptr(),
            #[cfg(feature = "mmap")]
            AllocInner::Mmap(mmap)          => mmap.as_ptr(),
        }
    }

    /// Returns a mutable raw pointer to the start of the allocation.
    ///
    /// # Safety
    ///
    /// The caller must ensure no other live reference into this allocation exists.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match &mut self.inner {
            AllocInner::ZeroSize            => NonNull::dangling().as_ptr(),
            AllocInner::Heap { ptr, .. }    => ptr.as_ptr(),
            #[cfg(feature = "mmap")]
            AllocInner::Mmap(mmap)          => mmap.as_mut_ptr(),
        }
    }

    // ─── Metadata ─────────────────────────────────────────────────────────────

    /// Byte length of this allocation (0 for zero-size handles).
    #[inline] pub fn len(&self) -> usize { self.len }
    /// Returns `true` if this is a zero-size allocation.
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }
    /// Minimum alignment of the allocation in bytes.
    #[inline] pub fn align(&self) -> usize { self.align }

    /// Returns the allocation strategy used for this handle.
    #[inline]
    pub fn strategy(&self) -> Strategy {
        match &self.inner {
            AllocInner::ZeroSize      => Strategy::ZeroSize,
            AllocInner::Heap { .. }   => Strategy::Heap,
            #[cfg(feature = "mmap")]
            AllocInner::Mmap(_)       => Strategy::Mmap,
        }
    }

    /// Returns `true` if the allocation start address is aligned to `align` bytes.
    #[inline]
    pub fn is_aligned_to(&self, align: usize) -> bool {
        (self.as_ptr() as usize) % align == 0
    }

    /// Returns the start pointer as a `NonNull<u8>`, or an error for zero-size.
    pub fn as_non_null(&self) -> MohuResult<NonNull<u8>> {
        if self.len == 0 {
            return Err(MohuError::bug("as_non_null called on a zero-size AllocHandle"));
        }
        Ok(unsafe { NonNull::new_unchecked(self.as_ptr() as *mut u8) })
    }

    // ─── Memory advice ────────────────────────────────────────────────────────

    /// Advises the kernel about the expected access pattern for this region.
    ///
    /// This is a performance hint — the kernel is free to ignore it.
    /// On non-Unix platforms, this is a no-op.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Before a sequential scan over a large array:
    /// handle.advise(MmapAdvice::Sequential).ok();
    /// ```
    pub fn advise(&self, advice: MmapAdvice) -> MohuResult<()> {
        if self.len == 0 {
            return Ok(());
        }

        #[cfg(unix)]
        {
            let ret = unsafe {
                libc::madvise(
                    self.as_ptr() as *mut libc::c_void,
                    self.len,
                    advice.to_libc(),
                )
            };
            if ret != 0 {
                return Err(MohuError::bug(format!(
                    "madvise({advice:?}) failed: errno={}",
                    std::io::Error::last_os_error()
                )));
            }
        }

        #[cfg(not(unix))]
        let _ = advice;

        Ok(())
    }

    // ─── Software prefetch ────────────────────────────────────────────────────

    /// Issues software prefetch instructions for the entire allocation.
    ///
    /// Prefetches one cache line at a time into L1/L2.  This is a
    /// non-blocking hint — the CPU may ignore it under load.
    ///
    /// # When to use
    ///
    /// Call before a tight loop over a buffer that is unlikely to already
    /// be in cache (e.g., after acquiring a cold buffer from the pool).
    #[inline]
    pub fn prefetch_read(&self) {
        if self.len == 0 { return; }
        let ptr = self.as_ptr();
        let len = self.len;

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let mut offset = 0usize;
            while offset < len {
                unsafe { _mm_prefetch(ptr.add(offset) as *const i8, _MM_HINT_T0); }
                offset += CACHE_LINE;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // AArch64: use data prefetch for load
            let mut offset = 0usize;
            while offset < len {
                unsafe {
                    let p = ptr.add(offset);
                    // Inline asm: PRFM PLDL1KEEP, [xN]
                    std::arch::asm!(
                        "prfm pldl1keep, [{p}]",
                        p = in(reg) p,
                        options(nostack, readonly)
                    );
                }
                offset += CACHE_LINE;
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let (_, _) = (ptr, len);
    }

    /// Issues write-intent prefetch instructions for the entire allocation.
    ///
    /// On x86_64, uses `PREFETCHW` to signal exclusive ownership intent.
    /// Reduces RFO (read-for-ownership) stalls in write-heavy loops.
    #[inline]
    pub fn prefetch_write(&self) {
        if self.len == 0 { return; }
        let ptr = self.as_ptr();
        let len = self.len;

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_ET0};
            let mut offset = 0usize;
            while offset < len {
                unsafe { _mm_prefetch(ptr.add(offset) as *const i8, _MM_HINT_ET0); }
                offset += CACHE_LINE;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let mut offset = 0usize;
            while offset < len {
                unsafe {
                    let p = ptr.add(offset);
                    std::arch::asm!(
                        "prfm pstl1keep, [{p}]",
                        p = in(reg) p,
                        options(nostack)
                    );
                }
                offset += CACHE_LINE;
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let (_, _) = (ptr, len);
    }

    // ─── Page pinning ─────────────────────────────────────────────────────────

    /// Pins the allocation's pages in RAM — prevents them from being swapped.
    ///
    /// Use for latency-critical buffers that must not incur page-fault overhead
    /// during a computation.  Call [`munlock`](Self::munlock) when done.
    ///
    /// Requires `CAP_IPC_LOCK` or sufficient `RLIMIT_MEMLOCK` on Linux.
    /// On non-Unix platforms, this is a no-op returning `Ok(())`.
    pub fn mlock(&mut self) -> MohuResult<()> {
        #[cfg(unix)]
        {
            if self.len == 0 || self.mlocked { return Ok(()); }
            let ret = unsafe {
                libc::mlock(self.as_ptr() as *const libc::c_void, self.len)
            };
            if ret != 0 {
                return Err(MohuError::bug(format!(
                    "mlock failed: {}",
                    std::io::Error::last_os_error()
                )));
            }
            self.mlocked = true;
        }
        Ok(())
    }

    /// Unpins the allocation's pages — re-allows swapping.
    ///
    /// Should be called before returning a buffer to the pool or dropping it.
    /// No-op if pages were never locked.
    pub fn munlock(&mut self) {
        #[cfg(unix)]
        {
            if self.len > 0 && self.mlocked {
                unsafe {
                    libc::munlock(self.as_ptr() as *const libc::c_void, self.len);
                }
                self.mlocked = false;
            }
        }
    }

    /// Returns `true` if this allocation's pages are currently locked in RAM.
    #[inline]
    pub fn is_mlocked(&self) -> bool {
        #[cfg(unix)] { self.mlocked }
        #[cfg(not(unix))] { false }
    }

    // ─── In-place grow (Linux mremap) ─────────────────────────────────────────

    /// Attempts to grow this allocation to `new_len` bytes **without copying**.
    ///
    /// Uses Linux `mremap(MREMAP_MAYMOVE)` — only available on mmap regions
    /// on Linux.  For heap allocations or non-Linux platforms, returns `false`
    /// (caller must allocate+copy manually).
    ///
    /// Returns `true` if the reallocation succeeded, `false` if not supported.
    /// On success, `self.len()` is updated to `new_len`.
    ///
    /// # Safety of returned data
    ///
    /// The old bytes are preserved; the new tail is **uninitialized**.
    pub fn try_grow(&mut self, new_len: usize) -> MohuResult<bool> {
        if new_len <= self.len {
            return Ok(true); // already large enough
        }

        // NOTE: mremap(2) on Linux could grow mmap regions in-place, but
        // memmap2::MmapMut cannot be reconstructed from a raw pointer after
        // mremap.  Falling through to Ok(false) lets the caller allocate a
        // new buffer and copy, which is always correct.
        #[cfg(target_os = "linux")]
        {
            let _ = &self.inner; // suppress unused-field warning
        }

        let _ = new_len; // suppress unused warning on non-linux
        Ok(false) // heap allocs and non-linux: caller must copy
    }

    // ─── Zero & recycle ───────────────────────────────────────────────────────

    /// Zeros the entire allocation.
    ///
    /// For mmap regions, this is a hint to the OS via `MADV_DONTNEED` (Linux)
    /// or an explicit `memset` on other platforms — the result is zeroed bytes.
    pub fn zero(&mut self) {
        if self.len == 0 { return; }

        #[cfg(target_os = "linux")]
        if matches!(self.inner, AllocInner::Mmap(_)) {
            // MADV_DONTNEED on Linux causes the pages to be re-zero-filled
            // on next access — effectively zero at no cost.
            self.advise(MmapAdvice::DontNeed).ok();
            return;
        }

        // Heap (or non-Linux mmap): explicit memset
        unsafe { self.as_mut_ptr().write_bytes(0, self.len); }
    }

    // ─── Debug poison ─────────────────────────────────────────────────────────

    /// Fills the allocation with [`POISON_BYTE`] (`0xDE`).
    ///
    /// In debug builds, call this when returning a buffer to the pool.
    /// Any use-after-free will produce obviously corrupt data rather than
    /// silently reading stale valid-looking bytes.
    ///
    /// This is a no-op in release builds (`#[cfg(debug_assertions)]`).
    #[inline]
    pub fn poison(&mut self) {
        #[cfg(debug_assertions)]
        if self.len > 0 {
            unsafe { self.as_mut_ptr().write_bytes(POISON_BYTE, self.len); }
        }
    }

    /// Returns `true` if **all** bytes in the allocation are [`POISON_BYTE`].
    ///
    /// Useful in tests to assert that a buffer was properly poisoned before
    /// being placed in the pool, confirming no aliasing occurred.
    pub fn check_poison(&self) -> bool {
        if self.len == 0 { return true; }
        let slice = unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len) };
        slice.iter().all(|&b| b == POISON_BYTE)
    }

    /// Returns `true` if any byte is the poison value.
    ///
    /// Use to detect use-after-free in debug scenarios.
    pub fn has_poison(&self) -> bool {
        if self.len == 0 { return false; }
        let slice = unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len) };
        slice.iter().any(|&b| b == POISON_BYTE)
    }

    // ─── Byte-range view ──────────────────────────────────────────────────────

    /// Returns a read-only byte slice covering the full allocation.
    ///
    /// # Safety
    ///
    /// For uninitialized or poisoned buffers, the bytes may not be valid for
    /// the element type the caller intends to interpret them as.
    pub fn as_byte_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len) }
    }
}

impl Drop for AllocHandle {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }

        // Auto-unlock before freeing — mlock'd pages must be unlocked first.
        #[cfg(unix)]
        self.munlock();

        let strategy = self.strategy();
        tracing::trace!(
            bytes = self.len, strategy = ?strategy,
            "mohu-buffer: free"
        );
        record_free(self.len, strategy);
        match &self.inner {
            AllocInner::Heap { ptr, layout } => {
                unsafe { alloc::dealloc(ptr.as_ptr(), *layout) };
            }
            #[cfg(feature = "mmap")]
            AllocInner::Mmap(_) => { /* Box<MmapMut> calls munmap on drop */ }
            AllocInner::ZeroSize => {}
        }
    }
}

impl std::fmt::Debug for AllocHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AllocHandle")
            .field("ptr",      &format_args!("{:p}", self.as_ptr()))
            .field("len",      &self.len)
            .field("align",    &self.align)
            .field("strategy", &self.strategy())
            .finish()
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

fn heap_alloc_inner(len: usize, align: usize) -> MohuResult<AllocInner> {
    let layout = StdLayout::from_size_align(len, align)
        .map_err(|_| MohuError::alloc(len))?;
    let raw = unsafe { alloc::alloc(layout) };
    let ptr = NonNull::new(raw).ok_or_else(|| MohuError::alloc(len))?;
    Ok(AllocInner::Heap { ptr, layout })
}

// ─── Non-temporal store helpers (x86_64) ─────────────────────────────────────

/// Fills `count` f32 values at `ptr` using non-temporal (streaming) stores.
///
/// Non-temporal stores bypass the CPU cache entirely — ideal for large fills
/// where we won't re-read the data soon.  This avoids cache pollution and
/// achieves peak memory bandwidth.
///
/// `ptr` must be 32-byte aligned. `count` must be a multiple of 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn fill_nontemporal_f32(ptr: *mut f32, count: usize, value: f32) {
    use std::arch::x86_64::*;
    debug_assert!(ptr as usize % 32 == 0, "ptr must be 32-byte aligned");
    debug_assert!(count % 8 == 0, "count must be multiple of 8");

    // SAFETY: caller guarantees ptr is 32-byte aligned and count is multiple of 8.
    unsafe {
        let vec = _mm256_set1_ps(value);
        let mut i = 0;
        while i + 8 <= count {
            _mm256_stream_ps(ptr.add(i), vec);
            i += 8;
        }
        _mm_sfence();
    }
}

/// Fills `count` f64 values at `ptr` using non-temporal stores (AVX2).
///
/// `ptr` must be 32-byte aligned. `count` must be a multiple of 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn fill_nontemporal_f64(ptr: *mut f64, count: usize, value: f64) {
    use std::arch::x86_64::*;
    debug_assert!(ptr as usize % 32 == 0, "ptr must be 32-byte aligned");
    debug_assert!(count % 4 == 0, "count must be multiple of 4");

    // SAFETY: caller guarantees ptr is 32-byte aligned and count is multiple of 4.
    unsafe {
        let vec = _mm256_set1_pd(value);
        let mut i = 0;
        while i + 4 <= count {
            _mm256_stream_pd(ptr.add(i), vec);
            i += 4;
        }
        _mm_sfence();
    }
}

// ─── Compile-time assertions ──────────────────────────────────────────────────

const _ASSERT_SEND_SYNC: () = {
    const fn check<T: Send + Sync>() {}
    check::<AllocHandle>();
};
