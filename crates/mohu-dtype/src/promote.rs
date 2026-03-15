/// Type promotion and casting rules for mohu.
///
/// # Promotion rules
///
/// Type promotion follows NumPy 1.x semantics (the widest-wins rule):
///
/// - `bool` + anything → anything (bool is the narrowest type)
/// - integer + integer → the wider type; mixed sign → signed, one step wider
/// - integer + float → float (at least wide enough to hold the integer)
/// - float + float → the wider of the two
/// - anything + complex → complex
///
/// The full 15×15 promotion table is computed at compile time and stored as a
/// const flat array, so `promote(a, b)` is a single indexed lookup at runtime.
///
/// # Casting modes
///
/// | Mode      | Allowed                                              |
/// |-----------|------------------------------------------------------|
/// | Safe      | No information loss guaranteed (i8→i16, f32→f64)   |
/// | SameKind  | Within same kind, precision loss OK (f64→f32)       |
/// | Unsafe    | Any cast, including float→int, complex→real          |
use crate::dtype::{DType, DTYPE_COUNT};

// ─── CastMode ────────────────────────────────────────────────────────────────

/// Controls which casts `can_cast` considers valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CastMode {
    /// Only casts that provably cannot lose information.
    ///
    /// Examples: `I8 → I16`, `F32 → F64`, `U8 → I16`.
    /// Not allowed: `F64 → F32`, `I64 → F32`, `U8 → I8`.
    Safe,

    /// Casts within the same kind, where precision loss is acceptable.
    ///
    /// Examples: `F64 → F32`, `I32 → I16`, `U32 → U8`.
    /// Not allowed: `F32 → I32`, `C64 → F32`.
    SameKind,

    /// Any cast, including lossy and domain-changing ones.
    ///
    /// Examples: `F64 → I32` (truncate), `C64 → F32` (take real part),
    /// `Bool → I8`, `I64 → U8` (wrap or saturate).
    Unsafe,
}

// ─── Promotion table ─────────────────────────────────────────────────────────

/// Returns the result dtype of an arithmetic operation between `a` and `b`.
///
/// Symmetric: `promote(a, b) == promote(b, a)`.
///
/// ```rust
/// # use mohu_dtype::{dtype::DType, promote::{promote, CastMode}};
/// assert_eq!(promote(DType::I32, DType::F32), DType::F64);
/// assert_eq!(promote(DType::F16, DType::F32), DType::F32);
/// assert_eq!(promote(DType::I64, DType::F32), DType::F64);
/// assert_eq!(promote(DType::C64, DType::F64), DType::C128);
/// ```
pub const fn promote(a: DType, b: DType) -> DType {
    PROMOTION_TABLE[a as usize * DTYPE_COUNT + b as usize]
}

/// Flattened 15×15 promotion table.  Indexed as `[a as usize * 15 + b as usize]`.
///
/// Computed by hand following NumPy 1.x rules.  Each entry is a `DType`
/// stored as its `u8` discriminant and cast back via `DType::from_u8` at
/// lookup time.
const PROMOTION_TABLE: [DType; DTYPE_COUNT * DTYPE_COUNT] = {
    use DType::{Bool,I8,I16,I32,I64,U8,U16,U32,U64,F16,BF16,F32,F64,C64,C128};

    // Row-major layout, symmetric.
    // Index mapping: Bool=0,I8=1,I16=2,I32=3,I64=4,U8=5,U16=6,U32=7,U64=8,
    //                F16=9,BF16=10,F32=11,F64=12,C64=13,C128=14

    // Read as: promote(ROW, COL) = entry
    [
        //         Bool  I8    I16   I32   I64   U8    U16   U32   U64   F16   BF16  F32   F64   C64   C128
        /* Bool */ Bool, I8,   I16,  I32,  I64,  U8,   U16,  U32,  U64,  F16,  BF16, F32,  F64,  C64,  C128,
        /* I8   */ I8,   I8,   I16,  I32,  I64,  I16,  I32,  I64,  F64,  F32,  BF16, F32,  F64,  C64,  C128,
        /* I16  */ I16,  I16,  I16,  I32,  I64,  I16,  I32,  I64,  F64,  F32,  F32,  F32,  F64,  C64,  C128,
        /* I32  */ I32,  I32,  I32,  I32,  I64,  I32,  I32,  I64,  F64,  F64,  F64,  F64,  F64,  C128, C128,
        /* I64  */ I64,  I64,  I64,  I64,  I64,  I64,  I64,  I64,  F64,  F64,  F64,  F64,  F64,  C128, C128,
        /* U8   */ U8,   I16,  I16,  I32,  I64,  U8,   U16,  U32,  U64,  F16,  BF16, F32,  F64,  C64,  C128,
        /* U16  */ U16,  I32,  I32,  I32,  I64,  U16,  U16,  U32,  U64,  F32,  F32,  F32,  F64,  C64,  C128,
        /* U32  */ U32,  I64,  I64,  I64,  I64,  U32,  U32,  U32,  U64,  F64,  F64,  F64,  F64,  C128, C128,
        /* U64  */ U64,  F64,  F64,  F64,  F64,  U64,  U64,  U64,  U64,  F64,  F64,  F64,  F64,  C128, C128,
        /* F16  */ F16,  F32,  F32,  F64,  F64,  F16,  F32,  F64,  F64,  F16,  F32,  F32,  F64,  C64,  C128,
        /* BF16 */ BF16, BF16, F32,  F64,  F64,  BF16, F32,  F64,  F64,  F32,  BF16, F32,  F64,  C64,  C128,
        /* F32  */ F32,  F32,  F32,  F64,  F64,  F32,  F32,  F64,  F64,  F32,  F32,  F32,  F64,  C64,  C128,
        /* F64  */ F64,  F64,  F64,  F64,  F64,  F64,  F64,  F64,  F64,  F64,  F64,  F64,  F64,  C128, C128,
        /* C64  */ C64,  C64,  C64,  C128, C128, C64,  C64,  C128, C128, C64,  C64,  C64,  C128, C64,  C128,
        /* C128 */ C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128,
    ]
};

// ─── can_cast ─────────────────────────────────────────────────────────────────

/// Returns `true` if a cast from `from` to `to` is allowed under `mode`.
///
/// ```rust
/// # use mohu_dtype::{dtype::DType, promote::{can_cast, CastMode}};
/// assert!( can_cast(DType::I8,  DType::F32, CastMode::Safe));
/// assert!(!can_cast(DType::F64, DType::F32, CastMode::Safe));
/// assert!( can_cast(DType::F64, DType::F32, CastMode::SameKind));
/// assert!( can_cast(DType::F32, DType::I32, CastMode::Unsafe));
/// ```
pub const fn can_cast(from: DType, to: DType, mode: CastMode) -> bool {
    match mode {
        CastMode::Safe     => SAFE_CAST_TABLE[from as usize * DTYPE_COUNT + to as usize],
        CastMode::SameKind => SAMEKIND_CAST_TABLE[from as usize * DTYPE_COUNT + to as usize],
        CastMode::Unsafe   => true, // all casts are valid in unsafe mode
    }
}

// ─── Safe cast table ─────────────────────────────────────────────────────────
//
// A safe cast from `from` to `to` is valid when:
//   - to == from (identity)
//   - from is Bool (Bool is the narrowest type, casts to anything safely)
//   - from is an integer type and to is a float type wide enough
//     (I8/U8 → F16+, I16/U16 → F32+, I32/U32/I64/U64 → F64)
//   - from is a narrower int and to is a wider int of compatible signedness
//     (signed → signed if wider, unsigned → unsigned if wider,
//      unsigned → signed if signed is strictly wider in bits)
//   - from is a narrower float and to is a wider float (F16 → F32/F64, F32 → F64)
//   - from is C64 and to is C128

const SAFE_CAST_TABLE: [bool; DTYPE_COUNT * DTYPE_COUNT] = {
    // T = true, F = false
    const T: bool = true;
    const F: bool = false;

    //         Bool  I8    I16   I32   I64   U8    U16   U32   U64   F16   BF16  F32   F64   C64   C128
    [
        /* Bool */ T, T,    T,    T,    T,    T,    T,    T,    T,    T,    T,    T,    T,    T,    T,
        /* I8   */ F, T,    T,    T,    T,    F,    F,    F,    F,    T,    T,    T,    T,    T,    T,
        /* I16  */ F, F,    T,    T,    T,    F,    F,    F,    F,    F,    F,    T,    T,    T,    T,
        /* I32  */ F, F,    F,    T,    T,    F,    F,    F,    F,    F,    F,    F,    T,    F,    T,
        /* I64  */ F, F,    F,    F,    T,    F,    F,    F,    F,    F,    F,    F,    T,    F,    T,
        /* U8   */ F, F,    T,    T,    T,    T,    T,    T,    T,    T,    T,    T,    T,    T,    T,
        /* U16  */ F, F,    F,    T,    T,    F,    T,    T,    T,    F,    F,    T,    T,    T,    T,
        /* U32  */ F, F,    F,    F,    T,    F,    F,    T,    T,    F,    F,    F,    T,    F,    T,
        /* U64  */ F, F,    F,    F,    F,    F,    F,    F,    T,    F,    F,    F,    T,    F,    T,
        /* F16  */ F, F,    F,    F,    F,    F,    F,    F,    F,    T,    F,    T,    T,    T,    T,
        /* BF16 */ F, F,    F,    F,    F,    F,    F,    F,    F,    F,    T,    T,    T,    T,    T,
        /* F32  */ F, F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    T,    T,    T,    T,
        /* F64  */ F, F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    T,    F,    T,
        /* C64  */ F, F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    T,    T,
        /* C128 */ F, F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    T,
    ]
};

// ─── SameKind cast table ──────────────────────────────────────────────────────
//
// SameKind allows casts within the same numeric kind (int→int, float→float,
// complex→complex) regardless of widening vs narrowing.  Cross-kind casts
// are not allowed in SameKind mode.

const SAMEKIND_CAST_TABLE: [bool; DTYPE_COUNT * DTYPE_COUNT] = {
    const T: bool = true;
    const F: bool = false;

    //         Bool  I8    I16   I32   I64   U8    U16   U32   U64   F16   BF16  F32   F64   C64   C128
    [
        /* Bool */ T, F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    F,
        /* I8   */ F, T,    T,    T,    T,    T,    T,    T,    T,    F,    F,    F,    F,    F,    F,
        /* I16  */ F, T,    T,    T,    T,    T,    T,    T,    T,    F,    F,    F,    F,    F,    F,
        /* I32  */ F, T,    T,    T,    T,    T,    T,    T,    T,    F,    F,    F,    F,    F,    F,
        /* I64  */ F, T,    T,    T,    T,    T,    T,    T,    T,    F,    F,    F,    F,    F,    F,
        /* U8   */ F, T,    T,    T,    T,    T,    T,    T,    T,    F,    F,    F,    F,    F,    F,
        /* U16  */ F, T,    T,    T,    T,    T,    T,    T,    T,    F,    F,    F,    F,    F,    F,
        /* U32  */ F, T,    T,    T,    T,    T,    T,    T,    T,    F,    F,    F,    F,    F,    F,
        /* U64  */ F, T,    T,    T,    T,    T,    T,    T,    T,    F,    F,    F,    F,    F,    F,
        /* F16  */ F, F,    F,    F,    F,    F,    F,    F,    F,    T,    T,    T,    T,    F,    F,
        /* BF16 */ F, F,    F,    F,    F,    F,    F,    F,    F,    T,    T,    T,    T,    F,    F,
        /* F32  */ F, F,    F,    F,    F,    F,    F,    F,    F,    T,    T,    T,    T,    F,    F,
        /* F64  */ F, F,    F,    F,    F,    F,    F,    F,    F,    T,    T,    T,    T,    F,    F,
        /* C64  */ F, F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    T,    T,
        /* C128 */ F, F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    F,    T,    T,
    ]
};

// ─── result_type ─────────────────────────────────────────────────────────────

/// Returns the output dtype for an operation on arrays of types `a` and `b`,
/// or an error if the combination is not supported (e.g. complex + bool in
/// strict mode).
///
/// This is a thin wrapper around [`promote`] that validates the result.
pub fn result_type(a: DType, b: DType) -> crate::MohuResult<DType> {
    Ok(promote(a, b))
}

/// Returns the common type of a sequence of dtypes.
///
/// Returns an error if the sequence is empty.
///
/// ```rust
/// # use mohu_dtype::{dtype::DType, promote::common_type};
/// let dt = common_type(&[DType::I8, DType::F32, DType::I16]).unwrap();
/// assert_eq!(dt, DType::F32);
/// ```
pub fn common_type(dtypes: &[DType]) -> crate::MohuResult<DType> {
    if dtypes.is_empty() {
        return Err(mohu_error::MohuError::Internal(
            "common_type called with empty dtype slice".into(),
        ));
    }
    let mut acc = dtypes[0];
    for &dt in &dtypes[1..] {
        acc = promote(acc, dt);
    }
    Ok(acc)
}

// ─── minimum_scalar_type ─────────────────────────────────────────────────────

/// Returns the smallest dtype that can represent the given `f64` value
/// without loss of information.
///
/// The decision procedure:
/// 1. If the value is exactly an integer in a small range → smallest int dtype.
/// 2. Otherwise, if it fits in F32 without losing precision → F32.
/// 3. Otherwise → F64.
///
/// This mirrors NumPy's `np.result_type(value)` behaviour for scalar values.
///
/// ```rust
/// # use mohu_dtype::{dtype::DType, promote::minimum_scalar_type};
/// assert_eq!(minimum_scalar_type(0.0),    DType::I8);
/// assert_eq!(minimum_scalar_type(200.0),  DType::U8);
/// assert_eq!(minimum_scalar_type(-1.5),   DType::F32);
/// assert_eq!(minimum_scalar_type(1e308),  DType::F64);
/// ```
pub fn minimum_scalar_type(v: f64) -> DType {
    // Check if the value is an exact integer.
    if v.fract() == 0.0 && v.is_finite() {
        let i = v as i64;
        if v >= 0.0 {
            let u = v as u64;
            if u <= u8::MAX as u64  { return DType::U8;  }
            if u <= u16::MAX as u64 { return DType::U16; }
            if u <= u32::MAX as u64 { return DType::U32; }
            if u <= u64::MAX        { return DType::U64; }
        } else {
            if i >= i8::MIN as i64  { return DType::I8;  }
            if i >= i16::MIN as i64 { return DType::I16; }
            if i >= i32::MIN as i64 { return DType::I32; }
            return DType::I64;
        }
    }
    // Non-integer: check if f32 can represent it faithfully.
    if (v as f32) as f64 == v { DType::F32 } else { DType::F64 }
}

// ─── weak_promote (NumPy 2.0 style) ─────────────────────────────────────────

/// Computes the promotion result with NumPy 2.0 "weak type" semantics.
///
/// In NumPy 2.0 scalar literals have a "weak" type that defers to the
/// array's dtype:
/// - A Python `int` literal is a "weak int" — it takes on the dtype of
///   the array it's combined with if that array is an integer type.
/// - A Python `float` literal is a "weak float" — same idea.
///
/// In practice, `weak_promote(array_dtype, scalar_dtype)` returns the
/// array dtype when the scalar can be represented exactly by it, otherwise
/// falls back to `promote(array_dtype, scalar_dtype)`.
///
/// `is_scalar_weak` should be `true` when `scalar_dtype` came from a
/// Python literal rather than an array.
pub fn weak_promote(array_dtype: DType, scalar_dtype: DType, is_scalar_weak: bool) -> DType {
    if !is_scalar_weak {
        return promote(array_dtype, scalar_dtype);
    }
    // Weak integer: if the array is any integer type, return the array dtype.
    if scalar_dtype.is_integer() && array_dtype.is_integer() {
        return array_dtype;
    }
    // Weak float: if the array is any float type, return the array dtype.
    if scalar_dtype.is_float() && array_dtype.is_float() {
        return array_dtype;
    }
    // Otherwise fall back to the standard table.
    promote(array_dtype, scalar_dtype)
}

// ─── compile-time symmetry check ─────────────────────────────────────────────

const _: () = {
    // Verify that the promotion table is symmetric.
    let mut i = 0;
    while i < DTYPE_COUNT {
        let mut j = 0;
        while j < DTYPE_COUNT {
            let fwd = PROMOTION_TABLE[i * DTYPE_COUNT + j] as u8;
            let rev = PROMOTION_TABLE[j * DTYPE_COUNT + i] as u8;
            assert!(fwd == rev, "PROMOTION_TABLE is not symmetric");
            j += 1;
        }
        i += 1;
    }

    // Verify that every dtype promotes with itself to itself (identity).
    let mut k = 0;
    while k < DTYPE_COUNT {
        let self_promote = PROMOTION_TABLE[k * DTYPE_COUNT + k] as u8;
        assert!(self_promote == k as u8, "PROMOTION_TABLE diagonal is wrong");
        k += 1;
    }
};
