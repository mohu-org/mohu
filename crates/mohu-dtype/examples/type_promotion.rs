// type_promotion.rs — Type promotion, casting rules, and scalar type detection
//
// Demonstrates mohu's NumPy-compatible type promotion system: how mixed-type
// operations resolve to a common output type, and how casting modes control
// which conversions are permitted.
//
// NumPy equivalents:
//   np.result_type(np.int32, np.float32)   # => float64
//   np.can_cast(np.float64, np.float32, casting='safe')  # => False
//   np.result_type(42)                     # => int8 (minimum scalar type)

use mohu_dtype::{
    DType,
    promote::{can_cast, common_type, minimum_scalar_type, promote, result_type, weak_promote, CastMode},
    cast::{cast_scalar, cast_slice},
};

fn main() {
    // ── Type promotion (np.result_type) ────────────────────────────────────
    // NumPy: np.result_type(np.int32, np.float32) => np.float64
    println!("── Type promotion ──");
    let pairs = [
        (DType::I32,  DType::F32),   // integer + float → wider float
        (DType::F16,  DType::F32),   // float + float → wider float
        (DType::I64,  DType::F32),   // wide int + float → F64
        (DType::C64,  DType::F64),   // complex + float → wider complex
        (DType::Bool, DType::I32),   // bool + anything → anything
        (DType::U8,   DType::I8),    // mixed sign → signed, one step wider
        (DType::U32,  DType::I32),   // U32 + I32 → I64
    ];
    for (a, b) in pairs {
        let result = promote(a, b);
        println!("  promote({}, {}) = {}", a, b, result);
    }

    // Promotion is symmetric
    println!("\n  Symmetry check:");
    println!("  promote(I32, F32) = {}", promote(DType::I32, DType::F32));
    println!("  promote(F32, I32) = {}", promote(DType::F32, DType::I32));
    assert_eq!(promote(DType::I32, DType::F32), promote(DType::F32, DType::I32));

    // ── common_type (reduce over multiple dtypes) ──────────────────────────
    // NumPy: np.result_type(np.int8, np.float32, np.int16)
    println!("\n── Common type ──");
    let dt = common_type(&[DType::I8, DType::F32, DType::I16]).unwrap();
    println!("  common_type(I8, F32, I16) = {dt}"); // F32

    let dt = common_type(&[DType::U8, DType::I32, DType::F16]).unwrap();
    println!("  common_type(U8, I32, F16) = {dt}"); // F64

    // ── result_type ────────────────────────────────────────────────────────
    let rt = result_type(DType::I16, DType::F32).unwrap();
    println!("\n  result_type(I16, F32) = {rt}");

    // ── Casting modes (np.can_cast) ────────────────────────────────────────
    // NumPy: np.can_cast(np.int8, np.float32, casting='safe')
    println!("\n── Casting modes ──");

    // Safe: no information loss
    println!("  Safe casts:");
    println!("    I8  → F32:  {}", can_cast(DType::I8, DType::F32, CastMode::Safe));   // true
    println!("    F64 → F32:  {}", can_cast(DType::F64, DType::F32, CastMode::Safe));  // false
    println!("    U8  → I16:  {}", can_cast(DType::U8, DType::I16, CastMode::Safe));   // true
    println!("    U8  → I8:   {}", can_cast(DType::U8, DType::I8, CastMode::Safe));    // false

    // SameKind: within same kind, precision loss OK
    println!("  SameKind casts:");
    println!("    F64 → F32:  {}", can_cast(DType::F64, DType::F32, CastMode::SameKind)); // true
    println!("    I32 → I16:  {}", can_cast(DType::I32, DType::I16, CastMode::SameKind)); // true
    println!("    F32 → I32:  {}", can_cast(DType::F32, DType::I32, CastMode::SameKind)); // false

    // Unsafe: any cast allowed
    println!("  Unsafe casts:");
    println!("    F32 → I32:  {}", can_cast(DType::F32, DType::I32, CastMode::Unsafe)); // true
    println!("    C64 → F32:  {}", can_cast(DType::C64, DType::F32, CastMode::Unsafe)); // true

    // ── Scalar casting (cast_scalar / cast_slice) ──────────────────────────
    // NumPy: int(np.float64(3.7))  => 3  (truncation)
    println!("\n── Scalar casting ──");

    let v: i32 = cast_scalar::<f64, i32>(3.7, CastMode::Unsafe).unwrap();
    println!("  cast f64(3.7) → i32 = {v}");  // 3 (truncated)

    let v: f64 = cast_scalar::<i16, f64>(42_i16, CastMode::Safe).unwrap();
    println!("  cast i16(42)  → f64 = {v}");  // 42.0

    // Slice casting
    let src: Vec<f32> = vec![1.1, 2.5, 3.9, -4.2];
    let mut dst = vec![0i32; 4];
    cast_slice::<f32, i32>(&src, &mut dst, CastMode::Unsafe).unwrap();
    println!("  cast_slice f32 → i32: {dst:?}");  // [1, 2, 3, -4]

    // ── minimum_scalar_type (np.result_type for scalars) ───────────────────
    // NumPy: np.result_type(42) => dtype('int8')
    println!("\n── Minimum scalar type ──");
    let values = [0.0, 200.0, -1.5, 70000.0, 1e308, -130.0];
    for v in values {
        println!("  minimum_scalar_type({v:>10}) = {}", minimum_scalar_type(v));
    }

    // ── Weak promotion (NumPy 2.0 semantics) ──────────────────────────────
    // NumPy 2.0: np.array([1,2,3], dtype=int8) + 1  => int8 (scalar is "weak")
    println!("\n── Weak promotion (NumPy 2.0) ──");
    let array_dt = DType::I8;
    let scalar_dt = DType::I64;
    let result = weak_promote(array_dt, scalar_dt, true);
    println!("  weak_promote(I8 array, I64 weak scalar) = {result}"); // I8

    let result = weak_promote(array_dt, scalar_dt, false);
    println!("  weak_promote(I8 array, I64 strong)      = {result}"); // I64

    println!("\nDone!");
}
