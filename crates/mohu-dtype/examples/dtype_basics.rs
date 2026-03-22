// dtype_basics.rs — DType construction, classification, and type metadata
//
// Demonstrates the core DType enum: how to create dtypes, inspect their
// properties, parse from NumPy-compatible strings, and query machine info
// via FloatInfo and IntInfo.
//
// NumPy equivalent:
//   dt = np.dtype('float32')
//   np.finfo(np.float32)
//   np.iinfo(np.int32)

use mohu_dtype::{
    DType, FloatInfo, IntInfo, ALL_DTYPES,
};

fn main() {
    // ── DType construction and display ─────────────────────────────────────
    // NumPy: np.dtype('float32')
    let f32_dt = DType::F32;
    println!("DType: {f32_dt}");                         // "float32"
    println!("  itemsize:  {} bytes", f32_dt.itemsize()); // 4
    println!("  alignment: {} bytes", f32_dt.alignment());// 4
    println!("  bit_width: {} bits",  f32_dt.bit_width()); // 32

    // ── Parsing from NumPy-compatible strings ──────────────────────────────
    // NumPy: np.dtype('int64'), np.dtype('f4'), np.dtype('complex128')
    let dt1 = DType::from_str("int64").unwrap();
    let dt2 = DType::from_str("f4").unwrap();        // shorthand for float32
    let dt3 = DType::from_str("complex128").unwrap();
    println!("\nParsed dtypes: {dt1}, {dt2}, {dt3}");

    // ── Classification predicates ──────────────────────────────────────────
    // NumPy: np.issubdtype(np.float32, np.floating)
    println!("\n── Classification ──");
    for &dt in &[DType::Bool, DType::I32, DType::U8, DType::F64, DType::C128, DType::BF16] {
        println!("{:>12}: integer={}, float={}, complex={}, numeric={}, ordered={}",
            dt.numpy_str(),
            dt.is_integer(),
            dt.is_float(),
            dt.is_complex(),
            dt.is_numeric(),
            dt.is_ordered(),
        );
    }

    // ── Type conversion helpers ────────────────────────────────────────────
    println!("\n── Type conversions ──");
    println!("F32.complex_dtype() = {}", DType::F32.complex_dtype()); // C64
    println!("C128.real_dtype()   = {}", DType::C128.real_dtype());   // F64
    println!("U16.as_signed()     = {}", DType::U16.as_signed());     // I16
    println!("I32.widen()         = {}", DType::I32.widen());          // I64
    println!("I64.to_float()      = {}", DType::I64.to_float());      // F64

    // ── NumPy string representations ───────────────────────────────────────
    // NumPy: np.dtype('float32').str  => '<f4'
    println!("\n── NumPy compat strings ──");
    let dt = DType::F32;
    println!("numpy_str:    {}", dt.numpy_str());            // "float32"
    println!("numpy_char:   {:?}", dt.numpy_char());          // Some('f')
    println!("kind_char:    {}", dt.kind_char());             // 'f'
    println!("typestr:      {}", dt.array_interface_typestr()); // "<f4"
    println!("buffer_format: {}", dt.buffer_format());         // "<f"

    // ── FloatInfo (np.finfo) ───────────────────────────────────────────────
    // NumPy: info = np.finfo(np.float32)
    println!("\n── FloatInfo (np.finfo equivalent) ──");
    let info = FloatInfo::of(DType::F32).unwrap();
    println!("{info}");
    println!("  eps:      {:.6e}", info.eps);
    println!("  max:      {:.6e}", info.max);
    println!("  tiny:     {:.6e}", info.tiny);
    println!("  nmant:    {}", info.nmant);
    println!("  nexp:     {}", info.nexp);
    println!("  precision:{}", info.precision);

    // Compare all float types
    println!("\n  All float types:");
    for &dt in &[DType::F16, DType::BF16, DType::F32, DType::F64] {
        let fi = FloatInfo::of(dt).unwrap();
        println!("    {:<10} bits={:2} nmant={:2} eps={:.3e} max={:.3e}",
            dt.numpy_str(), fi.bits, fi.nmant, fi.eps, fi.max);
    }

    // ── IntInfo (np.iinfo) ─────────────────────────────────────────────────
    // NumPy: info = np.iinfo(np.int32)
    println!("\n── IntInfo (np.iinfo equivalent) ──");
    let info = IntInfo::of(DType::I32).unwrap();
    println!("{info}");
    println!("  min: {}", info.min);
    println!("  max: {}", info.max);
    println!("  can_hold_i128(1000): {}", info.can_hold_i128(1000));

    // Compare all integer types
    println!("\n  All integer types:");
    for &dt in &[DType::I8, DType::I16, DType::I32, DType::I64,
                 DType::U8, DType::U16, DType::U32, DType::U64] {
        let ii = IntInfo::of(dt).unwrap();
        println!("    {:<10} bits={:2} signed={:5} min={:>21} max={}",
            dt.numpy_str(), ii.bits, ii.is_signed, ii.min, ii.max);
    }

    // ── Iterate all dtypes ─────────────────────────────────────────────────
    println!("\n── All {} DType variants ──", ALL_DTYPES.len());
    for dt in DType::all() {
        print!("{} ", dt.numpy_str());
    }
    println!();
}
