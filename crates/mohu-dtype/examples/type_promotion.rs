//! Demonstrates mohu-dtype parsing, promotion, and casting rules.

use mohu_dtype::{can_cast, promote, CastMode, DType};

fn parse_common_names() -> mohu_dtype::MohuResult<()> {
    println!("=== from_str ===");
    for name in ["float32", "int64", "complex128", "bool"] {
        let dt = DType::from_str(name)?;
        println!("{name} -> {dt} ({} bytes)", dt.itemsize());
    }
    Ok(())
}

fn promotion_examples() {
    println!("\n=== promote ===");
    let pairs = [
        (DType::I32, DType::F32, "int + float"),
        (DType::F32, DType::F64, "float + float"),
        (DType::F64, DType::C128, "real + complex"),
        (DType::Bool, DType::I8, "bool + int"),
    ];
    for (a, b, label) in pairs {
        println!("{label}: promote({a}, {b}) = {}", promote(a, b));
    }
}

fn casting_examples() -> mohu_dtype::MohuResult<()> {
    println!("\n=== can_cast ===");
    println!(
        "I32 -> F64 (Safe): {}",
        can_cast(DType::I32, DType::F64, CastMode::Safe)
    );
    println!(
        "F64 -> F32 (Safe): {}",
        can_cast(DType::F64, DType::F32, CastMode::Safe)
    );
    println!(
        "F64 -> F32 (SameKind): {}",
        can_cast(DType::F64, DType::F32, CastMode::SameKind)
    );
    Ok(())
}

fn main() -> mohu_dtype::MohuResult<()> {
    parse_common_names()?;
    promotion_examples();
    casting_examples()?;
    Ok(())
}
