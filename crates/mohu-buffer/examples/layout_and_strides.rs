//! Example: Understanding strides, layout, and memory order in mohu-buffer

use mohu_buffer::Buffer;
use mohu_buffer::SliceArg;
use mohu_buffer::strides::{NdIndexIter, c_strides, f_strides, ravel_multi_index, unravel_index};

fn main() {
    println!("=== 1. C-order vs F-order strides ===\n");

    let shape = [3, 4];
    let item_size = 4;

    let c_str = c_strides(&shape, item_size);
    let f_str = f_strides(&shape, item_size);

    println!("For a 3x4 matrix:");
    println!("  C-order strides: {:?}", c_str.as_slice());
    println!("  F-order strides: {:?}", f_str.as_slice());
    println!();

    println!("=== 2. Transpose as zero-copy stride swap ===\n");

    // Fixed: Using &[&[f64]] format
    let row1: &[f64] = &[1.0, 2.0, 3.0];
    let row2: &[f64] = &[4.0, 5.0, 6.0];
    let rows: &[&[f64]] = &[row1, row2];

    let original = Buffer::from_slice_2d(rows).unwrap();
    println!("Original 2x3 matrix:");
    println!("  Shape: {:?}", original.shape());
    println!("  Strides: {:?}", original.strides());

    let transposed = original.transpose();
    println!("\nAfter transpose (zero-copy):");
    println!("  Shape: {:?}", transposed.shape());
    println!("  Strides: {:?}", transposed.strides());
    println!();

    println!("=== 3. Broadcast with zero stride ===\n");

    let row = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0]).unwrap();
    println!("Original row shape: {:?}", row.shape());
    println!("Original row strides: {:?}", row.strides());

    let expanded = row.expand_dims(0).unwrap();
    println!("After expand_dims(0) shape: {:?}", expanded.shape());
    println!("After expand_dims(0) strides: {:?}", expanded.strides());

    let broadcasted = expanded.broadcast_to(&[4, 3]).unwrap();
    println!("\nBroadcasted to 4x3:");
    println!("  Shape: {:?}", broadcasted.shape());
    println!("  Strides: {:?}", broadcasted.strides());
    println!("  (Notice first stride is 0 - repeats the same row!)");
    println!();

    println!("=== 4. Slicing creates view with offset + stride ===\n");

    let data = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
    let reshaped = data.reshape(&[3, 3]).unwrap();
    println!("Original 3x3 matrix:");
    println!(
        "  Shape: {:?}, Strides: {:?}",
        reshaped.shape(),
        reshaped.strides()
    );

    // Slice: take every other row (step 2)
    let sliced = reshaped
        .slice_axis(
            0,
            SliceArg {
                start: Some(0),
                stop: None,
                step: Some(2),
            },
        )
        .unwrap();
    println!("\nAfter slicing [::2, :] (every other row):");
    println!(
        "  Shape: {:?}, Strides: {:?}",
        sliced.shape(),
        sliced.strides()
    );

    // Show the actual values
    println!("\n  Values in sliced view:");
    for i in 0..sliced.shape()[0] {
        print!("    ");
        for j in 0..sliced.shape()[1] {
            let val: f32 = sliced.get(&[i, j]).unwrap();
            print!("{} ", val);
        }
        println!();
    }
    println!();

    println!("=== 5. NdIndexIter walking indices in logical order ===\n");

    println!("Walking through all index positions of a 2x3 matrix (logical order):");
    let iter = NdIndexIter::new(&[2, 3]);
    for idx in iter {
        println!("  {:?}", idx.as_slice());
    }

    // Show with actual values
    println!("\nAccessing actual values using indices:");
    let matrix_data = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let matrix = matrix_data.reshape(&[2, 3]).unwrap();

    for idx in NdIndexIter::new(&[2, 3]) {
        let val: f32 = matrix.get(idx.as_slice()).unwrap();
        println!("  {:?} = {}", idx.as_slice(), val);
    }
    println!();

    println!("=== 6. ravel/unravel index conversion ===\n");

    let shape = [2, 3];
    println!("For a 2x3 matrix (6 total elements):\n");

    println!("Multi-index → Flat index (ravel_multi_index):");
    for i in 0..2 {
        for j in 0..3 {
            let flat = ravel_multi_index(&[i, j], &shape).unwrap();
            println!("  [{}, {}] → {}", i, j, flat);
        }
    }

    println!("\nFlat index → Multi-index (unravel_index):");
    for flat in 0..6 {
        let idx = unravel_index(flat, &shape).unwrap();
        println!(
            "  {} → [{}, {}]",
            flat,
            idx.as_slice()[0],
            idx.as_slice()[1]
        );
    }

    println!("\n✓ Example completed successfully!");
}
