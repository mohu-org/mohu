//! Example: Understanding strides, layout, and memory order in mohu-buffer.

use mohu_buffer::strides::{NdIndexIter, c_strides, f_strides, ravel_multi_index, unravel_index};
use mohu_buffer::{Buffer, SliceArg};

fn main() {
    println!("=== C-order vs F-order strides ===");
    let shape = [3, 4];
    println!("C: {:?}", c_strides(&shape, 4));
    println!("F: {:?}", f_strides(&shape, 4));

    println!("=== transpose ===");
    let source = Buffer::from_slice(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .reshape(&[2, 3])
        .unwrap();
    let transposed = source.transpose();
    println!(
        "shape {:?}, strides {:?}",
        transposed.shape(),
        transposed.strides()
    );

    println!("=== broadcast zero stride ===");
    let row = Buffer::from_slice(&[10.0_f32, 20.0, 30.0])
        .unwrap()
        .expand_dims(0)
        .unwrap()
        .broadcast_to(&[4, 3])
        .unwrap();
    println!("shape {:?}, strides {:?}", row.shape(), row.strides());

    println!("=== slicing offset and stride ===");
    let matrix = Buffer::from_slice(&(1..=9).map(|x| x as f32).collect::<Vec<_>>())
        .unwrap()
        .reshape(&[3, 3])
        .unwrap();
    let sliced = matrix
        .slice_axis(
            0,
            SliceArg {
                start: Some(0),
                stop: None,
                step: Some(2),
            },
        )
        .unwrap();
    println!("shape {:?}, strides {:?}", sliced.shape(), sliced.strides());
    println!(
        "values {:?}",
        sliced.to_contiguous().unwrap().as_slice::<f32>().unwrap()
    );

    println!("=== NdIndexIter ===");
    for index in NdIndexIter::new(&[2, 3]) {
        println!("{:?}", index.as_slice());
    }

    println!("=== ravel/unravel ===");
    let shape = [2, 3];
    for flat in 0..6 {
        let index = unravel_index(flat, &shape).unwrap();
        let round_trip = ravel_multi_index(index.as_slice(), &shape).unwrap();
        println!("{} -> {:?} -> {}", flat, index.as_slice(), round_trip);
    }
}
