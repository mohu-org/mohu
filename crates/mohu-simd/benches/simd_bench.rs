use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn fill_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("fill");
    for &len in &[64usize, 4096, 65536] {
        group.bench_function(format!("dispatch_{len}"), |b| {
            b.iter(|| {
                let mut values = vec![0.0f32; len];
                // SAFETY: `values` is valid for exclusive writes of `len` f32s.
                unsafe { mohu_simd::fill::fill_f32(values.as_mut_ptr(), len, black_box(1.25)) };
                black_box(values);
            })
        });
        group.bench_function(format!("slice_fill_{len}"), |b| {
            b.iter(|| {
                let mut values = vec![0.0f32; len];
                values.fill(black_box(1.25));
                black_box(values);
            })
        });
    }
    group.finish();
}

fn copy_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy");
    for &len in &[64usize, 4096, 65536] {
        let source: Vec<f64> = (0..len).map(|i| i as f64 + 0.5).collect();
        group.bench_function(format!("dispatch_{len}"), |b| {
            b.iter(|| {
                let mut values = vec![0.0f64; len];
                // SAFETY: source and destination are valid and non-overlapping.
                unsafe { mohu_simd::copy::copy_f64(values.as_mut_ptr(), source.as_ptr(), len) };
                black_box(values);
            })
        });
        group.bench_function(format!("slice_copy_{len}"), |b| {
            b.iter(|| {
                let mut values = vec![0.0f64; len];
                values.copy_from_slice(&source);
                black_box(values);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, fill_bench, copy_bench);
criterion_main!(benches);
