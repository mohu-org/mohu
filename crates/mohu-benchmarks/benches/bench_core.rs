use criterion::{black_box, criterion_group, criterion_main, Criterion};

// Lightweight core benchmarks scaffold. These are intentionally simple and
// provide maintainers a starting point to add mohu-array-specific benchmarks.

fn bench_arithmetic_add(c: &mut Criterion) {
    let size = 1_000_000usize;
    let a: Vec<f64> = (0..size).map(|i| (i as f64) * 0.5).collect();
    let b: Vec<f64> = (0..size).map(|i| (i as f64) * 1.5).collect();

    c.bench_function("vec_add_iter", |bencher| {
        bencher.iter(|| {
            let mut r = vec![0.0f64; size];
            for i in 0..size {
                r[i] = black_box(a[i]) + black_box(b[i]);
            }
            black_box(r)
        })
    });
}

fn bench_reduction_sum(c: &mut Criterion) {
    let size = 2_000_000usize;
    let a: Vec<f64> = (0..size).map(|i| (i as f64) * 0.25).collect();

    c.bench_function("vec_sum_iter", |bencher| {
        bencher.iter(|| {
            let mut s = 0.0f64;
            for &v in &a {
                s += black_box(v);
            }
            black_box(s)
        })
    });
}

fn bench_stride_access(c: &mut Criterion) {
    let rows = 10_000usize;
    let cols = 1_000usize;
    // simulate flat storage
    let mut data: Vec<f64> = vec![0.0; rows * cols];
    for i in 0..(rows * cols) { data[i] = (i as f64) % 100.0; }

    c.bench_function("row_major_sum", |bencher| {
        bencher.iter(|| {
            let mut s = 0.0f64;
            for r in 0..rows {
                for cidx in 0..cols {
                    s += black_box(data[r * cols + cidx]);
                }
            }
            black_box(s)
        })
    });

    c.bench_function("col_major_sum", |bencher| {
        bencher.iter(|| {
            let mut s = 0.0f64;
            for cidx in 0..cols {
                for r in 0..rows {
                    s += black_box(data[r * cols + cidx]);
                }
            }
            black_box(s)
        })
    });
}

criterion_group!(benches, bench_arithmetic_add, bench_reduction_sum, bench_stride_access);
criterion_main!(benches);
