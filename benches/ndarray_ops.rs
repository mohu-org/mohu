// Workspace-level benchmarks for mohu array operations.
// Run with: cargo bench -p mohu-buffer
//
// These benchmarks currently exercise the `mohu-buffer` allocation layer,
// since `mohu-array` (the public NdArray<T> type) is not yet implemented.
// Once NdArray lands, these should be extended/replaced with array-level
// benchmarks (creation, indexing, elementwise ops, etc).

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mohu_buffer::buffer::Buffer;
use mohu_dtype::dtype::DType;

fn bench_buffer_zeros(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_zeros_f64");
    for &n in &[100usize, 10_000, 1_000_000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| Buffer::zeros(DType::F64, &[n]).unwrap());
        });
    }
    group.finish();
}

fn bench_buffer_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_alloc_f64");
    for &n in &[100usize, 10_000, 1_000_000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| Buffer::alloc(DType::F64, &[n], mohu_buffer::layout::Order::C).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_buffer_zeros, bench_buffer_alloc);
criterion_main!(benches);
