use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mohu_simd::copy::{copy_f32, copy_f64};
use mohu_simd::fill::{fill_f32, fill_f64, fill_i32, fill_i64};

fn bench_fill(c: &mut Criterion) {
    let mut group = c.benchmark_group("fill");
    for size in [1, 64, 1024, 4096, 65536].iter() {
        group.bench_with_input(BenchmarkId::new("std::ptr::write_bytes", size), size, |b, &size| {
            let mut buf = vec![0.0f32; size];
            b.iter(|| {
                // write_bytes operates on bytes, so we write 0
                // To be fair, write_bytes is mostly for zeroing, but we benchmark it anyway
                unsafe {
                    std::ptr::write_bytes(buf.as_mut_ptr(), 0, size);
                }
                black_box(&mut buf);
            });
        });

        group.bench_with_input(BenchmarkId::new("mohu_simd::fill_f32", size), size, |b, &size| {
            let mut buf = vec![0.0f32; size];
            b.iter(|| {
                unsafe {
                    fill_f32(buf.as_mut_ptr(), size, 42.0);
                }
                black_box(&mut buf);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("mohu_simd::fill_f64", size), size, |b, &size| {
            let mut buf = vec![0.0f64; size];
            b.iter(|| {
                unsafe {
                    fill_f64(buf.as_mut_ptr(), size, 42.0);
                }
                black_box(&mut buf);
            });
        });
    }
    group.finish();
}

fn bench_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy");
    for size in [1, 64, 1024, 4096, 65536].iter() {
        group.bench_with_input(BenchmarkId::new("std::ptr::copy_nonoverlapping", size), size, |b, &size| {
            let src = vec![42.0f32; size];
            let mut dst = vec![0.0f32; size];
            b.iter(|| {
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), size);
                }
                black_box(&mut dst);
            });
        });

        group.bench_with_input(BenchmarkId::new("mohu_simd::copy_f32", size), size, |b, &size| {
            let src = vec![42.0f32; size];
            let mut dst = vec![0.0f32; size];
            b.iter(|| {
                unsafe {
                    copy_f32(dst.as_mut_ptr(), src.as_ptr(), size);
                }
                black_box(&mut dst);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_fill, bench_copy);
criterion_main!(benches);
