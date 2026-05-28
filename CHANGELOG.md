# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-01

### Added

#### Foundation Layer
- `mohu-error`: error types, error codes, and diagnostic reporter
- `mohu-dtype`: `DType` enum, scalar traits, and type-promotion rules
- `mohu-buffer`: aligned allocation, memory layout, strides, and DLPack v0.8 import/export
- `mohu-array`: N-dimensional array type built on `mohu-buffer`
- `mohu-core`: re-export facade unifying the four foundation crates

#### Dispatch & Protocol Layer
- `mohu-simd`: AVX2, AVX-512, and NEON SIMD kernel primitives
- `mohu-ufunc`: universal-function protocol supporting broadcast, reduce, and outer operations
- `mohu-index`: advanced indexing — fancy indexing, boolean masks, and take/put

#### Compute Layer
- `mohu-ops`: element-wise operations, matrix multiplication, and reductions
- `mohu-fft`: FFT, IFFT, RFFT, and 2-D FFT via `rustfft`
- `mohu-random`: PRNG engines (PCG64, Philox) and statistical distributions
- `mohu-special`: special math functions — erf, gamma, beta, Bessel, and more
- `mohu-stats`: descriptive statistics and hypothesis tests

#### Data Structure Extensions
- `mohu-sparse`: COO, CSR, and CSC sparse matrix formats
- `mohu-masked`: masked arrays with null/invalid value propagation

#### I/O & Interop
- `mohu-io`: read/write support for .npy, .npz, CSV, HDF5, and Parquet formats

#### Developer Tooling
- `mohu-testing`: test fixtures, property-based tests, and array comparison utilities

#### Infrastructure
- GitHub Actions CI pipeline: `fmt`, `clippy`, build, test, MSRV, `cargo-deny`, coverage, and fuzz-build jobs
- Automated PR review bot, DCO sign-off enforcement, PR title validation, and CodeRabbit integration
- Workspace-level build profiles: `release`, `dist-release` (fat LTO), and `bench`
