# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- (new features land here before release)

### Changed

- (behaviour changes land here before release)

### Fixed

- (bug fixes land here before release)

## [0.1.0] - 2026-01-01

### Added

- Workspace foundation crates: `mohu-error`, `mohu-dtype`, `mohu-buffer`, `mohu-array`, `mohu-core`
- Compute and dispatch crates: `mohu-simd`, `mohu-ufunc`, `mohu-index`, `mohu-ops`, `mohu-fft`, `mohu-random`, `mohu-special`, `mohu-stats`
- Data extensions: `mohu-sparse`, `mohu-masked`
- I/O crate `mohu-io` and developer tooling crate `mohu-testing`
- DLPack v0.8 import/export in `mohu-buffer`
- AVX2 / AVX-512 / NEON kernel stubs in `mohu-simd`
- GitHub Actions CI (format, clippy, build, test, MSRV, deny, coverage, fuzz-build)

[Unreleased]: https://github.com/mohu-org/mohu/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mohu-org/mohu/releases/tag/v0.1.0
