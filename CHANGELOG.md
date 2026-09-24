# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- `mohu-random`: `ChaCha8` and `SplitMix64` PRNG engines, completing the set
  of generators documented in the crate's module docs.

### Fixed

- `mohu-buffer`: `Buffer::eye` no longer panics (debug) or returns
  `IndexOutOfBounds` (release) when the requested diagonal offset lies
  entirely outside the matrix bounds; it now returns a valid zero-filled
  matrix in both build profiles.
