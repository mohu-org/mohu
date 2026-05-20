# Contributing to mohu

Thank you for your interest in contributing. This document covers everything you need to get from zero to a merged pull request.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setting up the environment](#setting-up-the-environment)
- [Workflow](#workflow)
- [DCO sign-off](#dco-sign-off)
- [Commit convention](#commit-convention)
- [Branch naming](#branch-naming)
- [Crate map](#crate-map)
- [Running CI checks locally](#running-ci-checks-locally)
- [Writing tests](#writing-tests)
- [Writing benchmarks](#writing-benchmarks)
- [Documentation](#documentation)
- [PR checklist](#pr-checklist)

---

## Prerequisites

| Tool | Minimum version | Install |
|------|----------------|---------|
| Rust stable | 1.85 (edition 2024) | `rustup update stable` |
| Python | 3.10+ | for `mohu-py` only |
| cargo-deny | latest | `cargo install cargo-deny` |
| cargo-machete | latest | `cargo install cargo-machete` |

Verify your setup:

```sh
rustc --version        # >= 1.85
cargo clippy --version
```

---

## Setting up the environment

```sh
# 1. Fork on GitHub, then clone your fork
git clone https://github.com/<you>/mohu.git
cd mohu

# 2. Add upstream
git remote add upstream https://github.com/mohu-org/mohu.git

# 3. Keep fork in sync
git fetch upstream
git rebase upstream/main
```

---

## Workflow

1. Sync with upstream: `git fetch upstream && git rebase upstream/main`
2. Create a branch (see [Branch naming](#branch-naming))
3. Make changes — run `cargo clippy` and `cargo test` before pushing
4. Push to your fork: `git push origin <branch>`
5. Open a PR against `mohu-org/mohu:main`
6. Address review feedback; push follow-up commits to the same branch

Do **not** force-push after a reviewer has left comments — add new commits instead.

---

## DCO sign-off

Every commit **must** carry a `Signed-off-by` line. CI will reject PRs without it.

```sh
git commit -s -m "feat(core): add strided slice iterator"
```

This certifies you wrote the code and have the right to contribute it under the project license per the [Developer Certificate of Origin](https://developercertificate.org/).

To fix a missing sign-off on the last commit:

```sh
git commit --amend -s --no-edit
git push --force-with-lease origin <branch>
```

---

## Commit convention

We use [Conventional Commits](https://www.conventionalcommits.org/). Subject line is **under 72 characters**, imperative mood, no trailing period.

```
feat(array): add strided slice iterator
fix(ops): correct broadcast shape for rank-0 arrays
perf(simd): use AVX2 path for f32 dot product
refactor(buffer): split layout from allocation
doc(error): document ErrorKind::is_recoverable
test(stats): add property tests for median
chore(deps): bump rand to 0.9.4
ci: add MSRV check job
```

| Type | When to use |
|------|-------------|
| `feat` | new user-visible functionality |
| `fix` | bug fix |
| `perf` | performance improvement |
| `refactor` | no behaviour change, no new feature |
| `doc` | documentation only |
| `test` | adding or fixing tests |
| `chore` | dependency bumps, tooling, config |
| `ci` | changes to GitHub Actions workflows |

**Breaking changes:** append `!` after the type/scope — `feat(core)!: rename Array → NdArray`

---

## Branch naming

```
feat/<short-description>        # new feature
fix/<short-description>         # bug fix
perf/<short-description>        # performance work
refactor/<short-description>    # refactoring
docs/<short-description>        # documentation
ci/<short-description>          # CI/tooling changes
```

---

## Crate map

The workspace is layered — each layer only depends on layers below it.

### Foundation

| Crate | Owns |
|-------|------|
| `mohu-error` | Shared error types; zero-dependency base for every crate |
| `mohu-dtype` | `DType` enum, scalar type traits, type promotion rules |
| `mohu-buffer` | Raw buffer allocation, memory layout, stride arithmetic |
| `mohu-array` | `NdArray<T>` — the core N-dimensional array type |
| `mohu-core` | Re-export facade for the four crates above |

### Dispatch & protocol

| Crate | Owns |
|-------|------|
| `mohu-simd` | AVX2 / AVX-512 / NEON SIMD kernel primitives |
| `mohu-ufunc` | Universal-function protocol: broadcast, reduce, accumulate, outer |
| `mohu-index` | Advanced indexing: fancy, boolean mask, take/put |

### Compute

| Crate | Owns |
|-------|------|
| `mohu-ops` | Element-wise arithmetic, comparison, logical, broadcasting |
| `mohu-fft` | FFT, IFFT, RFFT, 2-D transforms |
| `mohu-random` | PRNG engines and statistical distributions |
| `mohu-special` | Special math: erf, gamma, beta, Bessel, … |
| `mohu-stats` | Descriptive stats, hypothesis tests, sampling |

### Data structure extensions

| Crate | Owns |
|-------|------|
| `mohu-sparse` | COO / CSR / CSC sparse matrix formats |
| `mohu-masked` | Masked arrays — null/invalid value propagation |

### I/O & tooling

| Crate | Owns |
|-------|------|
| `mohu-io` | `.npy`/`.npz`, CSV, Arrow IPC, memory-mapped files |
| `mohu-testing` | Test fixtures, property-test helpers, array comparison utilities |

---

## Running CI checks locally

Run these before pushing — CI runs all of them and will fail on any error.

```sh
# Format
cargo fmt --all

# Lint (warnings = errors)
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Tests
cargo test --workspace --all-features

# Docs (doc warnings = errors)
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features

# Dependency audit (advisories, licenses, bans, sources)
cargo deny check

# Unused dependencies
cargo machete

# Benchmarks compile check
cargo bench --workspace --no-run --all-features
```

---

## Writing tests

- Unit tests go in `#[cfg(test)]` modules inside the source file being tested.
- Integration tests go in `crates/<name>/tests/`.
- Property-based tests use `proptest` — see `mohu-testing::strategies` for pre-built generators.
- Use `mohu-testing::approx` for floating-point array comparisons.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use mohu_testing::approx::assert_allclose;

    #[test]
    fn round_trip() {
        // ...
    }
}
```

---

## Writing benchmarks

Benchmarks live in `benches/` at the workspace root and use the `criterion` harness.

```sh
cargo bench                          # run all benchmarks
cargo bench -- array_ops             # run a specific group
cargo bench --no-run --all-features  # verify benchmarks compile
```

---

## Documentation

- Every public item needs a doc comment.
- Include at least one `# Example` section on non-trivial items.
- Avoid restating what the name already says — document *why* and *when*.

```rust
/// Returns the total number of elements across all dimensions.
///
/// # Example
///
/// ```rust
/// # use mohu_array::NdArray;
/// let a = NdArray::<f32>::zeros(&[3, 4]);
/// assert_eq!(a.size(), 12);
/// ```
pub fn size(&self) -> usize { ... }
```

---

## PR checklist

Before requesting review, confirm:

- [ ] Branch rebased on `upstream/main`
- [ ] All commits signed off (`git commit -s`)
- [ ] `cargo fmt --all` — no diff
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings` — clean
- [ ] `cargo test --workspace --all-features` — passes
- [ ] New public APIs have doc comments with examples
- [ ] `CHANGELOG.md` entry added for user-visible changes
- [ ] No `TODO`, `unimplemented!()`, or stub functions left in completed code
