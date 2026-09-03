# Contributing to mohu

Thank you for your interest in contributing. This document covers everything you need to get from zero to a merged pull request.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment variables](#environment-variables)
- [Setting up the environment](#setting-up-the-environment)
- [Workflow](#workflow)
- [DCO sign-off](#dco-sign-off)
- [Commit convention](#commit-convention)
- [Branch naming](#branch-naming)
- [Running the project locally](#running-the-project-locally)
- [Crate map](#crate-map)
- [Running CI checks locally](#running-ci-checks-locally)
- [Troubleshooting](#troubleshooting)
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
## Environment variables
mohu doesn't require any environment variables to build or test. A couple are
recognized optionally, to control error-message formatting from `mohu-error`:

| Variable | Effect |
|----------|--------|
| `MOHU_COLOR` | Forces colored error output on/off regardless of terminal detection. |
| `NO_COLOR` | Disables colored error output. Standard convention — see [no-color.org](https://no-color.org/). |

Neither needs to be set for normal development.

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

## Running the project locally

mohu is a library workspace, not an application — "running it locally" means
building it, running the example programs, and (optionally) previewing the
documentation book.

**Run an example:**
```sh
cargo run -p mohu-buffer --example buffer_basics
cargo run -p mohu-buffer --example alloc_and_pool
cargo run -p mohu-dtype --example dtype_basics
cargo run -p mohu-dtype --example type_promotion
```

> Note: the top-level `examples/` directory (`array_basics.rs`, `io_npy.rs`,
> `linalg_matmul.rs`) currently isn't wired into any crate's `Cargo.toml`, so
> `cargo run --example <name>` won't find them yet. The commands above use
> the examples that are registered and runnable today.

**Preview the documentation book** (`docs/` — built with [mdBook](https://rust-lang.github.io/mdBook/)):
```sh
cargo install mdbook   # one-time
mdbook serve docs       # serves locally, live-reloads on save
```

**Build and inspect API docs:**
```sh
cargo doc --workspace --no-deps --all-features --open
```

## Project Structure

Before making changes, take a moment to familiarize yourself with the project layout.

- [`CRATE_MAP.md`](CRATE_MAP.md) provides an overview of the workspace crates and explains the responsibility of each crate. It is the best place to start if you're unsure where a change belongs.
- [`CLAUDE.md`](CLAUDE.md) contains guidance for contributors using AI coding assistants. If you're using AI tools to help with development, please review it before submitting a pull request.

These documents can help you quickly identify the appropriate crate and follow the project's development practices.


### Quick Reference

| If you want to... | Start here |
|-------------------|------------|
| Understand the workspace architecture | `CRATE_MAP.md` |
| Use AI coding assistance | `CLAUDE.md` |
| Work on array operations | `crates/mohu-ops/` |
| Work on SIMD optimizations | `crates/mohu-simd/` |
| Add or update benchmarks | `benches/` |
| Improve documentation | `docs/` |
| Add or update tests | `tests/` or the relevant crate's `tests/` directory |

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

## Troubleshooting

If you encounter build or installation issues, try the solutions below before opening an issue.

### Rust toolchain version mismatch

**Problem**

The build fails because the required Rust toolchain is missing or outdated.

**Solution**
Ensure you're using Rust 1.85 or newer, as listed in the Prerequisites section.
Update Rust and verify your installation:

```sh
rustup update stable
rustup toolchain install stable
rustc --version
cargo --version
```

---

### Cargo dependency resolution errors

**Problem**

Cargo reports dependency conflicts or fails to resolve package versions.

**Solution**

Clean the workspace and update dependencies:

```sh
cargo clean
cargo update
cargo build
```

If the issue persists, make sure your branch is up to date with `upstream/main`.

---

### Missing system dependencies

Some crates require common system build tools.

On Ubuntu/Debian:

```sh
sudo apt update
sudo apt install build-essential pkg-config
```

Install any additional dependencies mentioned in compiler error messages.

---

### Python binding build failures

If you are working with `mohu-py`, ensure that Python 3.10 or newer is installed.

Verify your Python installation:

```sh
python --version
python -m pip install --upgrade pip
```

If multiple Python versions are installed, ensure Cargo is using the intended interpreter.

---

### Platform-specific notes

- **Linux:** Install the required build tools and keep Rust updated.
- **Windows:** Use the latest Rust toolchain installed via `rustup`. Running commands from PowerShell or Git Bash is recommended.
- **macOS:** Install Xcode Command Line Tools before building the project.

---

### Helpful Cargo commands

These commands are useful when debugging build issues:

```sh
cargo check
cargo build
cargo test
cargo clean
cargo tree
cargo metadata
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
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
