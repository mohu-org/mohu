# Contributing to mohu

Thank you for your interest in contributing. This document covers everything you need to get from zero to a merged pull request.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setting up the environment](#setting-up-the-environment)
- [Quick start](#quick-start)
- [Workflow](#workflow)
- [DCO sign-off](#dco-sign-off)
- [Commit convention](#commit-convention)
- [Branch naming](#branch-naming)
- [Crate map](#crate-map)
- [Running CI checks locally](#running-ci-checks-locally)
- [Writing tests](#writing-tests)
- [Writing benchmarks](#writing-benchmarks)
- [Documentation](#documentation)
- [CHANGELOG updates](#changelog-updates)
- [Troubleshooting](#troubleshooting)
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

## Quick start

### Try it out

```sh
# Build the entire workspace
cargo build --workspace

# Run all tests
cargo test --workspace

# Run an example to verify setup
cargo run --example array_basics

# Verify code passes all validations
make check    # or run individual checks below
```

### Your first contribution

1. Find something to work on — check [ROADMAP.md](ROADMAP.md) or GitHub issues
2. Create a branch: `git checkout -b feat/my-feature`
3. Make changes
4. Commit with sign-off: `git commit -s -m "feat(ops): add new operation"`
5. Validate locally: `make check`
6. Push and open a PR

Detailed steps in [Workflow](#workflow) below.

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

### Unsafe code validation (Miri)

For contributors working on `mohu-buffer` (allocator) or `mohu-simd` (intrinsics), validate unsafe code locally before pushing:

```sh
# Install nightly Rust
rustup toolchain install nightly
rustup +nightly component add miri

# Run Miri on mohu-buffer with strict provenance checking
MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" \
  cargo +nightly miri test -p mohu-buffer --all-features
```

Expected output: All tests pass. If Miri fails, there is undefined behavior in the unsafe code.

Run this before opening a PR if you modified any `unsafe` blocks.

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

## CHANGELOG updates

User-visible changes **must** include a `CHANGELOG.md` entry.

### What needs a CHANGELOG entry?

- ✓ New public APIs
- ✓ Behavior changes (including bug fixes)
- ✓ Performance improvements
- ✓ Deprecated APIs
- ✗ Internal refactoring (no user impact)
- ✗ Documentation fixes
- ✗ Test-only changes

### How to add an entry

Edit `CHANGELOG.md` — add a bullet under the **Unreleased** section in the appropriate category:

```markdown
## Unreleased

### Added
- New `einsum()` function for Einstein summation notation

### Fixed
- Correct SIMD alignment on aarch64 targets

### Changed
- `NdArray::reshape()` now validates dimension product

### Performance
- 3x speedup in f32 dot product via AVX2 kernel
```

Format follows [Keep a Changelog](https://keepachangelog.com/) conventions.

---

## Troubleshooting

### `error: failed to parse manifest at .../Cargo.toml`

Ensure you are using Rust 1.85 or later:

```sh
rustc --version     # Should show >= 1.85
rustup update stable
rustup default stable
```

### `error: could not compile because of -D warnings`

Clippy requires zero warnings. Fix all lints:

```sh
cargo clippy --workspace --all-targets --all-features --fix
```

If automatic fixes don't work, view the first few errors:

```sh
cargo clippy --workspace --all-targets --all-features 2>&1 | head -50
```

Common issues:
- Missing doc comments on public items → add `///` documentation
- Unused imports → remove or prefix with `_` if intentional
- Unused variables → remove or prefix with `_`

### Tests fail with `assertion: left == right` on floats

Do NOT use `assert_eq!` for floating-point comparisons. Use:

```rust
use mohu_testing::approx::assert_allclose;
assert_allclose(&result, &expected, 1e-6)?;
```

### `cargo machete` reports unused dependencies

Remove unused dependencies from `Cargo.toml`:

```sh
cargo machete                    # Lists unused deps
# Edit the crate's Cargo.toml to remove them
cargo test --workspace           # Re-test to confirm no impact
```

### `cargo deny check` fails

Check the error type:

```sh
cargo deny check advisories   # Security vulnerabilities
cargo deny check licenses     # License violations
cargo deny check bans         # Banned crates
cargo deny check sources      # Unexpected sources
```

To ignore a known issue, add to `deny.toml` (requires maintainer review). For security advisories, consider upgrading the crate instead.

### I can't push to my fork

Verify your remote is configured correctly:

```sh
git remote -v
# Should show:
# origin    https://github.com/<your-username>/mohu.git (fetch)
# origin    https://github.com/<your-username>/mohu.git (push)
# upstream  https://github.com/mohu-org/mohu.git (fetch)
# upstream  https://github.com/mohu-org/mohu.git (push)
```

If `origin` points to the wrong repo:

```sh
git remote set-url origin https://github.com/<your-username>/mohu.git
```

### I forgot to sign off a commit

Fix the last commit:

```sh
git commit --amend -s --no-edit
git push --force-with-lease origin <branch>
```

For multiple unsigned commits, rebase interactively:

```sh
git rebase -i HEAD~<N>      # Replace <N> with number of commits
# In the editor, mark commits as 'reword'
# Git will stop at each commit — just `:wq` and the sign-off will be added
```

### My PR says `DCO` check failed

All commits must have `Signed-off-by`. Re-sign all commits:

```sh
git rebase -i upstream/main
# Mark all commits as 'reword'
# For each commit, Git opens an editor — just `:wq` (keep the message)
# Git will automatically add the sign-off
git push --force-with-lease origin <branch>
```

### Do not force-push after reviews

If you accidentally force-pushed after reviewers left comments:

```sh
# Let maintainers know in the PR comment
# They may need to re-review
```

For future PRs: Only force-push before the first review comment. After that, add new commits.

### I need to test on a specific Rust version

CI runs on multiple Rust versions. To match locally:

```sh
# Test on Rust 1.85 (MSRV)
rustup toolchain install 1.85
cargo +1.85 build --workspace

# Test on nightly (for Miri)
rustup toolchain install nightly
cargo +nightly miri test -p mohu-buffer --all-features
```

Default is stable. Specify `+<toolchain>` to use a different version.

### Tests timeout or hang

If `cargo test` seems to hang, check for infinite loops or deadlocks:

```sh
# Run with a timeout (requires `timeout` command on Unix)
timeout 60 cargo test --workspace

# Run with verbose output to see which test is stuck
cargo test --workspace -- --nocapture --test-threads=1
```

If a specific test hangs, isolate it:

```sh
cargo test --workspace -- <test-name> --exact
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
