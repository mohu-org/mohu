# AGENTS.md — mohu

Guidance for AI coding agents (Codex, Claude, Copilot Workspace, etc.) working in this repository. Supplements `CLAUDE.md` with agent-specific behavior rules.

---

## Read first

Before writing any code, read:

1. `CLAUDE.md` — build commands, architecture, error handling, clippy rules, commit rules
2. `CRATE_MAP.md` — exact public API surface per crate
3. `CONTRIBUTING.md` — PR workflow, DCO, branch naming

---

## Repository orientation

```
mohu/
├── Cargo.toml          # workspace root — all dep versions here
├── crates/             # 17 workspace crates
│   ├── mohu-error/     # zero-dep error foundation
│   ├── mohu-dtype/     # scalar types and type promotion
│   ├── mohu-buffer/    # raw allocation, layout, strides (unsafe allowed here)
│   ├── mohu-array/     # NdArray<T> type
│   ├── mohu-core/      # re-export facade only
│   ├── mohu-simd/      # SIMD intrinsics (unsafe allowed here)
│   ├── mohu-ufunc/     # universal-function protocol
│   ├── mohu-index/     # advanced indexing
│   ├── mohu-ops/       # element-wise ops and broadcasting
│   ├── mohu-fft/       # FFT transforms
│   ├── mohu-random/    # PRNG and distributions
│   ├── mohu-special/   # special math functions
│   ├── mohu-stats/     # descriptive stats and hypothesis tests
│   ├── mohu-sparse/    # COO/CSR/CSC formats
│   ├── mohu-masked/    # masked arrays
│   ├── mohu-io/        # file I/O
│   └── mohu-testing/   # test utilities (never a non-test dep)
├── benches/            # workspace-level benchmarks
├── docs/               # mdBook source and RFCs
└── .github/
    ├── workflows/
    │   ├── ci.yml      # 13-job CI pipeline
    │   └── release.yml # tag-triggered cross-compilation + release
    ├── labeler.yml     # PR auto-labeling rules
    └── auto_assign.yml # PR auto-reviewer config
```

---

## Before writing code

1. **Read every file you will modify** — do not edit blind.
2. **Check the layer** — identify which crate owns the concept. Never add logic to the wrong layer.
3. **Check `CRATE_MAP.md`** — confirm the public items you need already exist before reimplementing them.
4. **Run `cargo check -p <crate>`** — verify the workspace builds before making changes.

---

## Crate layer rules (hard constraints)

| Layer | Crates | May depend on |
|-------|--------|---------------|
| Foundation | error, dtype, buffer, array, core | Only lower foundation crates |
| Dispatch | simd, ufunc, index | Foundation only |
| Compute | ops, fft, random, special, stats | Foundation + Dispatch |
| Extensions | sparse, masked | Foundation + Dispatch + Compute |
| IO/Tooling | io, testing | Foundation + Compute |

Violating layer order = immediate CI failure. Check before adding any `[dependencies]` entry.

---

## Code style rules

- No `unwrap()` or `expect()` in library code. Use `?` and `MohuError`.
- No `println!`/`eprintln!` in library code. Use `tracing::debug!` / `tracing::warn!`.
- No `todo!()` or `unimplemented!()` in non-test code. Implement it or do not add it.
- Every `unsafe` block needs a `// SAFETY:` comment. Unsafe is only permitted in `mohu-buffer` and `mohu-simd`.
- Float comparisons in tests must use `mohu_testing::approx::assert_allclose` — never `assert_eq!` on `f32`/`f64`.
- Public API items need doc comments with at least one `# Example` block.

---

## What to verify after every change

```sh
cargo check --workspace --all-features          # fast type check
cargo clippy -p <changed-crate> --all-targets -- -D warnings
cargo test -p <changed-crate> --all-features
cargo fmt --all -- --check
```

Run the full suite (`cargo test --workspace`) before opening a PR.

---

## Commit format (mandatory)

```
<type>(<scope>): <subject>
```

- `type`: feat / fix / perf / refactor / doc / test / chore / ci
- `scope`: crate name without `mohu-` prefix, e.g. `buffer`, `ops`, `error`
- `subject`: imperative, under 72 chars, no trailing period
- Must include `Signed-off-by` line — use `git commit -s`
- No mention of AI tools, Claude, Codex, or Copilot anywhere in the message

Examples:
```
feat(array): add strided window iterator
fix(buffer): align allocation to SIMD_ALIGN on aarch64
perf(ops): use AVX2 path for f32 horizontal sum
test(error): add proptest for MultiError extend_from
ci: add musl cross-compile target
```

---

## PR rules

- One logical change per PR. Do not bundle unrelated fixes.
- Title = commit subject format above.
- Target `mohu-org/mohu:main`.
- All commits must be signed (`Signed-off-by`). DCO check blocks merge.
- Required status: `CI Pass` (the `ci-pass` job aggregates all 13 checks).
- Do not force-push after review — amend is allowed only before first review comment.

---

## Things agents commonly get wrong in this repo

| Mistake | Correct approach |
|---------|-----------------|
| Adding `anyhow` or `Box<dyn Error>` | Use `MohuError` and `MohuResult<T>` |
| Using `unwrap()` in library code | Propagate with `?` and add context |
| Adding a dependency to the wrong layer | Check the layer table above first |
| Comparing floats with `assert_eq!` | Use `assert_allclose` from `mohu-testing` |
| Adding `println!` for debug output | Use `tracing::debug!` |
| Writing unsafe without `// SAFETY:` | Always document the invariant |
| Editing `Cargo.lock` directly | Never — `cargo` manages it |
| Using `git add .` | Stage files explicitly by name |
| Forgetting `-s` in commit | `git commit -s` — DCO is required |
| Adding logic to `mohu-core` | It is a re-export facade — no logic |
| Pinning versions in crate `Cargo.toml` | Use `{ workspace = true }` |

---

## CI pipeline summary

All 13 jobs must pass (aggregated under `ci-pass`):

| Job | What it checks |
|-----|---------------|
| `dco` | Every commit has `Signed-off-by` (PR only) |
| `fmt` | `rustfmt` — no diff |
| `clippy` | Zero warnings, `-D warnings` |
| `deny` | No banned deps, no unlicensed deps, no advisories |
| `unused-deps` | No unused `[dependencies]` entries |
| `build` | Workspace builds with all features |
| `doc` | Docs build with `-D warnings` |
| `msrv` | Builds on Rust 1.85 |
| `bench-check` | All benchmarks compile |
| `semver` | No accidental breaking API changes (PR only) |
| `cross` | Builds for aarch64-linux, aarch64-darwin, x86_64-musl |
| `coverage` | Test coverage reported to Codecov |
| `miri` | `mohu-buffer` unsafe code passes Miri strict provenance |
