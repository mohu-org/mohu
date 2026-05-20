# CLAUDE.md — mohu

AI assistant guide for the mohu workspace. Read this before touching any code.

---

## What this project is

mohu is a NumPy-equivalent scientific computing library written in Rust, with Python bindings via PyO3. It is a Cargo workspace of 17 crates arranged in strict dependency layers.

---

## Build & test commands

```sh
cargo build --workspace --all-features          # full build
cargo test --workspace --all-features           # all tests
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all
cargo doc --workspace --no-deps --all-features
cargo deny check                                # advisories + licenses + bans
cargo machete                                   # unused deps
cargo bench --workspace --no-run --all-features # verify benches compile
```

MSRV is **1.85** (Rust edition 2024). Never use syntax or APIs above 1.85.

---

## Crate dependency layers

Only depend downward. Never add an upward dependency.

```
Foundation:   mohu-error → mohu-dtype → mohu-buffer → mohu-array → mohu-core
Dispatch:     mohu-simd, mohu-ufunc, mohu-index
Compute:      mohu-ops, mohu-fft, mohu-random, mohu-special, mohu-stats
Extensions:   mohu-sparse, mohu-masked
IO/Tooling:   mohu-io, mohu-testing
```

`mohu-error` has zero internal dependencies. Every other crate may depend on it.
`mohu-core` is a re-export facade only — no logic lives there.

---

## Workspace Cargo.toml

All external dependency versions live in `[workspace.dependencies]` in the root `Cargo.toml`. Crates reference them as `dep = { workspace = true }`. Never pin a version inside a crate's own `Cargo.toml` unless there is a hard incompatibility.

---

## Error handling rules

- Use `MohuError` from `mohu-error` everywhere — never `anyhow`, never `Box<dyn Error>`.
- Wrap context with `.context("what was being attempted")` or `.with_context(|| ...)`.
- Never silently drop errors. Every `?` must propagate or be explicitly handled.
- Use `MultiError` when a pass must collect all failures before returning.
- Use `bail!` and `ensure!` macros from `mohu-error` — not manual `return Err(...)`.

---

## Unsafe code rules

- Unsafe is only allowed in `mohu-buffer` (allocation, DLPack FFI) and `mohu-simd` (intrinsics).
- Every `unsafe` block must have a `// SAFETY:` comment explaining the invariant.
- All new unsafe in `mohu-buffer` is covered by Miri in CI — keep it that way.
- Do not introduce unsafe in any other crate without opening an RFC first.

---

## Clippy configuration

`clippy.toml` sets:
- MSRV: 1.85
- `cognitive-complexity-threshold`: 30
- `too-many-arguments-threshold`: 10
- `type-complexity-threshold`: 400

CI runs `-D warnings` — zero warnings allowed. Fix clippy, do not `#[allow]` unless the lint is genuinely wrong, and document why.

---

## Testing rules

- Unit tests: `#[cfg(test)]` at the bottom of the source file being tested.
- Integration tests: `crates/<name>/tests/`.
- Property tests: use `proptest` via `mohu-testing::strategies`.
- Float comparison: use `mohu-testing::approx::assert_allclose` — never `==` on floats.
- No `unimplemented!()`, `todo!()`, or `panic!("not yet")` in non-test code.
- `mohu-testing` must not be a dependency of any non-test crate.

---

## Commit rules

- One-line subject only — no body.
- Imperative mood: "add", "fix", "update".
- Under 72 characters.
- Always sign off: `git commit -s` (DCO). Signed-off-by must appear.
- Conventional Commits prefix required: `feat`, `fix`, `perf`, `refactor`, `doc`, `test`, `chore`, `ci`.
- Never mention Claude, AI, or any tool in commit messages.

---

## PR workflow

- Branch off `main`. Name: `feat/<desc>`, `fix/<desc>`, `docs/<desc>`, `ci/<desc>`.
- Target `mohu-org/mohu:main` from your fork.
- DCO check runs on every PR — all commits must be signed.
- `ci-pass` job is the required status check for merge.
- Do not force-push after review comments land — add new commits.

---

## Do not do these

- Do not add `println!` or `eprintln!` in library code — use `tracing`.
- Do not use `unwrap()` or `expect()` outside tests and CLI entrypoints.
- Do not add `[dev-dependencies]` that leak into workspace — scope them to the crate.
- Do not create new crates without an RFC (see `docs/rfcs/`).
- Do not edit `Cargo.lock` manually.
- Do not commit with `--no-verify`.
- Do not use `git add .` or `git add -A` — stage files explicitly.

---

## Key files

| File | Purpose |
|------|---------|
| `Cargo.toml` | Workspace root — all dep versions live here |
| `deny.toml` | Allowed licenses, banned crates, advisory ignore list |
| `clippy.toml` | Clippy thresholds |
| `cliff.toml` | Changelog generation config (git-cliff) |
| `CRATE_MAP.md` | Full module and public API surface per crate |
| `.github/workflows/ci.yml` | 13-job CI pipeline |
| `.github/workflows/release.yml` | Tag-triggered release + cross builds |
| `.github/labeler.yml` | Auto-label rules for PRs |
| `.github/auto_assign.yml` | Auto-reviewer assignment config |
