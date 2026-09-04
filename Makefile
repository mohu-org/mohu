.PHONY: build release test lint fmt fmt-check check ci clean bench deny changelog size contrib-guide help

help:
	@echo "Available targets:"
	@echo "  build          Build the workspace"
	@echo "  release        Build the workspace in release mode"
	@echo "  test           Run workspace tests"
	@echo "  lint           Run workspace clippy"
	@echo "  fmt            Format all Rust code"
	@echo "  fmt-check      Check Rust formatting"
	@echo "  check          Run formatting, lint, and tests"
	@echo "  ci             Run the local CI validation loop"
	@echo "  clean          Remove build artifacts"
	@echo "  bench          Run benchmarks"
	@echo "  deny           Run cargo-deny"
	@echo "  changelog      Regenerate the changelog"
	@echo "  size           Show release library size"
	@echo "  contrib-guide  Print contributor references"
	@echo "Windows users: run make through WSL or Git Bash."

build:
	cargo build --workspace

release:
	cargo build --workspace --release

test:
	cargo test --workspace

lint:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

check: fmt-check lint test

ci: check

clean:
	cargo clean

bench:
	cargo bench --workspace

deny:
	cargo deny check

changelog:
	git cliff --output CHANGELOG.md

size: release
	@du -sh target/release/libmohu* 2>/dev/null || echo "no release artifacts found"

contrib-guide:
	@echo "Contributor Quick Reference"
	@echo "Read CONTRIBUTING.md"
	@echo "Workspace overview: CRATE_MAP.md"
	@echo "AI coding guide: CLAUDE.md"
