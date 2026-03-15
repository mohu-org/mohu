.PHONY: build release test lint fmt fmt-check check clean bench deny changelog

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
