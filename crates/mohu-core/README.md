# mohu-core

`mohu-core` is a convenience crate that re-exports the core Mohu foundation crates as a single dependency.

Instead of depending on individual crates such as `mohu-error`, `mohu-dtype`, `mohu-buffer`, and `mohu-array`, users can import them through `mohu-core`.

## When to use mohu-core

Use `mohu-core` when:

- You need functionality from multiple Mohu foundation crates.
- You prefer a single dependency and import path.

Use individual crates when:

- You only need one specific crate.
- You want to keep dependencies minimal.

## Example

```rust
use mohu_core::*;

fn main() {
    // Access re-exported items from Mohu crates.
}
```

## Dependency Graph

```text
mohu-core
├── mohu-array
├── mohu-buffer
├── mohu-dtype
└── mohu-error
```

## Contributing

See the workspace [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.