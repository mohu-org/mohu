# mohu-core

## Overview

`mohu-core` is the re-export facade that downstream users import instead of depending on individual mohu crates.
It re-exports the four foundation crates (`mohu-error`, `mohu-dtype`, `mohu-buffer`, `mohu-array`) as a single dependency. This allows users to import a single mohu crate instead of individual crates.

## When to use `mohu-core` vs individual crates

- Use `mohu-core` when you want to use a single dependency instead of importing many crates individually.
- It contains the four foundation crates (`mohu-error`, `mohu-dtype`, `mohu-buffer`, `mohu-array`).
- It is useful for the projects that requires different multiple capabilities such as error handling, data types, buffer and arrays.
- Use individual crates when the project requires a single component instead of many.
- Use individual crates when the project requires minimal dependencies.

# Example

```rust
use mohu_core::{  
    mohu_array,
    mohu_buffer,  // mohu_core contains all the four foundational crates here.
    mohu_dtype,
    mohu_error,
};

fn main() {
    // User can now access foundational crates here through mohu-core.
}
```

# Dependency Graph

```text
mohu-core
├── mohu-array
├── mohu-buffer
├── mohu-dtype
└── mohu-error
```

## Contributing

To contribute see [CONTRIBUTING.md] (https://github.com/mohu-org/mohu/blob/main/CONTRIBUTING.md) .
