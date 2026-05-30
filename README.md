# Mohu

**Rust-powered arrays for Python. Fast, parallel, and built for the future.**

Mohu is an early-stage NumPy replacement with its core written in Rust. The goal is simple — take everything Python's scientific stack does and do it without the bottlenecks that have been accepted for decades. No GIL. No single-threaded ops. No object overhead. Just arrays.

---

## Table of Contents
- [Why Mohu?](#why-mohu)
- [What's Coming](#whats-coming)
- [Current Status](#current-status)
- [Installation & Setup](#installation--setup)
- [Usage Example](#usage-example)
- [Built With](#built-with)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Why Mohu?

NumPy is written in C and hasn't fundamentally changed in 20 years. It's single-threaded by default, its string arrays are an afterthought, and parallelism requires reaching for other tools. The Python data ecosystem deserves a better foundation.

Polars proved you can rewrite the data layer in Rust and win. Mohu is that same bet, one layer down.

## What's Coming

- N-dimensional arrays with a NumPy-compatible API
- Parallel operations by default via Rayon
- First-class string arrays — not `dtype=object`
- Built on Apache Arrow — interop with Polars, DuckDB, and the rest of the ecosystem out of the box
- Zero-copy Python integration via PyO3
- SIMD-accelerated math ops
- Memory layouts NumPy can't express

## Current Status

**Early.** The foundation is being laid. If you believe the Python numerical stack deserves a rewrite, watch this repository or consider contributing to our efforts!

## Installation & Setup

### Prerequisites
Before building Mohu, ensure you have the following installed on your system:
- [Rust](https://www.rust-lang.org/tools/install) (latest stable)
- [Python](https://www.python.org/downloads/) 3.10+
- [Maturin](https://maturin.rs/installation.html) (for building the Python bindings)

### Build Instructions
To build and install the extension locally for development:

1. Clone the repository:
   ```bash
   git clone https://github.com/mohu-org/mohu.git
   cd mohu
   ```

2. Create a virtual environment and install Maturin:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install maturin
   ```

3. Build the development package:
   ```bash
   maturin develop
   ```
   *(Note: The Python bindings crate must be correctly configured in the workspace for this to compile).*

## Usage Example

Once installed, you can import and use Mohu just like NumPy:

```python
import mohu as mh

# Create a fast, parallel array
arr = mh.array([1, 2, 3])
print(arr)
```

## Built With

Mohu is proudly built on the shoulders of incredible open-source projects:
- **[Rust](https://rust-lang.org)** — Systems programming language
- **[PyO3](https://github.com/PyO3/pyo3)** — Python bindings for Rust
- **[arrow-rs](https://github.com/apache/arrow-rs)** — Columnar memory format and execution
- **[Rayon](https://github.com/rayon-rs/rayon)** — Data parallelism

## Contributing

We welcome contributions of all kinds—from reporting bugs and writing documentation, to implementing SIMD kernels and expanding the API. 

Please review our [CONTRIBUTING.md](./CONTRIBUTING.md) for details on the Pull Request workflow, our code style rules, and how to get your environment set up. Also, check out [AGENTS.md](./AGENTS.md) if you are using AI tools to assist your workflow.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for full details.

## Acknowledgments

Special thanks to the open-source community, particularly the maintainers of Polars, NumPy, and Apache Arrow, whose work has heavily inspired Mohu.
