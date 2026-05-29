# mohu

**Rust-powered arrays for Python. Fast, parallel, and built for modern data workloads.**

mohu is an early-stage attempt to build a NumPy-compatible array system with its core implemented in Rust. The goal is to explore a faster, more parallel-friendly foundation for Python numerical computing.

---

## 🚀 Overview

mohu focuses on:

- Rust-backed array execution
- Parallel operations by default
- Efficient memory layouts using Apache Arrow
- Python interoperability via PyO3

The project is inspired by the idea that modern hardware and modern data workloads deserve a modern numerical computing foundation.

---

## ❓ Why This Exists

Python’s numerical ecosystem is powerful, but much of its foundation was designed decades ago.

Some common limitations include:

- NumPy is primarily single-threaded for many operations
- Object-based arrays introduce significant overhead
- Parallel execution is not the default model
- Memory layouts are constrained by legacy design decisions

Modern workloads increasingly demand:

- Parallel execution
- Cache-efficient memory layouts
- Zero-copy interoperability
- Better utilization of modern CPUs

mohu explores an alternative approach using Rust as the core execution engine.

---

## 🗺️ Roadmap

This project is in active early development.

### Planned Features

- N-dimensional arrays with a NumPy-compatible API
- Parallel operations powered by Rayon
- First-class string arrays (without `dtype=object`)
- Apache Arrow-based memory model
- Zero-copy Python bindings through PyO3
- SIMD-accelerated numerical operations
- Flexible memory layouts beyond NumPy constraints

---

## ⚡ Quick Start

> ⚠️ APIs are unstable and may change frequently.

### 1. Clone the Repository

```bash
git clone https://github.com/<your-fork>/mohu.git
cd mohu
```

### 2. Install Dependencies

Requirements:

- Rust (latest stable)
- Python 3.9+
- maturin

Install maturin:

```bash
pip install maturin
```

### 3. Build the Python Extension

```bash
maturin develop
```

This compiles the Rust core and exposes it as a Python package.

### 4. Example Usage

```python
import mohu as mh

arr = mh.array([1, 2, 3, 4])

print(arr)

# Placeholder example
result = arr
```

---

## 🏗️ Architecture

| Component | Purpose |
|------------|----------|
| Rust | Core compute engine |
| PyO3 | Python bindings |
| Rayon | Parallel execution |
| Apache Arrow | Columnar memory model |

---

## 🧠 Design Principles

### Parallel by Default

Operations should automatically take advantage of available CPU cores.

### Zero-Copy Where Possible

Avoid unnecessary memory duplication and movement.

### Interoperability First

Designed to integrate naturally with the Arrow ecosystem.

### Performance-Oriented

SIMD acceleration, efficient memory layouts, and cache locality are core priorities.

---

## 📦 Project Status

mohu is currently experimental.

| Area | Status |
|--------|---------|
| Core Array Engine | 🚧 In Progress |
| Python Bindings | 🚧 In Progress |
| API Stability | ⚠️ Not Stable |
| Production Ready | ❌ No |

Expect frequent changes as development continues.

---

## 🤝 Contributing

Contributions are welcome.

Areas where help is especially valuable:

- Documentation improvements
- Python API design feedback
- Rust performance optimizations
- Testing and benchmarking
- Developer tooling

### Development Workflow

```bash
git clone https://github.com/<your-fork>/mohu.git
cd mohu

git checkout -b feature/my-change

# Make your changes

git add .
git commit -m "feat: describe your change"

git push origin feature/my-change
```

Then open a Pull Request.

---

## 📄 License

Licensed under the MIT License.

See the `LICENSE` file for details.

---

## ⚠️ Important Note

mohu is currently experimental and should not be considered a drop-in replacement for NumPy.

The project is still laying its foundations, and substantial changes should be expected as development progresses.
