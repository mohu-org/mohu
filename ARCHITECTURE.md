# Retrieval Architecture (Vector Search)

## Status

Planning stage. Tracked in [#286](https://github.com/mohu-org/mohu/issues/286).
No retrieval backend is implemented yet. This document exists to record the
design constraints and open questions before implementation starts, and to
claim the `mohu-retrieval` crate namespace so future contributors have a
clear home for this work.

## Background

mohu's core design pillars are:

- **Rayon parallelism by default** — operations use data-parallel execution
  without opt-in flags.
- **Arrow-native memory layout** — arrays share memory layout with Apache
  Arrow for zero-copy interop.
- **Zero-copy Python interop** — no unnecessary copies at the Rust/Python
  boundary.

Vector similarity search (approximate nearest-neighbor / ANN retrieval) is a
common downstream use case for an array library like mohu, but no retrieval
backend currently exists in this repository.

## Why this needs a design decision before implementation

The most widely-adopted ANN library, [FAISS](https://github.com/facebookresearch/faiss),
conflicts with mohu's design pillars in ways worth deciding on deliberately
rather than discovering after the fact:

1. **Not Arrow-native.** FAISS operates on contiguous float32 NumPy arrays
   with its own C++ memory layout. Using it from Arrow-backed mohu arrays
   would require an explicit copy at the index boundary (Arrow column →
   NumPy contiguous → FAISS), breaking the zero-copy goal.
2. **Single-threaded by default.** FAISS's Python bindings build indexes
   single-threaded (`IndexFlatL2`, `IndexIVFFlat`) unless
   `faiss.omp_set_num_threads()` is set explicitly, and it manages its own
   C++/OpenMP thread pool rather than integrating with Rayon.
3. **Not implemented in this repo.** There is currently no retrieval code
   (FAISS-based or otherwise) anywhere in this codebase to build on top of.

## Options under discussion

- **(a) A Rust-native ANN library** — e.g. [`usearch`](https://github.com/unum-cloud/usearch)
  or [`hnswlib`](https://github.com/nmslib/hnswlib) bindings — both have more
  Rust-friendly, Arrow-compatible buffer interfaces than FAISS.
- **(b) Wrap FAISS carefully** — with an explicit, clearly-documented
  Arrow → flat-copy boundary, accepting the copy as an unavoidable interop
  cost rather than pretending it doesn't exist.
- **(c) A native mohu implementation** — e.g. a Rayon-parallel HNSW index
  built as a `mohu-retrieval` crate, consistent with the rest of the
  workspace's "parallel by default" design.

No decision has been made yet — see issue #286 for ongoing discussion.

## This crate (`crates/mohu-retrieval`)

Currently an empty placeholder. It exists to:

- Reserve the `mohu-retrieval` namespace in the workspace.
- Signal where Arrow-native vector search is intended to live once a design
  is chosen.
- Give contributors following up on #286 a starting point instead of an
  ambiguous "where does this go?" question.

No behavior is implemented. This is intentionally additive-only.