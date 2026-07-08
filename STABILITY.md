# API Stability Policy

## Overview

The Mohu workspace is in an early stage of development. As the project evolves, APIs may change as functionality is added and refined.

## Current Status

Unless explicitly documented otherwise, public APIs should be considered **experimental** and may change between releases.

The authoritative source for the stability status of each crate or public API is
`CRATE_MAP.md`. It identifies whether a crate or component is considered
**Stable**, **Experimental**, or **Internal**. Any feature-specific exceptions
or stability overrides are also documented there.

## Future Stability Levels

As the project matures, APIs may be classified as:

- **Stable** – Intended for production use with compatibility guarantees.
- **Experimental** – Under active development and subject to change.
- **Internal** – Not intended for external use and may change without notice.

## Versioning

Mohu aims to follow Semantic Versioning (SemVer). Once APIs are designated as
stable, breaking changes will be reserved for major releases.

Refer to `CRATE_MAP.md` to determine the current stability classification of a
crate or API before relying on compatibility guarantees.
