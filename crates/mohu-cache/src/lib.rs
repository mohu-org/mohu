//! Incremental query cache for repeated mohu operations.
//!
//! Stores retrieval/computation outputs keyed by a deterministic fingerprint
//! of the query and its configuration, with optional TTL-based expiry.

mod key;
pub mod metrics;
mod store;

pub use key::CacheKey;
pub use store::{CacheConfig, QueryCache};
#[cfg(test)]
mod tests;
