use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A deterministic fingerprint identifying a unique query + config combination.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey(u64);

impl CacheKey {
    /// Builds a cache key from a normalized query string, config bytes,
    /// and a graph/index version tag.
    pub fn new(query: &str, config: &[u8], version: u64) -> Self {
        let mut h = DefaultHasher::new();
        query.trim().to_lowercase().hash(&mut h);
        config.hash(&mut h);
        version.hash(&mut h);
        CacheKey(h.finish())
    }
}
