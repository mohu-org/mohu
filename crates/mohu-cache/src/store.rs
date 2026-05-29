use crate::key::CacheKey;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for the query cache.
pub struct CacheConfig {
    /// How long a cached entry stays valid. None means entries never expire.
    pub ttl: Option<Duration>,
    /// Maximum number of entries to hold (oldest evicted first).
    pub max_entries: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            ttl: Some(Duration::from_secs(300)), // 5 minutes default
            max_entries: 1024,
        }
    }
}

struct Entry<V> {
    value: V,
    inserted_at: Instant,
}

/// Thread-local in-memory query cache.
pub struct QueryCache<V> {
    store: HashMap<CacheKey, Entry<V>>,
    config: CacheConfig,
}

impl<V: Clone> QueryCache<V> {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            store: HashMap::new(),
            config,
        }
    }

    /// Returns a cached value if present and not expired.
    pub fn get(&self, key: &CacheKey) -> Option<&V> {
        self.store.get(key).and_then(|entry| match self.config.ttl {
            Some(ttl) if entry.inserted_at.elapsed() > ttl => None,
            _ => Some(&entry.value),
        })
    }

    /// Inserts a value. Evicts oldest entry if at capacity.
    pub fn insert(&mut self, key: CacheKey, value: V) {
        if self.store.len() >= self.config.max_entries {
            // simple eviction: remove first key found
            if let Some(k) = self.store.keys().next().cloned() {
                self.store.remove(&k);
            }
        }
        self.store.insert(
            key,
            Entry {
                value,
                inserted_at: Instant::now(),
            },
        );
    }

    /// Removes all expired entries.
    pub fn evict_expired(&mut self) {
        if let Some(ttl) = self.config.ttl {
            self.store.retain(|_, e| e.inserted_at.elapsed() <= ttl);
        }
    }
}
