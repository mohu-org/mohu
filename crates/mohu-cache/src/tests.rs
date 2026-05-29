#[cfg(test)]
mod tests {
    use crate::{CacheConfig, CacheKey, QueryCache};
    use std::time::Duration;

    // Unit test: same inputs always produce same key
    #[test]
    fn cache_key_is_deterministic() {
        let k1 = CacheKey::new("my query", b"config", 1);
        let k2 = CacheKey::new("my query", b"config", 1);
        assert_eq!(k1, k2);
    }

    // Unit test: different inputs produce different keys
    #[test]
    fn cache_key_differs_on_different_input() {
        let k1 = CacheKey::new("query a", b"config", 1);
        let k2 = CacheKey::new("query b", b"config", 1);
        assert_ne!(k1, k2);
    }

    // Unit test: basic hit/miss
    #[test]
    fn cache_hit_and_miss() {
        let mut cache: QueryCache<String> = QueryCache::new(CacheConfig::default());
        let key = CacheKey::new("test", b"cfg", 0);
        assert!(cache.get(&key).is_none()); // miss
        cache.insert(key.clone(), "result".to_string());
        assert_eq!(cache.get(&key).unwrap(), "result"); // hit
    }

    // Unit test: TTL expiry
    #[test]
    fn cache_entry_expires_after_ttl() {
        let mut cache: QueryCache<String> = QueryCache::new(CacheConfig {
            ttl: Some(Duration::from_millis(1)),
            max_entries: 10,
        });
        let key = CacheKey::new("test", b"cfg", 0);
        cache.insert(key.clone(), "value".to_string());
        std::thread::sleep(Duration::from_millis(5));
        assert!(cache.get(&key).is_none()); // should have expired
    }

    // Unit test: graceful fallback (cache error doesn't crash caller)
    #[test]
    fn cache_miss_returns_none_not_panic() {
        let cache: QueryCache<String> = QueryCache::new(CacheConfig::default());
        let key = CacheKey::new("nonexistent", b"", 99);
        assert!(cache.get(&key).is_none()); // graceful, no panic
    }
}
