/// Cache performance counters for observability.
#[derive(Debug, Default)]
pub struct CacheMetrics {
    pub hits: u64,
    pub misses: u64,
}

impl CacheMetrics {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}
