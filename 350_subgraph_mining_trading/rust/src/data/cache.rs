//! Data caching for market data.

use crate::data::{Candle, Ticker};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Cache entry with timestamp
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    data: T,
    timestamp: i64,
    ttl: i64,
}

impl<T: Clone> CacheEntry<T> {
    fn new(data: T, ttl: i64) -> Self {
        Self {
            data,
            timestamp: chrono::Utc::now().timestamp(),
            ttl,
        }
    }

    fn is_expired(&self) -> bool {
        let now = chrono::Utc::now().timestamp();
        now - self.timestamp > self.ttl
    }
}

/// In-memory cache for market data
#[derive(Debug, Clone)]
pub struct DataCache {
    /// Candle cache: (symbol, interval) -> candles
    candles: Arc<RwLock<HashMap<(String, String), CacheEntry<Vec<Candle>>>>>,
    /// Ticker cache: symbol -> ticker
    tickers: Arc<RwLock<HashMap<String, CacheEntry<Ticker>>>>,
    /// Default TTL for candles (seconds)
    candle_ttl: i64,
    /// Default TTL for tickers (seconds)
    ticker_ttl: i64,
}

impl DataCache {
    /// Create a new cache with default TTLs
    pub fn new() -> Self {
        Self {
            candles: Arc::new(RwLock::new(HashMap::new())),
            tickers: Arc::new(RwLock::new(HashMap::new())),
            candle_ttl: 300, // 5 minutes
            ticker_ttl: 10,  // 10 seconds
        }
    }

    /// Set candle TTL
    pub fn with_candle_ttl(mut self, ttl: i64) -> Self {
        self.candle_ttl = ttl;
        self
    }

    /// Set ticker TTL
    pub fn with_ticker_ttl(mut self, ttl: i64) -> Self {
        self.ticker_ttl = ttl;
        self
    }

    /// Get cached candles
    pub fn get_candles(&self, symbol: &str, interval: &str) -> Option<Vec<Candle>> {
        let cache = self.candles.read().ok()?;
        let key = (symbol.to_string(), interval.to_string());

        cache.get(&key).and_then(|entry| {
            if entry.is_expired() {
                None
            } else {
                Some(entry.data.clone())
            }
        })
    }

    /// Cache candles
    pub fn set_candles(&self, symbol: &str, interval: &str, candles: Vec<Candle>) {
        if let Ok(mut cache) = self.candles.write() {
            let key = (symbol.to_string(), interval.to_string());
            cache.insert(key, CacheEntry::new(candles, self.candle_ttl));
        }
    }

    /// Get cached ticker
    pub fn get_ticker(&self, symbol: &str) -> Option<Ticker> {
        let cache = self.tickers.read().ok()?;

        cache.get(symbol).and_then(|entry| {
            if entry.is_expired() {
                None
            } else {
                Some(entry.data.clone())
            }
        })
    }

    /// Cache ticker
    pub fn set_ticker(&self, ticker: Ticker) {
        if let Ok(mut cache) = self.tickers.write() {
            let symbol = ticker.symbol.clone();
            cache.insert(symbol, CacheEntry::new(ticker, self.ticker_ttl));
        }
    }

    /// Clear all cached data
    pub fn clear(&self) {
        if let Ok(mut cache) = self.candles.write() {
            cache.clear();
        }
        if let Ok(mut cache) = self.tickers.write() {
            cache.clear();
        }
    }

    /// Remove expired entries
    pub fn cleanup(&self) {
        if let Ok(mut cache) = self.candles.write() {
            cache.retain(|_, entry| !entry.is_expired());
        }
        if let Ok(mut cache) = self.tickers.write() {
            cache.retain(|_, entry| !entry.is_expired());
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let candle_count = self.candles.read().map(|c| c.len()).unwrap_or(0);
        let ticker_count = self.tickers.read().map(|c| c.len()).unwrap_or(0);

        CacheStats {
            candle_entries: candle_count,
            ticker_entries: ticker_count,
        }
    }
}

impl Default for DataCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub candle_entries: usize,
    pub ticker_entries: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_candles() {
        let cache = DataCache::new().with_candle_ttl(60);

        let candles = vec![Candle::new(1609459200, 100.0, 110.0, 95.0, 105.0, 1000.0)];

        cache.set_candles("BTCUSDT", "1h", candles.clone());

        let cached = cache.get_candles("BTCUSDT", "1h");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 1);
    }

    #[test]
    fn test_cache_missing() {
        let cache = DataCache::new();

        let cached = cache.get_candles("NONEXISTENT", "1h");
        assert!(cached.is_none());
    }

    #[test]
    fn test_cache_clear() {
        let cache = DataCache::new();

        let candles = vec![Candle::new(1609459200, 100.0, 110.0, 95.0, 105.0, 1000.0)];
        cache.set_candles("BTCUSDT", "1h", candles);

        cache.clear();

        let cached = cache.get_candles("BTCUSDT", "1h");
        assert!(cached.is_none());
    }
}
