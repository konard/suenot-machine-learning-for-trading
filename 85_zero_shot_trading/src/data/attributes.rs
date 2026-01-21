//! Asset attributes for zero-shot learning.

use serde::{Deserialize, Serialize};

/// Asset type categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssetType {
    Cryptocurrency,
    Stock,
    Forex,
    Commodity,
    Index,
}

/// Market capitalization tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketCapTier {
    Large,
    Mid,
    Small,
    Micro,
}

/// Volatility regime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VolatilityRegime {
    Low,
    Medium,
    High,
    Extreme,
}

/// Sector classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sector {
    Technology,
    Finance,
    Healthcare,
    Energy,
    Consumer,
    Industrial,
    Materials,
    Utilities,
    RealEstate,
    Communications,
    DeFi,
    Layer1,
    Layer2,
    Meme,
    Other,
}

/// Asset attributes for semantic description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetAttributes {
    /// Asset type (crypto, stock, etc.)
    pub asset_type: AssetType,
    /// Market cap tier
    pub market_cap: MarketCapTier,
    /// Current volatility regime
    pub volatility: VolatilityRegime,
    /// Sector classification
    pub sector: Sector,
    /// Average daily volume (normalized, 0-1)
    pub volume_normalized: f64,
    /// Correlation with BTC/SPY (benchmark)
    pub benchmark_correlation: f64,
    /// Average spread as percentage
    pub avg_spread: f64,
    /// Trading hours (24 for crypto, less for stocks)
    pub trading_hours: f64,
}

impl AssetAttributes {
    /// Create new asset attributes.
    pub fn new(
        asset_type: AssetType,
        market_cap: MarketCapTier,
        volatility: VolatilityRegime,
        sector: Sector,
    ) -> Self {
        Self {
            asset_type,
            market_cap,
            volatility,
            sector,
            volume_normalized: 0.5,
            benchmark_correlation: 0.5,
            avg_spread: 0.001,
            trading_hours: 24.0,
        }
    }

    /// Create attributes for a major cryptocurrency.
    pub fn major_crypto(sector: Sector) -> Self {
        Self {
            asset_type: AssetType::Cryptocurrency,
            market_cap: MarketCapTier::Large,
            volatility: VolatilityRegime::High,
            sector,
            volume_normalized: 0.9,
            benchmark_correlation: 0.8,
            avg_spread: 0.0001,
            trading_hours: 24.0,
        }
    }

    /// Create attributes for a small cap crypto.
    pub fn small_cap_crypto(sector: Sector) -> Self {
        Self {
            asset_type: AssetType::Cryptocurrency,
            market_cap: MarketCapTier::Small,
            volatility: VolatilityRegime::Extreme,
            sector,
            volume_normalized: 0.3,
            benchmark_correlation: 0.5,
            avg_spread: 0.005,
            trading_hours: 24.0,
        }
    }

    /// Create attributes for a large cap stock.
    pub fn large_cap_stock(sector: Sector) -> Self {
        Self {
            asset_type: AssetType::Stock,
            market_cap: MarketCapTier::Large,
            volatility: VolatilityRegime::Low,
            sector,
            volume_normalized: 0.8,
            benchmark_correlation: 0.7,
            avg_spread: 0.0005,
            trading_hours: 6.5, // US market hours
        }
    }

    /// Convert to numerical vector for encoding.
    pub fn to_numerical(&self) -> Vec<f64> {
        vec![
            self.volume_normalized,
            self.benchmark_correlation,
            self.avg_spread * 100.0, // Scale up small values
            self.trading_hours / 24.0,
        ]
    }

    /// Get categorical indices for embedding lookup.
    pub fn to_categorical_indices(&self) -> Vec<usize> {
        vec![
            self.asset_type as usize,
            self.market_cap as usize,
            self.volatility as usize,
            self.sector as usize,
        ]
    }

    /// Get total number of categories for each attribute type.
    pub fn category_sizes() -> Vec<usize> {
        vec![
            5,  // AssetType variants
            4,  // MarketCapTier variants
            4,  // VolatilityRegime variants
            15, // Sector variants
        ]
    }
}

/// Registry of known assets and their attributes.
#[derive(Debug, Clone, Default)]
pub struct AssetRegistry {
    assets: std::collections::HashMap<String, AssetAttributes>,
}

impl AssetRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with common crypto assets.
    pub fn with_crypto_defaults() -> Self {
        let mut registry = Self::new();

        // Major cryptocurrencies
        registry.register("BTCUSDT", AssetAttributes::major_crypto(Sector::Layer1));
        registry.register("ETHUSDT", AssetAttributes::major_crypto(Sector::Layer1));
        registry.register("BNBUSDT", AssetAttributes::major_crypto(Sector::Layer1));
        registry.register("SOLUSDT", AssetAttributes::major_crypto(Sector::Layer1));
        registry.register("ADAUSDT", AssetAttributes::major_crypto(Sector::Layer1));

        // Layer 2
        registry.register("MATICUSDT", AssetAttributes {
            asset_type: AssetType::Cryptocurrency,
            market_cap: MarketCapTier::Mid,
            volatility: VolatilityRegime::High,
            sector: Sector::Layer2,
            volume_normalized: 0.6,
            benchmark_correlation: 0.7,
            avg_spread: 0.0005,
            trading_hours: 24.0,
        });
        registry.register("ARBUSDT", AssetAttributes {
            asset_type: AssetType::Cryptocurrency,
            market_cap: MarketCapTier::Mid,
            volatility: VolatilityRegime::High,
            sector: Sector::Layer2,
            volume_normalized: 0.5,
            benchmark_correlation: 0.65,
            avg_spread: 0.0008,
            trading_hours: 24.0,
        });

        // DeFi
        registry.register("UNIUSDT", AssetAttributes {
            asset_type: AssetType::Cryptocurrency,
            market_cap: MarketCapTier::Mid,
            volatility: VolatilityRegime::High,
            sector: Sector::DeFi,
            volume_normalized: 0.5,
            benchmark_correlation: 0.6,
            avg_spread: 0.001,
            trading_hours: 24.0,
        });
        registry.register("AAVEUSDT", AssetAttributes {
            asset_type: AssetType::Cryptocurrency,
            market_cap: MarketCapTier::Mid,
            volatility: VolatilityRegime::High,
            sector: Sector::DeFi,
            volume_normalized: 0.4,
            benchmark_correlation: 0.55,
            avg_spread: 0.001,
            trading_hours: 24.0,
        });

        // Meme coins
        registry.register("DOGEUSDT", AssetAttributes::small_cap_crypto(Sector::Meme));
        registry.register("SHIBUSDT", AssetAttributes::small_cap_crypto(Sector::Meme));

        registry
    }

    /// Register an asset with its attributes.
    pub fn register(&mut self, symbol: &str, attributes: AssetAttributes) {
        self.assets.insert(symbol.to_string(), attributes);
    }

    /// Get attributes for an asset.
    pub fn get(&self, symbol: &str) -> Option<&AssetAttributes> {
        self.assets.get(symbol)
    }

    /// Get or create default attributes for an asset.
    pub fn get_or_default(&self, symbol: &str) -> AssetAttributes {
        self.assets.get(symbol).cloned().unwrap_or_else(|| {
            // Infer attributes from symbol
            if symbol.ends_with("USDT") || symbol.ends_with("USD") {
                AssetAttributes::small_cap_crypto(Sector::Other)
            } else {
                AssetAttributes::large_cap_stock(Sector::Other)
            }
        })
    }

    /// List all registered symbols.
    pub fn symbols(&self) -> Vec<&String> {
        self.assets.keys().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_attributes() {
        let attrs = AssetAttributes::major_crypto(Sector::Layer1);
        assert_eq!(attrs.asset_type, AssetType::Cryptocurrency);
        assert_eq!(attrs.market_cap, MarketCapTier::Large);

        let numerical = attrs.to_numerical();
        assert_eq!(numerical.len(), 4);

        let categorical = attrs.to_categorical_indices();
        assert_eq!(categorical.len(), 4);
    }

    #[test]
    fn test_asset_registry() {
        let registry = AssetRegistry::with_crypto_defaults();

        let btc = registry.get("BTCUSDT");
        assert!(btc.is_some());
        assert_eq!(btc.unwrap().sector, Sector::Layer1);

        let unknown = registry.get_or_default("UNKNOWNUSDT");
        assert_eq!(unknown.asset_type, AssetType::Cryptocurrency);
    }
}
