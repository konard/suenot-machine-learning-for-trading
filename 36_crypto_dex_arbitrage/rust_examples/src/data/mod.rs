//! Data structures and utilities for DEX arbitrage
//!
//! This module provides common data types used throughout the library.

use serde::{Deserialize, Serialize};

/// A trading pair representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TradingPair {
    /// Base token (e.g., "ETH")
    pub base: String,
    /// Quote token (e.g., "USDC")
    pub quote: String,
}

impl TradingPair {
    /// Create a new trading pair
    pub fn new(base: &str, quote: &str) -> Self {
        Self {
            base: base.to_string(),
            quote: quote.to_string(),
        }
    }

    /// Get the pair symbol (e.g., "ETH/USDC")
    pub fn symbol(&self) -> String {
        format!("{}/{}", self.base, self.quote)
    }

    /// Get reversed pair
    pub fn reversed(&self) -> Self {
        Self {
            base: self.quote.clone(),
            quote: self.base.clone(),
        }
    }
}

impl std::fmt::Display for TradingPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.base, self.quote)
    }
}

/// Token pair (simplified without direction)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TokenPair {
    /// First token
    pub token_a: String,
    /// Second token
    pub token_b: String,
}

impl TokenPair {
    /// Create a new token pair
    pub fn new(token_a: &str, token_b: &str) -> Self {
        Self {
            token_a: token_a.to_string(),
            token_b: token_b.to_string(),
        }
    }

    /// Check if pair contains a specific token
    pub fn contains(&self, token: &str) -> bool {
        self.token_a == token || self.token_b == token
    }

    /// Get the other token in the pair
    pub fn other(&self, token: &str) -> Option<&str> {
        if self.token_a == token {
            Some(&self.token_b)
        } else if self.token_b == token {
            Some(&self.token_a)
        } else {
            None
        }
    }
}

/// Price data for a token pair on a specific DEX
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceData {
    /// DEX name
    pub dex: String,
    /// Trading pair
    pub pair: TradingPair,
    /// Current price (quote per base)
    pub price: f64,
    /// Available liquidity in base token
    pub liquidity_base: f64,
    /// Available liquidity in quote token
    pub liquidity_quote: f64,
    /// Timestamp (Unix milliseconds)
    pub timestamp: i64,
}

impl PriceData {
    /// Create new price data
    pub fn new(
        dex: &str,
        pair: TradingPair,
        price: f64,
        liquidity_base: f64,
        liquidity_quote: f64,
    ) -> Self {
        Self {
            dex: dex.to_string(),
            pair,
            price,
            liquidity_base,
            liquidity_quote,
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Total liquidity in quote terms
    pub fn total_liquidity_quote(&self) -> f64 {
        self.liquidity_base * self.price + self.liquidity_quote
    }

    /// Check if liquidity is sufficient for a trade size
    pub fn has_liquidity(&self, trade_size_base: f64) -> bool {
        self.liquidity_base >= trade_size_base
    }
}

/// DEX configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexConfig {
    /// DEX name
    pub name: String,
    /// Base swap fee rate
    pub fee_rate: f64,
    /// Estimated gas cost per swap (in ETH)
    pub gas_cost_eth: f64,
    /// Supported pairs
    pub pairs: Vec<TradingPair>,
}

impl DexConfig {
    /// Create a new DEX config
    pub fn new(name: &str, fee_rate: f64, gas_cost_eth: f64) -> Self {
        Self {
            name: name.to_string(),
            fee_rate,
            gas_cost_eth,
            pairs: Vec::new(),
        }
    }

    /// Add a supported pair
    pub fn add_pair(&mut self, pair: TradingPair) {
        self.pairs.push(pair);
    }

    /// Common DEX configurations
    pub fn uniswap_v2() -> Self {
        Self::new("UniswapV2", 0.003, 0.0003)
    }

    pub fn uniswap_v3() -> Self {
        Self::new("UniswapV3", 0.003, 0.00025)
    }

    pub fn sushiswap() -> Self {
        Self::new("SushiSwap", 0.003, 0.0003)
    }

    pub fn curve() -> Self {
        Self::new("Curve", 0.0004, 0.00035)
    }
}

/// Trade execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeResult {
    /// DEX where trade was executed
    pub dex: String,
    /// Trading pair
    pub pair: TradingPair,
    /// Amount in (input)
    pub amount_in: f64,
    /// Amount out (output)
    pub amount_out: f64,
    /// Effective price
    pub effective_price: f64,
    /// Slippage percentage
    pub slippage_pct: f64,
    /// Gas cost in ETH
    pub gas_cost_eth: f64,
    /// Success flag
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl TradeResult {
    /// Create a successful trade result
    pub fn success(
        dex: &str,
        pair: TradingPair,
        amount_in: f64,
        amount_out: f64,
        slippage_pct: f64,
        gas_cost_eth: f64,
    ) -> Self {
        Self {
            dex: dex.to_string(),
            pair,
            amount_in,
            amount_out,
            effective_price: amount_out / amount_in,
            slippage_pct,
            gas_cost_eth,
            success: true,
            error: None,
        }
    }

    /// Create a failed trade result
    pub fn failure(dex: &str, pair: TradingPair, amount_in: f64, error: &str) -> Self {
        Self {
            dex: dex.to_string(),
            pair,
            amount_in,
            amount_out: 0.0,
            effective_price: 0.0,
            slippage_pct: 0.0,
            gas_cost_eth: 0.0,
            success: false,
            error: Some(error.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_pair() {
        let pair = TradingPair::new("ETH", "USDC");
        assert_eq!(pair.symbol(), "ETH/USDC");

        let reversed = pair.reversed();
        assert_eq!(reversed.symbol(), "USDC/ETH");
    }

    #[test]
    fn test_token_pair() {
        let pair = TokenPair::new("ETH", "USDC");
        assert!(pair.contains("ETH"));
        assert!(pair.contains("USDC"));
        assert!(!pair.contains("BTC"));

        assert_eq!(pair.other("ETH"), Some("USDC"));
        assert_eq!(pair.other("BTC"), None);
    }

    #[test]
    fn test_price_data() {
        let pair = TradingPair::new("ETH", "USDC");
        let price = PriceData::new("Uniswap", pair, 2000.0, 1000.0, 2_000_000.0);

        assert_eq!(price.total_liquidity_quote(), 4_000_000.0);
        assert!(price.has_liquidity(500.0));
        assert!(!price.has_liquidity(1500.0));
    }
}
