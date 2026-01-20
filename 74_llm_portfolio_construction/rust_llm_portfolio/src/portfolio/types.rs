//! Portfolio-related type definitions

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Asset class enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetClass {
    Equity,
    Crypto,
    Bond,
    Commodity,
    Cash,
}

impl std::fmt::Display for AssetClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssetClass::Equity => write!(f, "equity"),
            AssetClass::Crypto => write!(f, "crypto"),
            AssetClass::Bond => write!(f, "bond"),
            AssetClass::Commodity => write!(f, "commodity"),
            AssetClass::Cash => write!(f, "cash"),
        }
    }
}

/// Represents a tradeable asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub symbol: String,
    pub name: String,
    pub asset_class: AssetClass,
    pub current_price: f64,
    pub market_cap: Option<f64>,
    pub sector: Option<String>,
}

impl Asset {
    /// Create a new asset
    pub fn new(symbol: &str, name: &str, asset_class: AssetClass, current_price: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            name: name.to_string(),
            asset_class,
            current_price,
            market_cap: None,
            sector: None,
        }
    }

    /// Create a new asset with all fields
    pub fn with_details(
        symbol: &str,
        name: &str,
        asset_class: AssetClass,
        current_price: f64,
        market_cap: Option<f64>,
        sector: Option<String>,
    ) -> Self {
        Self {
            symbol: symbol.to_string(),
            name: name.to_string(),
            asset_class,
            current_price,
            market_cap,
            sector,
        }
    }
}

/// Confidence level for LLM analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Confidence {
    Low,
    Medium,
    High,
}

impl std::fmt::Display for Confidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Confidence::Low => write!(f, "low"),
            Confidence::Medium => write!(f, "medium"),
            Confidence::High => write!(f, "high"),
        }
    }
}

/// LLM-generated scores for an asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetScore {
    pub symbol: String,
    pub fundamental_score: f64, // 1-10
    pub momentum_score: f64,    // 1-10
    pub sentiment_score: f64,   // 1-10
    pub risk_score: f64,        // 1-10 (higher = more risk)
    pub overall_score: f64,     // Weighted combination
    pub reasoning: String,      // LLM explanation
    pub confidence: Confidence, // low/medium/high
}

impl AssetScore {
    /// Calculate weighted composite score
    pub fn composite_score(&self) -> f64 {
        // Higher is better, so invert risk score
        let weights = (0.30, 0.25, 0.25, 0.20); // fundamental, momentum, sentiment, risk

        weights.0 * self.fundamental_score
            + weights.1 * self.momentum_score
            + weights.2 * self.sentiment_score
            + weights.3 * (10.0 - self.risk_score)
    }

    /// Create a default score
    pub fn default_for(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            fundamental_score: 5.0,
            momentum_score: 5.0,
            sentiment_score: 5.0,
            risk_score: 5.0,
            overall_score: 5.0,
            reasoning: "Default score".to_string(),
            confidence: Confidence::Low,
        }
    }
}

/// Represents a portfolio allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    pub weights: HashMap<String, f64>, // symbol -> weight
    pub cash_weight: f64,
    pub timestamp: String,
    pub metadata: HashMap<String, String>,
}

impl Portfolio {
    /// Create a new portfolio
    pub fn new(weights: HashMap<String, f64>) -> Self {
        let mut portfolio = Self {
            weights,
            cash_weight: 0.0,
            timestamp: chrono::Utc::now().to_rfc3339(),
            metadata: HashMap::new(),
        };
        portfolio.normalize();
        portfolio
    }

    /// Create a new portfolio with cash allocation
    pub fn with_cash(weights: HashMap<String, f64>, cash_weight: f64) -> Self {
        let mut portfolio = Self {
            weights,
            cash_weight,
            timestamp: chrono::Utc::now().to_rfc3339(),
            metadata: HashMap::new(),
        };
        portfolio.normalize();
        portfolio
    }

    /// Normalize weights to sum to 1
    fn normalize(&mut self) {
        let total: f64 = self.weights.values().sum::<f64>() + self.cash_weight;
        if total > 0.0 {
            for weight in self.weights.values_mut() {
                *weight /= total;
            }
            self.cash_weight /= total;
        }
    }

    /// Get weight for a symbol
    pub fn get_weight(&self, symbol: &str) -> f64 {
        *self.weights.get(symbol).unwrap_or(&0.0)
    }

    /// Get list of symbols in portfolio
    pub fn symbols(&self) -> Vec<&String> {
        self.weights.keys().collect()
    }

    /// Get sorted weights (highest first)
    pub fn sorted_weights(&self) -> Vec<(&String, &f64)> {
        let mut weights: Vec<_> = self.weights.iter().collect();
        weights.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        weights
    }

    /// Create equal-weight portfolio
    pub fn equal_weight(symbols: &[&str]) -> Self {
        let weight = 1.0 / symbols.len() as f64;
        let weights: HashMap<String, f64> = symbols.iter().map(|s| (s.to_string(), weight)).collect();
        Self::new(weights)
    }
}

impl std::fmt::Display for Portfolio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Portfolio Allocation:")?;
        for (symbol, weight) in self.sorted_weights() {
            writeln!(f, "  {}: {:.2}%", symbol, weight * 100.0)?;
        }
        if self.cash_weight > 0.0 {
            writeln!(f, "  CASH: {:.2}%", self.cash_weight * 100.0)?;
        }
        Ok(())
    }
}

/// Portfolio constraints for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioConstraints {
    pub max_weight: f64,
    pub min_weight: f64,
    pub max_assets: usize,
    pub min_score: f64,
    pub long_only: bool,
}

impl Default for PortfolioConstraints {
    fn default() -> Self {
        Self {
            max_weight: 0.30,
            min_weight: 0.02,
            max_assets: 10,
            min_score: 4.0,
            long_only: true,
        }
    }
}

/// Market data for an asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub return_7d: f64,
    pub return_30d: f64,
    pub volatility: f64,
    pub volume_24h: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_normalization() {
        let mut weights = HashMap::new();
        weights.insert("BTC".to_string(), 50.0);
        weights.insert("ETH".to_string(), 30.0);
        weights.insert("SOL".to_string(), 20.0);

        let portfolio = Portfolio::new(weights);

        let total: f64 = portfolio.weights.values().sum();
        assert!((total - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_equal_weight_portfolio() {
        let symbols = vec!["BTC", "ETH", "SOL", "BNB"];
        let portfolio = Portfolio::equal_weight(&symbols);

        for symbol in &symbols {
            let weight = portfolio.get_weight(*symbol);
            assert!((weight - 0.25).abs() < 0.0001);
        }
    }

    #[test]
    fn test_asset_score_composite() {
        let score = AssetScore {
            symbol: "BTC".to_string(),
            fundamental_score: 8.0,
            momentum_score: 7.0,
            sentiment_score: 8.0,
            risk_score: 4.0,
            overall_score: 7.5,
            reasoning: "Strong asset".to_string(),
            confidence: Confidence::High,
        };

        let composite = score.composite_score();
        // 0.30*8 + 0.25*7 + 0.25*8 + 0.20*(10-4) = 2.4 + 1.75 + 2.0 + 1.2 = 7.35
        assert!((composite - 7.35).abs() < 0.01);
    }
}
