//! Portfolio management and optimization.

use std::collections::HashMap;

/// Portfolio holding information
#[derive(Debug, Clone)]
pub struct Portfolio {
    /// Current positions (symbol -> quantity)
    pub positions: HashMap<String, f64>,
    /// Cash balance
    pub cash: f64,
    /// Initial capital
    pub initial_capital: f64,
}

impl Portfolio {
    /// Create a new portfolio with initial capital
    pub fn new(initial_capital: f64) -> Self {
        Self {
            positions: HashMap::new(),
            cash: initial_capital,
            initial_capital,
        }
    }

    /// Get total portfolio value given current prices
    pub fn total_value(&self, prices: &HashMap<String, f64>) -> f64 {
        let position_value: f64 = self
            .positions
            .iter()
            .map(|(symbol, &qty)| {
                let price = prices.get(symbol).copied().unwrap_or(0.0);
                qty * price
            })
            .sum();

        self.cash + position_value
    }

    /// Calculate portfolio return
    pub fn total_return(&self, prices: &HashMap<String, f64>) -> f64 {
        let current_value = self.total_value(prices);
        (current_value - self.initial_capital) / self.initial_capital
    }

    /// Update position
    pub fn update_position(&mut self, symbol: &str, quantity: f64, price: f64) {
        let current_qty = self.positions.get(symbol).copied().unwrap_or(0.0);
        let delta_qty = quantity - current_qty;
        let cost = delta_qty * price;

        if self.cash >= cost || delta_qty < 0.0 {
            self.positions.insert(symbol.to_string(), quantity);
            self.cash -= cost;
        }
    }

    /// Close all positions
    pub fn close_all(&mut self, prices: &HashMap<String, f64>) {
        for (symbol, &qty) in self.positions.clone().iter() {
            if let Some(&price) = prices.get(symbol) {
                self.cash += qty * price;
            }
        }
        self.positions.clear();
    }

    /// Get position weights
    pub fn weights(&self, prices: &HashMap<String, f64>) -> HashMap<String, f64> {
        let total = self.total_value(prices);
        if total <= 0.0 {
            return HashMap::new();
        }

        self.positions
            .iter()
            .map(|(symbol, &qty)| {
                let price = prices.get(symbol).copied().unwrap_or(0.0);
                let value = qty * price;
                (symbol.clone(), value / total)
            })
            .collect()
    }
}

/// Portfolio optimizer
pub struct PortfolioOptimizer {
    /// Risk aversion parameter
    risk_aversion: f64,
    /// Maximum position weight
    max_weight: f64,
    /// Minimum position weight
    min_weight: f64,
}

impl Default for PortfolioOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PortfolioOptimizer {
    /// Create a new optimizer
    pub fn new() -> Self {
        Self {
            risk_aversion: 1.0,
            max_weight: 0.25,
            min_weight: 0.0,
        }
    }

    /// Set risk aversion
    pub fn with_risk_aversion(mut self, risk_aversion: f64) -> Self {
        self.risk_aversion = risk_aversion;
        self
    }

    /// Set maximum weight
    pub fn with_max_weight(mut self, max_weight: f64) -> Self {
        self.max_weight = max_weight;
        self
    }

    /// Simple equal-weight allocation
    pub fn equal_weight(&self, symbols: &[String]) -> HashMap<String, f64> {
        let n = symbols.len();
        if n == 0 {
            return HashMap::new();
        }

        let weight = (1.0 / n as f64).min(self.max_weight);

        symbols.iter().map(|s| (s.clone(), weight)).collect()
    }

    /// Signal-based allocation
    pub fn signal_weight(
        &self,
        signals: &HashMap<String, f64>,
    ) -> HashMap<String, f64> {
        if signals.is_empty() {
            return HashMap::new();
        }

        // Normalize signals to weights
        let total_abs: f64 = signals.values().map(|s| s.abs()).sum();
        if total_abs <= 0.0 {
            return self.equal_weight(&signals.keys().cloned().collect::<Vec<_>>());
        }

        signals
            .iter()
            .map(|(symbol, &signal)| {
                let raw_weight = signal / total_abs;
                let clamped = raw_weight.max(-self.max_weight).min(self.max_weight);
                (symbol.clone(), clamped)
            })
            .collect()
    }

    /// Risk parity allocation (simplified)
    pub fn risk_parity(
        &self,
        volatilities: &HashMap<String, f64>,
    ) -> HashMap<String, f64> {
        if volatilities.is_empty() {
            return HashMap::new();
        }

        // Inverse volatility weighting
        let inv_vols: HashMap<String, f64> = volatilities
            .iter()
            .map(|(s, &v)| (s.clone(), if v > 0.0 { 1.0 / v } else { 0.0 }))
            .collect();

        let total_inv_vol: f64 = inv_vols.values().sum();
        if total_inv_vol <= 0.0 {
            return self.equal_weight(&volatilities.keys().cloned().collect::<Vec<_>>());
        }

        inv_vols
            .iter()
            .map(|(s, &iv)| {
                let weight = (iv / total_inv_vol).min(self.max_weight);
                (s.clone(), weight)
            })
            .collect()
    }

    /// Graph-aware allocation
    pub fn graph_weight(
        &self,
        centrality: &HashMap<String, f64>,
        signals: &HashMap<String, f64>,
    ) -> HashMap<String, f64> {
        // Combine centrality and signals
        let symbols: Vec<_> = centrality.keys().cloned().collect();

        let mut combined: HashMap<String, f64> = HashMap::new();
        for symbol in &symbols {
            let cent = centrality.get(symbol).copied().unwrap_or(0.0);
            let signal = signals.get(symbol).copied().unwrap_or(0.0);

            // Weight by both signal and centrality
            let combined_score = signal * (1.0 + cent * self.risk_aversion);
            combined.insert(symbol.clone(), combined_score);
        }

        self.signal_weight(&combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio() {
        let mut portfolio = Portfolio::new(10000.0);

        let mut prices = HashMap::new();
        prices.insert("BTC".to_string(), 40000.0);
        prices.insert("ETH".to_string(), 2000.0);

        // Buy some BTC
        portfolio.update_position("BTC", 0.1, 40000.0);

        assert!(portfolio.cash < 10000.0);
        assert_eq!(portfolio.positions.get("BTC"), Some(&0.1));

        let total = portfolio.total_value(&prices);
        assert!((total - 10000.0).abs() < 1.0);
    }

    #[test]
    fn test_equal_weight() {
        let optimizer = PortfolioOptimizer::new();
        let symbols = vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()];

        let weights = optimizer.equal_weight(&symbols);

        assert_eq!(weights.len(), 3);
        let total: f64 = weights.values().sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_signal_weight() {
        let optimizer = PortfolioOptimizer::new();

        let mut signals = HashMap::new();
        signals.insert("BTC".to_string(), 0.8);
        signals.insert("ETH".to_string(), 0.5);
        signals.insert("SOL".to_string(), -0.3);

        let weights = optimizer.signal_weight(&signals);

        // BTC should have highest weight
        assert!(weights["BTC"] > weights["ETH"]);
        assert!(weights["ETH"] > weights["SOL"]);
    }
}
