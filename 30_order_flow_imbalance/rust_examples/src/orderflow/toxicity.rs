//! # Flow Toxicity Indicators
//!
//! Composite indicators for detecting informed trading and market stress.

use crate::data::orderbook::OrderBook;
use crate::data::trade::TradeStats;

/// Composite toxicity indicator
#[derive(Debug, Clone, Default)]
pub struct ToxicityIndicator {
    /// Current VPIN value
    pub vpin: f64,
    /// OFI Z-score
    pub ofi_zscore: f64,
    /// Spread Z-score
    pub spread_zscore: f64,
    /// Depth imbalance
    pub depth_imbalance: f64,
    /// Trade imbalance
    pub trade_imbalance: f64,
    /// Large trade indicator (0 or 1)
    pub large_trade_flag: f64,
    /// Composite toxicity score
    pub composite_score: f64,
}

impl ToxicityIndicator {
    /// Create a new toxicity indicator
    pub fn new() -> Self {
        Self::default()
    }

    /// Update with new data
    pub fn update(
        &mut self,
        vpin: Option<f64>,
        ofi_zscore: Option<f64>,
        orderbook: &OrderBook,
        trade_stats: &TradeStats,
        avg_spread: f64,
        std_spread: f64,
    ) {
        self.vpin = vpin.unwrap_or(0.0);
        self.ofi_zscore = ofi_zscore.unwrap_or(0.0);

        // Spread Z-score
        let current_spread = orderbook.spread_bps().unwrap_or(0.0);
        self.spread_zscore = if std_spread > 0.0 {
            (current_spread - avg_spread) / std_spread
        } else {
            0.0
        };

        // Depth imbalance
        self.depth_imbalance = orderbook.depth_imbalance(5);

        // Trade imbalance
        self.trade_imbalance = trade_stats.trade_imbalance();

        // Large trade flag
        self.large_trade_flag = if trade_stats.max_size > 2.0 * trade_stats.avg_size {
            1.0
        } else {
            0.0
        };

        // Calculate composite score
        self.calculate_composite();
    }

    /// Calculate composite toxicity score
    fn calculate_composite(&mut self) {
        // Weighted combination of indicators
        // Higher weights for VPIN and OFI as they are primary indicators

        let vpin_contrib = self.vpin * 0.30;
        let ofi_contrib = self.ofi_zscore.abs().min(3.0) / 3.0 * 0.25;
        let spread_contrib = self.spread_zscore.max(0.0).min(3.0) / 3.0 * 0.15;
        let depth_contrib = self.depth_imbalance.abs() * 0.15;
        let trade_contrib = self.trade_imbalance.abs() * 0.10;
        let large_trade_contrib = self.large_trade_flag * 0.05;

        self.composite_score = vpin_contrib
            + ofi_contrib
            + spread_contrib
            + depth_contrib
            + trade_contrib
            + large_trade_contrib;

        // Clamp to [0, 1]
        self.composite_score = self.composite_score.clamp(0.0, 1.0);
    }

    /// Get toxicity level
    pub fn level(&self) -> ToxicityLevel {
        if self.composite_score > 0.7 {
            ToxicityLevel::Critical
        } else if self.composite_score > 0.5 {
            ToxicityLevel::High
        } else if self.composite_score > 0.3 {
            ToxicityLevel::Elevated
        } else {
            ToxicityLevel::Normal
        }
    }

    /// Check if market conditions are dangerous
    pub fn is_dangerous(&self) -> bool {
        self.composite_score > 0.6
    }

    /// Get trading recommendation
    pub fn recommendation(&self) -> TradingRecommendation {
        match self.level() {
            ToxicityLevel::Critical => TradingRecommendation::AvoidTrading,
            ToxicityLevel::High => TradingRecommendation::ReduceSize,
            ToxicityLevel::Elevated => TradingRecommendation::Caution,
            ToxicityLevel::Normal => TradingRecommendation::Normal,
        }
    }
}

/// Toxicity level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToxicityLevel {
    Normal,
    Elevated,
    High,
    Critical,
}

impl std::fmt::Display for ToxicityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToxicityLevel::Normal => write!(f, "Normal"),
            ToxicityLevel::Elevated => write!(f, "Elevated"),
            ToxicityLevel::High => write!(f, "High"),
            ToxicityLevel::Critical => write!(f, "Critical"),
        }
    }
}

/// Trading recommendation based on toxicity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingRecommendation {
    Normal,
    Caution,
    ReduceSize,
    AvoidTrading,
}

impl std::fmt::Display for TradingRecommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradingRecommendation::Normal => write!(f, "Normal trading"),
            TradingRecommendation::Caution => write!(f, "Trade with caution"),
            TradingRecommendation::ReduceSize => write!(f, "Reduce position size"),
            TradingRecommendation::AvoidTrading => write!(f, "Avoid trading"),
        }
    }
}

/// Adverse selection calculator
#[derive(Debug)]
pub struct AdverseSelectionCalculator {
    /// Window size for calculations
    window: usize,
    /// Recent execution prices
    exec_prices: Vec<f64>,
    /// Recent mid prices at execution
    mid_prices: Vec<f64>,
    /// Recent trade sides
    trade_sides: Vec<f64>, // 1.0 for buy, -1.0 for sell
}

impl AdverseSelectionCalculator {
    /// Create a new calculator
    pub fn new(window: usize) -> Self {
        Self {
            window,
            exec_prices: Vec::with_capacity(window),
            mid_prices: Vec::with_capacity(window),
            trade_sides: Vec::with_capacity(window),
        }
    }

    /// Add a trade observation
    pub fn add_trade(&mut self, exec_price: f64, mid_price: f64, is_buy: bool) {
        self.exec_prices.push(exec_price);
        self.mid_prices.push(mid_price);
        self.trade_sides.push(if is_buy { 1.0 } else { -1.0 });

        // Keep only window size
        if self.exec_prices.len() > self.window {
            self.exec_prices.remove(0);
            self.mid_prices.remove(0);
            self.trade_sides.remove(0);
        }
    }

    /// Calculate average realized spread (adverse selection cost)
    ///
    /// Positive value means market makers are earning spread
    /// Negative value means adverse selection (informed traders winning)
    pub fn realized_spread(&self, forward_periods: usize) -> Option<f64> {
        if self.exec_prices.len() < forward_periods + 1 {
            return None;
        }

        let n = self.exec_prices.len() - forward_periods;
        let mut total_spread = 0.0;
        let mut count = 0;

        for i in 0..n {
            let exec = self.exec_prices[i];
            let mid_at_exec = self.mid_prices[i];
            let mid_future = self.mid_prices[i + forward_periods];
            let side = self.trade_sides[i];

            // Realized spread = 2 * side * (mid_future - exec)
            // Positive if price moved against the taker (good for MM)
            // Negative if price moved with the taker (adverse selection)
            let spread = 2.0 * side * (mid_future - exec) / mid_at_exec * 10000.0; // in bps

            total_spread += spread;
            count += 1;
        }

        if count > 0 {
            Some(total_spread / count as f64)
        } else {
            None
        }
    }

    /// Calculate adverse selection component
    pub fn adverse_selection(&self, forward_periods: usize) -> Option<f64> {
        self.realized_spread(forward_periods).map(|rs| -rs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::orderbook::OrderBookLevel;
    use chrono::Utc;

    #[test]
    fn test_toxicity_levels() {
        let mut indicator = ToxicityIndicator::new();

        indicator.composite_score = 0.2;
        assert_eq!(indicator.level(), ToxicityLevel::Normal);

        indicator.composite_score = 0.4;
        assert_eq!(indicator.level(), ToxicityLevel::Elevated);

        indicator.composite_score = 0.6;
        assert_eq!(indicator.level(), ToxicityLevel::High);

        indicator.composite_score = 0.8;
        assert_eq!(indicator.level(), ToxicityLevel::Critical);
    }

    #[test]
    fn test_trading_recommendations() {
        let mut indicator = ToxicityIndicator::new();

        indicator.composite_score = 0.1;
        assert_eq!(indicator.recommendation(), TradingRecommendation::Normal);

        indicator.composite_score = 0.75;
        assert_eq!(
            indicator.recommendation(),
            TradingRecommendation::AvoidTrading
        );
    }

    #[test]
    fn test_adverse_selection() {
        let mut calc = AdverseSelectionCalculator::new(100);

        // Add trades where price moves against taker (good for MM)
        for i in 0..10 {
            let exec = 100.0 + (i as f64 * 0.1);
            let mid = 100.0 + (i as f64 * 0.05); // Mid moves less than exec
            calc.add_trade(exec, mid, true);
        }

        // Adverse selection should be calculable
        let as_cost = calc.adverse_selection(1);
        assert!(as_cost.is_some());
    }
}
