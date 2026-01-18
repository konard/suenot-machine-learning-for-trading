//! Portfolio management for risk hedging
//!
//! Tracks positions and portfolio state

use super::hedging::{HedgeAllocation, HedgeInstrument};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Single position in portfolio
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub entry_time: DateTime<Utc>,
    pub position_type: PositionType,
}

impl Position {
    /// Create new position
    pub fn new(
        symbol: String,
        quantity: f64,
        entry_price: f64,
        position_type: PositionType,
    ) -> Self {
        Self {
            symbol,
            quantity,
            entry_price,
            current_price: entry_price,
            entry_time: Utc::now(),
            position_type,
        }
    }

    /// Update current price
    pub fn update_price(&mut self, price: f64) {
        self.current_price = price;
    }

    /// Calculate notional value
    pub fn notional_value(&self) -> f64 {
        self.quantity * self.current_price
    }

    /// Calculate unrealized PnL
    pub fn unrealized_pnl(&self) -> f64 {
        let price_change = self.current_price - self.entry_price;
        match self.position_type {
            PositionType::Long => self.quantity * price_change,
            PositionType::Short => -self.quantity * price_change,
        }
    }

    /// Calculate unrealized PnL percentage
    pub fn unrealized_pnl_pct(&self) -> f64 {
        if self.entry_price == 0.0 {
            return 0.0;
        }
        (self.current_price - self.entry_price) / self.entry_price * 100.0
    }
}

/// Position direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionType {
    Long,
    Short,
}

/// Portfolio state
#[derive(Debug, Clone)]
pub struct Portfolio {
    /// All positions
    pub positions: HashMap<String, Position>,
    /// Hedge positions
    pub hedge_positions: HashMap<HedgeInstrument, f64>,
    /// Cash/stablecoin balance
    pub cash_balance: f64,
    /// Total portfolio value at start
    pub initial_value: f64,
    /// Creation time
    pub created_at: DateTime<Utc>,
}

impl Portfolio {
    /// Create new portfolio with initial cash
    pub fn new(initial_cash: f64) -> Self {
        Self {
            positions: HashMap::new(),
            hedge_positions: HashMap::new(),
            cash_balance: initial_cash,
            initial_value: initial_cash,
            created_at: Utc::now(),
        }
    }

    /// Add position
    pub fn add_position(&mut self, position: Position) {
        let cost = position.notional_value();
        self.cash_balance -= cost;
        self.positions.insert(position.symbol.clone(), position);
    }

    /// Close position
    pub fn close_position(&mut self, symbol: &str) -> Option<f64> {
        if let Some(pos) = self.positions.remove(symbol) {
            let value = pos.notional_value();
            let pnl = pos.unrealized_pnl();
            self.cash_balance += value;
            Some(pnl)
        } else {
            None
        }
    }

    /// Update all position prices
    pub fn update_prices(&mut self, prices: &HashMap<String, f64>) {
        for (symbol, price) in prices {
            if let Some(pos) = self.positions.get_mut(symbol) {
                pos.update_price(*price);
            }
        }
    }

    /// Calculate total portfolio value
    pub fn total_value(&self) -> f64 {
        let positions_value: f64 = self.positions.values().map(|p| p.notional_value()).sum();
        let hedge_value: f64 = self.hedge_positions.values().sum();
        self.cash_balance + positions_value + hedge_value
    }

    /// Calculate total unrealized PnL
    pub fn total_unrealized_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.unrealized_pnl()).sum()
    }

    /// Calculate return since inception
    pub fn total_return_pct(&self) -> f64 {
        if self.initial_value == 0.0 {
            return 0.0;
        }
        (self.total_value() - self.initial_value) / self.initial_value * 100.0
    }

    /// Calculate drawdown from peak
    pub fn current_drawdown(&self, peak_value: f64) -> f64 {
        if peak_value == 0.0 {
            return 0.0;
        }
        (peak_value - self.total_value()) / peak_value * 100.0
    }

    /// Apply hedge allocation
    pub fn apply_hedge(&mut self, allocation: &HedgeAllocation) {
        let portfolio_value = self.total_value();
        let amounts = allocation.dollar_amounts(portfolio_value);

        for (instrument, amount) in amounts {
            // For stablecoins, convert from positions
            if instrument == HedgeInstrument::Stablecoin {
                // Already counted in cash_balance
                continue;
            }

            let current = *self.hedge_positions.get(&instrument).unwrap_or(&0.0);
            let diff = amount - current;

            if diff > 0.0 {
                // Need to add hedge
                self.cash_balance -= diff;
                self.hedge_positions.insert(instrument, amount);
            } else if diff < 0.0 {
                // Need to reduce hedge
                self.cash_balance -= diff; // diff is negative, so this adds
                self.hedge_positions.insert(instrument, amount);
            }
        }
    }

    /// Get hedge ratio (hedge value / total value)
    pub fn hedge_ratio(&self) -> f64 {
        let total = self.total_value();
        if total == 0.0 {
            return 0.0;
        }

        let hedge_value: f64 = self.hedge_positions.values().sum();
        hedge_value / total
    }

    /// Get exposure summary
    pub fn exposure_summary(&self) -> ExposureSummary {
        let long_exposure: f64 = self
            .positions
            .values()
            .filter(|p| p.position_type == PositionType::Long)
            .map(|p| p.notional_value())
            .sum();

        let short_exposure: f64 = self
            .positions
            .values()
            .filter(|p| p.position_type == PositionType::Short)
            .map(|p| p.notional_value())
            .sum();

        let hedge_exposure: f64 = self.hedge_positions.values().sum();

        ExposureSummary {
            long: long_exposure,
            short: short_exposure,
            hedge: hedge_exposure,
            net: long_exposure - short_exposure - hedge_exposure,
            gross: long_exposure + short_exposure + hedge_exposure,
            cash: self.cash_balance,
        }
    }

    /// Format portfolio summary
    pub fn format(&self) -> String {
        let mut lines = vec![
            "=== Portfolio Summary ===".to_string(),
            format!("Total Value: ${:.2}", self.total_value()),
            format!("Cash Balance: ${:.2}", self.cash_balance),
            format!("Unrealized PnL: ${:.2}", self.total_unrealized_pnl()),
            format!("Total Return: {:.2}%", self.total_return_pct()),
            format!("Hedge Ratio: {:.1}%", self.hedge_ratio() * 100.0),
            String::new(),
            "Positions:".to_string(),
        ];

        for (symbol, pos) in &self.positions {
            lines.push(format!(
                "  {} {:?}: {:.4} @ ${:.2} (PnL: ${:.2})",
                symbol,
                pos.position_type,
                pos.quantity,
                pos.current_price,
                pos.unrealized_pnl()
            ));
        }

        if !self.hedge_positions.is_empty() {
            lines.push(String::new());
            lines.push("Hedges:".to_string());
            for (instrument, amount) in &self.hedge_positions {
                lines.push(format!("  {:?}: ${:.2}", instrument, amount));
            }
        }

        lines.join("\n")
    }
}

/// Exposure summary
#[derive(Debug, Clone)]
pub struct ExposureSummary {
    pub long: f64,
    pub short: f64,
    pub hedge: f64,
    pub net: f64,
    pub gross: f64,
    pub cash: f64,
}

impl ExposureSummary {
    /// Format for display
    pub fn format(&self) -> String {
        format!(
            "Long: ${:.0} | Short: ${:.0} | Hedge: ${:.0} | Net: ${:.0} | Cash: ${:.0}",
            self.long, self.short, self.hedge, self.net, self.cash
        )
    }
}

/// Portfolio tracker for risk monitoring
#[derive(Debug, Clone)]
pub struct PortfolioTracker {
    /// Portfolio
    portfolio: Portfolio,
    /// Peak value (for drawdown)
    peak_value: f64,
    /// Value history
    value_history: Vec<(DateTime<Utc>, f64)>,
    /// Max history size
    max_history: usize,
}

impl PortfolioTracker {
    /// Create new tracker
    pub fn new(portfolio: Portfolio) -> Self {
        let initial_value = portfolio.total_value();
        Self {
            portfolio,
            peak_value: initial_value,
            value_history: vec![(Utc::now(), initial_value)],
            max_history: 1000,
        }
    }

    /// Update with new prices
    pub fn update(&mut self, prices: &HashMap<String, f64>) {
        self.portfolio.update_prices(prices);

        let current_value = self.portfolio.total_value();

        // Update peak
        if current_value > self.peak_value {
            self.peak_value = current_value;
        }

        // Record history
        self.value_history.push((Utc::now(), current_value));
        if self.value_history.len() > self.max_history {
            self.value_history.remove(0);
        }
    }

    /// Get current drawdown
    pub fn current_drawdown(&self) -> f64 {
        self.portfolio.current_drawdown(self.peak_value)
    }

    /// Get maximum drawdown from history
    pub fn max_drawdown(&self) -> f64 {
        if self.value_history.is_empty() {
            return 0.0;
        }

        let mut peak = self.value_history[0].1;
        let mut max_dd = 0.0;

        for (_, value) in &self.value_history {
            if *value > peak {
                peak = *value;
            }
            let dd = (peak - *value) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd * 100.0
    }

    /// Get portfolio reference
    pub fn portfolio(&self) -> &Portfolio {
        &self.portfolio
    }

    /// Get mutable portfolio reference
    pub fn portfolio_mut(&mut self) -> &mut Portfolio {
        &mut self.portfolio
    }

    /// Get risk metrics summary
    pub fn risk_metrics(&self) -> RiskMetrics {
        let returns: Vec<f64> = self
            .value_history
            .windows(2)
            .map(|w| (w[1].1 - w[0].1) / w[0].1)
            .collect();

        let volatility = if returns.len() > 1 {
            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 =
                returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
            variance.sqrt() * (252.0_f64).sqrt() // Annualized
        } else {
            0.0
        };

        RiskMetrics {
            total_value: self.portfolio.total_value(),
            peak_value: self.peak_value,
            current_drawdown: self.current_drawdown(),
            max_drawdown: self.max_drawdown(),
            volatility_annualized: volatility * 100.0,
            sharpe_ratio: 0.0, // Simplified
            hedge_ratio: self.portfolio.hedge_ratio(),
        }
    }
}

/// Risk metrics
#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub total_value: f64,
    pub peak_value: f64,
    pub current_drawdown: f64,
    pub max_drawdown: f64,
    pub volatility_annualized: f64,
    pub sharpe_ratio: f64,
    pub hedge_ratio: f64,
}

impl RiskMetrics {
    /// Format for display
    pub fn format(&self) -> String {
        format!(
            "Value: ${:.0} | Peak: ${:.0} | DD: {:.1}% | MaxDD: {:.1}% | Vol: {:.1}% | Hedge: {:.1}%",
            self.total_value,
            self.peak_value,
            self.current_drawdown,
            self.max_drawdown,
            self.volatility_annualized,
            self.hedge_ratio * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position() {
        let mut pos = Position::new("BTCUSDT".into(), 1.0, 50000.0, PositionType::Long);
        pos.update_price(55000.0);

        assert_eq!(pos.unrealized_pnl(), 5000.0);
        assert!((pos.unrealized_pnl_pct() - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_portfolio() {
        let mut portfolio = Portfolio::new(100_000.0);

        let pos = Position::new("BTCUSDT".into(), 0.5, 50000.0, PositionType::Long);
        portfolio.add_position(pos);

        assert!((portfolio.cash_balance - 75_000.0).abs() < 0.01);
        assert!((portfolio.total_value() - 100_000.0).abs() < 0.01);
    }

    #[test]
    fn test_portfolio_tracker() {
        let portfolio = Portfolio::new(100_000.0);
        let tracker = PortfolioTracker::new(portfolio);

        let metrics = tracker.risk_metrics();
        assert_eq!(metrics.total_value, 100_000.0);
        assert_eq!(metrics.current_drawdown, 0.0);
    }
}
