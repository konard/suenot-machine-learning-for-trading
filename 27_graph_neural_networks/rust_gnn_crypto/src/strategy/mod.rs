//! Trading strategy module.

mod momentum;

pub use momentum::{MomentumStrategy, Signal, TradingStrategy};

use std::collections::HashMap;

/// Trade record for backtesting.
#[derive(Debug, Clone)]
pub struct Trade {
    /// Timestamp of trade
    pub timestamp: i64,
    /// Symbol traded
    pub symbol: String,
    /// Trade action
    pub action: TradeAction,
    /// Position size (in quote currency)
    pub size: f64,
    /// Entry/exit price
    pub price: f64,
    /// PnL for closed trades
    pub pnl: Option<f64>,
    /// Confidence score
    pub confidence: f64,
}

/// Trade action type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeAction {
    /// Open long position
    Long,
    /// Open short position
    Short,
    /// Close position
    Close,
}

/// Open position.
#[derive(Debug, Clone)]
pub struct Position {
    /// Symbol
    pub symbol: String,
    /// Entry timestamp
    pub entry_time: i64,
    /// Entry price
    pub entry_price: f64,
    /// Position size
    pub size: f64,
    /// Direction (1 for long, -1 for short)
    pub direction: i32,
}

impl Position {
    /// Calculate current PnL given current price.
    pub fn current_pnl(&self, current_price: f64) -> f64 {
        let price_change = (current_price - self.entry_price) / self.entry_price;
        self.size * price_change * self.direction as f64
    }

    /// Calculate return percentage.
    pub fn return_pct(&self, current_price: f64) -> f64 {
        let price_change = (current_price - self.entry_price) / self.entry_price;
        price_change * self.direction as f64
    }
}

/// Portfolio for backtesting.
#[derive(Debug)]
pub struct Portfolio {
    /// Available capital
    pub capital: f64,
    /// Open positions
    pub positions: HashMap<String, Position>,
    /// Trade history
    pub trades: Vec<Trade>,
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// Transaction cost rate
    pub transaction_cost: f64,
    /// Maximum positions allowed
    pub max_positions: usize,
    /// Position size as fraction of capital
    pub position_size_pct: f64,
}

impl Portfolio {
    /// Create a new portfolio.
    pub fn new(
        initial_capital: f64,
        transaction_cost: f64,
        max_positions: usize,
        position_size_pct: f64,
    ) -> Self {
        Self {
            capital: initial_capital,
            positions: HashMap::new(),
            trades: Vec::new(),
            equity_curve: vec![initial_capital],
            transaction_cost,
            max_positions,
            position_size_pct,
        }
    }

    /// Open a new position.
    pub fn open_position(
        &mut self,
        symbol: &str,
        timestamp: i64,
        price: f64,
        direction: i32,
        confidence: f64,
    ) -> Option<Trade> {
        // Check if we can open a new position
        if self.positions.len() >= self.max_positions {
            return None;
        }

        if self.positions.contains_key(symbol) {
            return None;
        }

        // Calculate position size
        let size = self.capital * self.position_size_pct;
        let cost = size * self.transaction_cost;

        if size + cost > self.capital {
            return None;
        }

        // Deduct cost
        self.capital -= cost;

        let position = Position {
            symbol: symbol.to_string(),
            entry_time: timestamp,
            entry_price: price,
            size,
            direction,
        };

        self.positions.insert(symbol.to_string(), position);

        let trade = Trade {
            timestamp,
            symbol: symbol.to_string(),
            action: if direction > 0 {
                TradeAction::Long
            } else {
                TradeAction::Short
            },
            size,
            price,
            pnl: None,
            confidence,
        };

        self.trades.push(trade.clone());
        Some(trade)
    }

    /// Close a position.
    pub fn close_position(
        &mut self,
        symbol: &str,
        timestamp: i64,
        price: f64,
    ) -> Option<Trade> {
        let position = self.positions.remove(symbol)?;

        // Calculate PnL
        let pnl = position.current_pnl(price);
        let cost = position.size * self.transaction_cost;

        self.capital += position.size + pnl - cost;

        let trade = Trade {
            timestamp,
            symbol: symbol.to_string(),
            action: TradeAction::Close,
            size: position.size,
            price,
            pnl: Some(pnl),
            confidence: 0.0,
        };

        self.trades.push(trade.clone());
        Some(trade)
    }

    /// Update equity curve.
    pub fn update_equity(&mut self, current_prices: &HashMap<String, f64>) {
        let mut equity = self.capital;

        for (symbol, position) in &self.positions {
            if let Some(&price) = current_prices.get(symbol) {
                equity += position.size + position.current_pnl(price);
            }
        }

        self.equity_curve.push(equity);
    }

    /// Get current equity.
    pub fn current_equity(&self, current_prices: &HashMap<String, f64>) -> f64 {
        let mut equity = self.capital;

        for (symbol, position) in &self.positions {
            if let Some(&price) = current_prices.get(symbol) {
                equity += position.size + position.current_pnl(price);
            }
        }

        equity
    }

    /// Calculate strategy metrics.
    pub fn calculate_metrics(&self) -> BacktestMetrics {
        let initial = self.equity_curve.first().copied().unwrap_or(0.0);
        let final_equity = self.equity_curve.last().copied().unwrap_or(0.0);

        let total_return = (final_equity - initial) / initial;

        // Calculate daily returns
        let returns: Vec<f64> = self
            .equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean_return = if returns.is_empty() {
            0.0
        } else {
            returns.iter().sum::<f64>() / returns.len() as f64
        };

        let std_return = if returns.len() < 2 {
            0.0
        } else {
            let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
                / (returns.len() - 1) as f64;
            variance.sqrt()
        };

        let sharpe_ratio = if std_return > 0.0 {
            mean_return / std_return * (365.0_f64).sqrt()
        } else {
            0.0
        };

        // Maximum drawdown
        let mut max_equity = initial;
        let mut max_drawdown = 0.0;
        for &eq in &self.equity_curve {
            max_equity = max_equity.max(eq);
            let drawdown = (max_equity - eq) / max_equity;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Win rate
        let closed_trades: Vec<&Trade> = self
            .trades
            .iter()
            .filter(|t| t.action == TradeAction::Close)
            .collect();

        let wins = closed_trades.iter().filter(|t| t.pnl.unwrap_or(0.0) > 0.0).count();
        let win_rate = if closed_trades.is_empty() {
            0.0
        } else {
            wins as f64 / closed_trades.len() as f64
        };

        BacktestMetrics {
            total_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            num_trades: closed_trades.len(),
            avg_trade_pnl: closed_trades
                .iter()
                .filter_map(|t| t.pnl)
                .sum::<f64>()
                / closed_trades.len().max(1) as f64,
        }
    }
}

/// Backtest performance metrics.
#[derive(Debug, Clone)]
pub struct BacktestMetrics {
    /// Total return
    pub total_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Average trade PnL
    pub avg_trade_pnl: f64,
}

impl std::fmt::Display for BacktestMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Backtest Results:")?;
        writeln!(f, "  Total Return:    {:.2}%", self.total_return * 100.0)?;
        writeln!(f, "  Sharpe Ratio:    {:.2}", self.sharpe_ratio)?;
        writeln!(f, "  Max Drawdown:    {:.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "  Win Rate:        {:.2}%", self.win_rate * 100.0)?;
        writeln!(f, "  Number of Trades: {}", self.num_trades)?;
        writeln!(f, "  Avg Trade PnL:   ${:.2}", self.avg_trade_pnl)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio() {
        let mut portfolio = Portfolio::new(100000.0, 0.001, 5, 0.1);

        // Open a long position
        let trade = portfolio.open_position("BTCUSDT", 0, 50000.0, 1, 0.8);
        assert!(trade.is_some());
        assert_eq!(portfolio.positions.len(), 1);

        // Close the position at higher price
        let close = portfolio.close_position("BTCUSDT", 1, 51000.0);
        assert!(close.is_some());
        assert!(close.unwrap().pnl.unwrap() > 0.0);
        assert!(portfolio.positions.is_empty());
    }
}
