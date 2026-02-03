//! Trading performance metrics.

use serde::{Deserialize, Serialize};

/// A single trade record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Entry timestamp
    pub entry_time: i64,
    /// Exit timestamp
    pub exit_time: i64,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Trade direction (1 for long, -1 for short)
    pub direction: i32,
    /// Position size
    pub size: f64,
    /// Profit/loss in currency
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
}

impl Trade {
    /// Check if trade was profitable.
    pub fn is_profitable(&self) -> bool {
        self.pnl > 0.0
    }

    /// Calculate holding period in milliseconds.
    pub fn holding_period(&self) -> i64 {
        self.exit_time - self.entry_time
    }
}

/// Trading performance metrics calculator.
#[derive(Debug, Clone, Default)]
pub struct TradingMetrics {
    trades: Vec<Trade>,
    equity_curve: Vec<f64>,
    initial_capital: f64,
}

impl TradingMetrics {
    /// Create a new trading metrics calculator.
    pub fn new(initial_capital: f64) -> Self {
        Self {
            trades: Vec::new(),
            equity_curve: vec![initial_capital],
            initial_capital,
        }
    }

    /// Add a trade.
    pub fn add_trade(&mut self, trade: Trade) {
        let current_equity = *self.equity_curve.last().unwrap_or(&self.initial_capital);
        self.equity_curve.push(current_equity + trade.pnl);
        self.trades.push(trade);
    }

    /// Set equity curve directly.
    pub fn set_equity_curve(&mut self, equity: Vec<f64>) {
        self.equity_curve = equity;
    }

    /// Calculate total return.
    pub fn total_return(&self) -> f64 {
        if let Some(final_equity) = self.equity_curve.last() {
            (final_equity / self.initial_capital) - 1.0
        } else {
            0.0
        }
    }

    /// Calculate annualized return.
    pub fn annualized_return(&self, periods_per_year: f64) -> f64 {
        let total_return = self.total_return();
        let n_periods = self.equity_curve.len() as f64;

        if n_periods <= 1.0 {
            return 0.0;
        }

        let years = n_periods / periods_per_year;
        (1.0 + total_return).powf(1.0 / years) - 1.0
    }

    /// Calculate Sharpe ratio.
    pub fn sharpe_ratio(&self, periods_per_year: f64, risk_free_rate: f64) -> f64 {
        let returns = self.period_returns();

        if returns.len() < 2 {
            return 0.0;
        }

        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_dev = self.std_dev(&returns);

        if std_dev < 1e-10 {
            return 0.0;
        }

        let excess_return = mean_return - risk_free_rate / periods_per_year;
        excess_return * periods_per_year.sqrt() / std_dev
    }

    /// Calculate Sortino ratio.
    pub fn sortino_ratio(&self, periods_per_year: f64, risk_free_rate: f64) -> f64 {
        let returns = self.period_returns();

        if returns.len() < 2 {
            return 0.0;
        }

        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;

        // Calculate downside deviation
        let downside_returns: Vec<f64> = returns
            .iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();

        if downside_returns.is_empty() {
            return f64::INFINITY;
        }

        let downside_dev = self.std_dev(&downside_returns);

        if downside_dev < 1e-10 {
            return f64::INFINITY;
        }

        let excess_return = mean_return - risk_free_rate / periods_per_year;
        excess_return * periods_per_year.sqrt() / downside_dev
    }

    /// Calculate maximum drawdown.
    pub fn max_drawdown(&self) -> f64 {
        if self.equity_curve.len() < 2 {
            return 0.0;
        }

        let mut peak = self.equity_curve[0];
        let mut max_dd = 0.0;

        for &equity in &self.equity_curve {
            if equity > peak {
                peak = equity;
            }
            let drawdown = (peak - equity) / peak;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }

        max_dd
    }

    /// Calculate Calmar ratio.
    pub fn calmar_ratio(&self, periods_per_year: f64) -> f64 {
        let annual_return = self.annualized_return(periods_per_year);
        let max_dd = self.max_drawdown();

        if max_dd < 1e-10 {
            return f64::INFINITY;
        }

        annual_return / max_dd
    }

    /// Calculate win rate.
    pub fn win_rate(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }

        let winners = self.trades.iter().filter(|t| t.is_profitable()).count();
        winners as f64 / self.trades.len() as f64
    }

    /// Calculate profit factor.
    pub fn profit_factor(&self) -> f64 {
        let gross_profit: f64 = self
            .trades
            .iter()
            .filter(|t| t.pnl > 0.0)
            .map(|t| t.pnl)
            .sum();

        let gross_loss: f64 = self
            .trades
            .iter()
            .filter(|t| t.pnl < 0.0)
            .map(|t| t.pnl.abs())
            .sum();

        if gross_loss < 1e-10 {
            return f64::INFINITY;
        }

        gross_profit / gross_loss
    }

    /// Calculate average winning trade.
    pub fn average_win(&self) -> f64 {
        let winners: Vec<_> = self.trades.iter().filter(|t| t.pnl > 0.0).collect();

        if winners.is_empty() {
            return 0.0;
        }

        winners.iter().map(|t| t.return_pct).sum::<f64>() / winners.len() as f64
    }

    /// Calculate average losing trade.
    pub fn average_loss(&self) -> f64 {
        let losers: Vec<_> = self.trades.iter().filter(|t| t.pnl < 0.0).collect();

        if losers.is_empty() {
            return 0.0;
        }

        losers.iter().map(|t| t.return_pct).sum::<f64>() / losers.len() as f64
    }

    /// Get total number of trades.
    pub fn total_trades(&self) -> usize {
        self.trades.len()
    }

    /// Get number of winning trades.
    pub fn winning_trades(&self) -> usize {
        self.trades.iter().filter(|t| t.is_profitable()).count()
    }

    /// Get number of losing trades.
    pub fn losing_trades(&self) -> usize {
        self.trades.iter().filter(|t| !t.is_profitable()).count()
    }

    /// Get final equity.
    pub fn final_equity(&self) -> f64 {
        *self.equity_curve.last().unwrap_or(&self.initial_capital)
    }

    /// Get equity curve.
    pub fn equity_curve(&self) -> &[f64] {
        &self.equity_curve
    }

    /// Get all trades.
    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }

    /// Calculate period returns from equity curve.
    fn period_returns(&self) -> Vec<f64> {
        if self.equity_curve.len() < 2 {
            return Vec::new();
        }

        self.equity_curve
            .windows(2)
            .map(|w| w[1] / w[0] - 1.0)
            .collect()
    }

    /// Calculate standard deviation.
    fn std_dev(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (data.len() - 1) as f64;

        variance.sqrt()
    }

    /// Generate a summary report.
    pub fn summary(&self) -> TradingSummary {
        TradingSummary {
            total_trades: self.total_trades(),
            winning_trades: self.winning_trades(),
            losing_trades: self.losing_trades(),
            win_rate: self.win_rate(),
            total_return: self.total_return(),
            final_equity: self.final_equity(),
            max_drawdown: self.max_drawdown(),
            sharpe_ratio: self.sharpe_ratio(252.0, 0.0), // Assuming daily periods
            sortino_ratio: self.sortino_ratio(252.0, 0.0),
            calmar_ratio: self.calmar_ratio(252.0),
            profit_factor: self.profit_factor(),
            average_win: self.average_win(),
            average_loss: self.average_loss(),
        }
    }
}

/// Summary of trading performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSummary {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub total_return: f64,
    pub final_equity: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub profit_factor: f64,
    pub average_win: f64,
    pub average_loss: f64,
}

impl std::fmt::Display for TradingSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Trading Performance Summary")?;
        writeln!(f, "===========================")?;
        writeln!(f)?;
        writeln!(f, "Performance Metrics:")?;
        writeln!(f, "  Total Return: {:.2}%", self.total_return * 100.0)?;
        writeln!(f, "  Final Equity: ${:.2}", self.final_equity)?;
        writeln!(f, "  Sharpe Ratio: {:.3}", self.sharpe_ratio)?;
        writeln!(f, "  Sortino Ratio: {:.3}", self.sortino_ratio)?;
        writeln!(f, "  Max Drawdown: {:.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "  Calmar Ratio: {:.3}", self.calmar_ratio)?;
        writeln!(f)?;
        writeln!(f, "Trade Statistics:")?;
        writeln!(f, "  Total Trades: {}", self.total_trades)?;
        writeln!(f, "  Winning Trades: {}", self.winning_trades)?;
        writeln!(f, "  Losing Trades: {}", self.losing_trades)?;
        writeln!(f, "  Win Rate: {:.2}%", self.win_rate * 100.0)?;
        writeln!(f, "  Profit Factor: {:.3}", self.profit_factor)?;
        writeln!(f, "  Avg Win: {:.2}%", self.average_win * 100.0)?;
        writeln!(f, "  Avg Loss: {:.2}%", self.average_loss * 100.0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_return() {
        let mut metrics = TradingMetrics::new(100000.0);
        metrics.set_equity_curve(vec![100000.0, 105000.0, 110000.0]);

        assert!((metrics.total_return() - 0.10).abs() < 1e-10);
    }

    #[test]
    fn test_max_drawdown() {
        let mut metrics = TradingMetrics::new(100000.0);
        metrics.set_equity_curve(vec![100000.0, 110000.0, 88000.0, 100000.0]);

        // Peak was 110000, trough was 88000
        // Drawdown = (110000 - 88000) / 110000 = 0.2
        assert!((metrics.max_drawdown() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_win_rate() {
        let mut metrics = TradingMetrics::new(100000.0);

        metrics.add_trade(Trade {
            entry_time: 0,
            exit_time: 1,
            entry_price: 100.0,
            exit_price: 105.0,
            direction: 1,
            size: 1.0,
            pnl: 500.0,
            return_pct: 0.05,
        });

        metrics.add_trade(Trade {
            entry_time: 1,
            exit_time: 2,
            entry_price: 100.0,
            exit_price: 98.0,
            direction: 1,
            size: 1.0,
            pnl: -200.0,
            return_pct: -0.02,
        });

        assert!((metrics.win_rate() - 0.5).abs() < 1e-10);
    }
}
