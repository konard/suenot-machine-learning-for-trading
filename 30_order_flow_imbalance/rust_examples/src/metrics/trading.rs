//! # Trading Metrics
//!
//! Performance metrics for trading strategies.

use serde::{Deserialize, Serialize};

/// Trading performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TradingMetrics {
    /// Total number of trades
    pub total_trades: usize,
    /// Winning trades
    pub winning_trades: usize,
    /// Losing trades
    pub losing_trades: usize,
    /// Gross profit
    pub gross_profit: f64,
    /// Gross loss
    pub gross_loss: f64,
    /// Net P&L
    pub net_pnl: f64,
    /// Transaction costs
    pub costs: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Peak equity
    pub peak_equity: f64,
    /// Returns for Sharpe calculation
    returns: Vec<f64>,
}

impl TradingMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a trade
    pub fn record_trade(&mut self, pnl: f64, cost: f64) {
        self.total_trades += 1;
        self.costs += cost;

        let net = pnl - cost;

        if net > 0.0 {
            self.winning_trades += 1;
            self.gross_profit += pnl;
        } else {
            self.losing_trades += 1;
            self.gross_loss += (-pnl).max(0.0);
        }

        self.net_pnl += net;
        self.returns.push(net);

        // Update drawdown
        if self.net_pnl > self.peak_equity {
            self.peak_equity = self.net_pnl;
        }
        let drawdown = self.peak_equity - self.net_pnl;
        if drawdown > self.max_drawdown {
            self.max_drawdown = drawdown;
        }
    }

    /// Win rate
    pub fn win_rate(&self) -> f64 {
        if self.total_trades > 0 {
            self.winning_trades as f64 / self.total_trades as f64
        } else {
            0.0
        }
    }

    /// Profit factor
    pub fn profit_factor(&self) -> f64 {
        if self.gross_loss > 0.0 {
            self.gross_profit / self.gross_loss
        } else if self.gross_profit > 0.0 {
            f64::INFINITY
        } else {
            1.0
        }
    }

    /// Average win
    pub fn avg_win(&self) -> f64 {
        if self.winning_trades > 0 {
            self.gross_profit / self.winning_trades as f64
        } else {
            0.0
        }
    }

    /// Average loss
    pub fn avg_loss(&self) -> f64 {
        if self.losing_trades > 0 {
            self.gross_loss / self.losing_trades as f64
        } else {
            0.0
        }
    }

    /// Win/loss ratio
    pub fn win_loss_ratio(&self) -> f64 {
        let avg_loss = self.avg_loss();
        if avg_loss > 0.0 {
            self.avg_win() / avg_loss
        } else {
            f64::INFINITY
        }
    }

    /// Expectancy per trade
    pub fn expectancy(&self) -> f64 {
        if self.total_trades > 0 {
            self.net_pnl / self.total_trades as f64
        } else {
            0.0
        }
    }

    /// Sharpe ratio (assuming risk-free rate = 0)
    pub fn sharpe_ratio(&self) -> f64 {
        if self.returns.len() < 2 {
            return 0.0;
        }

        let n = self.returns.len() as f64;
        let mean: f64 = self.returns.iter().sum::<f64>() / n;
        let variance: f64 = self.returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        if std > 0.0 {
            mean / std * (252.0_f64).sqrt() // Annualized
        } else {
            0.0
        }
    }

    /// Sortino ratio (downside deviation)
    pub fn sortino_ratio(&self) -> f64 {
        if self.returns.len() < 2 {
            return 0.0;
        }

        let n = self.returns.len() as f64;
        let mean: f64 = self.returns.iter().sum::<f64>() / n;

        let downside_returns: Vec<f64> = self.returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        if downside_returns.is_empty() {
            return f64::INFINITY;
        }

        let downside_variance: f64 =
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64;
        let downside_std = downside_variance.sqrt();

        if downside_std > 0.0 {
            mean / downside_std * (252.0_f64).sqrt()
        } else {
            0.0
        }
    }

    /// Maximum drawdown percentage
    pub fn max_drawdown_pct(&self, initial_capital: f64) -> f64 {
        if initial_capital > 0.0 {
            self.max_drawdown / initial_capital * 100.0
        } else {
            0.0
        }
    }

    /// Calmar ratio (return / max drawdown)
    pub fn calmar_ratio(&self) -> f64 {
        if self.max_drawdown > 0.0 {
            self.net_pnl / self.max_drawdown
        } else if self.net_pnl > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        format!(
            r#"Trading Performance Summary
═══════════════════════════════════════
Total Trades:    {}
Winning Trades:  {} ({:.1}%)
Losing Trades:   {} ({:.1}%)

Gross Profit:    ${:.2}
Gross Loss:      ${:.2}
Net P&L:         ${:.2}
Transaction Costs: ${:.2}

Win Rate:        {:.1}%
Profit Factor:   {:.2}
Avg Win:         ${:.2}
Avg Loss:        ${:.2}
Win/Loss Ratio:  {:.2}
Expectancy:      ${:.2}

Sharpe Ratio:    {:.2}
Sortino Ratio:   {:.2}
Max Drawdown:    ${:.2}
Calmar Ratio:    {:.2}
═══════════════════════════════════════"#,
            self.total_trades,
            self.winning_trades,
            self.win_rate() * 100.0,
            self.losing_trades,
            (1.0 - self.win_rate()) * 100.0,
            self.gross_profit,
            self.gross_loss,
            self.net_pnl,
            self.costs,
            self.win_rate() * 100.0,
            self.profit_factor(),
            self.avg_win(),
            self.avg_loss(),
            self.win_loss_ratio(),
            self.expectancy(),
            self.sharpe_ratio(),
            self.sortino_ratio(),
            self.max_drawdown,
            self.calmar_ratio()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_metrics() {
        let mut metrics = TradingMetrics::new();

        // Simulate trades
        metrics.record_trade(100.0, 1.0); // Win: $99 net
        metrics.record_trade(-50.0, 1.0); // Loss: -$51 net
        metrics.record_trade(80.0, 1.0); // Win: $79 net
        metrics.record_trade(-30.0, 1.0); // Loss: -$31 net
        metrics.record_trade(120.0, 1.0); // Win: $119 net

        assert_eq!(metrics.total_trades, 5);
        assert_eq!(metrics.winning_trades, 3);
        assert_eq!(metrics.losing_trades, 2);

        // Win rate should be 60%
        assert!((metrics.win_rate() - 0.6).abs() < 0.01);

        // Net P&L should be positive
        assert!(metrics.net_pnl > 0.0);
    }

    #[test]
    fn test_sharpe_ratio() {
        let mut metrics = TradingMetrics::new();

        // Consistent small wins = high Sharpe
        for _ in 0..100 {
            metrics.record_trade(10.0, 0.1);
        }

        let sharpe = metrics.sharpe_ratio();
        assert!(sharpe > 0.0);
    }
}
