//! Backtest report generation

use super::engine::Trade;

/// Backtest report with performance metrics
#[derive(Debug, Clone)]
pub struct BacktestReport {
    // Basic metrics
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,

    // Trade statistics
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_trade_pnl: f64,
    pub avg_winner: f64,
    pub avg_loser: f64,

    // Time series
    pub equity_curve: Vec<(i64, f64)>,
    pub returns: Vec<f64>,
    pub drawdown: Vec<f64>,

    // Additional metrics
    pub volatility: f64,
    pub max_consecutive_wins: usize,
    pub max_consecutive_losses: usize,
}

impl BacktestReport {
    /// Generate report from trade history
    pub fn from_trades(
        trades: &[Trade],
        equity_history: &[(i64, f64)],
        initial_capital: f64,
    ) -> Self {
        let equity_curve = equity_history.to_vec();

        // Calculate returns
        let returns: Vec<f64> = if equity_history.len() > 1 {
            equity_history.windows(2)
                .map(|w| (w[1].1 - w[0].1) / w[0].1)
                .collect()
        } else {
            vec![]
        };

        // Calculate drawdown
        let mut peak = initial_capital;
        let drawdown: Vec<f64> = equity_history.iter()
            .map(|&(_, equity)| {
                if equity > peak { peak = equity; }
                (equity - peak) / peak
            })
            .collect();

        let max_drawdown = drawdown.iter().cloned().fold(0.0, f64::min);

        // Basic metrics
        let final_equity = equity_history.last().map(|e| e.1).unwrap_or(initial_capital);
        let total_return = (final_equity / initial_capital) - 1.0;

        // Sharpe ratio (annualized, assuming 1-min data)
        let mean_return: f64 = if returns.is_empty() { 0.0 } else {
            returns.iter().sum::<f64>() / returns.len() as f64
        };
        let std_return: f64 = if returns.len() > 1 {
            let variance: f64 = returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / returns.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };
        let sharpe_ratio = if std_return > 0.0 {
            mean_return / std_return * (525600.0_f64).sqrt()  // Annualize
        } else {
            0.0
        };

        // Sortino ratio
        let negative_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();
        let downside_std = if negative_returns.len() > 1 {
            let mean: f64 = negative_returns.iter().sum::<f64>() / negative_returns.len() as f64;
            (negative_returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / negative_returns.len() as f64).sqrt()
        } else {
            std_return
        };
        let sortino_ratio = if downside_std > 0.0 {
            mean_return / downside_std * (525600.0_f64).sqrt()
        } else {
            sharpe_ratio
        };

        // Calmar ratio
        let calmar_ratio = if max_drawdown != 0.0 {
            total_return / max_drawdown.abs()
        } else {
            0.0
        };

        // Trade statistics
        let winning_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let total_trades = trades.len();
        let n_winners = winning_trades.len();
        let n_losers = losing_trades.len();
        let win_rate = if total_trades > 0 { n_winners as f64 / total_trades as f64 } else { 0.0 };

        let gross_profit: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else { f64::INFINITY };

        let avg_trade_pnl = if total_trades > 0 {
            trades.iter().map(|t| t.pnl_pct).sum::<f64>() / total_trades as f64
        } else { 0.0 };

        let avg_winner = if n_winners > 0 {
            winning_trades.iter().map(|t| t.pnl_pct).sum::<f64>() / n_winners as f64
        } else { 0.0 };

        let avg_loser = if n_losers > 0 {
            losing_trades.iter().map(|t| t.pnl_pct).sum::<f64>() / n_losers as f64
        } else { 0.0 };

        // Consecutive wins/losses
        let (max_consecutive_wins, max_consecutive_losses) = Self::calculate_consecutive(trades);

        Self {
            total_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            total_trades,
            winning_trades: n_winners,
            losing_trades: n_losers,
            win_rate,
            profit_factor,
            avg_trade_pnl,
            avg_winner,
            avg_loser,
            equity_curve,
            returns,
            drawdown,
            volatility: std_return * (525600.0_f64).sqrt(),
            max_consecutive_wins,
            max_consecutive_losses,
        }
    }

    /// Calculate max consecutive wins and losses
    fn calculate_consecutive(trades: &[Trade]) -> (usize, usize) {
        let mut max_wins = 0;
        let mut max_losses = 0;
        let mut current_wins = 0;
        let mut current_losses = 0;

        for trade in trades {
            if trade.pnl > 0.0 {
                current_wins += 1;
                current_losses = 0;
                max_wins = max_wins.max(current_wins);
            } else {
                current_losses += 1;
                current_wins = 0;
                max_losses = max_losses.max(current_losses);
            }
        }

        (max_wins, max_losses)
    }

    /// Generate summary string
    pub fn summary(&self) -> String {
        format!(
            r#"
Backtest Results Summary
========================
Total Return:     {:.2}%
Sharpe Ratio:     {:.2}
Sortino Ratio:    {:.2}
Max Drawdown:     {:.2}%
Calmar Ratio:     {:.2}

Trade Statistics:
  Total Trades:   {}
  Win Rate:       {:.2}%
  Profit Factor:  {:.2}
  Avg Trade PnL:  {:.2}%
  Avg Winner:     {:.2}%
  Avg Loser:      {:.2}%
"#,
            self.total_return * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.calmar_ratio,
            self.total_trades,
            self.win_rate * 100.0,
            self.profit_factor,
            self.avg_trade_pnl * 100.0,
            self.avg_winner * 100.0,
            self.avg_loser * 100.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::signal::SignalType;
    use crate::backtest::engine::TradeStatus;

    #[test]
    fn test_report_generation() {
        let trades = vec![
            Trade {
                entry_time: 0,
                exit_time: Some(1),
                symbol: "TEST".to_string(),
                side: SignalType::Long,
                entry_price: 100.0,
                exit_price: Some(110.0),
                size: 1.0,
                pnl: 10.0,
                pnl_pct: 0.1,
                status: TradeStatus::Closed,
            },
            Trade {
                entry_time: 2,
                exit_time: Some(3),
                symbol: "TEST".to_string(),
                side: SignalType::Long,
                entry_price: 110.0,
                exit_price: Some(105.0),
                size: 1.0,
                pnl: -5.0,
                pnl_pct: -0.045,
                status: TradeStatus::Closed,
            },
        ];

        let equity_history = vec![
            (0, 10000.0),
            (1, 10100.0),
            (2, 10050.0),
            (3, 10025.0),
        ];

        let report = BacktestReport::from_trades(&trades, &equity_history, 10000.0);

        assert_eq!(report.total_trades, 2);
        assert_eq!(report.winning_trades, 1);
        assert_eq!(report.losing_trades, 1);
        assert_eq!(report.win_rate, 0.5);
    }
}
