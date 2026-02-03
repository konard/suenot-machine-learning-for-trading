//! Backtesting engine for attention-based trading strategies.

use crate::model::{AttentionTransformer, AttentionWeights};
use crate::trading::{Signal, TradingStrategy};

/// A single trade record.
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_idx: usize,
    pub exit_idx: usize,
    pub entry_price: f64,
    pub exit_price: f64,
    pub signal: Signal,
    pub pnl: f64,
    pub confidence: f64,
}

/// Trading performance metrics.
#[derive(Debug, Clone)]
pub struct TradingMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub num_trades: usize,
    pub avg_trade_duration: f64,
    pub avg_confidence: f64,
}

impl TradingMetrics {
    /// Create metrics from backtest results.
    pub fn from_results(
        equity_curve: &[f64],
        trades: &[Trade],
        confidences: &[f64],
        n_samples: usize,
        initial_capital: f64,
    ) -> Self {
        // Compute returns
        let returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Total and annualized return
        let total_return = (equity_curve.last().unwrap_or(&initial_capital) - initial_capital)
            / initial_capital;
        let annualized_return = (1.0 + total_return).powf(252.0 / n_samples as f64) - 1.0;

        // Sharpe ratio (annualized)
        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
        let variance: f64 = returns
            .iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len().max(1) as f64;
        let std = variance.sqrt();
        let sharpe_ratio = if std > 0.0 {
            (mean_return / std) * (252_f64).sqrt()
        } else {
            0.0
        };

        // Sortino ratio (annualized)
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
        let downside_variance: f64 = downside_returns
            .iter()
            .map(|&r| r.powi(2))
            .sum::<f64>()
            / downside_returns.len().max(1) as f64;
        let downside_std = downside_variance.sqrt();
        let sortino_ratio = if downside_std > 0.0 {
            (mean_return / downside_std) * (252_f64).sqrt()
        } else {
            0.0
        };

        // Maximum drawdown
        let mut peak = equity_curve[0];
        let mut max_drawdown = 0.0;
        for &equity in equity_curve {
            if equity > peak {
                peak = equity;
            }
            let drawdown = (peak - equity) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Trade metrics
        let num_trades = trades.len();
        let (win_rate, profit_factor, avg_duration, avg_confidence) = if num_trades > 0 {
            let winning: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
            let losing: Vec<&Trade> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

            let win_rate = winning.len() as f64 / num_trades as f64;

            let gross_profit: f64 = winning.iter().map(|t| t.pnl).sum();
            let gross_loss: f64 = losing.iter().map(|t| t.pnl.abs()).sum();
            let profit_factor = if gross_loss > 0.0 {
                gross_profit / gross_loss
            } else {
                gross_profit
            };

            let avg_duration: f64 = trades
                .iter()
                .map(|t| (t.exit_idx - t.entry_idx) as f64)
                .sum::<f64>()
                / num_trades as f64;

            let avg_confidence: f64 =
                trades.iter().map(|t| t.confidence).sum::<f64>() / num_trades as f64;

            (win_rate, profit_factor, avg_duration, avg_confidence)
        } else {
            let avg_conf = if !confidences.is_empty() {
                confidences.iter().sum::<f64>() / confidences.len() as f64
            } else {
                0.0
            };
            (0.0, 0.0, 0.0, avg_conf)
        };

        TradingMetrics {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            num_trades,
            avg_trade_duration: avg_duration,
            avg_confidence,
        }
    }

    /// Print a summary of the metrics.
    pub fn print_summary(&self) {
        println!("=== Trading Metrics ===");
        println!("Total Return:        {:.2}%", self.total_return * 100.0);
        println!("Annualized Return:   {:.2}%", self.annualized_return * 100.0);
        println!("Sharpe Ratio:        {:.2}", self.sharpe_ratio);
        println!("Sortino Ratio:       {:.2}", self.sortino_ratio);
        println!("Max Drawdown:        {:.2}%", self.max_drawdown * 100.0);
        println!("Win Rate:            {:.2}%", self.win_rate * 100.0);
        println!("Profit Factor:       {:.2}", self.profit_factor);
        println!("Number of Trades:    {}", self.num_trades);
        println!("Avg Trade Duration:  {:.1} periods", self.avg_trade_duration);
        println!("Avg Confidence:      {:.2}", self.avg_confidence);
    }
}

/// Backtesting engine for attention-based strategies.
pub struct Backtester {
    model: AttentionTransformer,
    strategy: TradingStrategy,
    initial_capital: f64,
    transaction_cost: f64,
}

impl Backtester {
    /// Create a new backtester.
    pub fn new(
        model: AttentionTransformer,
        strategy: TradingStrategy,
        initial_capital: f64,
        transaction_cost: f64,
    ) -> Self {
        Self {
            model,
            strategy,
            initial_capital,
            transaction_cost,
        }
    }

    /// Run backtest on historical data.
    ///
    /// # Arguments
    ///
    /// * `sequences` - Feature sequences of shape (n_samples, seq_len, n_features)
    /// * `prices` - Price array of shape (n_samples,)
    ///
    /// # Returns
    ///
    /// Tuple of (metrics, trades, equity_curve)
    pub fn run(
        &self,
        sequences: &[Vec<Vec<f64>>],
        prices: &[f64],
    ) -> (TradingMetrics, Vec<Trade>, Vec<f64>) {
        let n_samples = sequences.len();
        let seq_len = sequences.first().map(|s| s.len()).unwrap_or(0);

        let mut equity = self.initial_capital;
        let mut equity_curve = vec![equity];
        let mut position = Signal::Hold;
        let mut entry_price = 0.0;
        let mut entry_idx = 0;
        let mut entry_confidence = 0.0;
        let mut trades = Vec::new();
        let mut confidences = Vec::new();

        for (i, seq) in sequences.iter().enumerate() {
            // Get model predictions
            let (pred_return, pred_direction, _pred_vol, attention) = self.model.forward(seq);

            // Generate signal
            let (signal, confidence) =
                self.strategy
                    .generate_signal(pred_return, pred_direction, Some(&attention), seq_len);
            confidences.push(confidence);

            let current_price = prices[i];

            // Handle position changes
            if position == Signal::Hold && signal != Signal::Hold {
                // Enter position
                position = signal;
                entry_price = current_price;
                entry_idx = i;
                entry_confidence = confidence;
                equity *= 1.0 - self.transaction_cost;
            } else if position != Signal::Hold && (signal != position || i == n_samples - 1) {
                // Exit position
                let pnl = match position {
                    Signal::Long => (current_price - entry_price) / entry_price,
                    Signal::Short => (entry_price - current_price) / entry_price,
                    Signal::Hold => 0.0,
                };

                equity *= 1.0 + pnl * self.strategy.position_size();
                equity *= 1.0 - self.transaction_cost;

                trades.push(Trade {
                    entry_idx,
                    exit_idx: i,
                    entry_price,
                    exit_price: current_price,
                    signal: position,
                    pnl,
                    confidence: entry_confidence,
                });

                // Enter new position if signal is not HOLD
                if signal != Signal::Hold && i < n_samples - 1 {
                    position = signal;
                    entry_price = current_price;
                    entry_idx = i;
                    entry_confidence = confidence;
                    equity *= 1.0 - self.transaction_cost;
                } else {
                    position = Signal::Hold;
                }
            }

            equity_curve.push(equity);
        }

        let metrics = TradingMetrics::from_results(
            &equity_curve,
            &trades,
            &confidences,
            n_samples,
            self.initial_capital,
        );

        (metrics, trades, equity_curve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TransformerConfig;

    #[test]
    fn test_backtest() {
        let config = TransformerConfig {
            input_dim: 8,
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            ..Default::default()
        };
        let model = AttentionTransformer::new(config);
        let strategy = TradingStrategy::new();
        let backtester = Backtester::new(model, strategy, 10000.0, 0.001);

        // Generate dummy data
        let seq_len = 30;
        let n_samples = 100;
        let sequences: Vec<Vec<Vec<f64>>> = (0..n_samples)
            .map(|_| {
                (0..seq_len)
                    .map(|_| (0..8).map(|_| rand::random::<f64>()).collect())
                    .collect()
            })
            .collect();

        let prices: Vec<f64> = (0..n_samples)
            .map(|i| 100.0 + (i as f64 * 0.1) + rand::random::<f64>())
            .collect();

        let (metrics, trades, equity_curve) = backtester.run(&sequences, &prices);

        assert_eq!(equity_curve.len(), n_samples + 1);
        assert!(metrics.max_drawdown >= 0.0 && metrics.max_drawdown <= 1.0);
    }
}
