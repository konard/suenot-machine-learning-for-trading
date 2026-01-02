//! # Backtesting Module
//!
//! Framework for backtesting score matching trading strategies.

use crate::trading::{ScoreMatchingTrader, TradingSignal};
use crate::utils::Candle;
use chrono::{DateTime, Utc};

/// Backtest result for a single trade
#[derive(Debug, Clone)]
pub struct Trade {
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Exit timestamp
    pub exit_time: DateTime<Utc>,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size (-1 to 1)
    pub position: f64,
    /// Profit/Loss
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
}

/// Daily backtest result
#[derive(Debug, Clone)]
pub struct DailyResult {
    /// Date
    pub timestamp: DateTime<Utc>,
    /// Closing price
    pub price: f64,
    /// Signal generated
    pub signal: f64,
    /// Confidence
    pub confidence: f64,
    /// Whether in distribution
    pub in_distribution: bool,
    /// Position held
    pub position: f64,
    /// Daily PnL
    pub pnl: f64,
    /// Cumulative PnL
    pub cumulative_pnl: f64,
}

/// Complete backtest results
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Daily results
    pub daily_results: Vec<DailyResult>,
    /// Individual trades
    pub trades: Vec<Trade>,
    /// Performance metrics
    pub metrics: BacktestMetrics,
}

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct BacktestMetrics {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Number of trades
    pub n_trades: usize,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Percentage of time in distribution
    pub in_distribution_ratio: f64,
    /// Average confidence when trading
    pub avg_confidence: f64,
}

/// Backtesting configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Maximum position size (as fraction of capital)
    pub max_position: f64,
    /// Trading cost per trade (as percentage)
    pub trading_cost: f64,
    /// Warmup period (candles before starting backtest)
    pub warmup: usize,
    /// Whether to allow shorting
    pub allow_short: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            max_position: 1.0,
            trading_cost: 0.001, // 0.1%
            warmup: 100,
            allow_short: true,
        }
    }
}

/// Backtester for score matching strategy
pub struct Backtester {
    /// Trader to backtest
    trader: ScoreMatchingTrader,
    /// Backtest configuration
    config: BacktestConfig,
}

impl Backtester {
    /// Create new backtester
    pub fn new(trader: ScoreMatchingTrader, config: BacktestConfig) -> Self {
        Self { trader, config }
    }

    /// Run backtest on price data
    ///
    /// # Arguments
    ///
    /// * `candles` - Historical candle data
    ///
    /// # Returns
    ///
    /// Backtest results with daily data and metrics
    pub fn run(&self, candles: &[Candle]) -> BacktestResult {
        let lookback = self.trader.config().lookback;
        let warmup = self.config.warmup.max(lookback + 1);

        if candles.len() < warmup + 1 {
            return BacktestResult {
                daily_results: vec![],
                trades: vec![],
                metrics: BacktestMetrics::default(),
            };
        }

        let mut daily_results = Vec::new();
        let mut trades = Vec::new();

        let mut position = 0.0;
        let mut cumulative_pnl = 0.0;
        let mut entry_price = 0.0;
        let mut entry_time = candles[warmup].timestamp;

        for i in warmup..candles.len() {
            let window = &candles[i.saturating_sub(lookback + 1)..=i];

            // Generate signal
            let signal = self.trader.generate_signal(window).unwrap_or(TradingSignal {
                signal: 0.0,
                confidence: 0.0,
                in_distribution: false,
                raw_score: vec![],
                denoised_state: None,
                timestamp: Some(candles[i].timestamp),
            });

            // Calculate PnL from previous position
            let pnl = if i > warmup && position != 0.0 {
                let price_return = (candles[i].close - candles[i - 1].close) / candles[i - 1].close;
                position * price_return * self.config.initial_capital
            } else {
                0.0
            };

            cumulative_pnl += pnl;

            // Determine new position
            let new_position = if signal.in_distribution && signal.confidence >= self.trader.config().confidence_threshold {
                let raw_position = signal.signal * self.config.max_position;
                if !self.config.allow_short && raw_position < 0.0 {
                    0.0
                } else {
                    raw_position.clamp(-self.config.max_position, self.config.max_position)
                }
            } else {
                0.0
            };

            // Record trade if position changed
            if position != 0.0 && new_position != position {
                let exit_price = candles[i].close;
                let trade_return = if position > 0.0 {
                    (exit_price - entry_price) / entry_price
                } else {
                    (entry_price - exit_price) / entry_price
                };

                trades.push(Trade {
                    entry_time,
                    exit_time: candles[i].timestamp,
                    entry_price,
                    exit_price,
                    position,
                    pnl: trade_return * position.abs() * self.config.initial_capital,
                    return_pct: trade_return * 100.0,
                });
            }

            // Update entry info if opening new position
            if new_position != 0.0 && position == 0.0 {
                entry_price = candles[i].close;
                entry_time = candles[i].timestamp;
            }

            position = new_position;

            daily_results.push(DailyResult {
                timestamp: candles[i].timestamp,
                price: candles[i].close,
                signal: signal.signal,
                confidence: signal.confidence,
                in_distribution: signal.in_distribution,
                position,
                pnl,
                cumulative_pnl,
            });
        }

        // Calculate metrics
        let metrics = self.calculate_metrics(&daily_results, &trades);

        BacktestResult {
            daily_results,
            trades,
            metrics,
        }
    }

    /// Calculate performance metrics
    fn calculate_metrics(&self, daily: &[DailyResult], trades: &[Trade]) -> BacktestMetrics {
        if daily.is_empty() {
            return BacktestMetrics::default();
        }

        // Total return
        let total_return = daily.last().map(|d| d.cumulative_pnl).unwrap_or(0.0);

        // Daily returns for Sharpe/Sortino
        let returns: Vec<f64> = daily.iter().map(|d| d.pnl).collect();
        let n_days = returns.len() as f64;

        let mean_return = returns.iter().sum::<f64>() / n_days;
        let std_return = {
            let variance = returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / n_days;
            variance.sqrt()
        };

        // Sharpe ratio (annualized)
        let sharpe_ratio = if std_return > 0.0 {
            mean_return / std_return * 252_f64.sqrt()
        } else {
            0.0
        };

        // Sortino ratio (only downside deviation)
        let downside: Vec<f64> = returns.iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();
        let downside_std = if !downside.is_empty() {
            let sum_sq: f64 = downside.iter().map(|r| r.powi(2)).sum();
            (sum_sq / downside.len() as f64).sqrt()
        } else {
            0.0
        };
        let sortino_ratio = if downside_std > 0.0 {
            mean_return / downside_std * 252_f64.sqrt()
        } else {
            0.0
        };

        // Maximum drawdown
        let mut peak = 0.0_f64;
        let mut max_drawdown = 0.0_f64;
        for d in daily {
            peak = peak.max(d.cumulative_pnl);
            let drawdown = peak - d.cumulative_pnl;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Trade statistics
        let n_trades = trades.len();
        let winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
        let win_rate = if n_trades > 0 {
            winning_trades as f64 / n_trades as f64
        } else {
            0.0
        };

        let avg_trade_return = if n_trades > 0 {
            trades.iter().map(|t| t.return_pct).sum::<f64>() / n_trades as f64
        } else {
            0.0
        };

        // Profit factor
        let gross_profit: f64 = trades.iter()
            .filter(|t| t.pnl > 0.0)
            .map(|t| t.pnl)
            .sum();
        let gross_loss: f64 = trades.iter()
            .filter(|t| t.pnl < 0.0)
            .map(|t| t.pnl.abs())
            .sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // In-distribution ratio
        let in_dist_count = daily.iter().filter(|d| d.in_distribution).count();
        let in_distribution_ratio = in_dist_count as f64 / n_days;

        // Average confidence when trading
        let trading_days: Vec<&DailyResult> = daily.iter()
            .filter(|d| d.position != 0.0)
            .collect();
        let avg_confidence = if !trading_days.is_empty() {
            trading_days.iter().map(|d| d.confidence).sum::<f64>() / trading_days.len() as f64
        } else {
            0.0
        };

        // Annualized return (assuming 252 trading days)
        let annualized_return = total_return / self.config.initial_capital * (252.0 / n_days);

        BacktestMetrics {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            n_trades,
            avg_trade_return,
            profit_factor,
            in_distribution_ratio,
            avg_confidence,
        }
    }
}

impl BacktestMetrics {
    /// Print metrics summary
    pub fn print_summary(&self) {
        println!("\n=== Backtest Results ===");
        println!("Total Return:        ${:.2}", self.total_return);
        println!("Annualized Return:   {:.2}%", self.annualized_return * 100.0);
        println!("Sharpe Ratio:        {:.2}", self.sharpe_ratio);
        println!("Sortino Ratio:       {:.2}", self.sortino_ratio);
        println!("Max Drawdown:        ${:.2}", self.max_drawdown);
        println!("Win Rate:            {:.1}%", self.win_rate * 100.0);
        println!("Number of Trades:    {}", self.n_trades);
        println!("Avg Trade Return:    {:.2}%", self.avg_trade_return);
        println!("Profit Factor:       {:.2}", self.profit_factor);
        println!("In-Distribution:     {:.1}%", self.in_distribution_ratio * 100.0);
        println!("Avg Confidence:      {:.2}", self.avg_confidence);
        println!("========================\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::MultiScaleScoreNetwork;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        let mut candles = Vec::new();
        let mut price = 100.0;

        for i in 0..n {
            let change = (i as f64 * 0.1).sin() * 2.0;
            price += change;

            candles.push(Candle {
                timestamp: Utc::now(),
                open: price - 0.5,
                high: price + 1.0,
                low: price - 1.0,
                close: price,
                volume: 1000.0,
            });
        }

        candles
    }

    #[test]
    fn test_backtest_config_default() {
        let config = BacktestConfig::default();
        assert_eq!(config.initial_capital, 10000.0);
        assert_eq!(config.warmup, 100);
    }

    #[test]
    fn test_backtest_run() {
        let network = MultiScaleScoreNetwork::new(10, 32, 2, 5, 0.01, 1.0);
        let trader = ScoreMatchingTrader::new(network);
        let config = BacktestConfig::default();
        let backtester = Backtester::new(trader, config);

        let candles = create_test_candles(200);
        let result = backtester.run(&candles);

        assert!(!result.daily_results.is_empty());
    }
}
