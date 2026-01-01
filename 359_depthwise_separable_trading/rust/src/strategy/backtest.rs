//! Backtesting framework
//!
//! Provides comprehensive backtesting capabilities for trading strategies.

use ndarray::Array1;

use crate::data::Candle;

use super::{Portfolio, Position, Signal, SignalGenerator, StrategyError, Trade};

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Total return percentage
    pub total_return: f64,
    /// Annualized return
    pub annual_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Total trades
    pub total_trades: usize,
    /// Average trade duration
    pub avg_trade_duration: f64,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Equity curve
    pub equity_curve: Array1<f64>,
    /// All trades
    pub trades: Vec<Trade>,
    /// Drawdown curve
    pub drawdown_curve: Array1<f64>,
}

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Commission rate (e.g., 0.001 = 0.1%)
    pub commission: f64,
    /// Position size as fraction of equity (0.0 to 1.0)
    pub position_size: f64,
    /// Enable short selling
    pub allow_short: bool,
    /// Stop loss percentage (e.g., 0.02 = 2%)
    pub stop_loss: Option<f64>,
    /// Take profit percentage
    pub take_profit: Option<f64>,
    /// Maximum position hold time (in bars)
    pub max_hold_time: Option<usize>,
    /// Risk-free rate for Sharpe calculation
    pub risk_free_rate: f64,
    /// Trading days per year (252 for stocks, 365 for crypto)
    pub trading_days: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            commission: 0.001,
            position_size: 0.95,
            allow_short: true,
            stop_loss: Some(0.02),
            take_profit: Some(0.05),
            max_hold_time: None,
            risk_free_rate: 0.02,
            trading_days: 365, // Crypto
        }
    }
}

/// Backtester
pub struct Backtest {
    config: BacktestConfig,
}

impl Backtest {
    /// Create new backtester
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest with default config
    pub fn with_defaults() -> Self {
        Self::new(BacktestConfig::default())
    }

    /// Run backtest
    pub fn run<S: SignalGenerator>(
        &self,
        strategy: &S,
        candles: &[Candle],
    ) -> Result<BacktestResult, StrategyError> {
        let signals = strategy.generate_signals(candles)?;
        self.run_with_signals(candles, &signals)
    }

    /// Run backtest with pre-generated signals
    pub fn run_with_signals(
        &self,
        candles: &[Candle],
        signals: &[(Signal, f64)],
    ) -> Result<BacktestResult, StrategyError> {
        let mut portfolio = Portfolio::new(self.config.initial_capital, self.config.commission);

        let mut entry_idx = 0;
        let mut entry_price = 0.0;
        let mut bars_in_trade = 0;

        // Align signals with candles (signals start after window)
        let offset = candles.len() - signals.len();

        for (i, (signal, confidence)) in signals.iter().enumerate() {
            let candle_idx = offset + i;
            let price = candles[candle_idx].close;

            // Update position hold time
            if !portfolio.position.is_flat() {
                bars_in_trade += 1;
            }

            // Check stop loss / take profit
            if !portfolio.position.is_flat() {
                let pnl_pct = match portfolio.position {
                    Position::Long(_) => (price - entry_price) / entry_price,
                    Position::Short(_) => (entry_price - price) / entry_price,
                    Position::Flat => 0.0,
                };

                // Stop loss
                if let Some(sl) = self.config.stop_loss {
                    if pnl_pct <= -sl {
                        portfolio.close_position(price, entry_idx, candle_idx, entry_price);
                        bars_in_trade = 0;
                    }
                }

                // Take profit
                if let Some(tp) = self.config.take_profit {
                    if pnl_pct >= tp {
                        portfolio.close_position(price, entry_idx, candle_idx, entry_price);
                        bars_in_trade = 0;
                    }
                }

                // Max hold time
                if let Some(max_hold) = self.config.max_hold_time {
                    if bars_in_trade >= max_hold {
                        portfolio.close_position(price, entry_idx, candle_idx, entry_price);
                        bars_in_trade = 0;
                    }
                }
            }

            // Process signals
            if portfolio.position.is_flat() {
                // Open new position
                if signal.is_buy() {
                    let size = portfolio.cash * self.config.position_size / price;
                    portfolio.open_long(price, size, candle_idx);
                    entry_idx = candle_idx;
                    entry_price = price;
                    bars_in_trade = 0;
                } else if signal.is_sell() && self.config.allow_short {
                    let size = portfolio.cash * self.config.position_size / price;
                    portfolio.open_short(price, size, candle_idx);
                    entry_idx = candle_idx;
                    entry_price = price;
                    bars_in_trade = 0;
                }
            } else {
                // Close position on opposite signal
                if portfolio.position.is_long() && signal.is_sell() {
                    portfolio.close_position(price, entry_idx, candle_idx, entry_price);
                    bars_in_trade = 0;

                    // Open short if allowed
                    if self.config.allow_short {
                        let size = portfolio.cash * self.config.position_size / price;
                        portfolio.open_short(price, size, candle_idx);
                        entry_idx = candle_idx;
                        entry_price = price;
                    }
                } else if portfolio.position.is_short() && signal.is_buy() {
                    portfolio.close_position(price, entry_idx, candle_idx, entry_price);
                    bars_in_trade = 0;

                    // Open long
                    let size = portfolio.cash * self.config.position_size / price;
                    portfolio.open_long(price, size, candle_idx);
                    entry_idx = candle_idx;
                    entry_price = price;
                }
            }

            // Update equity
            portfolio.update_equity(price);
        }

        // Close any remaining position at the end
        if !portfolio.position.is_flat() {
            let last_price = candles.last().unwrap().close;
            portfolio.close_position(last_price, entry_idx, candles.len() - 1, entry_price);
        }

        self.calculate_metrics(portfolio)
    }

    /// Calculate performance metrics
    fn calculate_metrics(&self, portfolio: Portfolio) -> Result<BacktestResult, StrategyError> {
        let equity = portfolio.equity_curve();
        let n = equity.len();

        // Returns
        let mut returns = Array1::zeros(n);
        for i in 1..n {
            if equity[i - 1] != 0.0 {
                returns[i] = (equity[i] - equity[i - 1]) / equity[i - 1];
            }
        }

        // Total return
        let total_return = if equity[0] != 0.0 {
            (equity[n - 1] - equity[0]) / equity[0]
        } else {
            0.0
        };

        // Annualized return
        let years = n as f64 / self.config.trading_days as f64;
        let annual_return = if years > 0.0 {
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        // Sharpe ratio
        let returns_mean = returns.mean().unwrap_or(0.0);
        let returns_std = returns.mapv(|x| (x - returns_mean).powi(2)).mean().unwrap_or(1.0).sqrt();
        let daily_rf = self.config.risk_free_rate / self.config.trading_days as f64;
        let sharpe_ratio = if returns_std > 1e-10 {
            (returns_mean - daily_rf) / returns_std * (self.config.trading_days as f64).sqrt()
        } else {
            0.0
        };

        // Sortino ratio (using downside deviation)
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
        let downside_std = if !downside_returns.is_empty() {
            let mean: f64 = downside_returns.iter().sum::<f64>() / downside_returns.len() as f64;
            (downside_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / downside_returns.len() as f64)
                .sqrt()
        } else {
            1.0
        };
        let sortino_ratio = if downside_std > 1e-10 {
            (returns_mean - daily_rf) / downside_std * (self.config.trading_days as f64).sqrt()
        } else {
            0.0
        };

        // Maximum drawdown
        let mut peak = equity[0];
        let mut drawdown_curve = Array1::zeros(n);
        let mut max_drawdown = 0.0;

        for i in 0..n {
            if equity[i] > peak {
                peak = equity[i];
            }
            let dd = if peak > 0.0 {
                (peak - equity[i]) / peak
            } else {
                0.0
            };
            drawdown_curve[i] = dd;
            if dd > max_drawdown {
                max_drawdown = dd;
            }
        }

        // Trade statistics
        let trades = &portfolio.trades;
        let total_trades = trades.len();

        let winning_trades: Vec<&Trade> = trades.iter().filter(|t| t.is_profitable()).collect();
        let win_rate = if total_trades > 0 {
            winning_trades.len() as f64 / total_trades as f64
        } else {
            0.0
        };

        let gross_profit: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = trades
            .iter()
            .filter(|t| !t.is_profitable())
            .map(|t| t.pnl.abs())
            .sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            1.0
        };

        let avg_trade_duration = if total_trades > 0 {
            trades.iter().map(|t| t.duration() as f64).sum::<f64>() / total_trades as f64
        } else {
            0.0
        };

        let avg_trade_return = if total_trades > 0 {
            trades.iter().map(|t| t.pnl_pct).sum::<f64>() / total_trades as f64
        } else {
            0.0
        };

        Ok(BacktestResult {
            total_return,
            annual_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            total_trades,
            avg_trade_duration,
            avg_trade_return,
            equity_curve: equity,
            trades: portfolio.trades,
            drawdown_curve,
        })
    }
}

impl BacktestResult {
    /// Print summary report
    pub fn print_summary(&self) {
        println!("========== Backtest Results ==========");
        println!("Total Return:     {:>10.2}%", self.total_return * 100.0);
        println!("Annual Return:    {:>10.2}%", self.annual_return * 100.0);
        println!("Sharpe Ratio:     {:>10.2}", self.sharpe_ratio);
        println!("Sortino Ratio:    {:>10.2}", self.sortino_ratio);
        println!("Max Drawdown:     {:>10.2}%", self.max_drawdown * 100.0);
        println!("---------------------------------------");
        println!("Total Trades:     {:>10}", self.total_trades);
        println!("Win Rate:         {:>10.2}%", self.win_rate * 100.0);
        println!("Profit Factor:    {:>10.2}", self.profit_factor);
        println!("Avg Trade Return: {:>10.2}%", self.avg_trade_return);
        println!("Avg Hold Time:    {:>10.1} bars", self.avg_trade_duration);
        println!("=======================================");
    }

    /// Check if strategy is profitable
    pub fn is_profitable(&self) -> bool {
        self.total_return > 0.0
    }

    /// Check if strategy beats buy-and-hold
    pub fn beats_benchmark(&self, benchmark_return: f64) -> bool {
        self.total_return > benchmark_return
    }

    /// Get risk-adjusted score
    pub fn risk_adjusted_score(&self) -> f64 {
        if self.max_drawdown > 0.0 {
            self.total_return / self.max_drawdown
        } else if self.total_return > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use crate::data::Timeframe;

    fn create_uptrending_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                let base = 100.0 + i as f64 * 0.5;
                Candle::new(
                    "BTCUSDT",
                    Utc::now(),
                    Timeframe::H1,
                    base,
                    base + 1.0,
                    base - 0.5,
                    base + 0.3,
                    1000.0,
                )
            })
            .collect()
    }

    fn create_simple_signals(n: usize) -> Vec<(Signal, f64)> {
        (0..n)
            .map(|i| {
                if i % 20 < 10 {
                    (Signal::Buy, 0.8)
                } else {
                    (Signal::Sell, 0.7)
                }
            })
            .collect()
    }

    #[test]
    fn test_backtest_default() {
        let backtest = Backtest::with_defaults();
        assert_eq!(backtest.config.initial_capital, 100_000.0);
    }

    #[test]
    fn test_backtest_run() {
        let config = BacktestConfig {
            initial_capital: 10_000.0,
            commission: 0.001,
            position_size: 0.9,
            allow_short: false,
            stop_loss: None,
            take_profit: None,
            max_hold_time: None,
            risk_free_rate: 0.0,
            trading_days: 365,
        };

        let backtest = Backtest::new(config);
        let candles = create_uptrending_candles(100);
        let signals = create_simple_signals(100);

        let result = backtest.run_with_signals(&candles, &signals);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.total_trades > 0);
    }

    #[test]
    fn test_backtest_metrics() {
        let config = BacktestConfig::default();
        let backtest = Backtest::new(config);

        let candles = create_uptrending_candles(200);
        let signals = create_simple_signals(200);

        let result = backtest.run_with_signals(&candles, &signals).unwrap();

        // Check that metrics are calculated
        assert!(result.sharpe_ratio.is_finite());
        assert!(result.max_drawdown >= 0.0);
        assert!(result.win_rate >= 0.0 && result.win_rate <= 1.0);
    }
}
