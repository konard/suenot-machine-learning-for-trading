//! Backtest engine implementation.
//!
//! This module provides the main backtesting functionality for evaluating
//! CML trading strategies on historical data.

use crate::trading::strategy::{CMLStrategy, TradeAction, Position};
use crate::data::features::TradingFeatures;
use crate::data::Kline;
use crate::MarketRegime;

/// Configuration for backtesting.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital.
    pub initial_capital: f64,
    /// Trading fee (as fraction, e.g., 0.001 = 0.1%).
    pub trading_fee: f64,
    /// Slippage (as fraction).
    pub slippage: f64,
    /// Whether to reinvest profits.
    pub reinvest_profits: bool,
    /// Risk-free rate for Sharpe calculation (annualized).
    pub risk_free_rate: f64,
    /// Trading days per year (for annualization).
    pub trading_days_per_year: f64,
    /// Whether to log individual trades.
    pub log_trades: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
            reinvest_profits: true,
            risk_free_rate: 0.02,
            trading_days_per_year: 252.0,
            log_trades: true,
        }
    }
}

/// A single trade log entry.
#[derive(Debug, Clone)]
pub struct TradeLog {
    /// Trade index.
    pub index: usize,
    /// Entry timestamp.
    pub entry_time: i64,
    /// Exit timestamp.
    pub exit_time: Option<i64>,
    /// Entry price.
    pub entry_price: f64,
    /// Exit price.
    pub exit_price: Option<f64>,
    /// Position size.
    pub size: f64,
    /// Position direction (1 for long, -1 for short).
    pub direction: f64,
    /// Profit/loss.
    pub pnl: f64,
    /// Return percentage.
    pub return_pct: f64,
    /// Market regime at entry.
    pub regime: MarketRegime,
    /// Fees paid.
    pub fees: f64,
}

/// Result of a backtest.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Final portfolio value.
    pub final_value: f64,
    /// Total return.
    pub total_return: f64,
    /// Annualized return.
    pub annualized_return: f64,
    /// Sharpe ratio.
    pub sharpe_ratio: f64,
    /// Sortino ratio.
    pub sortino_ratio: f64,
    /// Maximum drawdown.
    pub max_drawdown: f64,
    /// Win rate.
    pub win_rate: f64,
    /// Profit factor.
    pub profit_factor: f64,
    /// Number of trades.
    pub num_trades: usize,
    /// Average trade return.
    pub avg_trade_return: f64,
    /// Total fees paid.
    pub total_fees: f64,
    /// Trade logs.
    pub trades: Vec<TradeLog>,
    /// Equity curve.
    pub equity_curve: Vec<f64>,
    /// Daily returns.
    pub daily_returns: Vec<f64>,
    /// Regime performance.
    pub regime_performance: Vec<(MarketRegime, f64, usize)>,
}

/// Backtester for CML strategies.
pub struct Backtester {
    /// Configuration.
    config: BacktestConfig,
    /// Current capital.
    capital: f64,
    /// Equity curve.
    equity_curve: Vec<f64>,
    /// Trade logs.
    trades: Vec<TradeLog>,
    /// Current trade (if any).
    current_trade: Option<TradeLog>,
    /// Daily returns.
    daily_returns: Vec<f64>,
    /// Peak equity (for drawdown calculation).
    peak_equity: f64,
    /// Maximum drawdown.
    max_drawdown: f64,
    /// Returns by regime.
    regime_returns: Vec<(MarketRegime, Vec<f64>)>,
    /// Trade counter.
    trade_counter: usize,
}

impl Backtester {
    /// Create a new backtester.
    pub fn new(config: BacktestConfig) -> Self {
        let initial = config.initial_capital;
        Self {
            config,
            capital: initial,
            equity_curve: vec![initial],
            trades: Vec::new(),
            current_trade: None,
            daily_returns: Vec::new(),
            peak_equity: initial,
            max_drawdown: 0.0,
            regime_returns: Vec::new(),
            trade_counter: 0,
        }
    }

    /// Run backtest on historical data.
    pub fn run(&mut self, strategy: &mut CMLStrategy, klines: &[Kline], features: &TradingFeatures) -> BacktestResult {
        // Reset state
        self.capital = self.config.initial_capital;
        self.equity_curve = vec![self.config.initial_capital];
        self.trades.clear();
        self.current_trade = None;
        self.daily_returns.clear();
        self.peak_equity = self.config.initial_capital;
        self.max_drawdown = 0.0;
        self.regime_returns.clear();
        self.trade_counter = 0;

        let mut prev_equity = self.capital;

        // Run through each data point
        for (i, kline) in klines.iter().enumerate() {
            if i >= features.len() {
                break;
            }

            let price = kline.close;
            let regime = features.get_regime(i).unwrap_or(MarketRegime::Sideways);

            // Get action from strategy
            let action = strategy.step(features, i, price);

            // Execute action
            self.execute_action(action, price, kline.start_time, regime, strategy.position());

            // Update equity
            let position_value = self.calculate_position_value(strategy.position(), price);
            let equity = self.capital + position_value;
            self.equity_curve.push(equity);

            // Calculate daily return
            if prev_equity > 0.0 {
                let daily_return = (equity - prev_equity) / prev_equity;
                self.daily_returns.push(daily_return);

                // Track returns by regime
                self.add_regime_return(regime, daily_return);
            }

            // Update drawdown
            if equity > self.peak_equity {
                self.peak_equity = equity;
            }
            let drawdown = (self.peak_equity - equity) / self.peak_equity;
            if drawdown > self.max_drawdown {
                self.max_drawdown = drawdown;
            }

            prev_equity = equity;
        }

        // Close any open position
        if !strategy.position().is_flat() && !klines.is_empty() {
            let last_kline = klines.last().unwrap();
            self.close_position(last_kline.close, last_kline.start_time);
        }

        // Calculate final metrics
        self.calculate_result()
    }

    /// Execute a trading action.
    #[allow(unused_variables)]
    fn execute_action(
        &mut self,
        action: TradeAction,
        price: f64,
        timestamp: i64,
        regime: MarketRegime,
        position: Position,
    ) {
        match action {
            TradeAction::OpenLong(size) => {
                self.open_trade(price, size, 1.0, timestamp, regime);
            }
            TradeAction::OpenShort(size) => {
                self.open_trade(price, size, -1.0, timestamp, regime);
            }
            TradeAction::ClosePosition => {
                self.close_position(price, timestamp);
            }
            TradeAction::IncreasePosition(delta) => {
                // Calculate fees first
                let fees = self.calculate_fees(price, delta);
                // Add to current trade (simplified)
                if let Some(ref mut trade) = self.current_trade {
                    trade.size += delta;
                    trade.fees += fees;
                }
                self.capital -= fees;
            }
            TradeAction::DecreasePosition(delta) => {
                // Extract values needed for calculation
                let (entry_price, direction, close_size) = if let Some(ref trade) = self.current_trade {
                    let close_size = delta.min(trade.size);
                    (trade.entry_price, trade.direction, close_size)
                } else {
                    return;
                };

                // Calculate pnl and fees
                let pnl = self.calculate_pnl(entry_price, price, close_size, direction);
                let fees = self.calculate_fees(price, close_size);

                // Now update the trade
                if let Some(ref mut trade) = self.current_trade {
                    trade.size -= close_size;
                    trade.fees += fees;
                }
                self.capital += pnl;
                self.capital -= fees;
            }
            TradeAction::Hold => {}
        }
    }

    /// Open a new trade.
    fn open_trade(&mut self, price: f64, size: f64, direction: f64, timestamp: i64, regime: MarketRegime) {
        // Apply slippage
        let exec_price = if direction > 0.0 {
            price * (1.0 + self.config.slippage)
        } else {
            price * (1.0 - self.config.slippage)
        };

        // Calculate fees
        let fees = self.calculate_fees(exec_price, size);
        self.capital -= fees;

        self.trade_counter += 1;

        self.current_trade = Some(TradeLog {
            index: self.trade_counter,
            entry_time: timestamp,
            exit_time: None,
            entry_price: exec_price,
            exit_price: None,
            size,
            direction,
            pnl: 0.0,
            return_pct: 0.0,
            regime,
            fees,
        });
    }

    /// Close current position.
    fn close_position(&mut self, price: f64, timestamp: i64) {
        if let Some(mut trade) = self.current_trade.take() {
            // Apply slippage
            let exec_price = if trade.direction > 0.0 {
                price * (1.0 - self.config.slippage)
            } else {
                price * (1.0 + self.config.slippage)
            };

            // Calculate PnL
            let pnl = self.calculate_pnl(trade.entry_price, exec_price, trade.size, trade.direction);
            let fees = self.calculate_fees(exec_price, trade.size);

            trade.exit_time = Some(timestamp);
            trade.exit_price = Some(exec_price);
            trade.pnl = pnl - fees;
            trade.return_pct = if trade.entry_price > 0.0 {
                (exec_price - trade.entry_price) / trade.entry_price * trade.direction
            } else {
                0.0
            };
            trade.fees += fees;

            // Update capital
            self.capital += pnl - fees;

            if self.config.log_trades {
                self.trades.push(trade);
            }
        }
    }

    /// Calculate position value.
    fn calculate_position_value(&self, position: Position, price: f64) -> f64 {
        match position {
            Position::Long(size) => {
                if let Some(ref trade) = self.current_trade {
                    let pnl = (price - trade.entry_price) * size;
                    pnl
                } else {
                    0.0
                }
            }
            Position::Short(size) => {
                if let Some(ref trade) = self.current_trade {
                    let pnl = (trade.entry_price - price) * size;
                    pnl
                } else {
                    0.0
                }
            }
            Position::Flat => 0.0,
        }
    }

    /// Calculate PnL for a trade.
    fn calculate_pnl(&self, entry: f64, exit: f64, size: f64, direction: f64) -> f64 {
        (exit - entry) * size * direction * self.config.initial_capital
    }

    /// Calculate trading fees.
    fn calculate_fees(&self, price: f64, size: f64) -> f64 {
        price * size * self.config.trading_fee * self.config.initial_capital
    }

    /// Add return to regime tracking.
    fn add_regime_return(&mut self, regime: MarketRegime, return_val: f64) {
        if let Some(entry) = self.regime_returns.iter_mut().find(|(r, _)| *r == regime) {
            entry.1.push(return_val);
        } else {
            self.regime_returns.push((regime, vec![return_val]));
        }
    }

    /// Calculate final backtest result.
    fn calculate_result(&self) -> BacktestResult {
        let final_value = *self.equity_curve.last().unwrap_or(&self.config.initial_capital);
        let total_return = (final_value - self.config.initial_capital) / self.config.initial_capital;

        // Annualized return
        let num_days = self.daily_returns.len() as f64;
        let annualized_return = if num_days > 0.0 {
            (1.0 + total_return).powf(self.config.trading_days_per_year / num_days) - 1.0
        } else {
            0.0
        };

        // Sharpe ratio
        let sharpe_ratio = self.calculate_sharpe();

        // Sortino ratio
        let sortino_ratio = self.calculate_sortino();

        // Win rate and profit factor
        let (win_rate, profit_factor, avg_return) = self.calculate_trade_stats();

        // Total fees
        let total_fees: f64 = self.trades.iter().map(|t| t.fees).sum();

        // Regime performance
        let regime_performance: Vec<(MarketRegime, f64, usize)> = self
            .regime_returns
            .iter()
            .map(|(regime, returns)| {
                let avg = if returns.is_empty() {
                    0.0
                } else {
                    returns.iter().sum::<f64>() / returns.len() as f64
                };
                (*regime, avg, returns.len())
            })
            .collect();

        BacktestResult {
            final_value,
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown: self.max_drawdown,
            win_rate,
            profit_factor,
            num_trades: self.trades.len(),
            avg_trade_return: avg_return,
            total_fees,
            trades: self.trades.clone(),
            equity_curve: self.equity_curve.clone(),
            daily_returns: self.daily_returns.clone(),
            regime_performance,
        }
    }

    /// Calculate Sharpe ratio.
    fn calculate_sharpe(&self) -> f64 {
        if self.daily_returns.is_empty() {
            return 0.0;
        }

        let mean = self.daily_returns.iter().sum::<f64>() / self.daily_returns.len() as f64;
        let variance = self.daily_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / self.daily_returns.len() as f64;
        let std = variance.sqrt();

        if std <= 0.0 {
            return 0.0;
        }

        let daily_rf = self.config.risk_free_rate / self.config.trading_days_per_year;
        let excess_return = mean - daily_rf;

        (excess_return / std) * (self.config.trading_days_per_year).sqrt()
    }

    /// Calculate Sortino ratio.
    fn calculate_sortino(&self) -> f64 {
        if self.daily_returns.is_empty() {
            return 0.0;
        }

        let mean = self.daily_returns.iter().sum::<f64>() / self.daily_returns.len() as f64;

        // Downside deviation (only negative returns)
        let downside: Vec<f64> = self.daily_returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        if downside.is_empty() {
            return f64::INFINITY;
        }

        let downside_variance = downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
        let downside_std = downside_variance.sqrt();

        if downside_std <= 0.0 {
            return 0.0;
        }

        let daily_rf = self.config.risk_free_rate / self.config.trading_days_per_year;
        let excess_return = mean - daily_rf;

        (excess_return / downside_std) * (self.config.trading_days_per_year).sqrt()
    }

    /// Calculate trade statistics.
    fn calculate_trade_stats(&self) -> (f64, f64, f64) {
        if self.trades.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let winning: Vec<&TradeLog> = self.trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing: Vec<&TradeLog> = self.trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let win_rate = winning.len() as f64 / self.trades.len() as f64;

        let gross_profit: f64 = winning.iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = losing.iter().map(|t| t.pnl.abs()).sum();

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_return = self.trades.iter().map(|t| t.return_pct).sum::<f64>() / self.trades.len() as f64;

        (win_rate, profit_factor, avg_return)
    }

    /// Get configuration.
    pub fn config(&self) -> &BacktestConfig {
        &self.config
    }

    /// Get current capital.
    pub fn capital(&self) -> f64 {
        self.capital
    }
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new(BacktestConfig::default())
    }
}

/// Print a formatted backtest report.
pub fn print_report(result: &BacktestResult) {
    println!("\n=== Backtest Report ===\n");
    println!("Performance Metrics:");
    println!("  Final Value:      ${:.2}", result.final_value);
    println!("  Total Return:     {:.2}%", result.total_return * 100.0);
    println!("  Annualized Return: {:.2}%", result.annualized_return * 100.0);
    println!("  Sharpe Ratio:     {:.3}", result.sharpe_ratio);
    println!("  Sortino Ratio:    {:.3}", result.sortino_ratio);
    println!("  Max Drawdown:     {:.2}%", result.max_drawdown * 100.0);
    println!("\nTrading Statistics:");
    println!("  Total Trades:     {}", result.num_trades);
    println!("  Win Rate:         {:.2}%", result.win_rate * 100.0);
    println!("  Profit Factor:    {:.3}", result.profit_factor);
    println!("  Avg Trade Return: {:.4}%", result.avg_trade_return * 100.0);
    println!("  Total Fees:       ${:.2}", result.total_fees);

    if !result.regime_performance.is_empty() {
        println!("\nPerformance by Regime:");
        for (regime, avg_return, count) in &result.regime_performance {
            println!(
                "  {:?}: avg {:.4}% ({} samples)",
                regime,
                avg_return * 100.0,
                count
            );
        }
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CMLConfig;
    use crate::continual::learner::ContinualMetaLearner;
    use crate::trading::strategy::{CMLStrategy, StrategyConfig};
    use crate::data::features::FeatureConfig;

    fn create_test_klines() -> Vec<Kline> {
        let prices = vec![
            100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 110.0,
            112.0, 111.0, 113.0, 115.0, 114.0, 116.0, 118.0, 117.0, 120.0, 122.0,
            121.0, 123.0, 125.0, 124.0, 126.0, 128.0, 127.0, 130.0, 132.0, 131.0,
        ];

        prices
            .iter()
            .enumerate()
            .map(|(i, &p)| Kline {
                start_time: (i as i64) * 3600000,
                open: p - 0.5,
                high: p + 1.0,
                low: p - 1.0,
                close: p,
                volume: 1000.0,
                turnover: p * 1000.0,
            })
            .collect()
    }

    #[test]
    fn test_backtester_creation() {
        let backtester = Backtester::new(BacktestConfig::default());
        assert_eq!(backtester.capital(), 10000.0);
    }

    #[test]
    fn test_backtest_run() {
        let klines = create_test_klines();
        let features = TradingFeatures::from_klines(&klines, FeatureConfig::default());

        let learner = ContinualMetaLearner::new(CMLConfig {
            input_size: 9,
            hidden_size: 16,
            output_size: 1,
            ..Default::default()
        });

        let strategy_config = StrategyConfig {
            warmup_samples: 5,
            ..Default::default()
        };

        let mut strategy = CMLStrategy::new(learner, strategy_config);
        let mut backtester = Backtester::new(BacktestConfig::default());

        let result = backtester.run(&mut strategy, &klines, &features);

        assert!(result.final_value > 0.0);
        assert!(!result.equity_curve.is_empty());
    }

    #[test]
    fn test_sharpe_calculation() {
        let mut backtester = Backtester::new(BacktestConfig::default());
        backtester.daily_returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.002];

        let sharpe = backtester.calculate_sharpe();
        // Sharpe should be positive for positive average return
        assert!(sharpe.is_finite());
    }
}
