//! Backtesting framework for regime-based strategies.
//!
//! Provides tools to evaluate regime classification strategies
//! on historical data.

use crate::classifier::{MarketRegime, RegimeResult};
use crate::data::OHLCVData;
use crate::signals::SignalGenerator;
use chrono::{DateTime, Utc};

/// Individual trade record.
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_date: DateTime<Utc>,
    pub exit_date: Option<DateTime<Utc>>,
    pub entry_price: f64,
    pub exit_price: Option<f64>,
    pub position_size: f64,
    pub regime_at_entry: MarketRegime,
    pub pnl: Option<f64>,
    pub pnl_pct: Option<f64>,
}

/// Results from a backtest run.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Total return as decimal
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown as decimal
    pub max_drawdown: f64,
    /// Win rate (0-1)
    pub win_rate: f64,
    /// Number of trades
    pub num_trades: usize,
    /// List of trades
    pub trades: Vec<Trade>,
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// Regime history
    pub regime_history: Vec<MarketRegime>,
    /// Daily returns
    pub daily_returns: Vec<f64>,
}

impl BacktestResult {
    /// Get profit factor.
    pub fn profit_factor(&self) -> f64 {
        let gross_profit: f64 = self
            .trades
            .iter()
            .filter_map(|t| t.pnl)
            .filter(|&pnl| pnl > 0.0)
            .sum();

        let gross_loss: f64 = self
            .trades
            .iter()
            .filter_map(|t| t.pnl)
            .filter(|&pnl| pnl < 0.0)
            .map(|pnl| pnl.abs())
            .sum();

        if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else {
            f64::INFINITY
        }
    }

    /// Get average trade PnL.
    pub fn avg_trade_pnl(&self) -> f64 {
        let pnls: Vec<f64> = self.trades.iter().filter_map(|t| t.pnl).collect();
        if pnls.is_empty() {
            0.0
        } else {
            pnls.iter().sum::<f64>() / pnls.len() as f64
        }
    }
}

/// Backtester for regime-based strategies.
pub struct Backtester {
    /// Initial capital
    initial_capital: f64,
    /// Commission rate per trade
    commission: f64,
    /// Slippage rate per trade
    slippage: f64,
    /// Annual risk-free rate
    risk_free_rate: f64,
}

impl Backtester {
    /// Create a new backtester.
    pub fn new(initial_capital: f64, commission: f64, slippage: f64) -> Self {
        Self {
            initial_capital,
            commission,
            slippage,
            risk_free_rate: 0.02,
        }
    }

    /// Run backtest on historical data.
    pub fn run(
        &self,
        data: &OHLCVData,
        regime_results: &[RegimeResult],
        signal_generator: &SignalGenerator,
    ) -> BacktestResult {
        let prices = data.close_prices();
        let timestamps: Vec<DateTime<Utc>> = data.bars.iter().map(|b| b.timestamp).collect();

        // Generate signals
        let signals = signal_generator.generate_signals(regime_results);

        // Ensure alignment
        let min_len = prices.len().min(signals.len());
        if min_len == 0 {
            return self.empty_result();
        }

        let prices = &prices[..min_len];
        let signals = &signals[..min_len];
        let timestamps = &timestamps[..min_len];

        // Initialize tracking
        let mut capital = self.initial_capital;
        let mut position: f64 = 0.0;
        let mut entry_price: f64 = 0.0;
        let mut entry_date = timestamps[0];
        let mut current_regime = MarketRegime::Sideways;

        let mut trades: Vec<Trade> = Vec::new();
        let mut equity_curve: Vec<f64> = Vec::new();
        let mut regime_history: Vec<MarketRegime> = Vec::new();
        let mut daily_returns: Vec<f64> = Vec::new();
        let mut prev_equity = capital;

        for i in 0..min_len {
            let price = prices[i];
            let signal = &signals[i];
            let date = timestamps[i];
            let regime = regime_results[i].regime;
            regime_history.push(regime);

            // Calculate current equity
            let current_equity = if position != 0.0 {
                capital + position * price
            } else {
                capital
            };
            equity_curve.push(current_equity);

            // Daily return
            let daily_ret = if prev_equity > 0.0 {
                (current_equity - prev_equity) / prev_equity
            } else {
                0.0
            };
            daily_returns.push(daily_ret);
            prev_equity = current_equity;

            // Check for position change
            let target_position = signal.position_size;
            let should_trade = self.should_trade(position, target_position);

            if should_trade {
                // Close existing position
                if position != 0.0 {
                    let exit_price = if position > 0.0 {
                        price * (1.0 - self.slippage)
                    } else {
                        price * (1.0 + self.slippage)
                    };

                    let pnl = position * (exit_price - entry_price)
                        - (position.abs() * exit_price * self.commission);

                    trades.push(Trade {
                        entry_date,
                        exit_date: Some(date),
                        entry_price,
                        exit_price: Some(exit_price),
                        position_size: position,
                        regime_at_entry: current_regime,
                        pnl: Some(pnl),
                        pnl_pct: Some(pnl / (position.abs() * entry_price)),
                    });

                    capital += position * exit_price;
                    capital -= position.abs() * exit_price * self.commission;
                    position = 0.0;
                }

                // Open new position
                if target_position.abs() > 0.1 {
                    let available = capital * target_position.abs();
                    entry_price = if target_position > 0.0 {
                        price * (1.0 + self.slippage)
                    } else {
                        price * (1.0 - self.slippage)
                    };

                    position = available / entry_price;
                    if target_position < 0.0 {
                        position = -position;
                    }

                    capital -= position.abs() * entry_price * self.commission;
                    entry_date = date;
                    current_regime = regime;
                }
            }
        }

        // Close remaining position
        if position != 0.0 && !prices.is_empty() {
            let final_price = prices[prices.len() - 1];
            let exit_price = if position > 0.0 {
                final_price * (1.0 - self.slippage)
            } else {
                final_price * (1.0 + self.slippage)
            };

            let pnl = position * (exit_price - entry_price);

            trades.push(Trade {
                entry_date,
                exit_date: Some(timestamps[timestamps.len() - 1]),
                entry_price,
                exit_price: Some(exit_price),
                position_size: position,
                regime_at_entry: current_regime,
                pnl: Some(pnl),
                pnl_pct: Some(pnl / (position.abs() * entry_price)),
            });

            capital += position * exit_price;
        }

        // Calculate metrics
        self.calculate_metrics(
            capital,
            trades,
            equity_curve,
            regime_history,
            daily_returns,
        )
    }

    fn should_trade(&self, current: f64, target: f64) -> bool {
        if current == 0.0 && target.abs() > 0.1 {
            return true;
        }
        if current != 0.0 && target.abs() < 0.1 {
            return true;
        }
        if current > 0.0 && target < -0.1 {
            return true;
        }
        if current < 0.0 && target > 0.1 {
            return true;
        }
        false
    }

    fn calculate_metrics(
        &self,
        final_capital: f64,
        trades: Vec<Trade>,
        equity_curve: Vec<f64>,
        regime_history: Vec<MarketRegime>,
        daily_returns: Vec<f64>,
    ) -> BacktestResult {
        let total_return = (final_capital - self.initial_capital) / self.initial_capital;

        let num_days = equity_curve.len();
        let annualized_return = if num_days > 0 {
            (1.0 + total_return).powf(252.0 / num_days as f64) - 1.0
        } else {
            0.0
        };

        // Sharpe ratio
        let sharpe_ratio = self.calculate_sharpe(&daily_returns);

        // Max drawdown
        let max_drawdown = self.calculate_max_drawdown(&equity_curve);

        // Win rate
        let winning = trades.iter().filter(|t| t.pnl.unwrap_or(0.0) > 0.0).count();
        let win_rate = if !trades.is_empty() {
            winning as f64 / trades.len() as f64
        } else {
            0.0
        };

        BacktestResult {
            total_return,
            annualized_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            num_trades: trades.len(),
            trades,
            equity_curve,
            regime_history,
            daily_returns,
        }
    }

    fn calculate_sharpe(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }

        let daily_rf = self.risk_free_rate / 252.0;
        let excess_returns: Vec<f64> = returns.iter().map(|r| r - daily_rf).collect();

        let mean = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
        let variance = excess_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / (excess_returns.len() - 1) as f64;
        let std = variance.sqrt();

        if std > 0.0 {
            252.0_f64.sqrt() * mean / std
        } else {
            0.0
        }
    }

    fn calculate_max_drawdown(&self, equity: &[f64]) -> f64 {
        if equity.is_empty() {
            return 0.0;
        }

        let mut max_equity = equity[0];
        let mut max_drawdown = 0.0_f64;

        for &e in equity {
            max_equity = max_equity.max(e);
            let drawdown = (max_equity - e) / max_equity;
            max_drawdown = max_drawdown.max(drawdown);
        }

        max_drawdown
    }

    fn empty_result(&self) -> BacktestResult {
        BacktestResult {
            total_return: 0.0,
            annualized_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            num_trades: 0,
            trades: Vec::new(),
            equity_curve: Vec::new(),
            regime_history: Vec::new(),
            daily_returns: Vec::new(),
        }
    }
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new(100000.0, 0.001, 0.0005)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::YahooFinanceLoader;

    #[test]
    fn test_backtester() {
        let loader = YahooFinanceLoader::new();
        let data = loader.generate_mock_data("SPY", 100);

        let classifier = crate::classifier::StatisticalClassifier::default();

        // Generate regime results
        let mut results = Vec::new();
        for i in 20..data.len() {
            let window = data.slice(i - 20, i + 1);
            results.push(classifier.classify(&window));
        }

        let signal_gen = SignalGenerator::default();
        let backtester = Backtester::default();

        let trimmed_data = data.slice(20, data.len());
        let result = backtester.run(&trimmed_data, &results, &signal_gen);

        assert!(result.equity_curve.len() > 0);
    }
}
