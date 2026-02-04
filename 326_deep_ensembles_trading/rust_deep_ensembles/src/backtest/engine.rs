//! Backtesting engine

use super::metrics::TradingMetrics;
use super::report::{BacktestReport, TradeRecord};
use crate::ensemble::model::DeepEnsemble;
use crate::strategy::signal::SignalType;
use crate::strategy::uncertainty::{MarketRegime, RegimeDetector, UncertaintyStrategy};
use ndarray::Array2;

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub trading_fee: f64,
    pub slippage: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
        }
    }
}

/// Position tracking
struct Position {
    size: f64,
    entry_price: f64,
    entry_time: i64,
}

/// Backtesting engine
pub struct BacktestEngine {
    pub config: BacktestConfig,
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest
    pub fn run(
        &self,
        ensemble: &mut DeepEnsemble,
        strategy: &UncertaintyStrategy,
        features: &Array2<f64>,
        prices: &[f64],
        symbol: &str,
    ) -> BacktestReport {
        let n_steps = features.nrows().min(prices.len());

        let mut capital = self.config.initial_capital;
        let mut equity_curve = vec![capital];
        let mut trades = Vec::new();
        let mut returns = Vec::new();

        let mut position: Option<Position> = None;
        let mut regime_detector = RegimeDetector::default();

        for t in 0..n_steps {
            let current_price = prices[t];

            // Get prediction for current time step
            let features_t = features.slice(ndarray::s![t..t+1, ..]).to_owned();
            let predictions = ensemble.predict(&features_t);

            // Update regime detector
            regime_detector.update(&predictions);
            let regime = regime_detector.detect();

            // Generate signal
            let signals = strategy.generate_signals(&predictions, &[symbol.to_string()]);
            let signal = &signals[0];

            // Check exit conditions
            if let Some(ref pos) = position {
                let should_exit = match regime {
                    MarketRegime::RegimeChange => true,
                    _ => {
                        (signal.signal_type == SignalType::Short && pos.size > 0.0)
                            || (signal.signal_type == SignalType::Long && pos.size < 0.0)
                    }
                };

                if should_exit {
                    // Close position
                    let exit_price = if pos.size > 0.0 {
                        current_price * (1.0 - self.config.slippage)
                    } else {
                        current_price * (1.0 + self.config.slippage)
                    };

                    let fee = pos.size.abs() * exit_price * self.config.trading_fee;
                    let pnl = pos.size * (exit_price - pos.entry_price) - fee;

                    capital += pnl;
                    returns.push(pnl / (pos.size.abs() * pos.entry_price));

                    trades.push(TradeRecord {
                        symbol: symbol.to_string(),
                        entry_time: pos.entry_time,
                        exit_time: t as i64,
                        entry_price: pos.entry_price,
                        exit_price,
                        size: pos.size,
                        pnl,
                        return_pct: pnl / (pos.size.abs() * pos.entry_price),
                    });

                    position = None;
                }
            }

            // Check entry conditions
            if position.is_none() && regime != MarketRegime::RegimeChange && regime != MarketRegime::Unknown {
                let position_size = strategy.calculate_position_size(signal, capital);

                if position_size.abs() > 0.0 {
                    let entry_price = if position_size > 0.0 {
                        current_price * (1.0 + self.config.slippage)
                    } else {
                        current_price * (1.0 - self.config.slippage)
                    };

                    let fee = position_size.abs() * self.config.trading_fee;
                    capital -= fee;

                    position = Some(Position {
                        size: position_size / entry_price,
                        entry_price,
                        entry_time: t as i64,
                    });
                }
            }

            // Update equity curve
            let current_equity = if let Some(ref pos) = position {
                capital + pos.size * (current_price - pos.entry_price)
            } else {
                capital
            };
            equity_curve.push(current_equity);
        }

        // Close any remaining position at the end
        if let Some(pos) = position {
            let exit_price = prices.last().unwrap_or(&0.0);
            let pnl = pos.size * (exit_price - pos.entry_price);
            capital += pnl;
            equity_curve.push(capital);
        }

        // Calculate metrics
        let metrics = TradingMetrics::compute(
            &returns,
            &equity_curve,
            self.config.initial_capital,
            capital,
            trades.len(),
            0.02, // risk-free rate
            252.0, // annualization factor
        );

        BacktestReport::new(
            self.config.initial_capital,
            capital,
            equity_curve,
            trades,
            metrics,
        )
    }

    /// Run a simple backtest with synthetic signals
    pub fn run_simple(
        &self,
        predictions: &[(f64, f64)], // (predicted_return, uncertainty)
        prices: &[f64],
        symbol: &str,
    ) -> BacktestReport {
        let n_steps = predictions.len().min(prices.len());

        let mut capital = self.config.initial_capital;
        let mut equity_curve = vec![capital];
        let mut trades = Vec::new();
        let mut returns = Vec::new();

        let mut position: Option<Position> = None;

        for t in 0..n_steps {
            let current_price = prices[t];
            let (pred_return, uncertainty) = predictions[t];

            // Simple signal logic
            let signal_strength = if uncertainty > 0.0 {
                pred_return / uncertainty
            } else {
                0.0
            };

            // Exit conditions
            if let Some(ref pos) = position {
                let should_exit = (signal_strength < -1.0 && pos.size > 0.0)
                    || (signal_strength > 1.0 && pos.size < 0.0)
                    || uncertainty > 0.5; // High uncertainty = exit

                if should_exit {
                    let exit_price = current_price;
                    let fee = pos.size.abs() * exit_price * self.config.trading_fee;
                    let pnl = pos.size * (exit_price - pos.entry_price) - fee;

                    capital += pnl;
                    returns.push(pnl / (pos.size.abs() * pos.entry_price));

                    trades.push(TradeRecord {
                        symbol: symbol.to_string(),
                        entry_time: pos.entry_time,
                        exit_time: t as i64,
                        entry_price: pos.entry_price,
                        exit_price,
                        size: pos.size,
                        pnl,
                        return_pct: pnl / (pos.size.abs() * pos.entry_price),
                    });

                    position = None;
                }
            }

            // Entry conditions
            if position.is_none() && uncertainty < 0.3 {
                let position_value = capital * 0.1 * (1.0 - uncertainty);

                if signal_strength > 1.5 {
                    // Long
                    let entry_price = current_price * (1.0 + self.config.slippage);
                    let fee = position_value * self.config.trading_fee;
                    capital -= fee;

                    position = Some(Position {
                        size: position_value / entry_price,
                        entry_price,
                        entry_time: t as i64,
                    });
                } else if signal_strength < -1.5 {
                    // Short
                    let entry_price = current_price * (1.0 - self.config.slippage);
                    let fee = position_value * self.config.trading_fee;
                    capital -= fee;

                    position = Some(Position {
                        size: -position_value / entry_price,
                        entry_price,
                        entry_time: t as i64,
                    });
                }
            }

            // Update equity
            let current_equity = if let Some(ref pos) = position {
                capital + pos.size * (current_price - pos.entry_price)
            } else {
                capital
            };
            equity_curve.push(current_equity);
        }

        // Close remaining position
        if let Some(pos) = position {
            let exit_price = prices.last().unwrap_or(&0.0);
            let pnl = pos.size * (exit_price - pos.entry_price);
            capital += pnl;
        }

        // Calculate metrics
        let metrics = TradingMetrics::compute(
            &returns,
            &equity_curve,
            self.config.initial_capital,
            capital,
            trades.len(),
            0.02,
            252.0,
        );

        BacktestReport::new(
            self.config.initial_capital,
            capital,
            equity_curve,
            trades,
            metrics,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_backtest() {
        let config = BacktestConfig::default();
        let engine = BacktestEngine::new(config);

        // Generate synthetic data
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0)
            .collect();

        let predictions: Vec<(f64, f64)> = prices
            .windows(2)
            .map(|w| {
                let ret = (w[1] - w[0]) / w[0];
                (ret * 0.8, 0.1) // Slightly noisy predictions
            })
            .collect();

        let report = engine.run_simple(&predictions, &prices[1..], "BTCUSDT");

        assert!(report.final_capital > 0.0);
        assert!(!report.equity_curve.is_empty());
    }
}
