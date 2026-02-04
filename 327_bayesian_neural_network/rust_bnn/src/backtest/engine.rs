//! Backtesting engine.

use chrono::{DateTime, Utc};

use crate::strategy::signal::SignalType;
use crate::strategy::trading::{Portfolio, TradingStrategy};
use super::report::{BacktestReport, Trade};

/// Backtesting engine for trading strategies.
#[derive(Debug)]
pub struct BacktestEngine {
    /// Trading strategy
    pub strategy: TradingStrategy,
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost (as fraction)
    pub transaction_cost: f64,
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self {
            strategy: TradingStrategy::default(),
            initial_capital: 10000.0,
            transaction_cost: 0.001,
        }
    }
}

impl BacktestEngine {
    /// Create a new backtest engine.
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            ..Default::default()
        }
    }

    /// Set trading strategy.
    pub fn strategy(mut self, strategy: TradingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set transaction cost.
    pub fn transaction_cost(mut self, cost: f64) -> Self {
        self.transaction_cost = cost;
        self
    }

    /// Run backtest on historical data.
    ///
    /// # Arguments
    /// * `prices` - Price series
    /// * `timestamps` - Timestamps for each price
    /// * `predictions` - Model predictions (probabilities) for each timestep
    /// * `uncertainties` - Uncertainty estimates for each timestep
    ///
    /// # Returns
    /// Backtest report with performance metrics
    pub fn run(
        &self,
        prices: &[f64],
        timestamps: &[DateTime<Utc>],
        predictions: &[Vec<f64>],
        uncertainties: &[f64],
    ) -> BacktestReport {
        let mut portfolio = Portfolio::new(self.initial_capital);
        let mut trades = Vec::new();
        let mut equity_curve = Vec::with_capacity(prices.len());
        let mut returns = Vec::with_capacity(prices.len());

        let mut entry_time: Option<DateTime<Utc>> = None;
        let mut entry_signal: Option<(f64, f64)> = None; // (confidence, uncertainty)

        for i in 0..prices.len() {
            let price = prices[i];
            let timestamp = timestamps[i];

            // Update portfolio value
            portfolio.update_value(price);
            equity_curve.push(portfolio.total_value);

            if i > 0 {
                let ret = (portfolio.total_value - equity_curve[i - 1]) / equity_curve[i - 1];
                returns.push(ret);
            }

            // Get prediction and uncertainty
            let probs = &predictions[i];
            let uncertainty = uncertainties[i];
            let confidence = 1.0 / (1.0 + uncertainty);

            // Generate signal
            let signal = crate::strategy::signal::Signal::new(
                timestamp,
                if probs.len() >= 3 {
                    let prob_up = probs[0];
                    let prob_down = probs[2];
                    if prob_up > self.strategy.signal_generator.prob_threshold
                        && prob_up > prob_down
                    {
                        SignalType::Long
                    } else if prob_down > self.strategy.signal_generator.prob_threshold
                        && prob_down > prob_up
                    {
                        SignalType::Short
                    } else {
                        SignalType::Neutral
                    }
                } else {
                    SignalType::Neutral
                },
                if probs.is_empty() { 0.0 } else { probs[0] },
                uncertainty,
                if confidence >= self.strategy.signal_generator.conf_threshold {
                    self.strategy.signal_generator.kelly_position_size(
                        if probs.is_empty() { 0.0 } else { probs[0] },
                        confidence,
                        uncertainty,
                    )
                } else {
                    0.0
                },
            );

            // Check for exit conditions
            if portfolio.position != 0.0 {
                let (should_close, reason) = self.strategy.should_close_position(
                    portfolio.entry_price,
                    price,
                    portfolio.position_type,
                    &signal,
                );

                if should_close {
                    let pnl = portfolio.close_position(price, self.transaction_cost);
                    let pnl_pct = pnl / (portfolio.entry_price * portfolio.position.abs());

                    if let (Some(entry_t), Some((entry_conf, entry_unc))) =
                        (entry_time, entry_signal)
                    {
                        trades.push(Trade {
                            entry_time: entry_t,
                            exit_time: timestamp,
                            entry_price: portfolio.entry_price,
                            exit_price: price,
                            position_size: portfolio.position.abs(),
                            signal_type: portfolio.position_type,
                            pnl,
                            pnl_pct,
                            holding_period: (timestamp - entry_t).num_hours() as u32,
                            entry_confidence: entry_conf,
                            entry_uncertainty: entry_unc,
                            exit_reason: reason.to_string(),
                        });
                    }

                    entry_time = None;
                    entry_signal = None;
                }
            }

            // Check for entry conditions
            if portfolio.position == 0.0
                && signal.signal_type != SignalType::Neutral
                && signal.is_tradeable()
            {
                let position_value = self.strategy.calculate_position_value(
                    portfolio.total_value,
                    &signal,
                    portfolio.current_drawdown(),
                );

                portfolio.open_position(
                    signal.signal_type,
                    position_value,
                    price,
                    self.transaction_cost,
                );

                entry_time = Some(timestamp);
                entry_signal = Some((signal.confidence, signal.uncertainty));
            }
        }

        // Close any remaining position
        if portfolio.position != 0.0 && !prices.is_empty() {
            let final_price = *prices.last().unwrap();
            portfolio.close_position(final_price, self.transaction_cost);
        }

        // Create report
        let mut report = BacktestReport::new(
            trades,
            equity_curve,
            returns,
            self.initial_capital,
        );
        report.calculate_metrics();
        report
    }
}

impl crate::strategy::signal::SignalGenerator {
    /// Kelly position size (exposed for backtest engine).
    pub fn kelly_position_size(
        &self,
        prob_win: f64,
        confidence: f64,
        uncertainty: f64,
    ) -> f64 {
        let win_loss_ratio = 2.0;
        let b = win_loss_ratio;
        let q = 1.0 - prob_win;
        let kelly = (prob_win * b - q) / b;
        let uncertainty_penalty = uncertainty.sqrt();
        let adjusted_kelly = kelly * (1.0 - uncertainty_penalty) * confidence;
        adjusted_kelly.max(0.0).min(self.max_position_size)
    }
}
