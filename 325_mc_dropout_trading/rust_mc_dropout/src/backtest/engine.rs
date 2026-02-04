//! Backtesting engine

use super::report::BacktestReport;
use crate::model::dropout::MCDropoutModel;
use crate::strategy::signal::{Signal, SignalGenerator, SignalType};
use ndarray::Array1;

/// Trade record
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_time: i64,
    pub exit_time: Option<i64>,
    pub signal_type: SignalType,
    pub entry_price: f64,
    pub exit_price: Option<f64>,
    pub position_size: f64,
    pub predicted_return: f64,
    pub uncertainty: f64,
    pub actual_return: Option<f64>,
    pub pnl: Option<f64>,
}

impl Trade {
    /// Close the trade
    pub fn close(&mut self, exit_time: i64, exit_price: f64) {
        self.exit_time = Some(exit_time);
        self.exit_price = Some(exit_price);

        let return_pct = match self.signal_type {
            SignalType::Long => (exit_price - self.entry_price) / self.entry_price,
            SignalType::Short => (self.entry_price - exit_price) / self.entry_price,
            SignalType::Hold => 0.0,
        };

        self.actual_return = Some(return_pct);
        self.pnl = Some(return_pct * self.position_size);
    }
}

/// Backtesting engine configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Commission per trade (fraction)
    pub commission: f64,
    /// Slippage (fraction)
    pub slippage: f64,
    /// Number of bars to hold each position
    pub holding_period: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100000.0,
            commission: 0.001,
            slippage: 0.0005,
            holding_period: 1,
        }
    }
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
        model: &MCDropoutModel,
        signal_generator: &SignalGenerator,
        features: &[Array1<f64>],
        prices: &[f64],
        timestamps: Option<&[i64]>,
    ) -> BacktestReport {
        let n_samples = features.len().min(prices.len());
        if n_samples < self.config.holding_period + 1 {
            return BacktestReport::default();
        }

        let timestamps: Vec<i64> = timestamps
            .map(|t| t.to_vec())
            .unwrap_or_else(|| (0..n_samples as i64).collect());

        let mut capital = self.config.initial_capital;
        let mut equity_history = vec![capital];
        let mut trades: Vec<Trade> = Vec::new();
        let mut open_trade: Option<Trade> = None;
        let mut bars_held = 0;

        // Track for uncertainty analysis
        let mut predictions: Vec<f64> = Vec::new();
        let mut uncertainties: Vec<f64> = Vec::new();
        let mut actual_returns: Vec<f64> = Vec::new();

        for i in 0..(n_samples - self.config.holding_period) {
            let current_price = prices[i];
            let current_time = timestamps[i];

            // Check if we should close existing position
            if let Some(ref mut trade) = open_trade {
                bars_held += 1;

                if bars_held >= self.config.holding_period {
                    // Close position
                    let slippage_mult = match trade.signal_type {
                        SignalType::Long => 1.0 - self.config.slippage,
                        SignalType::Short => 1.0 + self.config.slippage,
                        SignalType::Hold => 1.0,
                    };

                    let exit_price = prices[i] * slippage_mult;
                    trade.close(current_time, exit_price);

                    // Apply commission
                    let trade_pnl = trade.pnl.unwrap_or(0.0) - self.config.commission * 2.0;
                    capital += capital * trade_pnl;

                    // Track for analysis
                    if let Some(actual_ret) = trade.actual_return {
                        actual_returns.push(actual_ret);
                    }

                    trades.push(trade.clone());
                    open_trade = None;
                    bars_held = 0;
                }
            }

            // Generate signal if no open position
            if open_trade.is_none() {
                let prediction = model.predict_with_uncertainty(&features[i], None);
                let signal = signal_generator.generate(&prediction);

                if signal.is_actionable() {
                    // Open position
                    let slippage_mult = match signal.signal_type {
                        SignalType::Long => 1.0 + self.config.slippage,
                        SignalType::Short => 1.0 - self.config.slippage,
                        SignalType::Hold => 1.0,
                    };

                    let entry_price = current_price * slippage_mult;

                    open_trade = Some(Trade {
                        entry_time: current_time,
                        exit_time: None,
                        signal_type: signal.signal_type,
                        entry_price,
                        exit_price: None,
                        position_size: signal.position_size,
                        predicted_return: signal.predicted_return,
                        uncertainty: signal.uncertainty,
                        actual_return: None,
                        pnl: None,
                    });

                    predictions.push(signal.predicted_return);
                    uncertainties.push(signal.uncertainty);
                }
            }

            equity_history.push(capital);
        }

        // Close any remaining position
        if let Some(ref mut trade) = open_trade {
            let exit_price = *prices.last().unwrap();
            trade.close(*timestamps.last().unwrap(), exit_price);
            if let Some(actual_ret) = trade.actual_return {
                actual_returns.push(actual_ret);
            }
            trades.push(trade.clone());
        }

        // Calculate metrics
        self.calculate_metrics(&trades, &equity_history, &predictions, &uncertainties, &actual_returns)
    }

    fn calculate_metrics(
        &self,
        trades: &[Trade],
        equity_history: &[f64],
        predictions: &[f64],
        uncertainties: &[f64],
        actual_returns: &[f64],
    ) -> BacktestReport {
        let mut report = BacktestReport::default();

        // Trade statistics
        report.total_trades = trades.len();
        report.winning_trades = trades.iter().filter(|t| t.pnl.unwrap_or(0.0) > 0.0).count();
        report.losing_trades = trades.iter().filter(|t| t.pnl.unwrap_or(0.0) <= 0.0).count();

        if report.total_trades > 0 {
            report.win_rate = report.winning_trades as f64 / report.total_trades as f64;
        }

        // Returns
        let initial = equity_history.first().copied().unwrap_or(1.0);
        let final_val = equity_history.last().copied().unwrap_or(1.0);
        report.total_return = (final_val - initial) / initial;

        // Annualized (assuming hourly data)
        let n_years = equity_history.len() as f64 / (24.0 * 365.0);
        if n_years > 0.0 {
            report.annualized_return = (1.0 + report.total_return).powf(1.0 / n_years) - 1.0;
        }

        // Calculate returns for Sharpe/Sortino
        let returns: Vec<f64> = equity_history
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        if !returns.is_empty() {
            let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > 0.0 {
                report.sharpe_ratio = mean_return / std_dev * (24.0 * 365.0_f64).sqrt();
            }

            // Sortino (downside deviation)
            let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
            if !downside.is_empty() {
                let downside_variance: f64 = downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
                let downside_std = downside_variance.sqrt();
                if downside_std > 0.0 {
                    report.sortino_ratio = mean_return / downside_std * (24.0 * 365.0_f64).sqrt();
                }
            }
        }

        // Maximum drawdown
        let mut peak = equity_history[0];
        let mut max_dd = 0.0;
        for &equity in equity_history {
            if equity > peak {
                peak = equity;
            }
            let dd = (peak - equity) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        report.max_drawdown = max_dd;

        // Profit factor
        let gross_profit: f64 = trades.iter()
            .filter_map(|t| t.pnl)
            .filter(|&p| p > 0.0)
            .sum();
        let gross_loss: f64 = trades.iter()
            .filter_map(|t| t.pnl)
            .filter(|&p| p < 0.0)
            .map(|p| -p)
            .sum();
        if gross_loss > 0.0 {
            report.profit_factor = gross_profit / gross_loss;
        }

        // Uncertainty metrics
        if !uncertainties.is_empty() {
            report.mean_uncertainty = uncertainties.iter().sum::<f64>() / uncertainties.len() as f64;
        }

        if predictions.len() == actual_returns.len() && !predictions.is_empty() {
            // 95% coverage
            let mut within_ci = 0;
            for i in 0..predictions.len() {
                let lower = predictions[i] - 1.96 * uncertainties[i];
                let upper = predictions[i] + 1.96 * uncertainties[i];
                if actual_returns[i] >= lower && actual_returns[i] <= upper {
                    within_ci += 1;
                }
            }
            report.uncertainty_coverage_95 = within_ci as f64 / predictions.len() as f64;

            // Correlation
            let errors: Vec<f64> = predictions.iter()
                .zip(actual_returns.iter())
                .map(|(p, a)| (a - p).abs())
                .collect();

            let mean_unc: f64 = uncertainties.iter().sum::<f64>() / uncertainties.len() as f64;
            let mean_err: f64 = errors.iter().sum::<f64>() / errors.len() as f64;

            let cov: f64 = uncertainties.iter()
                .zip(errors.iter())
                .map(|(u, e)| (u - mean_unc) * (e - mean_err))
                .sum::<f64>() / uncertainties.len() as f64;

            let var_unc: f64 = uncertainties.iter().map(|u| (u - mean_unc).powi(2)).sum::<f64>() / uncertainties.len() as f64;
            let var_err: f64 = errors.iter().map(|e| (e - mean_err).powi(2)).sum::<f64>() / errors.len() as f64;

            let std_unc = var_unc.sqrt();
            let std_err = var_err.sqrt();

            if std_unc > 0.0 && std_err > 0.0 {
                report.uncertainty_correlation = cov / (std_unc * std_err);
            }
        }

        report
    }
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self::new(BacktestConfig::default())
    }
}
