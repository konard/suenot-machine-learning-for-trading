//! Trading Strategy and Backtesting
//!
//! Provides a causality-driven trading strategy using the PCMCI algorithm
//! and a backtesting framework for evaluation.

use crate::data::MarketData;
use crate::model::{CausalGraph, CausalLink, IndependenceTest, PCMCICausalDiscovery, PCMCIConfig};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Configuration for backtesting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f32,
    /// Position sizing (fraction of capital)
    pub position_size: f32,
    /// Stop loss percentage
    pub stop_loss: Option<f32>,
    /// Take profit percentage
    pub take_profit: Option<f32>,
    /// Trading fee percentage
    pub fee_rate: f32,
    /// Slippage percentage
    pub slippage: f32,
    /// Lookback window for causal analysis
    pub lookback_window: usize,
    /// Re-estimation frequency (bars between causal graph updates)
    pub reestimate_freq: usize,
    /// Minimum causal link strength to generate a signal
    pub min_link_strength: f64,
    /// Significance level for causal links
    pub significance_level: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size: 0.1,
            stop_loss: Some(0.02),
            take_profit: Some(0.04),
            fee_rate: 0.001,
            slippage: 0.0005,
            lookback_window: 200,
            reestimate_freq: 50,
            min_link_strength: 0.15,
            significance_level: 0.05,
        }
    }
}

/// Trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Entry timestamp
    pub entry_time: i64,
    /// Exit timestamp
    pub exit_time: i64,
    /// Entry price
    pub entry_price: f32,
    /// Exit price
    pub exit_price: f32,
    /// Position size (positive for long, negative for short)
    pub size: f32,
    /// Profit/loss
    pub pnl: f32,
    /// Return percentage
    pub return_pct: f32,
    /// Signal strength at entry
    pub signal_strength: f64,
    /// Exit reason
    pub exit_reason: String,
    /// Number of causal links used for signal
    pub n_causal_links: usize,
}

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// Buy/long signal
    Long,
    /// Sell/short signal
    Short,
    /// No action
    Hold,
}

/// Causal signal with metadata
#[derive(Debug, Clone)]
pub struct CausalSignal {
    /// Trading direction
    pub signal: Signal,
    /// Aggregate signal strength
    pub strength: f64,
    /// Number of causal links contributing
    pub n_links: usize,
    /// The strongest causal link driving the signal
    pub primary_cause: Option<CausalLink>,
}

/// Trading strategy based on PCMCI causal discovery
pub struct CausalTradingStrategy {
    pcmci_config: PCMCIConfig,
    backtest_config: BacktestConfig,
    current_graph: Option<CausalGraph>,
    bars_since_update: usize,
    /// Variable names for interpretation
    variable_names: Vec<String>,
}

impl CausalTradingStrategy {
    /// Create a new causal trading strategy
    pub fn new(pcmci_config: PCMCIConfig, backtest_config: BacktestConfig) -> Self {
        Self {
            pcmci_config,
            backtest_config,
            current_graph: None,
            bars_since_update: 0,
            variable_names: vec![
                "returns".to_string(),
                "volume_change".to_string(),
                "volatility".to_string(),
                "spread".to_string(),
            ],
        }
    }

    /// Update the causal graph using recent data
    pub fn update_causal_graph(&mut self, data: &Array2<f64>) {
        let mut pcmci = PCMCICausalDiscovery::new(self.pcmci_config.clone());
        self.current_graph = Some(pcmci.fit(data));
        self.bars_since_update = 0;
    }

    /// Check if the causal graph needs to be re-estimated
    pub fn needs_update(&self) -> bool {
        self.current_graph.is_none()
            || self.bars_since_update >= self.backtest_config.reestimate_freq
    }

    /// Generate a trading signal based on the current causal graph and recent data
    ///
    /// The signal is derived from causal links pointing to the returns variable (index 0):
    /// - Positive causal effects from increasing variables -> Long
    /// - Negative causal effects from increasing variables -> Short
    pub fn generate_signal(&mut self, causal_data: &Array2<f64>) -> CausalSignal {
        self.bars_since_update += 1;

        let graph = match &self.current_graph {
            Some(g) => g,
            None => {
                return CausalSignal {
                    signal: Signal::Hold,
                    strength: 0.0,
                    n_links: 0,
                    primary_cause: None,
                };
            }
        };

        // Get significant causal links pointing to returns (variable 0)
        let return_parents = graph.get_parents(0, self.backtest_config.significance_level);

        if return_parents.is_empty() {
            return CausalSignal {
                signal: Signal::Hold,
                strength: 0.0,
                n_links: 0,
                primary_cause: None,
            };
        }

        // Compute aggregate signal by weighting recent observations by causal strength
        let n_rows = causal_data.nrows();
        let mut signal_value = 0.0;
        let mut total_weight = 0.0;

        for link in &return_parents {
            if link.strength.abs() < self.backtest_config.min_link_strength {
                continue;
            }

            let source = link.source;
            let lag = link.lag;

            // Get the recent value of the source variable (at the correct lag)
            if n_rows > lag && source < causal_data.ncols() {
                let source_value = causal_data[[n_rows - 1 - lag, source]];
                signal_value += link.strength * source_value;
                total_weight += link.strength.abs();
            }
        }

        if total_weight < 1e-10 {
            return CausalSignal {
                signal: Signal::Hold,
                strength: 0.0,
                n_links: return_parents.len(),
                primary_cause: return_parents.into_iter().next(),
            };
        }

        // Normalize signal
        let normalized_signal = signal_value / total_weight;

        // Determine direction
        let signal = if normalized_signal > 0.0 {
            Signal::Long
        } else if normalized_signal < 0.0 {
            Signal::Short
        } else {
            Signal::Hold
        };

        let primary_cause = return_parents
            .iter()
            .max_by(|a, b| a.strength.abs().partial_cmp(&b.strength.abs()).unwrap())
            .cloned();

        CausalSignal {
            signal,
            strength: normalized_signal.abs(),
            n_links: return_parents.len(),
            primary_cause,
        }
    }

    /// Get the current causal graph
    pub fn graph(&self) -> Option<&CausalGraph> {
        self.current_graph.as_ref()
    }

    /// Get variable names
    pub fn variable_names(&self) -> &[String] {
        &self.variable_names
    }

    /// Set custom variable names
    pub fn set_variable_names(&mut self, names: Vec<String>) {
        self.variable_names = names;
    }

    /// Print a summary of the current causal graph
    pub fn print_graph_summary(&self) {
        if let Some(graph) = &self.current_graph {
            let summary = graph.summary();
            println!("Causal Graph Summary:");
            println!("  Variables: {}", summary.n_vars);
            println!("  Max lag: {}", summary.max_lag);
            println!("  Significant links: {}", summary.n_significant_links);
            println!("  Average link strength: {:.4}", summary.avg_link_strength);
            println!("  Max link strength: {:.4}", summary.max_link_strength);

            let significant = graph.get_significant_links(self.backtest_config.significance_level);
            if !significant.is_empty() {
                println!("  Significant causal links:");
                for link in &significant {
                    let source_name = self
                        .variable_names
                        .get(link.source)
                        .cloned()
                        .unwrap_or_else(|| format!("Var{}", link.source));
                    let target_name = self
                        .variable_names
                        .get(link.target)
                        .cloned()
                        .unwrap_or_else(|| format!("Var{}", link.target));
                    println!(
                        "    {} (t-{}) -> {}: strength={:.4}, p={:.4}",
                        source_name, link.lag, target_name, link.strength, link.p_value
                    );
                }
            }
        } else {
            println!("No causal graph estimated yet.");
        }
    }
}

/// Backtest result metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Total return
    pub total_return: f32,
    /// Annualized return
    pub annual_return: f32,
    /// Sharpe ratio
    pub sharpe_ratio: f32,
    /// Maximum drawdown
    pub max_drawdown: f32,
    /// Win rate
    pub win_rate: f32,
    /// Profit factor
    pub profit_factor: f32,
    /// Total number of trades
    pub total_trades: usize,
    /// Average trade return
    pub avg_trade_return: f32,
    /// All trades
    pub trades: Vec<Trade>,
    /// Equity curve
    pub equity_curve: Vec<f32>,
    /// Timestamps for equity curve
    pub timestamps: Vec<i64>,
    /// Number of causal graph updates
    pub n_graph_updates: usize,
    /// Average number of significant causal links per update
    pub avg_significant_links: f32,
}

/// Backtester for evaluating causal trading strategies
pub struct Backtester {
    config: BacktestConfig,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest on market data using the causal trading strategy
    pub fn run(&self, strategy: &mut CausalTradingStrategy, data: &MarketData) -> BacktestResult {
        let mut capital = self.config.initial_capital;
        let mut position: Option<(f32, f32, i64, f64, usize)> = None;
        // (size, entry_price, entry_time, signal_strength, n_links)
        let mut trades: Vec<Trade> = Vec::new();
        let mut equity_curve: Vec<f32> = Vec::new();
        let mut timestamps: Vec<i64> = Vec::new();
        let mut n_graph_updates: usize = 0;
        let mut total_significant_links: usize = 0;

        let lookback = self.config.lookback_window;

        // Need enough data for lookback window plus some buffer
        if data.len() < lookback + 10 {
            return self.empty_result();
        }

        // Extract causal variables from market data
        let all_causal_vars = data.to_causal_variables();
        let n_causal_rows = all_causal_vars.nrows();

        // The causal variables start 2 bars into the original data
        let offset = 2;

        for i in lookback..n_causal_rows {
            let bar_idx = i + offset;
            if bar_idx >= data.bars.len() {
                break;
            }

            let current_bar = &data.bars[bar_idx];
            let current_price = current_bar.close;

            // Update causal graph if needed
            if strategy.needs_update() && i >= lookback {
                let window_start = i - lookback;
                let window_data = all_causal_vars
                    .slice(ndarray::s![window_start..i, ..])
                    .to_owned();

                // Standardize the window data
                let standardized = crate::data::standardize(&window_data);
                strategy.update_causal_graph(&standardized);
                n_graph_updates += 1;

                if let Some(graph) = strategy.graph() {
                    let sig_links = graph
                        .get_significant_links(self.config.significance_level)
                        .len();
                    total_significant_links += sig_links;
                }
            }

            // Record equity
            let unrealized_pnl = if let Some((size, entry_price, _, _, _)) = &position {
                size * (current_price - entry_price)
            } else {
                0.0
            };
            equity_curve.push(capital + unrealized_pnl);
            timestamps.push(current_bar.timestamp);

            // Check exit conditions for existing position
            if let Some((size, entry_price, entry_time, strength, n_links)) = position.take() {
                let mut should_exit = false;
                let mut exit_reason = String::new();

                let pnl_pct = if size > 0.0 {
                    (current_price - entry_price) / entry_price
                } else {
                    (entry_price - current_price) / entry_price
                };

                // Check stop loss
                if let Some(stop_loss) = self.config.stop_loss {
                    if pnl_pct < -stop_loss {
                        should_exit = true;
                        exit_reason = "stop_loss".to_string();
                    }
                }

                // Check take profit
                if let Some(take_profit) = self.config.take_profit {
                    if pnl_pct > take_profit {
                        should_exit = true;
                        exit_reason = "take_profit".to_string();
                    }
                }

                // Check end of data
                if i >= n_causal_rows - 2 {
                    should_exit = true;
                    exit_reason = "end_of_data".to_string();
                }

                if should_exit {
                    let exit_price = current_price * (1.0 - self.config.slippage * size.signum());
                    let raw_pnl = size * (exit_price - entry_price);
                    let fees = (size.abs() * entry_price + size.abs() * exit_price) * self.config.fee_rate;
                    let net_pnl = raw_pnl - fees;

                    capital += net_pnl;

                    trades.push(Trade {
                        entry_time,
                        exit_time: current_bar.timestamp,
                        entry_price,
                        exit_price,
                        size,
                        pnl: net_pnl,
                        return_pct: net_pnl / (size.abs() * entry_price),
                        signal_strength: strength,
                        exit_reason,
                        n_causal_links: n_links,
                    });
                } else {
                    position = Some((size, entry_price, entry_time, strength, n_links));
                }
            }

            // Generate signal if no position
            if position.is_none() && i >= lookback {
                let recent_data = all_causal_vars
                    .slice(ndarray::s![(i.saturating_sub(20))..i, ..])
                    .to_owned();

                let causal_signal = strategy.generate_signal(&recent_data);

                if causal_signal.strength >= self.config.min_link_strength as f64 {
                    match causal_signal.signal {
                        Signal::Long => {
                            let size = (capital * self.config.position_size) / current_price;
                            let entry_price = current_price * (1.0 + self.config.slippage);
                            position = Some((
                                size,
                                entry_price,
                                current_bar.timestamp,
                                causal_signal.strength,
                                causal_signal.n_links,
                            ));
                        }
                        Signal::Short => {
                            let size = -(capital * self.config.position_size) / current_price;
                            let entry_price = current_price * (1.0 - self.config.slippage);
                            position = Some((
                                size,
                                entry_price,
                                current_bar.timestamp,
                                causal_signal.strength,
                                causal_signal.n_links,
                            ));
                        }
                        Signal::Hold => {}
                    }
                }
            }
        }

        // Close any remaining position
        if let Some((size, entry_price, entry_time, strength, n_links)) = position {
            let last_bar = data.bars.last().unwrap();
            let exit_price = last_bar.close * (1.0 - self.config.slippage * size.signum());
            let raw_pnl = size * (exit_price - entry_price);
            let fees = (size.abs() * entry_price + size.abs() * exit_price) * self.config.fee_rate;
            let net_pnl = raw_pnl - fees;

            capital += net_pnl;

            trades.push(Trade {
                entry_time,
                exit_time: last_bar.timestamp,
                entry_price,
                exit_price,
                size,
                pnl: net_pnl,
                return_pct: net_pnl / (size.abs() * entry_price),
                signal_strength: strength,
                exit_reason: "end_of_backtest".to_string(),
                n_causal_links: n_links,
            });
        }

        let avg_significant_links = if n_graph_updates > 0 {
            total_significant_links as f32 / n_graph_updates as f32
        } else {
            0.0
        };

        self.calculate_metrics(capital, trades, equity_curve, timestamps, n_graph_updates, avg_significant_links)
    }

    fn empty_result(&self) -> BacktestResult {
        BacktestResult {
            total_return: 0.0,
            annual_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            total_trades: 0,
            avg_trade_return: 0.0,
            trades: vec![],
            equity_curve: vec![self.config.initial_capital],
            timestamps: vec![],
            n_graph_updates: 0,
            avg_significant_links: 0.0,
        }
    }

    fn calculate_metrics(
        &self,
        final_capital: f32,
        trades: Vec<Trade>,
        equity_curve: Vec<f32>,
        timestamps: Vec<i64>,
        n_graph_updates: usize,
        avg_significant_links: f32,
    ) -> BacktestResult {
        let total_return = (final_capital - self.config.initial_capital) / self.config.initial_capital;

        // Calculate max drawdown
        let mut max_equity = self.config.initial_capital;
        let mut max_drawdown = 0.0_f32;
        for &equity in &equity_curve {
            max_equity = max_equity.max(equity);
            let drawdown = (max_equity - equity) / max_equity;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Calculate trade statistics
        let total_trades = trades.len();
        let winning_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let win_rate = if total_trades > 0 {
            winning_trades.len() as f32 / total_trades as f32
        } else {
            0.0
        };

        let gross_profit: f32 = winning_trades.iter().map(|t| t.pnl).sum();
        let gross_loss: f32 = losing_trades.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f32::INFINITY
        } else {
            0.0
        };

        let avg_trade_return = if total_trades > 0 {
            trades.iter().map(|t| t.return_pct).sum::<f32>() / total_trades as f32
        } else {
            0.0
        };

        // Calculate returns for Sharpe ratio
        let returns: Vec<f32> = equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean_return = if returns.is_empty() {
            0.0
        } else {
            returns.iter().sum::<f32>() / returns.len() as f32
        };

        let std_return = if returns.len() > 1 {
            let variance: f32 = returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f32>()
                / (returns.len() - 1) as f32;
            variance.sqrt()
        } else {
            0.0
        };

        let sharpe_ratio = if std_return > 0.0 {
            (mean_return / std_return) * (252.0_f32).sqrt()
        } else {
            0.0
        };

        // Annualized return (assuming daily data, 252 trading days)
        let n_periods = equity_curve.len().max(1);
        let annual_return = ((1.0 + total_return).powf(252.0 / n_periods as f32)) - 1.0;

        BacktestResult {
            total_return,
            annual_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            total_trades,
            avg_trade_return,
            trades,
            equity_curve,
            timestamps,
            n_graph_updates,
            avg_significant_links,
        }
    }
}

/// Run walk-forward backtest with causal discovery
pub fn walk_forward_causal_backtest(
    data: &MarketData,
    pcmci_config: PCMCIConfig,
    backtest_config: BacktestConfig,
    n_splits: usize,
    train_ratio: f32,
) -> Vec<BacktestResult> {
    let n = data.len();
    if n == 0 || n_splits == 0 || n_splits > n {
        return Vec::new();
    }
    if !(0.0..1.0).contains(&train_ratio) {
        return Vec::new();
    }

    let split_size = n / n_splits;
    let mut results = Vec::new();

    for i in 0..n_splits {
        let split_end = ((i + 1) * split_size).min(n);
        let split_data = MarketData::new(
            data.symbol.clone(),
            data.bars[..split_end].to_vec(),
        );

        let train_end = (split_data.len() as f32 * train_ratio) as usize;
        let test_data = MarketData::new(
            data.symbol.clone(),
            split_data.bars[train_end..].to_vec(),
        );

        if test_data.len() < backtest_config.lookback_window + 20 {
            continue;
        }

        let mut strategy = CausalTradingStrategy::new(
            pcmci_config.clone(),
            backtest_config.clone(),
        );

        let backtester = Backtester::new(backtest_config.clone());
        let result = backtester.run(&mut strategy, &test_data);
        results.push(result);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::generate_synthetic_data;

    #[test]
    fn test_causal_trading_strategy_creation() {
        let pcmci_config = PCMCIConfig {
            max_lag: 3,
            significance_level: 0.05,
            test_type: IndependenceTest::ParCorr,
            max_conds_dim: Some(3),
        };

        let backtest_config = BacktestConfig::default();

        let strategy = CausalTradingStrategy::new(pcmci_config, backtest_config);
        assert!(strategy.graph().is_none());
        assert!(strategy.needs_update());
    }

    #[test]
    fn test_backtester() {
        let pcmci_config = PCMCIConfig {
            max_lag: 2,
            significance_level: 0.1,
            test_type: IndependenceTest::ParCorr,
            max_conds_dim: Some(2),
        };

        let backtest_config = BacktestConfig {
            initial_capital: 10000.0,
            position_size: 0.1,
            stop_loss: Some(0.02),
            take_profit: Some(0.04),
            fee_rate: 0.001,
            slippage: 0.0005,
            lookback_window: 50,
            reestimate_freq: 25,
            min_link_strength: 0.1,
            significance_level: 0.1,
        };

        let mut strategy = CausalTradingStrategy::new(pcmci_config, backtest_config.clone());

        let data = generate_synthetic_data(200, 0.02);
        let backtester = Backtester::new(backtest_config);
        let result = backtester.run(&mut strategy, &data);

        // Basic sanity checks
        assert!(!result.equity_curve.is_empty());
        assert!(result.max_drawdown >= 0.0);
        assert!(result.max_drawdown <= 1.0);
        assert!(result.n_graph_updates > 0);
    }

    #[test]
    fn test_signal_generation() {
        let pcmci_config = PCMCIConfig {
            max_lag: 2,
            significance_level: 0.1,
            test_type: IndependenceTest::ParCorr,
            max_conds_dim: Some(2),
        };

        let backtest_config = BacktestConfig {
            min_link_strength: 0.05,
            significance_level: 0.1,
            ..BacktestConfig::default()
        };

        let mut strategy = CausalTradingStrategy::new(pcmci_config, backtest_config);

        // Generate data and update causal graph
        let data = generate_synthetic_data(200, 0.02);
        let causal_vars = data.to_causal_variables();

        if causal_vars.nrows() > 50 {
            let window = causal_vars.slice(ndarray::s![..50, ..]).to_owned();
            let standardized = crate::data::standardize(&window);
            strategy.update_causal_graph(&standardized);

            // Generate a signal
            let recent = causal_vars.slice(ndarray::s![40..50, ..]).to_owned();
            let signal = strategy.generate_signal(&recent);

            // Signal should be valid
            assert!(signal.strength >= 0.0);
        }
    }

    #[test]
    fn test_backtest_config_default() {
        let config = BacktestConfig::default();
        assert_eq!(config.initial_capital, 100_000.0);
        assert_eq!(config.position_size, 0.1);
        assert_eq!(config.lookback_window, 200);
        assert_eq!(config.reestimate_freq, 50);
    }
}
