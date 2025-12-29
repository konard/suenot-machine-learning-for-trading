//! Momentum propagation trading strategy.

use super::{Portfolio, Trade, TradeAction};
use crate::graph::CryptoGraph;
use crate::model::{GNNModel, Prediction};
use std::collections::HashMap;
use tch::Tensor;

/// Trading signal.
#[derive(Debug, Clone)]
pub struct Signal {
    /// Symbol
    pub symbol: String,
    /// Signal direction (1: long, -1: short, 0: neutral)
    pub direction: i32,
    /// Confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Reason for signal
    pub reason: String,
}

/// Trait for trading strategies.
pub trait TradingStrategy {
    /// Generate trading signals.
    fn generate_signals(
        &self,
        features: &Tensor,
        edge_index: &Tensor,
        symbols: &[String],
    ) -> Vec<Signal>;
}

/// Momentum propagation strategy.
///
/// Uses GNN predictions to identify cryptocurrencies that will follow
/// leaders in momentum movements.
pub struct MomentumStrategy<M: GNNModel> {
    /// GNN model for predictions
    model: M,
    /// Confidence threshold for signals
    threshold: f64,
    /// Minimum momentum for leaders
    min_leader_momentum: f64,
}

impl<M: GNNModel> MomentumStrategy<M> {
    /// Create a new momentum strategy.
    pub fn new(model: M, threshold: f64) -> Self {
        Self {
            model,
            threshold,
            min_leader_momentum: 0.02, // 2% minimum movement
        }
    }

    /// Set minimum leader momentum.
    pub fn with_min_momentum(mut self, min_momentum: f64) -> Self {
        self.min_leader_momentum = min_momentum;
        self
    }

    /// Get model predictions.
    pub fn get_predictions(
        &self,
        features: &Tensor,
        edge_index: &Tensor,
        symbols: &[String],
    ) -> Vec<Prediction> {
        self.model.predict(features, edge_index, None, symbols)
    }

    /// Identify leader cryptocurrencies.
    pub fn identify_leaders(
        &self,
        features: &Tensor,
        graph: &CryptoGraph,
        symbols: &[String],
    ) -> Vec<(String, f64)> {
        // Get momentum from features (assuming first feature is momentum)
        let momentum_vec: Vec<f64> =
            Vec::<f64>::try_from(features.select(1, 0)).unwrap();

        let mut leaders: Vec<(String, f64)> = symbols
            .iter()
            .enumerate()
            .filter_map(|(i, symbol)| {
                let momentum = momentum_vec.get(i).copied().unwrap_or(0.0);
                let degree = graph.degree(symbol);

                // Leaders have high momentum and high connectivity
                if momentum.abs() > self.min_leader_momentum && degree > 2 {
                    Some((symbol.clone(), momentum))
                } else {
                    None
                }
            })
            .collect();

        leaders.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        leaders
    }

    /// Find laggers that should follow leaders.
    pub fn find_laggers(
        &self,
        leaders: &[(String, f64)],
        predictions: &[Prediction],
        graph: &CryptoGraph,
    ) -> Vec<Signal> {
        let mut signals = Vec::new();

        for (leader, leader_momentum) in leaders {
            let neighbors = graph.neighbors(leader);

            for neighbor in neighbors {
                // Find prediction for this neighbor
                let pred = predictions.iter().find(|p| p.symbol == neighbor);

                if let Some(prediction) = pred {
                    // Check if prediction aligns with leader momentum
                    let expected_direction = if *leader_momentum > 0.0 { 2 } else { 0 };

                    if prediction.predicted_class == expected_direction
                        && prediction.confidence() > self.threshold
                    {
                        let direction = if expected_direction == 2 { 1 } else { -1 };

                        signals.push(Signal {
                            symbol: neighbor.clone(),
                            direction,
                            confidence: prediction.confidence(),
                            reason: format!(
                                "Following {} (momentum: {:.2}%)",
                                leader,
                                leader_momentum * 100.0
                            ),
                        });
                    }
                }
            }
        }

        // Remove duplicates, keeping highest confidence
        let mut unique_signals: HashMap<String, Signal> = HashMap::new();
        for signal in signals {
            unique_signals
                .entry(signal.symbol.clone())
                .and_modify(|existing| {
                    if signal.confidence > existing.confidence {
                        *existing = signal.clone();
                    }
                })
                .or_insert(signal);
        }

        unique_signals.into_values().collect()
    }
}

impl<M: GNNModel> TradingStrategy for MomentumStrategy<M> {
    fn generate_signals(
        &self,
        features: &Tensor,
        edge_index: &Tensor,
        symbols: &[String],
    ) -> Vec<Signal> {
        let predictions = self.get_predictions(features, edge_index, symbols);

        // Simple approach: use model predictions directly
        predictions
            .into_iter()
            .filter_map(|pred| {
                if pred.confidence() > self.threshold {
                    let direction = match pred.predicted_class {
                        0 => -1, // Short
                        2 => 1,  // Long
                        _ => return None,
                    };

                    Some(Signal {
                        symbol: pred.symbol,
                        direction,
                        confidence: pred.confidence(),
                        reason: format!(
                            "GNN prediction: {} ({:.1}%)",
                            pred.direction(),
                            pred.confidence() * 100.0
                        ),
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Backtest the momentum strategy.
pub fn backtest_strategy<M: GNNModel>(
    strategy: &MomentumStrategy<M>,
    historical_features: &[Tensor],
    historical_edge_indices: &[Tensor],
    historical_prices: &[HashMap<String, f64>],
    symbols: &[String],
    config: BacktestConfig,
) -> super::BacktestMetrics {
    let mut portfolio = Portfolio::new(
        config.initial_capital,
        config.transaction_cost,
        config.max_positions,
        config.position_size_pct,
    );

    let timestamps: Vec<i64> = (0..historical_features.len() as i64).collect();

    for (i, (features, edge_index)) in historical_features
        .iter()
        .zip(historical_edge_indices.iter())
        .enumerate()
    {
        if i >= historical_prices.len() {
            break;
        }

        let prices = &historical_prices[i];
        let timestamp = timestamps[i];

        // Generate signals
        let signals = strategy.generate_signals(features, edge_index, symbols);

        // Process signals
        for signal in &signals {
            if let Some(&price) = prices.get(&signal.symbol) {
                // Check if we should close existing position
                if let Some(position) = portfolio.positions.get(&signal.symbol) {
                    // Close if direction changed
                    if (position.direction > 0 && signal.direction < 0)
                        || (position.direction < 0 && signal.direction > 0)
                    {
                        portfolio.close_position(&signal.symbol, timestamp, price);
                    }
                }

                // Open new position
                if !portfolio.positions.contains_key(&signal.symbol) {
                    portfolio.open_position(
                        &signal.symbol,
                        timestamp,
                        price,
                        signal.direction,
                        signal.confidence,
                    );
                }
            }
        }

        // Close positions for symbols with low confidence
        let symbols_to_close: Vec<String> = portfolio
            .positions
            .keys()
            .filter(|sym| {
                !signals.iter().any(|s| &s.symbol == *sym && s.confidence > config.exit_threshold)
            })
            .cloned()
            .collect();

        for symbol in symbols_to_close {
            if let Some(&price) = prices.get(&symbol) {
                portfolio.close_position(&symbol, timestamp, price);
            }
        }

        // Update equity
        portfolio.update_equity(prices);
    }

    portfolio.calculate_metrics()
}

/// Backtest configuration.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost rate
    pub transaction_cost: f64,
    /// Maximum number of positions
    pub max_positions: usize,
    /// Position size as percentage of capital
    pub position_size_pct: f64,
    /// Threshold for exiting positions
    pub exit_threshold: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100000.0,
            transaction_cost: 0.001,
            max_positions: 5,
            position_size_pct: 0.1,
            exit_threshold: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{GCN, GNNConfig};
    use tch::Device;

    #[test]
    fn test_momentum_strategy() {
        let config = GNNConfig::default();
        let model = GCN::new(&config, Device::Cpu);
        let strategy = MomentumStrategy::new(model, 0.6);

        let features = Tensor::randn(&[5, 10], (tch::Kind::Float, Device::Cpu));
        let edge_index = Tensor::from_slice2(&[[0i64, 1, 2, 3], [1, 2, 3, 4]]);
        let symbols = vec![
            "BTC".to_string(),
            "ETH".to_string(),
            "SOL".to_string(),
            "AVAX".to_string(),
            "MATIC".to_string(),
        ];

        let signals = strategy.generate_signals(&features, &edge_index, &symbols);
        // Signals may or may not be generated depending on random model output
        assert!(signals.len() <= symbols.len());
    }
}
