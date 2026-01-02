//! Trading signal generation from graph analysis.

use crate::graph::{GraphMetrics, MarketGraph};
use std::collections::HashMap;

/// Graph-based trading signals
pub struct GraphSignals<'a> {
    graph: &'a MarketGraph,
    metrics: GraphMetrics<'a>,
}

impl<'a> GraphSignals<'a> {
    /// Create a new signal generator
    pub fn new(graph: &'a MarketGraph) -> Self {
        Self {
            graph,
            metrics: GraphMetrics::new(graph),
        }
    }

    /// Get betweenness centrality signals
    pub fn betweenness_centrality(&self) -> HashMap<String, f64> {
        self.metrics.betweenness_centrality()
    }

    /// Get degree centrality signals
    pub fn degree_centrality(&self) -> HashMap<String, f64> {
        self.metrics.degree_centrality()
    }

    /// Detect hub assets (high centrality)
    pub fn detect_hubs(&self, top_k: usize) -> Vec<(String, f64)> {
        self.metrics.detect_hubs(top_k)
    }

    /// Detect communities using simple label propagation
    pub fn detect_communities(&self) -> Vec<Vec<String>> {
        let symbols = self.graph.symbols();
        let n = symbols.len();

        if n == 0 {
            return Vec::new();
        }

        // Initialize each node with its own label
        let mut labels: HashMap<String, usize> = symbols
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        // Iterate until convergence
        for _ in 0..20 {
            let mut changed = false;

            for symbol in &symbols {
                let neighbors = self.graph.neighbors(symbol);
                if neighbors.is_empty() {
                    continue;
                }

                // Count neighbor labels
                let mut label_counts: HashMap<usize, usize> = HashMap::new();
                for neighbor in &neighbors {
                    if let Some(&label) = labels.get(neighbor) {
                        *label_counts.entry(label).or_insert(0) += 1;
                    }
                }

                // Find most common label
                if let Some((&most_common, _)) = label_counts.iter().max_by_key(|(_, &c)| c) {
                    if labels.get(symbol) != Some(&most_common) {
                        labels.insert(symbol.clone(), most_common);
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        // Group by label
        let mut communities: HashMap<usize, Vec<String>> = HashMap::new();
        for (symbol, label) in labels {
            communities.entry(label).or_default().push(symbol);
        }

        communities.into_values().collect()
    }

    /// Calculate graph momentum signal
    ///
    /// Nodes with high centrality and positive momentum indicate strength
    pub fn graph_momentum(
        &self,
        returns: &HashMap<String, f64>,
    ) -> HashMap<String, f64> {
        let centrality = self.betweenness_centrality();

        self.graph
            .symbols()
            .into_iter()
            .map(|symbol| {
                let ret = returns.get(&symbol).copied().unwrap_or(0.0);
                let cent = centrality.get(&symbol).copied().unwrap_or(0.0);

                // Momentum signal = return * centrality weight
                let signal = ret * (1.0 + cent);
                (symbol, signal)
            })
            .collect()
    }

    /// Calculate network stress indicator
    ///
    /// High average correlation + high density = stressed market
    pub fn network_stress(&self) -> f64 {
        let density = self.graph.density();
        let avg_weight = self.graph.average_edge_weight();
        let clustering = self.metrics.average_clustering();

        // Combine metrics into stress score (0-1)
        let stress = (density + avg_weight.abs() + clustering) / 3.0;
        stress.min(1.0).max(0.0)
    }

    /// Generate long/short signals based on centrality ranking
    pub fn centrality_signals(&self, long_pct: f64, short_pct: f64) -> HashMap<String, f64> {
        let centrality = self.betweenness_centrality();
        let mut ranked: Vec<_> = centrality.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let n = ranked.len();
        let n_long = ((n as f64 * long_pct) as usize).max(1);
        let n_short = ((n as f64 * short_pct) as usize).max(1);

        let mut signals = HashMap::new();

        // Long top centrality
        for (symbol, _) in ranked.iter().take(n_long) {
            signals.insert(symbol.clone(), 1.0);
        }

        // Short bottom centrality
        for (symbol, _) in ranked.iter().rev().take(n_short) {
            signals.insert(symbol.clone(), -1.0);
        }

        // Neutral for others
        for (symbol, _) in &ranked {
            signals.entry(symbol.clone()).or_insert(0.0);
        }

        signals
    }

    /// Generate community momentum signals
    pub fn community_momentum_signals(
        &self,
        returns: &HashMap<String, f64>,
    ) -> HashMap<String, f64> {
        let communities = self.detect_communities();

        // Calculate community momentum
        let community_momentum: Vec<(usize, f64)> = communities
            .iter()
            .enumerate()
            .map(|(i, community)| {
                let total_return: f64 = community
                    .iter()
                    .filter_map(|s| returns.get(s))
                    .sum();
                let avg = total_return / community.len().max(1) as f64;
                (i, avg)
            })
            .collect();

        // Rank communities
        let mut ranked = community_momentum.clone();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_community = ranked.first().map(|x| x.0);
        let bottom_community = ranked.last().map(|x| x.0);

        // Assign signals
        let mut signals = HashMap::new();
        for (i, community) in communities.iter().enumerate() {
            let signal = if Some(i) == top_community {
                1.0
            } else if Some(i) == bottom_community {
                -0.5
            } else {
                0.0
            };

            for symbol in community {
                signals.insert(symbol.clone(), signal);
            }
        }

        signals
    }

    /// Get regime indicator based on graph structure
    pub fn regime_indicator(&self) -> MarketRegime {
        let stress = self.network_stress();
        let density = self.graph.density();
        let avg_clustering = self.metrics.average_clustering();

        if stress > 0.7 && density > 0.6 {
            MarketRegime::Crisis
        } else if stress > 0.5 {
            MarketRegime::RiskOff
        } else if avg_clustering < 0.3 && density < 0.4 {
            MarketRegime::RiskOn
        } else {
            MarketRegime::Normal
        }
    }
}

/// Market regime based on graph structure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    RiskOn,
    Normal,
    RiskOff,
    Crisis,
}

impl MarketRegime {
    /// Get recommended position sizing multiplier
    pub fn position_multiplier(&self) -> f64 {
        match self {
            MarketRegime::RiskOn => 1.2,
            MarketRegime::Normal => 1.0,
            MarketRegime::RiskOff => 0.7,
            MarketRegime::Crisis => 0.3,
        }
    }

    /// Get recommended leverage
    pub fn max_leverage(&self) -> f64 {
        match self {
            MarketRegime::RiskOn => 3.0,
            MarketRegime::Normal => 2.0,
            MarketRegime::RiskOff => 1.5,
            MarketRegime::Crisis => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> MarketGraph {
        let mut graph = MarketGraph::new();
        graph.add_edge("BTC", "ETH", 0.85);
        graph.add_edge("BTC", "SOL", 0.72);
        graph.add_edge("ETH", "SOL", 0.78);
        graph.add_edge("DOGE", "SHIB", 0.9);
        graph
    }

    #[test]
    fn test_detect_communities() {
        let graph = create_test_graph();
        let signals = GraphSignals::new(&graph);
        let communities = signals.detect_communities();

        // Should detect 2 communities: (BTC, ETH, SOL) and (DOGE, SHIB)
        assert_eq!(communities.len(), 2);
    }

    #[test]
    fn test_centrality_signals() {
        let graph = create_test_graph();
        let signals = GraphSignals::new(&graph);
        let trading_signals = signals.centrality_signals(0.3, 0.3);

        assert!(trading_signals.contains_key("BTC"));
    }

    #[test]
    fn test_network_stress() {
        let graph = create_test_graph();
        let signals = GraphSignals::new(&graph);
        let stress = signals.network_stress();

        assert!(stress >= 0.0 && stress <= 1.0);
    }

    #[test]
    fn test_regime_indicator() {
        let graph = create_test_graph();
        let signals = GraphSignals::new(&graph);
        let regime = signals.regime_indicator();

        // Should be one of the valid regimes
        assert!(matches!(
            regime,
            MarketRegime::RiskOn | MarketRegime::Normal | MarketRegime::RiskOff | MarketRegime::Crisis
        ));
    }
}
