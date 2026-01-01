//! Network Topology Module
//!
//! Defines different connectivity patterns for neural networks.

use crate::neuron::synapse::{Synapse, SynapseConfig};
use rand::Rng;

/// Connectivity pattern types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConnectivityPattern {
    /// Full connectivity (all-to-all)
    Dense,
    /// Sparse random connectivity
    Sparse(f64),  // connection probability
    /// One-to-one connectivity
    OneToOne,
    /// Local connectivity (nearby neurons)
    Local(usize),  // neighborhood size
}

/// Topology builder for creating network connections
pub struct TopologyBuilder {
    pattern: ConnectivityPattern,
    weight_init: WeightInitialization,
    delay_range: (f64, f64),
}

/// Weight initialization strategies
#[derive(Debug, Clone, Copy)]
pub enum WeightInitialization {
    /// Uniform random in range
    Uniform(f64, f64),
    /// Normal distribution
    Normal(f64, f64),  // mean, std
    /// Constant value
    Constant(f64),
    /// Xavier initialization
    Xavier,
}

impl Default for TopologyBuilder {
    fn default() -> Self {
        Self {
            pattern: ConnectivityPattern::Dense,
            weight_init: WeightInitialization::Uniform(0.1, 0.5),
            delay_range: (1.0, 1.0),
        }
    }
}

impl TopologyBuilder {
    /// Create a new topology builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set connectivity pattern
    pub fn pattern(mut self, pattern: ConnectivityPattern) -> Self {
        self.pattern = pattern;
        self
    }

    /// Set weight initialization
    pub fn weight_init(mut self, init: WeightInitialization) -> Self {
        self.weight_init = init;
        self
    }

    /// Set delay range
    pub fn delay_range(mut self, min: f64, max: f64) -> Self {
        self.delay_range = (min, max);
        self
    }

    /// Build connections between two layers
    pub fn build(&self, pre_size: usize, post_size: usize, pre_offset: usize) -> Vec<Synapse> {
        let mut rng = rand::thread_rng();
        let mut synapses = Vec::new();

        match self.pattern {
            ConnectivityPattern::Dense => {
                for i in 0..pre_size {
                    for j in 0..post_size {
                        let weight = self.init_weight(&mut rng, pre_size);
                        let delay = rng.gen_range(self.delay_range.0..=self.delay_range.1);
                        synapses.push(Synapse::new(
                            pre_offset + i,
                            j,
                            SynapseConfig {
                                weight,
                                delay,
                                ..Default::default()
                            },
                        ));
                    }
                }
            }
            ConnectivityPattern::Sparse(prob) => {
                for i in 0..pre_size {
                    for j in 0..post_size {
                        if rng.gen::<f64>() < prob {
                            let weight = self.init_weight(&mut rng, pre_size);
                            let delay = rng.gen_range(self.delay_range.0..=self.delay_range.1);
                            synapses.push(Synapse::new(
                                pre_offset + i,
                                j,
                                SynapseConfig {
                                    weight,
                                    delay,
                                    ..Default::default()
                                },
                            ));
                        }
                    }
                }
            }
            ConnectivityPattern::OneToOne => {
                let n = pre_size.min(post_size);
                for i in 0..n {
                    let weight = self.init_weight(&mut rng, pre_size);
                    let delay = rng.gen_range(self.delay_range.0..=self.delay_range.1);
                    synapses.push(Synapse::new(
                        pre_offset + i,
                        i,
                        SynapseConfig {
                            weight,
                            delay,
                            ..Default::default()
                        },
                    ));
                }
            }
            ConnectivityPattern::Local(neighborhood) => {
                for i in 0..pre_size {
                    let center = (i as f64 / pre_size as f64 * post_size as f64) as usize;
                    let start = center.saturating_sub(neighborhood / 2);
                    let end = (center + neighborhood / 2 + 1).min(post_size);

                    for j in start..end {
                        let weight = self.init_weight(&mut rng, pre_size);
                        let delay = rng.gen_range(self.delay_range.0..=self.delay_range.1);
                        synapses.push(Synapse::new(
                            pre_offset + i,
                            j,
                            SynapseConfig {
                                weight,
                                delay,
                                ..Default::default()
                            },
                        ));
                    }
                }
            }
        }

        synapses
    }

    /// Initialize a weight value
    fn init_weight(&self, rng: &mut impl Rng, fan_in: usize) -> f64 {
        match self.weight_init {
            WeightInitialization::Uniform(min, max) => rng.gen_range(min..max),
            WeightInitialization::Normal(mean, std) => {
                use rand_distr::{Distribution, Normal};
                let normal = Normal::new(mean, std).unwrap();
                normal.sample(rng)
            }
            WeightInitialization::Constant(val) => val,
            WeightInitialization::Xavier => {
                let std = (2.0 / fan_in as f64).sqrt();
                use rand_distr::{Distribution, Normal};
                let normal = Normal::new(0.0, std).unwrap();
                normal.sample(rng).abs()  // Positive weights only
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_topology() {
        let builder = TopologyBuilder::new().pattern(ConnectivityPattern::Dense);
        let synapses = builder.build(10, 5, 0);
        assert_eq!(synapses.len(), 50);  // 10 * 5
    }

    #[test]
    fn test_sparse_topology() {
        let builder = TopologyBuilder::new().pattern(ConnectivityPattern::Sparse(0.5));
        let synapses = builder.build(100, 50, 0);
        // Should be approximately 2500 connections (50% of 5000)
        assert!(synapses.len() > 2000 && synapses.len() < 3000);
    }

    #[test]
    fn test_one_to_one() {
        let builder = TopologyBuilder::new().pattern(ConnectivityPattern::OneToOne);
        let synapses = builder.build(10, 10, 0);
        assert_eq!(synapses.len(), 10);
    }
}
