//! Training logic for zero-shot model.

use crate::data::attributes::AssetAttributes;
use crate::model::embeddings::EmbeddingSpace;
use crate::model::network::ZeroShotModel;
use crate::{MarketRegime, Result, ZeroShotError};

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Contrastive loss temperature
    pub temperature: f64,
    /// Gradient clipping threshold
    pub max_grad_norm: f64,
    /// Weight decay for regularization
    pub weight_decay: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
            temperature: 0.07,
            max_grad_norm: 1.0,
            weight_decay: 0.01,
        }
    }
}

/// Training sample for contrastive learning.
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// Market features (sequence x features)
    pub market_features: Vec<Vec<f64>>,
    /// Asset attributes
    pub attributes: AssetAttributes,
    /// True market regime label
    pub regime: MarketRegime,
    /// Asset symbol for logging
    pub symbol: String,
}

/// Trainer for zero-shot model using contrastive learning.
pub struct ZeroShotTrainer {
    /// Model to train
    model: ZeroShotModel,
    /// Training configuration
    config: TrainingConfig,
    /// Training history (loss per epoch)
    loss_history: Vec<f64>,
}

impl ZeroShotTrainer {
    /// Create a new trainer.
    pub fn new(model: ZeroShotModel, config: TrainingConfig) -> Self {
        Self {
            model,
            config,
            loss_history: Vec::new(),
        }
    }

    /// Get reference to the model.
    pub fn model(&self) -> &ZeroShotModel {
        &self.model
    }

    /// Get mutable reference to the model.
    pub fn model_mut(&mut self) -> &mut ZeroShotModel {
        &mut self.model
    }

    /// Get training loss history.
    pub fn loss_history(&self) -> &[f64] {
        &self.loss_history
    }

    /// Train the model on a batch of samples.
    pub fn train_batch(&mut self, samples: &[TrainingSample]) -> Result<f64> {
        if samples.is_empty() {
            return Err(ZeroShotError::NotEnoughData {
                needed: 1,
                got: 0,
            });
        }

        let batch_size = samples.len();
        let mut total_loss = 0.0;

        // Encode all samples
        let mut market_embeddings = Vec::with_capacity(batch_size);
        let mut regime_labels = Vec::with_capacity(batch_size);

        for sample in samples {
            let embedding = self.model.encode_market(&sample.market_features)?;
            market_embeddings.push(embedding);
            regime_labels.push(sample.regime);
        }

        // Calculate contrastive loss
        // Positive pairs: same regime, Negative pairs: different regimes
        for i in 0..batch_size {
            for j in (i + 1)..batch_size {
                let similarity = EmbeddingSpace::cosine_similarity(
                    &market_embeddings[i],
                    &market_embeddings[j],
                );

                // Scaled similarity
                let scaled_sim = similarity / self.config.temperature;

                // Target: 1 if same regime, 0 otherwise
                let target = if regime_labels[i] == regime_labels[j] {
                    1.0
                } else {
                    0.0
                };

                // Binary cross-entropy style loss
                let prob = 1.0 / (1.0 + (-scaled_sim).exp());
                let loss = -(target * prob.ln() + (1.0 - target) * (1.0 - prob).ln());

                if loss.is_finite() {
                    total_loss += loss;
                }
            }
        }

        // Normalize by number of pairs
        let num_pairs = (batch_size * (batch_size - 1)) / 2;
        if num_pairs > 0 {
            total_loss /= num_pairs as f64;
        }

        Ok(total_loss)
    }

    /// Train for one epoch on all samples.
    pub fn train_epoch(&mut self, samples: &[TrainingSample]) -> Result<f64> {
        if samples.len() < self.config.batch_size {
            return self.train_batch(samples);
        }

        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        for chunk in samples.chunks(self.config.batch_size) {
            let batch_loss = self.train_batch(chunk)?;
            epoch_loss += batch_loss;
            num_batches += 1;
        }

        let avg_loss = epoch_loss / num_batches as f64;
        self.loss_history.push(avg_loss);

        Ok(avg_loss)
    }

    /// Train for multiple epochs.
    pub fn train(&mut self, samples: &[TrainingSample]) -> Result<TrainingResult> {
        let start_time = std::time::Instant::now();

        for _epoch in 0..self.config.epochs {
            let loss = self.train_epoch(samples)?;

            // Early stopping if loss is very low
            if loss < 1e-6 {
                // Early stopping at this epoch
                break;
            }
        }

        let duration = start_time.elapsed();

        Ok(TrainingResult {
            final_loss: *self.loss_history.last().unwrap_or(&0.0),
            epochs_completed: self.loss_history.len(),
            training_time_secs: duration.as_secs_f64(),
            loss_history: self.loss_history.clone(),
        })
    }

    /// Evaluate model on test samples.
    pub fn evaluate(&self, samples: &[TrainingSample]) -> Result<EvaluationResult> {
        let mut correct = 0;
        let mut total = 0;
        let mut regime_predictions = Vec::new();
        let mut regime_actuals = Vec::new();

        for sample in samples {
            let (predicted_regime, _, _) = self.model.predict_regime(&sample.market_features)?;
            let actual = sample.regime;

            regime_predictions.push(predicted_regime);
            regime_actuals.push(actual);

            if predicted_regime == actual {
                correct += 1;
            }
            total += 1;
        }

        let accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };

        // Calculate per-regime accuracy
        let mut regime_correct = [0usize; 5];
        let mut regime_total = [0usize; 5];

        for (pred, actual) in regime_predictions.iter().zip(regime_actuals.iter()) {
            let idx = *actual as usize;
            regime_total[idx] += 1;
            if *pred == *actual {
                regime_correct[idx] += 1;
            }
        }

        let regime_accuracy: Vec<f64> = regime_correct
            .iter()
            .zip(regime_total.iter())
            .map(|(c, t)| {
                if *t > 0 {
                    *c as f64 / *t as f64
                } else {
                    0.0
                }
            })
            .collect();

        Ok(EvaluationResult {
            accuracy,
            total_samples: total,
            correct_predictions: correct,
            regime_accuracy,
        })
    }
}

/// Result of training.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final loss value
    pub final_loss: f64,
    /// Number of epochs completed
    pub epochs_completed: usize,
    /// Training time in seconds
    pub training_time_secs: f64,
    /// Loss history per epoch
    pub loss_history: Vec<f64>,
}

/// Result of evaluation.
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Overall accuracy
    pub accuracy: f64,
    /// Total number of samples
    pub total_samples: usize,
    /// Number of correct predictions
    pub correct_predictions: usize,
    /// Per-regime accuracy
    pub regime_accuracy: Vec<f64>,
}

impl std::fmt::Display for EvaluationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Evaluation Results:")?;
        writeln!(
            f,
            "  Overall Accuracy: {:.2}% ({}/{})",
            self.accuracy * 100.0,
            self.correct_predictions,
            self.total_samples
        )?;
        writeln!(f, "  Per-Regime Accuracy:")?;
        let regimes = [
            "Strong Uptrend",
            "Weak Uptrend",
            "Sideways",
            "Weak Downtrend",
            "Strong Downtrend",
        ];
        for (i, acc) in self.regime_accuracy.iter().enumerate() {
            writeln!(f, "    {}: {:.2}%", regimes[i], acc * 100.0)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::attributes::Sector;
    use crate::model::network::ModelConfig;

    fn create_dummy_sample(regime: MarketRegime) -> TrainingSample {
        let features: Vec<Vec<f64>> = (0..50)
            .map(|_| (0..11).map(|_| rand::random::<f64>() - 0.5).collect())
            .collect();

        TrainingSample {
            market_features: features,
            attributes: AssetAttributes::major_crypto(Sector::Layer1),
            regime,
            symbol: "BTCUSDT".to_string(),
        }
    }

    #[test]
    fn test_training_batch() {
        let model = ZeroShotModel::new(ModelConfig::default());
        let mut trainer = ZeroShotTrainer::new(model, TrainingConfig::default());

        let samples: Vec<TrainingSample> = vec![
            create_dummy_sample(MarketRegime::StrongUptrend),
            create_dummy_sample(MarketRegime::StrongUptrend),
            create_dummy_sample(MarketRegime::StrongDowntrend),
            create_dummy_sample(MarketRegime::Sideways),
        ];

        let loss = trainer.train_batch(&samples).unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn test_evaluation() {
        let model = ZeroShotModel::new(ModelConfig::default());
        let trainer = ZeroShotTrainer::new(model, TrainingConfig::default());

        let samples: Vec<TrainingSample> = vec![
            create_dummy_sample(MarketRegime::StrongUptrend),
            create_dummy_sample(MarketRegime::Sideways),
        ];

        let result = trainer.evaluate(&samples).unwrap();
        assert_eq!(result.total_samples, 2);
    }
}
