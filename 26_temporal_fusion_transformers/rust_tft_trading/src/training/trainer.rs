//! Training utilities
//!
//! Trainer для обучения TFT модели.

use crate::data::Dataset;
use crate::model::{QuantilePrediction, TFTModel};
use serde::{Deserialize, Serialize};

/// Конфигурация обучения
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Скорость обучения
    pub learning_rate: f64,

    /// Размер батча
    pub batch_size: usize,

    /// Максимальное количество эпох
    pub max_epochs: usize,

    /// Early stopping patience
    pub patience: usize,

    /// Минимальное улучшение для early stopping
    pub min_delta: f64,

    /// Gradient clipping
    pub gradient_clip_val: Option<f64>,

    /// Логировать каждые N батчей
    pub log_every: usize,

    /// Валидация каждые N эпох
    pub validate_every: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            max_epochs: 100,
            patience: 10,
            min_delta: 1e-4,
            gradient_clip_val: Some(1.0),
            log_every: 10,
            validate_every: 1,
        }
    }
}

/// История обучения
#[derive(Debug, Clone, Default)]
pub struct TrainingHistory {
    /// Train loss по эпохам
    pub train_loss: Vec<f64>,

    /// Validation loss по эпохам
    pub val_loss: Vec<f64>,

    /// Лучший validation loss
    pub best_val_loss: f64,

    /// Эпоха с лучшим результатом
    pub best_epoch: usize,

    /// Время обучения (секунды)
    pub training_time: f64,
}

impl TrainingHistory {
    /// Добавляет результат эпохи
    pub fn add_epoch(&mut self, train_loss: f64, val_loss: f64) {
        self.train_loss.push(train_loss);
        self.val_loss.push(val_loss);

        if val_loss < self.best_val_loss || self.best_val_loss == 0.0 {
            self.best_val_loss = val_loss;
            self.best_epoch = self.train_loss.len();
        }
    }

    /// Проверяет, нужно ли остановить обучение
    pub fn should_stop(&self, patience: usize, min_delta: f64) -> bool {
        if self.val_loss.len() < patience {
            return false;
        }

        let recent = &self.val_loss[self.val_loss.len() - patience..];
        let improvement = recent.first().unwrap() - recent.last().unwrap();

        improvement < min_delta
    }

    /// Выводит статистику
    pub fn print_summary(&self) {
        println!("\n=== Training Summary ===");
        println!("Total epochs: {}", self.train_loss.len());
        println!("Best epoch: {}", self.best_epoch);
        println!("Best val loss: {:.6}", self.best_val_loss);
        if let (Some(first), Some(last)) = (self.train_loss.first(), self.train_loss.last()) {
            println!("Train loss: {:.6} -> {:.6}", first, last);
        }
        println!("Training time: {:.1}s", self.training_time);
    }
}

/// Trainer для TFT модели
pub struct Trainer {
    /// Конфигурация
    config: TrainingConfig,

    /// История обучения
    history: TrainingHistory,
}

impl Trainer {
    /// Создает новый Trainer
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            history: TrainingHistory::default(),
        }
    }

    /// Обучает модель
    ///
    /// Примечание: Это упрощенная версия без реального градиентного спуска.
    /// Для production использования рекомендуется использовать PyTorch через tch-rs.
    pub fn train(
        &mut self,
        model: &mut TFTModel,
        train_data: &Dataset,
        val_data: &Dataset,
    ) -> &TrainingHistory {
        use std::time::Instant;

        let start_time = Instant::now();

        println!(
            "\nStarting training with {} train samples and {} val samples",
            train_data.len(),
            val_data.len()
        );
        println!("Config: {:?}", self.config);

        for epoch in 0..self.config.max_epochs {
            // Train phase
            let train_loss = self.train_epoch(model, train_data);

            // Validation phase
            let val_loss = if epoch % self.config.validate_every == 0 {
                self.evaluate(model, val_data)
            } else {
                self.history.val_loss.last().copied().unwrap_or(f64::MAX)
            };

            self.history.add_epoch(train_loss, val_loss);

            // Log progress
            if epoch % self.config.log_every == 0 {
                println!(
                    "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}",
                    epoch + 1,
                    self.config.max_epochs,
                    train_loss,
                    val_loss
                );
            }

            // Early stopping
            if self.history.should_stop(self.config.patience, self.config.min_delta) {
                println!("Early stopping at epoch {}", epoch + 1);
                break;
            }
        }

        self.history.training_time = start_time.elapsed().as_secs_f64();
        self.history.print_summary();

        &self.history
    }

    /// Одна эпоха обучения
    fn train_epoch(&self, model: &mut TFTModel, data: &Dataset) -> f64 {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for batch in data.batches(self.config.batch_size) {
            let batch_loss: f64 = batch.iter().map(|sample| model.compute_loss(sample)).sum();

            total_loss += batch_loss / batch.len() as f64;
            num_batches += 1;
        }

        if num_batches > 0 {
            total_loss / num_batches as f64
        } else {
            0.0
        }
    }

    /// Оценка на валидационных данных
    fn evaluate(&self, model: &mut TFTModel, data: &Dataset) -> f64 {
        let mut total_loss = 0.0;
        let num_samples = data.len();

        for sample in &data.samples {
            total_loss += model.compute_loss(sample);
        }

        if num_samples > 0 {
            total_loss / num_samples as f64
        } else {
            0.0
        }
    }

    /// Возвращает историю обучения
    pub fn history(&self) -> &TrainingHistory {
        &self.history
    }

    /// Делает предсказания на данных
    pub fn predict(&self, model: &mut TFTModel, data: &Dataset) -> Vec<QuantilePrediction> {
        data.samples.iter().map(|s| model.forward(s)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_history() {
        let mut history = TrainingHistory::default();

        history.add_epoch(1.0, 0.9);
        history.add_epoch(0.8, 0.7);
        history.add_epoch(0.6, 0.6);

        assert_eq!(history.train_loss.len(), 3);
        assert_eq!(history.best_epoch, 3);
        assert!((history.best_val_loss - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_early_stopping() {
        let mut history = TrainingHistory::default();

        // Improving
        history.add_epoch(1.0, 0.9);
        history.add_epoch(0.8, 0.8);
        history.add_epoch(0.6, 0.7);
        assert!(!history.should_stop(3, 0.01));

        // Не улучшается
        history.add_epoch(0.5, 0.701);
        history.add_epoch(0.4, 0.702);
        history.add_epoch(0.3, 0.703);
        assert!(history.should_stop(3, 0.01));
    }

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::default();

        assert_eq!(config.max_epochs, 100);
        assert_eq!(config.patience, 10);
        assert!(config.learning_rate > 0.0);
    }
}
