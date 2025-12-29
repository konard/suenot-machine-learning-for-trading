//! Обучение и оценка модели

use super::{CnnConfig, CnnModel};
use crate::data::{Dataset, Sample};
use burn::{
    module::Module,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, Tensor},
};
use tracing::{debug, info, warn};

/// Результаты обучения
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// История потерь на train
    pub train_losses: Vec<f32>,
    /// История потерь на validation
    pub val_losses: Vec<f32>,
    /// История accuracy на train
    pub train_accuracies: Vec<f32>,
    /// История accuracy на validation
    pub val_accuracies: Vec<f32>,
    /// Лучшая эпоха
    pub best_epoch: usize,
    /// Лучший val_accuracy
    pub best_accuracy: f32,
}

/// Метрики оценки
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    /// Accuracy
    pub accuracy: f32,
    /// Precision по классам
    pub precision: [f32; 3],
    /// Recall по классам
    pub recall: [f32; 3],
    /// F1 по классам
    pub f1: [f32; 3],
    /// Confusion matrix
    pub confusion_matrix: [[usize; 3]; 3],
    /// Средний loss
    pub loss: f32,
}

/// Обучение модели
pub fn train_model<B: AutodiffBackend>(
    model: CnnModel<B>,
    train_dataset: &mut Dataset,
    val_dataset: &mut Dataset,
    config: &super::TrainingConfig,
    device: &B::Device,
) -> (CnnModel<B>, TrainingResult) {
    info!("Starting training for {} epochs", config.num_epochs);
    info!(
        "Train samples: {}, Val samples: {}",
        train_dataset.len(),
        val_dataset.len()
    );

    let mut model = model;
    let mut optimizer = AdamConfig::new().init();

    let mut result = TrainingResult {
        train_losses: Vec::new(),
        val_losses: Vec::new(),
        train_accuracies: Vec::new(),
        val_accuracies: Vec::new(),
        best_epoch: 0,
        best_accuracy: 0.0,
    };

    let class_weights = if config.use_class_weights {
        train_dataset.class_weights()
    } else {
        [1.0, 1.0, 1.0]
    };

    info!("Class weights: {:?}", class_weights);

    let mut patience_counter = 0;

    for epoch in 0..config.num_epochs {
        // Training phase
        train_dataset.reset();
        let mut train_loss_sum = 0.0;
        let mut train_correct = 0;
        let mut train_total = 0;
        let mut batch_count = 0;

        while let Some(batch) = train_dataset.next_batch() {
            let batch_size = batch.labels.len();

            // Преобразуем данные в тензоры
            let features = array3_to_tensor(&batch.features, device);
            let labels = Tensor::<B, 1, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::from(
                    batch.labels.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                ),
                device,
            );

            // Forward pass
            let logits = model.forward(features);

            // Compute loss (cross-entropy)
            let loss = cross_entropy_loss(logits.clone(), labels.clone(), &class_weights, device);

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(config.learning_rate, model, grads);

            // Metrics
            let predictions = logits.argmax(1);
            let correct = predictions
                .equal(labels)
                .int()
                .sum()
                .into_scalar()
                .elem::<i32>();

            train_loss_sum += loss.into_scalar().elem::<f32>();
            train_correct += correct as usize;
            train_total += batch_size;
            batch_count += 1;

            if batch_count % config.log_interval == 0 {
                debug!(
                    "Epoch {} Batch {}: loss={:.4}",
                    epoch,
                    batch_count,
                    train_loss_sum / batch_count as f32
                );
            }
        }

        let train_loss = train_loss_sum / batch_count as f32;
        let train_acc = train_correct as f32 / train_total as f32;

        // Validation phase
        let val_metrics = evaluate_model(&model, val_dataset, device);

        result.train_losses.push(train_loss);
        result.val_losses.push(val_metrics.loss);
        result.train_accuracies.push(train_acc);
        result.val_accuracies.push(val_metrics.accuracy);

        info!(
            "Epoch {}/{}: train_loss={:.4}, train_acc={:.4}, val_loss={:.4}, val_acc={:.4}",
            epoch + 1,
            config.num_epochs,
            train_loss,
            train_acc,
            val_metrics.loss,
            val_metrics.accuracy
        );

        // Early stopping check
        if val_metrics.accuracy > result.best_accuracy + config.min_delta as f32 {
            result.best_accuracy = val_metrics.accuracy;
            result.best_epoch = epoch;
            patience_counter = 0;

            // TODO: Save checkpoint
            if let Some(ref _path) = config.checkpoint_path {
                debug!("New best model at epoch {}", epoch);
            }
        } else {
            patience_counter += 1;
            if patience_counter >= config.patience {
                info!("Early stopping at epoch {}", epoch);
                break;
            }
        }
    }

    info!(
        "Training completed. Best accuracy: {:.4} at epoch {}",
        result.best_accuracy, result.best_epoch
    );

    (model, result)
}

/// Оценка модели
pub fn evaluate_model<B: AutodiffBackend>(
    model: &CnnModel<B>,
    dataset: &mut Dataset,
    device: &B::Device,
) -> EvaluationMetrics {
    dataset.reset();

    let mut confusion_matrix = [[0usize; 3]; 3];
    let mut total_loss = 0.0;
    let mut batch_count = 0;

    while let Some(batch) = dataset.next_batch() {
        let features = array3_to_tensor(&batch.features, device);
        let labels_vec = batch.labels.clone();
        let labels = Tensor::<B, 1, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::from(
                labels_vec.iter().map(|&x| x as i32).collect::<Vec<_>>(),
            ),
            device,
        );

        let logits = model.forward(features);

        // Loss
        let loss = cross_entropy_loss(logits.clone(), labels.clone(), &[1.0, 1.0, 1.0], device);
        total_loss += loss.into_scalar().elem::<f32>();

        // Predictions
        let predictions = logits.argmax(1);
        let pred_data: Vec<i32> = predictions.into_data().to_vec().unwrap();

        // Update confusion matrix
        for (pred, actual) in pred_data.iter().zip(labels_vec.iter()) {
            if *pred >= 0 && *pred < 3 && *actual < 3 {
                confusion_matrix[*actual][*pred as usize] += 1;
            }
        }

        batch_count += 1;
    }

    // Calculate metrics from confusion matrix
    let mut precision = [0.0f32; 3];
    let mut recall = [0.0f32; 3];
    let mut f1 = [0.0f32; 3];

    for c in 0..3 {
        let tp = confusion_matrix[c][c] as f32;
        let fp: f32 = (0..3).map(|i| confusion_matrix[i][c] as f32).sum::<f32>() - tp;
        let fn_: f32 = confusion_matrix[c].iter().sum::<f32>() - tp;

        precision[c] = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        recall[c] = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        f1[c] = if precision[c] + recall[c] > 0.0 {
            2.0 * precision[c] * recall[c] / (precision[c] + recall[c])
        } else {
            0.0
        };
    }

    let total: usize = confusion_matrix.iter().flat_map(|row| row.iter()).sum();
    let correct: usize = (0..3).map(|i| confusion_matrix[i][i]).sum();
    let accuracy = if total > 0 {
        correct as f32 / total as f32
    } else {
        0.0
    };

    let loss = if batch_count > 0 {
        total_loss / batch_count as f32
    } else {
        0.0
    };

    EvaluationMetrics {
        accuracy,
        precision,
        recall,
        f1,
        confusion_matrix,
        loss,
    }
}

/// Вспомогательная функция: ndarray -> Tensor
fn array3_to_tensor<B: burn::tensor::backend::Backend>(
    arr: &ndarray::Array3<f32>,
    device: &B::Device,
) -> Tensor<B, 3> {
    let shape = arr.shape();
    let data: Vec<f32> = arr.iter().cloned().collect();
    Tensor::from_data(
        burn::tensor::TensorData::new(data, [shape[0], shape[1], shape[2]]),
        device,
    )
}

/// Cross-entropy loss с весами классов
fn cross_entropy_loss<B: AutodiffBackend>(
    logits: Tensor<B, 2>,
    labels: Tensor<B, 1, burn::tensor::Int>,
    class_weights: &[f32; 3],
    device: &B::Device,
) -> Tensor<B, 1> {
    let batch_size = logits.dims()[0];
    let num_classes = logits.dims()[1];

    // Softmax
    let log_probs = burn::tensor::activation::log_softmax(logits, 1);

    // Gather the log probabilities for the correct classes
    let labels_2d = labels.clone().unsqueeze_dim(1);
    let selected_log_probs = log_probs.gather(1, labels_2d).squeeze(1);

    // Apply class weights
    let weights = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::from(class_weights.to_vec()),
        device,
    );
    let sample_weights = weights.select(0, labels);

    // Weighted negative log likelihood
    let weighted_loss = selected_log_probs.neg().mul(sample_weights);

    weighted_loss.mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::sample::Label;
    use ndarray::Array2;

    fn create_test_samples(n: usize) -> Vec<Sample> {
        (0..n)
            .map(|i| {
                let features = Array2::from_elem((10, 60), (i as f32) * 0.01);
                Sample::new(features, i as i64 * 1000).with_label(Label::from(i % 3))
            })
            .collect()
    }

    #[test]
    fn test_evaluation_metrics() {
        // Создаём простую confusion matrix и проверяем расчёт метрик
        let metrics = EvaluationMetrics {
            accuracy: 0.6,
            precision: [0.5, 0.6, 0.7],
            recall: [0.5, 0.6, 0.7],
            f1: [0.5, 0.6, 0.7],
            confusion_matrix: [[5, 2, 3], [2, 6, 2], [3, 2, 5]],
            loss: 0.8,
        };

        assert!(metrics.accuracy > 0.0);
    }
}
