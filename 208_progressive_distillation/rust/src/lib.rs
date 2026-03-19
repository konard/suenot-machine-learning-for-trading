use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// FlexibleNetwork — feedforward network with configurable layer sizes
// ---------------------------------------------------------------------------

/// A feedforward neural network with arbitrary layer sizes.
/// Uses ReLU activations for hidden layers and linear output.
#[derive(Clone)]
pub struct FlexibleNetwork {
    pub weights: Vec<Array2<f64>>,
    pub biases: Vec<Array1<f64>>,
    pub layer_sizes: Vec<usize>,
}

impl FlexibleNetwork {
    /// Create a new network with Xavier-initialized random weights.
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "Need at least input and output layers"
        );
        let mut rng = rand::thread_rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let fan_in = layer_sizes[i];
            let fan_out = layer_sizes[i + 1];
            let scale = (2.0 / fan_in as f64).sqrt();
            let w = Array2::from_shape_fn((fan_in, fan_out), |_| rng.gen::<f64>() * scale - scale / 2.0);
            let b = Array1::zeros(fan_out);
            weights.push(w);
            biases.push(b);
        }

        FlexibleNetwork {
            weights,
            biases,
            layer_sizes: layer_sizes.to_vec(),
        }
    }

    /// Forward pass for a single sample.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut x = input.clone();
        let n_layers = self.weights.len();
        for i in 0..n_layers {
            x = x.dot(&self.weights[i]) + &self.biases[i];
            // ReLU for hidden layers, linear for output
            if i < n_layers - 1 {
                x.mapv_inplace(|v| v.max(0.0));
            }
        }
        x
    }

    /// Forward pass for a batch of samples (rows = samples).
    pub fn forward_batch(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut x = input.clone();
        let n_layers = self.weights.len();
        for i in 0..n_layers {
            x = x.dot(&self.weights[i]) + &self.biases[i];
            if i < n_layers - 1 {
                x.mapv_inplace(|v| v.max(0.0));
            }
        }
        x
    }

    /// Total number of trainable parameters (weights + biases).
    pub fn param_count(&self) -> usize {
        let w: usize = self.weights.iter().map(|w| w.len()).sum();
        let b: usize = self.biases.iter().map(|b| b.len()).sum();
        w + b
    }

    /// Train with supervised learning using simple gradient-free perturbation.
    /// This is a simplified training loop for demonstration purposes.
    pub fn train_supervised(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        epochs: usize,
        lr: f64,
    ) -> f64 {
        let n = x_train.nrows();
        let mut best_loss = self.compute_mse(x_train, y_train);

        for _epoch in 0..epochs {
            // Perturb each weight matrix and keep improvements
            for layer in 0..self.weights.len() {
                let (rows, cols) = self.weights[layer].dim();
                for r in 0..rows {
                    for c in 0..cols {
                        let original = self.weights[layer][[r, c]];

                        // Try positive perturbation
                        self.weights[layer][[r, c]] = original + lr;
                        let loss_plus = self.compute_mse(x_train, y_train);

                        // Try negative perturbation
                        self.weights[layer][[r, c]] = original - lr;
                        let loss_minus = self.compute_mse(x_train, y_train);

                        // Keep the best
                        if loss_plus < best_loss && loss_plus <= loss_minus {
                            self.weights[layer][[r, c]] = original + lr;
                            best_loss = loss_plus;
                        } else if loss_minus < best_loss {
                            self.weights[layer][[r, c]] = original - lr;
                            best_loss = loss_minus;
                        } else {
                            self.weights[layer][[r, c]] = original;
                        }
                    }
                }

                // Perturb biases
                let bias_len = self.biases[layer].len();
                for b in 0..bias_len {
                    let original = self.biases[layer][b];

                    self.biases[layer][b] = original + lr;
                    let loss_plus = self.compute_mse(x_train, y_train);

                    self.biases[layer][b] = original - lr;
                    let loss_minus = self.compute_mse(x_train, y_train);

                    if loss_plus < best_loss && loss_plus <= loss_minus {
                        self.biases[layer][b] = original + lr;
                        best_loss = loss_plus;
                    } else if loss_minus < best_loss {
                        self.biases[layer][b] = original - lr;
                        best_loss = loss_minus;
                    } else {
                        self.biases[layer][b] = original;
                    }
                }
            }
        }

        let _ = n; // suppress unused warning
        best_loss
    }

    /// Train via distillation: match the teacher's outputs instead of ground truth.
    pub fn train_distill(
        &mut self,
        teacher: &FlexibleNetwork,
        x_train: &Array2<f64>,
        epochs: usize,
        lr: f64,
    ) -> f64 {
        // Generate teacher soft targets
        let teacher_out = teacher.forward_batch(x_train);
        let teacher_targets: Array1<f64> = teacher_out.column(0).to_owned();

        self.train_supervised(x_train, &teacher_targets, epochs, lr)
    }

    /// Compute mean squared error against targets.
    pub fn compute_mse(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let pred = self.forward_batch(x);
        let n = y.len() as f64;
        let mut sum = 0.0;
        for i in 0..y.len() {
            let diff = pred[[i, 0]] - y[i];
            sum += diff * diff;
        }
        sum / n
    }

    /// Compute R-squared on given data.
    pub fn r_squared(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let pred = self.forward_batch(x);
        let y_mean = y.mean().unwrap_or(0.0);
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for i in 0..y.len() {
            let diff = y[i] - pred[[i, 0]];
            ss_res += diff * diff;
            let diff_mean = y[i] - y_mean;
            ss_tot += diff_mean * diff_mean;
        }
        if ss_tot == 0.0 {
            return 0.0;
        }
        1.0 - ss_res / ss_tot
    }
}

// ---------------------------------------------------------------------------
// Stage metrics
// ---------------------------------------------------------------------------

/// Metrics collected at each distillation stage.
#[derive(Debug, Clone)]
pub struct StageMetrics {
    pub stage: usize,
    pub param_count: usize,
    pub compression_ratio: f64,
    pub distillation_loss: f64,
    pub r_squared: f64,
    pub accuracy_retention: f64,
}

impl std::fmt::Display for StageMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Stage {} | Params: {:>6} | Compression: {:>6.1}x | Distill Loss: {:.6} | R²: {:.4} | Retention: {:.1}%",
            self.stage,
            self.param_count,
            self.compression_ratio,
            self.distillation_loss,
            self.r_squared,
            self.accuracy_retention * 100.0
        )
    }
}

// ---------------------------------------------------------------------------
// Progressive Distiller
// ---------------------------------------------------------------------------

/// Orchestrates multi-stage progressive distillation.
pub struct ProgressiveDistiller {
    /// Layer configurations for each stage (index 0 = first student, not teacher).
    pub stages: Vec<Vec<usize>>,
    /// Trained models: index 0 = teacher, then one per stage.
    pub models: Vec<FlexibleNetwork>,
    /// Metrics for each distillation stage.
    pub stage_metrics: Vec<StageMetrics>,
}

impl ProgressiveDistiller {
    /// Create a new progressive distiller with the given stage architectures.
    /// `stages[0]` is the architecture for the first student (stage 1),
    /// `stages[1]` for stage 2, etc.
    pub fn new(stages: Vec<Vec<usize>>) -> Self {
        ProgressiveDistiller {
            stages,
            models: Vec::new(),
            stage_metrics: Vec::new(),
        }
    }

    /// Run the full progressive distillation pipeline.
    ///
    /// - `teacher`: the trained teacher model (stage 0)
    /// - `x_train`: training features
    /// - `y_train`: training targets (for R² evaluation)
    /// - `epochs_per_stage`: how many epochs to train each student
    /// - `lr`: learning rate
    pub fn run(
        &mut self,
        teacher: &FlexibleNetwork,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        epochs_per_stage: usize,
        lr: f64,
    ) {
        let teacher_params = teacher.param_count();
        let teacher_r2 = teacher.r_squared(x_train, y_train);

        self.models.push(teacher.clone());
        self.stage_metrics.push(StageMetrics {
            stage: 0,
            param_count: teacher_params,
            compression_ratio: 1.0,
            distillation_loss: 0.0,
            r_squared: teacher_r2,
            accuracy_retention: 1.0,
        });

        for (i, stage_arch) in self.stages.iter().enumerate() {
            let stage_num = i + 1;
            let current_teacher = &self.models[i];

            let mut student = FlexibleNetwork::new(stage_arch);
            let distill_loss = student.train_distill(current_teacher, x_train, epochs_per_stage, lr);

            let student_params = student.param_count();
            let student_r2 = student.r_squared(x_train, y_train);
            let compression = teacher_params as f64 / student_params as f64;
            let retention = if teacher_r2.abs() > 1e-10 {
                (student_r2 / teacher_r2).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let metrics = StageMetrics {
                stage: stage_num,
                param_count: student_params,
                compression_ratio: compression,
                distillation_loss: distill_loss,
                r_squared: student_r2,
                accuracy_retention: retention,
            };

            self.stage_metrics.push(metrics);
            self.models.push(student);
        }
    }

    /// Print a summary table of all stages.
    pub fn print_summary(&self) {
        println!("\n=== Progressive Distillation Summary ===");
        for m in &self.stage_metrics {
            println!("  {}", m);
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
// One-shot distillation (for comparison)
// ---------------------------------------------------------------------------

/// Perform one-shot distillation: train a student of the given architecture
/// directly from the original teacher.
pub fn one_shot_distill(
    teacher: &FlexibleNetwork,
    student_arch: &[usize],
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
    epochs: usize,
    lr: f64,
) -> (FlexibleNetwork, StageMetrics) {
    let teacher_params = teacher.param_count();
    let teacher_r2 = teacher.r_squared(x_train, y_train);

    let mut student = FlexibleNetwork::new(student_arch);
    let distill_loss = student.train_distill(teacher, x_train, epochs, lr);

    let student_params = student.param_count();
    let student_r2 = student.r_squared(x_train, y_train);
    let compression = teacher_params as f64 / student_params as f64;
    let retention = if teacher_r2.abs() > 1e-10 {
        (student_r2 / teacher_r2).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let metrics = StageMetrics {
        stage: 0, // one-shot has a single stage
        param_count: student_params,
        compression_ratio: compression,
        distillation_loss: distill_loss,
        r_squared: student_r2,
        accuracy_retention: retention,
    };

    (student, metrics)
}

// ---------------------------------------------------------------------------
// Bybit API data fetching
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// A single OHLCV candle from Bybit.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch OHLCV kline data from Bybit API.
///
/// - `symbol`: e.g., "BTCUSDT"
/// - `interval`: e.g., "5" for 5-minute candles
/// - `limit`: number of candles (max 200)
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .context("Failed to build HTTP client")?;

    let resp: BybitResponse = client
        .get(&url)
        .send()
        .context("Failed to send request to Bybit API")?
        .json()
        .context("Failed to parse Bybit API response")?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: retCode={}", resp.ret_code);
    }

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(Candle {
                timestamp: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order
    candles.reverse();

    Ok(candles)
}

/// Convert candles to feature matrix and target vector.
///
/// Features per row: [return, volatility, volume_change, momentum_5]
/// Target: next-period return
pub fn candles_to_features(candles: &[Candle]) -> (Array2<f64>, Array1<f64>) {
    let n = candles.len();
    assert!(n > 6, "Need at least 7 candles for feature computation");

    let mut features = Vec::new();
    let mut targets = Vec::new();

    // Start at index 5 so we have 5 periods of history for momentum
    for i in 5..n - 1 {
        let ret = (candles[i].close - candles[i - 1].close) / candles[i - 1].close;
        let volatility = (candles[i].high - candles[i].low) / candles[i].close;
        let vol_change = if candles[i - 1].volume > 0.0 {
            (candles[i].volume - candles[i - 1].volume) / candles[i - 1].volume
        } else {
            0.0
        };
        let momentum = (candles[i].close - candles[i - 5].close) / candles[i - 5].close;

        features.push(vec![ret, volatility, vol_change, momentum]);

        let next_ret = (candles[i + 1].close - candles[i].close) / candles[i].close;
        targets.push(next_ret);
    }

    let n_samples = features.len();
    let n_features = 4;
    let flat: Vec<f64> = features.into_iter().flatten().collect();
    let x = Array2::from_shape_vec((n_samples, n_features), flat).unwrap();
    let y = Array1::from_vec(targets);

    (x, y)
}

/// Generate synthetic trading data for testing (when API is unavailable).
pub fn generate_synthetic_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = rand::thread_rng();

    let x = Array2::from_shape_fn((n_samples, n_features), |_| {
        rng.gen::<f64>() * 2.0 - 1.0
    });

    // Target is a nonlinear function of features
    let y = Array1::from_shape_fn(n_samples, |i| {
        let row = x.row(i);
        let signal = 0.5 * row[0] - 0.3 * row[1] + 0.2 * row[0] * row[1];
        let noise = (rng.gen::<f64>() - 0.5) * 0.1;
        signal + noise
    });

    (x, y)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let net = FlexibleNetwork::new(&[4, 32, 16, 1]);
        assert_eq!(net.layer_sizes, vec![4, 32, 16, 1]);
        assert_eq!(net.weights.len(), 3);
        assert_eq!(net.biases.len(), 3);
    }

    #[test]
    fn test_param_count() {
        let net = FlexibleNetwork::new(&[4, 32, 16, 1]);
        // Weights: 4*32 + 32*16 + 16*1 = 128 + 512 + 16 = 656
        // Biases: 32 + 16 + 1 = 49
        assert_eq!(net.param_count(), 656 + 49);
    }

    #[test]
    fn test_forward_pass() {
        let net = FlexibleNetwork::new(&[4, 8, 1]);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let output = net.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_forward_batch() {
        let net = FlexibleNetwork::new(&[4, 8, 1]);
        let input = Array2::from_shape_vec((3, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ]).unwrap();
        let output = net.forward_batch(&input);
        assert_eq!(output.dim(), (3, 1));
    }

    #[test]
    fn test_progressive_distillation_reduces_params() {
        let (x, y) = generate_synthetic_data(50, 4);

        let mut teacher = FlexibleNetwork::new(&[4, 64, 32, 1]);
        teacher.train_supervised(&x, &y, 2, 0.01);

        let stages = vec![
            vec![4, 32, 16, 1],  // medium
            vec![4, 16, 8, 1],   // small
            vec![4, 8, 1],       // tiny
        ];

        let mut distiller = ProgressiveDistiller::new(stages);
        distiller.run(&teacher, &x, &y, 2, 0.01);

        // Should have teacher + 3 students = 4 models
        assert_eq!(distiller.models.len(), 4);

        // Each model should have fewer params than the previous
        for i in 1..distiller.models.len() {
            assert!(
                distiller.models[i].param_count() < distiller.models[i - 1].param_count(),
                "Stage {} should have fewer params than stage {}",
                i, i - 1
            );
        }
    }

    #[test]
    fn test_compression_ratio_increases() {
        let (x, y) = generate_synthetic_data(50, 4);

        let mut teacher = FlexibleNetwork::new(&[4, 64, 32, 1]);
        teacher.train_supervised(&x, &y, 2, 0.01);

        let stages = vec![
            vec![4, 32, 16, 1],
            vec![4, 16, 8, 1],
            vec![4, 8, 1],
        ];

        let mut distiller = ProgressiveDistiller::new(stages);
        distiller.run(&teacher, &x, &y, 2, 0.01);

        // Compression ratio should increase at each stage
        for i in 2..distiller.stage_metrics.len() {
            assert!(
                distiller.stage_metrics[i].compression_ratio
                    > distiller.stage_metrics[i - 1].compression_ratio,
                "Compression ratio should increase at stage {}",
                i
            );
        }
    }

    #[test]
    fn test_one_shot_distillation() {
        let (x, y) = generate_synthetic_data(50, 4);

        let mut teacher = FlexibleNetwork::new(&[4, 64, 32, 1]);
        teacher.train_supervised(&x, &y, 2, 0.01);

        let (student, metrics) = one_shot_distill(
            &teacher,
            &[4, 8, 1],
            &x,
            &y,
            2,
            0.01,
        );

        assert!(metrics.compression_ratio > 1.0);
        assert_eq!(student.layer_sizes, vec![4, 8, 1]);
    }

    #[test]
    fn test_synthetic_data_generation() {
        let (x, y) = generate_synthetic_data(100, 4);
        assert_eq!(x.dim(), (100, 4));
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_model_size_tracking() {
        let sizes_large = &[4, 128, 64, 32, 1];
        let sizes_small = &[4, 16, 1];

        let large = FlexibleNetwork::new(sizes_large);
        let small = FlexibleNetwork::new(sizes_small);

        assert!(large.param_count() > small.param_count());
        let ratio = large.param_count() as f64 / small.param_count() as f64;
        assert!(ratio > 5.0, "Large model should be significantly bigger");
    }

    #[test]
    fn test_stage_metrics_display() {
        let metrics = StageMetrics {
            stage: 1,
            param_count: 1000,
            compression_ratio: 5.0,
            distillation_loss: 0.001,
            r_squared: 0.85,
            accuracy_retention: 0.92,
        };
        let display = format!("{}", metrics);
        assert!(display.contains("Stage 1"));
        assert!(display.contains("1000"));
    }

    #[test]
    fn test_r_squared_computation() {
        let net = FlexibleNetwork::new(&[2, 4, 1]);
        let x = Array2::from_shape_vec((4, 2), vec![
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            0.0, 0.0,
        ]).unwrap();
        let y = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        let r2 = net.r_squared(&x, &y);
        // R² can be negative for bad models, just check it's finite
        assert!(r2.is_finite());
    }
}
