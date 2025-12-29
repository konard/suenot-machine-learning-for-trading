//! DDPM (Denoising Diffusion Probabilistic Model) implementation.

use tch::{nn, Tensor, Kind, Device};
use tch::nn::{Module, ModuleT};

use super::schedule::{NoiseSchedule, ScheduleTensors};

/// Sinusoidal position embeddings for timestep encoding.
fn sinusoidal_embedding(timesteps: &Tensor, dim: i64) -> Tensor {
    let device = timesteps.device();
    let half_dim = dim / 2;

    let emb = (10000.0_f64.ln() / (half_dim - 1) as f64).neg();
    let emb = (Tensor::arange(half_dim, (Kind::Float, device)) * emb).exp();
    let emb = timesteps.unsqueeze(-1).to_kind(Kind::Float) * emb.unsqueeze(0);

    Tensor::cat(&[emb.sin(), emb.cos()], -1)
}

/// Time embedding MLP.
#[derive(Debug)]
struct TimeEmbedding {
    linear1: nn::Linear,
    linear2: nn::Linear,
    time_dim: i64,
}

impl TimeEmbedding {
    fn new(vs: &nn::Path, time_dim: i64, hidden_dim: i64) -> Self {
        let linear1 = nn::linear(vs / "time_linear1", time_dim, hidden_dim, Default::default());
        let linear2 = nn::linear(vs / "time_linear2", hidden_dim, hidden_dim, Default::default());

        Self { linear1, linear2, time_dim }
    }

    fn forward(&self, t: &Tensor) -> Tensor {
        let emb = sinusoidal_embedding(t, self.time_dim);
        let emb = self.linear1.forward(&emb).silu();
        self.linear2.forward(&emb)
    }
}

/// Condition encoder using LSTM.
#[derive(Debug)]
struct ConditionEncoder {
    lstm: nn::LSTM,
    output_dim: i64,
}

impl ConditionEncoder {
    fn new(vs: &nn::Path, input_dim: i64, hidden_dim: i64) -> Self {
        let lstm_config = nn::RNNConfig {
            num_layers: 2,
            dropout: 0.1,
            batch_first: true,
            ..Default::default()
        };

        let lstm = nn::lstm(vs / "lstm", input_dim, hidden_dim, lstm_config);

        Self { lstm, output_dim: hidden_dim }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let (_, (h_n, _)) = self.lstm.seq(x);
        // Get the last layer's hidden state
        h_n.select(0, -1)
    }
}

/// Denoising network.
#[derive(Debug)]
struct DenoiseNet {
    input_proj: nn::Linear,
    layers: Vec<(nn::Linear, nn::LayerNorm)>,
    output_proj: nn::Linear,
}

impl DenoiseNet {
    fn new(vs: &nn::Path, input_dim: i64, hidden_dim: i64, output_dim: i64, num_layers: i64) -> Self {
        let input_proj = nn::linear(vs / "input_proj", input_dim, hidden_dim, Default::default());

        let mut layers = Vec::new();
        for i in 0..num_layers {
            let linear = nn::linear(vs / format!("layer_{}", i), hidden_dim, hidden_dim, Default::default());
            let norm = nn::layer_norm(vs / format!("norm_{}", i), vec![hidden_dim], Default::default());
            layers.push((linear, norm));
        }

        let output_proj = nn::linear(vs / "output_proj", hidden_dim, output_dim, Default::default());

        Self { input_proj, layers, output_proj }
    }

    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let mut h = self.input_proj.forward(x);

        for (linear, norm) in &self.layers {
            let residual = &h;
            h = linear.forward(&h);
            h = norm.forward(&h);
            h = h.silu();
            if train {
                h = h.dropout(0.1, train);
            }
            h = h + residual;
        }

        self.output_proj.forward(&h)
    }
}

/// Conditional DDPM for time series forecasting.
pub struct DDPM {
    vs: nn::VarStore,
    time_embedding: TimeEmbedding,
    condition_encoder: ConditionEncoder,
    denoise_net: DenoiseNet,
    schedule: NoiseSchedule,
    schedule_tensors: ScheduleTensors,
    forecast_horizon: i64,
    hidden_dim: i64,
}

impl DDPM {
    /// Create a new DDPM model.
    ///
    /// # Arguments
    /// * `input_dim` - Number of input features
    /// * `seq_length` - Length of historical sequence
    /// * `forecast_horizon` - Number of steps to forecast
    /// * `hidden_dim` - Hidden dimension size
    /// * `schedule` - Noise schedule
    /// * `device` - Device to run on
    pub fn new(
        input_dim: i64,
        seq_length: i64,
        forecast_horizon: i64,
        hidden_dim: i64,
        schedule: &NoiseSchedule,
        device: Device,
    ) -> Self {
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        let time_dim = 64;
        let time_embedding = TimeEmbedding::new(&root, time_dim, hidden_dim);
        let condition_encoder = ConditionEncoder::new(&root, input_dim, hidden_dim);

        // Input to denoise net: noisy forecast + time emb + condition emb
        let denoise_input_dim = forecast_horizon + hidden_dim * 2;
        let denoise_net = DenoiseNet::new(&root, denoise_input_dim, hidden_dim, forecast_horizon, 4);

        let schedule_tensors = schedule.to_tensors(device);

        Self {
            vs,
            time_embedding,
            condition_encoder,
            denoise_net,
            schedule: schedule.clone(),
            schedule_tensors,
            forecast_horizon,
            hidden_dim,
        }
    }

    /// Get the variable store.
    pub fn vs(&self) -> &nn::VarStore {
        &self.vs
    }

    /// Get mutable variable store.
    pub fn vs_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    /// Forward pass: predict noise given noisy input, timestep, and condition.
    pub fn forward(&self, x_noisy: &Tensor, t: &Tensor, condition: &Tensor, train: bool) -> Tensor {
        let t_emb = self.time_embedding.forward(t);
        let cond_emb = self.condition_encoder.forward(condition);

        let combined = Tensor::cat(&[x_noisy, &t_emb, &cond_emb], -1);
        self.denoise_net.forward(&combined, train)
    }

    /// Add noise to targets at timestep t.
    pub fn add_noise(&self, x_0: &Tensor, t: &Tensor) -> (Tensor, Tensor) {
        let noise = Tensor::randn_like(x_0);
        let x_noisy = self.schedule_tensors.add_noise(x_0, t, &noise);
        (x_noisy, noise)
    }

    /// Compute training loss.
    pub fn compute_loss(&self, condition: &Tensor, target: &Tensor) -> Tensor {
        let batch_size = condition.size()[0];
        let device = condition.device();

        // Sample random timesteps
        let t = Tensor::randint(
            self.schedule.num_steps as i64,
            &[batch_size],
            (Kind::Int64, device),
        );

        // Add noise
        let (x_noisy, noise) = self.add_noise(target, &t);

        // Predict noise
        let noise_pred = self.forward(&x_noisy, &t, condition, true);

        // MSE loss
        (&noise_pred - &noise).pow_tensor_scalar(2).mean(Kind::Float)
    }

    /// Sample from the model (DDPM sampling).
    pub fn sample(&self, condition: &Tensor, num_samples: i64) -> Tensor {
        let device = condition.device();

        // Expand condition for all samples
        let condition = condition.expand(&[num_samples, -1, -1], false);

        // Start from pure noise
        let mut x = Tensor::randn(&[num_samples, self.forecast_horizon], (Kind::Float, device));

        // Iteratively denoise
        for t in (0..self.schedule.num_steps).rev() {
            let t_tensor = Tensor::full(&[num_samples], t as i64, (Kind::Int64, device));

            // Predict noise
            let noise_pred = self.forward(&x, &t_tensor, &condition, false);

            // Get schedule parameters
            let alpha = self.schedule.alphas[t];
            let alpha_cumprod = self.schedule.alphas_cumprod[t];
            let beta = self.schedule.betas[t];

            // Denoise
            let coef1 = 1.0 / alpha.sqrt();
            let coef2 = beta / (1.0 - alpha_cumprod).sqrt();

            x = coef1 * (&x - coef2 * &noise_pred);

            // Add noise (except for last step)
            if t > 0 {
                let noise = Tensor::randn_like(&x);
                x = x + beta.sqrt() * noise;
            }
        }

        x
    }

    /// Generate probabilistic forecast.
    pub fn forecast(&self, condition: &Tensor, num_samples: i64) -> ForecastResult {
        let samples = self.sample(condition, num_samples);

        // Compute statistics
        let mean = samples.mean_dim(Some([0].as_slice()), false, Kind::Float);
        let std = samples.std_dim(Some([0].as_slice()), true, false);

        // Compute percentiles
        let samples_sorted = samples.sort(0, false).0;
        let p5_idx = ((num_samples as f64 * 0.05) as i64).max(0);
        let p25_idx = ((num_samples as f64 * 0.25) as i64).max(0);
        let p50_idx = ((num_samples as f64 * 0.50) as i64).max(0);
        let p75_idx = ((num_samples as f64 * 0.75) as i64).min(num_samples - 1);
        let p95_idx = ((num_samples as f64 * 0.95) as i64).min(num_samples - 1);

        ForecastResult {
            samples,
            mean,
            std,
            p5: samples_sorted.select(0, p5_idx),
            p25: samples_sorted.select(0, p25_idx),
            median: samples_sorted.select(0, p50_idx),
            p75: samples_sorted.select(0, p75_idx),
            p95: samples_sorted.select(0, p95_idx),
        }
    }

    /// Save model to file.
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        self.vs.save(path)?;
        Ok(())
    }

    /// Load model from file.
    pub fn load(&mut self, path: &str) -> anyhow::Result<()> {
        self.vs.load(path)?;
        Ok(())
    }
}

/// Result of probabilistic forecasting.
pub struct ForecastResult {
    /// All samples [num_samples, forecast_horizon]
    pub samples: Tensor,
    /// Mean forecast
    pub mean: Tensor,
    /// Standard deviation
    pub std: Tensor,
    /// 5th percentile
    pub p5: Tensor,
    /// 25th percentile
    pub p25: Tensor,
    /// Median (50th percentile)
    pub median: Tensor,
    /// 75th percentile
    pub p75: Tensor,
    /// 95th percentile
    pub p95: Tensor,
}

impl ForecastResult {
    /// Convert mean forecast to Vec.
    pub fn mean_vec(&self) -> Vec<f64> {
        Vec::<f64>::try_from(&self.mean).unwrap_or_default()
    }

    /// Convert std to Vec.
    pub fn std_vec(&self) -> Vec<f64> {
        Vec::<f64>::try_from(&self.std).unwrap_or_default()
    }
}
