//! Noise schedules for diffusion models.

use tch::{Tensor, Kind, Device};
use std::f64::consts::PI;

/// Noise schedule types.
#[derive(Debug, Clone, Copy)]
pub enum ScheduleType {
    Linear,
    Cosine,
    Sigmoid,
}

/// Noise schedule for diffusion models.
#[derive(Debug, Clone)]
pub struct NoiseSchedule {
    /// Number of diffusion steps
    pub num_steps: usize,
    /// Beta values (noise levels)
    pub betas: Vec<f64>,
    /// Alpha values (1 - beta)
    pub alphas: Vec<f64>,
    /// Cumulative product of alphas
    pub alphas_cumprod: Vec<f64>,
    /// Square root of cumulative alphas
    pub sqrt_alphas_cumprod: Vec<f64>,
    /// Square root of (1 - cumulative alphas)
    pub sqrt_one_minus_alphas_cumprod: Vec<f64>,
    /// Schedule type
    pub schedule_type: ScheduleType,
}

impl NoiseSchedule {
    /// Create a linear noise schedule.
    ///
    /// β increases linearly from β_start to β_end.
    pub fn linear(num_steps: usize) -> Self {
        Self::linear_with_params(num_steps, 0.0001, 0.02)
    }

    /// Create a linear noise schedule with custom parameters.
    pub fn linear_with_params(num_steps: usize, beta_start: f64, beta_end: f64) -> Self {
        let betas: Vec<f64> = (0..num_steps)
            .map(|i| beta_start + (beta_end - beta_start) * i as f64 / (num_steps - 1) as f64)
            .collect();

        Self::from_betas(betas, ScheduleType::Linear)
    }

    /// Create a cosine noise schedule.
    ///
    /// Better for smaller sequences and images.
    pub fn cosine(num_steps: usize) -> Self {
        Self::cosine_with_params(num_steps, 0.008)
    }

    /// Create a cosine noise schedule with custom offset.
    pub fn cosine_with_params(num_steps: usize, s: f64) -> Self {
        let steps = num_steps + 1;
        let t: Vec<f64> = (0..steps)
            .map(|i| i as f64 / num_steps as f64)
            .collect();

        let alphas_cumprod: Vec<f64> = t
            .iter()
            .map(|&ti| ((ti + s) / (1.0 + s) * PI / 2.0).cos().powi(2))
            .collect();

        // Normalize
        let alpha_0 = alphas_cumprod[0];
        let alphas_cumprod: Vec<f64> = alphas_cumprod
            .iter()
            .map(|&a| a / alpha_0)
            .collect();

        // Compute betas
        let betas: Vec<f64> = (1..steps)
            .map(|i| {
                let beta = 1.0 - alphas_cumprod[i] / alphas_cumprod[i - 1];
                beta.clamp(0.0001, 0.9999)
            })
            .collect();

        Self::from_betas(betas, ScheduleType::Cosine)
    }

    /// Create a sigmoid noise schedule.
    pub fn sigmoid(num_steps: usize) -> Self {
        Self::sigmoid_with_params(num_steps, 0.0001, 0.02)
    }

    /// Create a sigmoid noise schedule with custom parameters.
    pub fn sigmoid_with_params(num_steps: usize, beta_start: f64, beta_end: f64) -> Self {
        let betas: Vec<f64> = (0..num_steps)
            .map(|i| {
                let t = -6.0 + 12.0 * i as f64 / (num_steps - 1) as f64;
                let sigmoid = 1.0 / (1.0 + (-t).exp());
                sigmoid * (beta_end - beta_start) + beta_start
            })
            .collect();

        Self::from_betas(betas, ScheduleType::Sigmoid)
    }

    /// Create a schedule from beta values.
    fn from_betas(betas: Vec<f64>, schedule_type: ScheduleType) -> Self {
        let num_steps = betas.len();

        let alphas: Vec<f64> = betas.iter().map(|b| 1.0 - b).collect();

        let mut alphas_cumprod = Vec::with_capacity(num_steps);
        let mut prod = 1.0;
        for &alpha in &alphas {
            prod *= alpha;
            alphas_cumprod.push(prod);
        }

        let sqrt_alphas_cumprod: Vec<f64> = alphas_cumprod.iter().map(|a| a.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod: Vec<f64> = alphas_cumprod
            .iter()
            .map(|a| (1.0 - a).sqrt())
            .collect();

        Self {
            num_steps,
            betas,
            alphas,
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            schedule_type,
        }
    }

    /// Get tensors for the schedule on a specific device.
    pub fn to_tensors(&self, device: Device) -> ScheduleTensors {
        ScheduleTensors {
            betas: Tensor::from_slice(&self.betas).to_kind(Kind::Float).to(device),
            alphas: Tensor::from_slice(&self.alphas).to_kind(Kind::Float).to(device),
            alphas_cumprod: Tensor::from_slice(&self.alphas_cumprod).to_kind(Kind::Float).to(device),
            sqrt_alphas_cumprod: Tensor::from_slice(&self.sqrt_alphas_cumprod).to_kind(Kind::Float).to(device),
            sqrt_one_minus_alphas_cumprod: Tensor::from_slice(&self.sqrt_one_minus_alphas_cumprod).to_kind(Kind::Float).to(device),
        }
    }

    /// Get the signal-to-noise ratio at each timestep.
    pub fn snr(&self) -> Vec<f64> {
        self.alphas_cumprod
            .iter()
            .map(|a| a / (1.0 - a + 1e-10))
            .collect()
    }
}

/// Tensor versions of the schedule parameters.
pub struct ScheduleTensors {
    pub betas: Tensor,
    pub alphas: Tensor,
    pub alphas_cumprod: Tensor,
    pub sqrt_alphas_cumprod: Tensor,
    pub sqrt_one_minus_alphas_cumprod: Tensor,
}

impl ScheduleTensors {
    /// Add noise to data at timestep t.
    pub fn add_noise(&self, x_0: &Tensor, t: &Tensor, noise: &Tensor) -> Tensor {
        let sqrt_alpha = self.sqrt_alphas_cumprod.index_select(0, t).unsqueeze(-1);
        let sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.index_select(0, t).unsqueeze(-1);

        sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_schedule() {
        let schedule = NoiseSchedule::linear(100);

        assert_eq!(schedule.num_steps, 100);
        assert!(schedule.betas[0] < schedule.betas[99]);
        assert!(schedule.alphas_cumprod[0] > schedule.alphas_cumprod[99]);
    }

    #[test]
    fn test_cosine_schedule() {
        let schedule = NoiseSchedule::cosine(100);

        assert_eq!(schedule.num_steps, 100);
        // Cosine schedule starts slower
        assert!(schedule.alphas_cumprod[0] > 0.99);
        assert!(schedule.alphas_cumprod[99] < 0.01);
    }

    #[test]
    fn test_snr() {
        let schedule = NoiseSchedule::cosine(100);
        let snr = schedule.snr();

        // SNR should decrease over time
        assert!(snr[0] > snr[50]);
        assert!(snr[50] > snr[99]);
    }
}
