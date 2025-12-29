//! Training metrics for monitoring GAN progress
//!
//! Provides structures for tracking and logging training progress.

use std::collections::VecDeque;

/// Metrics collected during training
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Generator losses per epoch
    pub gen_losses: Vec<f64>,
    /// Discriminator losses per epoch
    pub disc_losses: Vec<f64>,
    /// Discriminator accuracy on real samples
    pub disc_real_acc: Vec<f64>,
    /// Discriminator accuracy on fake samples
    pub disc_fake_acc: Vec<f64>,
}

impl TrainingMetrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record epoch metrics
    pub fn record_epoch(
        &mut self,
        gen_loss: f64,
        disc_loss: f64,
        real_acc: f64,
        fake_acc: f64,
    ) {
        self.gen_losses.push(gen_loss);
        self.disc_losses.push(disc_loss);
        self.disc_real_acc.push(real_acc);
        self.disc_fake_acc.push(fake_acc);
    }

    /// Get number of recorded epochs
    pub fn num_epochs(&self) -> usize {
        self.gen_losses.len()
    }

    /// Get latest generator loss
    pub fn latest_gen_loss(&self) -> Option<f64> {
        self.gen_losses.last().copied()
    }

    /// Get latest discriminator loss
    pub fn latest_disc_loss(&self) -> Option<f64> {
        self.disc_losses.last().copied()
    }

    /// Calculate moving average of generator loss
    pub fn gen_loss_ma(&self, window: usize) -> f64 {
        moving_average(&self.gen_losses, window)
    }

    /// Calculate moving average of discriminator loss
    pub fn disc_loss_ma(&self, window: usize) -> f64 {
        moving_average(&self.disc_losses, window)
    }

    /// Check if training appears to have collapsed
    ///
    /// Mode collapse indicators:
    /// - Discriminator loss very low (can easily distinguish)
    /// - Generator loss very high (can't fool discriminator)
    pub fn check_mode_collapse(&self, window: usize) -> bool {
        if self.num_epochs() < window {
            return false;
        }

        let disc_ma = self.disc_loss_ma(window);
        let gen_ma = self.gen_loss_ma(window);

        // Heuristic thresholds for mode collapse detection
        disc_ma < 0.1 && gen_ma > 5.0
    }

    /// Check if training is balanced
    ///
    /// Good training has discriminator accuracy around 50-70%
    pub fn is_balanced(&self, window: usize) -> bool {
        if self.num_epochs() < window {
            return true;
        }

        let recent_real: Vec<_> = self.disc_real_acc.iter().rev().take(window).copied().collect();
        let recent_fake: Vec<_> = self.disc_fake_acc.iter().rev().take(window).copied().collect();

        let avg_real: f64 = recent_real.iter().sum::<f64>() / recent_real.len() as f64;
        let avg_fake: f64 = recent_fake.iter().sum::<f64>() / recent_fake.len() as f64;

        // Balanced if both accuracies are in reasonable range
        (0.3..0.9).contains(&avg_real) && (0.3..0.9).contains(&avg_fake)
    }

    /// Save metrics to CSV file
    pub fn save_csv(&self, path: &str) -> anyhow::Result<()> {
        let mut writer = csv::Writer::from_path(path)?;

        writer.write_record(["epoch", "gen_loss", "disc_loss", "real_acc", "fake_acc"])?;

        for i in 0..self.num_epochs() {
            writer.write_record([
                (i + 1).to_string(),
                self.gen_losses[i].to_string(),
                self.disc_losses[i].to_string(),
                self.disc_real_acc[i].to_string(),
                self.disc_fake_acc[i].to_string(),
            ])?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load metrics from CSV file
    pub fn load_csv(path: &str) -> anyhow::Result<Self> {
        let mut reader = csv::Reader::from_path(path)?;
        let mut metrics = Self::new();

        for result in reader.records() {
            let record = result?;
            metrics.gen_losses.push(record[1].parse()?);
            metrics.disc_losses.push(record[2].parse()?);
            metrics.disc_real_acc.push(record[3].parse()?);
            metrics.disc_fake_acc.push(record[4].parse()?);
        }

        Ok(metrics)
    }
}

/// Exponential moving average tracker
#[derive(Debug)]
pub struct EMATracker {
    value: f64,
    alpha: f64,
    initialized: bool,
}

impl EMATracker {
    /// Create new EMA tracker
    ///
    /// # Arguments
    ///
    /// * `alpha` - Smoothing factor (0 < alpha <= 1). Higher = more weight on recent
    pub fn new(alpha: f64) -> Self {
        Self {
            value: 0.0,
            alpha: alpha.clamp(0.001, 1.0),
            initialized: false,
        }
    }

    /// Update with new value
    pub fn update(&mut self, new_value: f64) {
        if !self.initialized {
            self.value = new_value;
            self.initialized = true;
        } else {
            self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value;
        }
    }

    /// Get current EMA value
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.initialized = false;
    }
}

/// Calculate moving average of last `window` values
fn moving_average(values: &[f64], window: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let n = window.min(values.len());
    let sum: f64 = values.iter().rev().take(n).sum();
    sum / n as f64
}

/// Rolling statistics tracker
#[derive(Debug)]
pub struct RollingStats {
    window: VecDeque<f64>,
    max_size: usize,
    sum: f64,
    sum_sq: f64,
}

impl RollingStats {
    /// Create new rolling statistics tracker
    pub fn new(max_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(max_size),
            max_size,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Add a new value
    pub fn push(&mut self, value: f64) {
        if self.window.len() >= self.max_size {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }

        self.window.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
    }

    /// Get mean of values in window
    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            self.sum / self.window.len() as f64
        }
    }

    /// Get variance of values in window
    pub fn variance(&self) -> f64 {
        if self.window.len() < 2 {
            0.0
        } else {
            let n = self.window.len() as f64;
            let mean = self.sum / n;
            (self.sum_sq / n) - (mean * mean)
        }
    }

    /// Get standard deviation
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get number of values in window
    pub fn len(&self) -> usize {
        self.window.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new();

        metrics.record_epoch(1.5, 0.8, 0.6, 0.7);
        metrics.record_epoch(1.3, 0.75, 0.65, 0.68);

        assert_eq!(metrics.num_epochs(), 2);
        assert_eq!(metrics.latest_gen_loss(), Some(1.3));
    }

    #[test]
    fn test_ema_tracker() {
        let mut ema = EMATracker::new(0.5);

        ema.update(10.0);
        assert_eq!(ema.value(), 10.0);

        ema.update(20.0);
        assert_eq!(ema.value(), 15.0); // 0.5 * 20 + 0.5 * 10
    }

    #[test]
    fn test_rolling_stats() {
        let mut stats = RollingStats::new(3);

        stats.push(1.0);
        stats.push(2.0);
        stats.push(3.0);

        assert_eq!(stats.mean(), 2.0);

        stats.push(4.0); // removes 1.0
        assert_eq!(stats.mean(), 3.0); // (2 + 3 + 4) / 3
    }
}
