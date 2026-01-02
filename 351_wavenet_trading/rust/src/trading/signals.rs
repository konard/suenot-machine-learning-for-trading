//! Генерация торговых сигналов

use crate::types::Signal;

/// Генератор сигналов на основе предсказанной доходности
#[derive(Debug, Clone)]
pub struct ThresholdSignalGenerator {
    pub buy_threshold: f64,
    pub sell_threshold: f64,
}

impl ThresholdSignalGenerator {
    pub fn new(buy_threshold: f64, sell_threshold: f64) -> Self {
        Self {
            buy_threshold,
            sell_threshold,
        }
    }

    /// Сгенерировать сигнал на основе предсказанной доходности
    pub fn generate(&self, predicted_return: f64) -> Signal {
        if predicted_return > self.buy_threshold {
            Signal::Buy
        } else if predicted_return < self.sell_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    /// Сгенерировать сигналы для последовательности
    pub fn generate_sequence(&self, predictions: &[f64]) -> Vec<Signal> {
        predictions.iter().map(|&p| self.generate(p)).collect()
    }
}

impl Default for ThresholdSignalGenerator {
    fn default() -> Self {
        Self::new(0.001, -0.001) // 0.1% threshold
    }
}

/// Генератор сигналов на основе классификации
#[derive(Debug, Clone)]
pub struct ClassificationSignalGenerator {
    pub confidence_threshold: f64,
}

impl ClassificationSignalGenerator {
    pub fn new(confidence_threshold: f64) -> Self {
        Self {
            confidence_threshold,
        }
    }

    /// Сгенерировать сигнал на основе вероятностей [sell, hold, buy]
    pub fn generate(&self, probabilities: &[f64; 3]) -> Signal {
        let (max_idx, &max_prob) = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        if max_prob < self.confidence_threshold {
            return Signal::Hold;
        }

        match max_idx {
            0 => Signal::Sell,
            1 => Signal::Hold,
            2 => Signal::Buy,
            _ => Signal::Hold,
        }
    }
}

impl Default for ClassificationSignalGenerator {
    fn default() -> Self {
        Self::new(0.5)
    }
}

/// Сглаживание сигналов (фильтрация шума)
pub struct SignalSmoother {
    pub min_holding_period: usize,
}

impl SignalSmoother {
    pub fn new(min_holding_period: usize) -> Self {
        Self { min_holding_period }
    }

    /// Сгладить последовательность сигналов
    pub fn smooth(&self, signals: &[Signal]) -> Vec<Signal> {
        if signals.is_empty() {
            return Vec::new();
        }

        let mut smoothed = Vec::with_capacity(signals.len());
        let mut current_signal = signals[0];
        let mut holding_count = 0;

        for &signal in signals {
            if signal != current_signal {
                if holding_count >= self.min_holding_period {
                    current_signal = signal;
                    holding_count = 0;
                }
            }

            smoothed.push(current_signal);
            holding_count += 1;
        }

        smoothed
    }
}

/// Комбинирование нескольких сигналов (ансамбль)
pub fn combine_signals(signals: &[Signal], weights: &[f64]) -> Signal {
    assert_eq!(signals.len(), weights.len());

    let mut buy_weight = 0.0;
    let mut sell_weight = 0.0;
    let mut hold_weight = 0.0;

    for (signal, &weight) in signals.iter().zip(weights.iter()) {
        match signal {
            Signal::Buy => buy_weight += weight,
            Signal::Sell => sell_weight += weight,
            Signal::Hold => hold_weight += weight,
        }
    }

    if buy_weight > sell_weight && buy_weight > hold_weight {
        Signal::Buy
    } else if sell_weight > buy_weight && sell_weight > hold_weight {
        Signal::Sell
    } else {
        Signal::Hold
    }
}

/// Преобразование сигналов в позиции
pub fn signals_to_positions(signals: &[Signal]) -> Vec<f64> {
    signals.iter().map(|s| s.to_position()).collect()
}

/// Подсчёт статистики сигналов
#[derive(Debug, Default)]
pub struct SignalStats {
    pub total: usize,
    pub buy_count: usize,
    pub sell_count: usize,
    pub hold_count: usize,
    pub transitions: usize,
}

impl SignalStats {
    pub fn from_signals(signals: &[Signal]) -> Self {
        let mut stats = Self::default();
        stats.total = signals.len();

        for (i, signal) in signals.iter().enumerate() {
            match signal {
                Signal::Buy => stats.buy_count += 1,
                Signal::Sell => stats.sell_count += 1,
                Signal::Hold => stats.hold_count += 1,
            }

            if i > 0 && signals[i] != signals[i - 1] {
                stats.transitions += 1;
            }
        }

        stats
    }

    pub fn print(&self) {
        println!("=== Signal Statistics ===");
        println!("Total signals:  {}", self.total);
        println!("Buy signals:    {} ({:.1}%)", self.buy_count, 100.0 * self.buy_count as f64 / self.total as f64);
        println!("Sell signals:   {} ({:.1}%)", self.sell_count, 100.0 * self.sell_count as f64 / self.total as f64);
        println!("Hold signals:   {} ({:.1}%)", self.hold_count, 100.0 * self.hold_count as f64 / self.total as f64);
        println!("Transitions:    {}", self.transitions);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_generator() {
        let gen = ThresholdSignalGenerator::new(0.01, -0.01);

        assert_eq!(gen.generate(0.02), Signal::Buy);
        assert_eq!(gen.generate(-0.02), Signal::Sell);
        assert_eq!(gen.generate(0.005), Signal::Hold);
    }

    #[test]
    fn test_signal_smoother() {
        let signals = vec![
            Signal::Buy,
            Signal::Sell, // Quick change - should be smoothed
            Signal::Buy,
            Signal::Buy,
            Signal::Buy,
        ];

        let smoother = SignalSmoother::new(2);
        let smoothed = smoother.smooth(&signals);

        // First signal stays, others should be smoothed
        assert_eq!(smoothed[0], Signal::Buy);
    }

    #[test]
    fn test_combine_signals() {
        let signals = vec![Signal::Buy, Signal::Buy, Signal::Sell];
        let weights = vec![1.0, 1.0, 1.0];

        assert_eq!(combine_signals(&signals, &weights), Signal::Buy);
    }

    #[test]
    fn test_signal_stats() {
        let signals = vec![Signal::Buy, Signal::Buy, Signal::Sell, Signal::Hold];
        let stats = SignalStats::from_signals(&signals);

        assert_eq!(stats.total, 4);
        assert_eq!(stats.buy_count, 2);
        assert_eq!(stats.sell_count, 1);
        assert_eq!(stats.hold_count, 1);
        assert_eq!(stats.transitions, 2);
    }
}
