//! Генератор торговых сигналов

use ndarray::Array2;

/// Торговый сигнал
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingSignal {
    /// Покупка
    Long,
    /// Продажа
    Short,
    /// Без позиции
    Neutral,
}

impl TradingSignal {
    /// Возвращает позицию как число (-1, 0, 1)
    pub fn position(&self) -> f64 {
        match self {
            TradingSignal::Long => 1.0,
            TradingSignal::Short => -1.0,
            TradingSignal::Neutral => 0.0,
        }
    }
}

/// Генератор торговых сигналов на основе прогнозов модели
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    /// Порог для открытия long позиции
    long_threshold: f64,
    /// Порог для открытия short позиции
    short_threshold: f64,
    /// Минимальная уверенность для сигнала
    min_confidence: f64,
    /// Использовать усреднение прогнозов
    use_averaging: bool,
}

impl SignalGenerator {
    /// Создаёт генератор с параметрами по умолчанию
    pub fn new() -> Self {
        Self {
            long_threshold: 0.001,   // 0.1% expected return
            short_threshold: -0.001, // -0.1% expected return
            min_confidence: 0.0,
            use_averaging: true,
        }
    }

    /// Создаёт генератор с кастомными порогами
    pub fn with_thresholds(long_threshold: f64, short_threshold: f64) -> Self {
        Self {
            long_threshold,
            short_threshold,
            ..Self::new()
        }
    }

    /// Устанавливает минимальную уверенность
    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Генерирует сигнал из прогноза
    ///
    /// # Arguments
    ///
    /// * `predictions` - Прогнозы модели [pred_len]
    /// * `confidence` - Опциональная уверенность модели
    pub fn generate(
        &self,
        predictions: &[f64],
        confidence: Option<f64>,
    ) -> TradingSignal {
        if predictions.is_empty() {
            return TradingSignal::Neutral;
        }

        // Проверяем уверенность
        if let Some(conf) = confidence {
            if conf < self.min_confidence {
                return TradingSignal::Neutral;
            }
        }

        // Агрегируем прогнозы
        let prediction = if self.use_averaging {
            // Среднее всех прогнозов
            predictions.iter().sum::<f64>() / predictions.len() as f64
        } else {
            // Только первый (следующий) прогноз
            predictions[0]
        };

        // Генерируем сигнал
        if prediction > self.long_threshold {
            TradingSignal::Long
        } else if prediction < self.short_threshold {
            TradingSignal::Short
        } else {
            TradingSignal::Neutral
        }
    }

    /// Генерирует сигналы для батча прогнозов
    pub fn generate_batch(&self, predictions: &Array2<f64>) -> Vec<TradingSignal> {
        let batch_size = predictions.dim().0;

        (0..batch_size)
            .map(|i| {
                let preds: Vec<f64> = predictions.row(i).iter().copied().collect();
                self.generate(&preds, None)
            })
            .collect()
    }

    /// Возвращает агрегированный прогноз
    pub fn aggregate_prediction(&self, predictions: &[f64]) -> f64 {
        if predictions.is_empty() {
            return 0.0;
        }

        if self.use_averaging {
            predictions.iter().sum::<f64>() / predictions.len() as f64
        } else {
            predictions[0]
        }
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let generator = SignalGenerator::new();

        // Positive prediction -> Long
        let signal = generator.generate(&[0.005, 0.003], None);
        assert_eq!(signal, TradingSignal::Long);

        // Negative prediction -> Short
        let signal = generator.generate(&[-0.005, -0.003], None);
        assert_eq!(signal, TradingSignal::Short);

        // Near zero -> Neutral
        let signal = generator.generate(&[0.0001, -0.0001], None);
        assert_eq!(signal, TradingSignal::Neutral);
    }

    #[test]
    fn test_signal_with_threshold() {
        let generator = SignalGenerator::with_thresholds(0.01, -0.01);

        // Below threshold -> Neutral
        let signal = generator.generate(&[0.005], None);
        assert_eq!(signal, TradingSignal::Neutral);

        // Above threshold -> Long
        let signal = generator.generate(&[0.015], None);
        assert_eq!(signal, TradingSignal::Long);
    }

    #[test]
    fn test_signal_with_confidence() {
        let generator = SignalGenerator::new()
            .with_min_confidence(0.5);

        // Low confidence -> Neutral
        let signal = generator.generate(&[0.01], Some(0.3));
        assert_eq!(signal, TradingSignal::Neutral);

        // High confidence -> Signal
        let signal = generator.generate(&[0.01], Some(0.7));
        assert_eq!(signal, TradingSignal::Long);
    }

    #[test]
    fn test_signal_position() {
        assert_eq!(TradingSignal::Long.position(), 1.0);
        assert_eq!(TradingSignal::Short.position(), -1.0);
        assert_eq!(TradingSignal::Neutral.position(), 0.0);
    }

    #[test]
    fn test_batch_generation() {
        let generator = SignalGenerator::new();

        let predictions = ndarray::array![
            [0.005, 0.003],   // Long
            [-0.005, -0.003], // Short
            [0.0001, -0.0001] // Neutral
        ];

        let signals = generator.generate_batch(&predictions);

        assert_eq!(signals.len(), 3);
        assert_eq!(signals[0], TradingSignal::Long);
        assert_eq!(signals[1], TradingSignal::Short);
        assert_eq!(signals[2], TradingSignal::Neutral);
    }
}
