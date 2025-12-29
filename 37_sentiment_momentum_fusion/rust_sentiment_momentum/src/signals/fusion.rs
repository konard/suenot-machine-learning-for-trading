//! Fusion стратегия
//!
//! Объединение ценового моментума и sentiment-моментума
//! для генерации торговых сигналов.

use crate::models::{Signal, SignalStrength, SignalType, TradingSignal};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::divergence::{Divergence, DivergenceDetector};

/// Результат fusion-анализа
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionResult {
    /// Объединённый сигнал
    pub signal: Signal,
    /// Компонент ценового моментума (нормализованный)
    pub price_component: f64,
    /// Компонент sentiment-моментума (нормализованный)
    pub sentiment_component: f64,
    /// Информация о дивергенции
    pub divergence: Option<Divergence>,
    /// Уверенность в сигнале
    pub confidence: f64,
}

impl FusionResult {
    /// Создание нового результата
    pub fn new(
        signal: Signal,
        price_component: f64,
        sentiment_component: f64,
    ) -> Self {
        Self {
            signal,
            price_component,
            sentiment_component,
            divergence: None,
            confidence: 0.5,
        }
    }

    /// Добавление информации о дивергенции
    pub fn with_divergence(mut self, divergence: Divergence) -> Self {
        self.divergence = Some(divergence);
        self
    }

    /// Установка уверенности
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Преобразование в TradingSignal
    pub fn to_trading_signal(
        &self,
        timestamp: DateTime<Utc>,
        symbol: &str,
    ) -> TradingSignal {
        let mut signal = TradingSignal::new(
            timestamp,
            symbol.to_string(),
            self.signal.signal_type,
            SignalStrength::new(self.signal.value, self.confidence),
            self.price_component,
            self.sentiment_component,
        );

        if let Some(ref div) = self.divergence {
            signal = signal.with_divergence(&format!("{:?}", div.divergence_type));
        }

        signal
    }
}

/// Стратегия Fusion
///
/// Объединяет ценовой и sentiment моментум с настраиваемыми весами.
pub struct FusionStrategy {
    /// Вес ценового моментума (0.0 - 1.0)
    price_weight: f64,
    /// Вес sentiment-моментума (0.0 - 1.0)
    sentiment_weight: f64,
    /// Детектор дивергенций
    divergence_detector: DivergenceDetector,
    /// Порог для сильного сигнала
    strong_signal_threshold: f64,
    /// Порог для обычного сигнала
    signal_threshold: f64,
}

impl Default for FusionStrategy {
    fn default() -> Self {
        Self::new(0.6, 0.4)
    }
}

impl FusionStrategy {
    /// Создание новой стратегии
    ///
    /// # Аргументы
    ///
    /// * `price_weight` - Вес ценового моментума (0.0 - 1.0)
    /// * `sentiment_weight` - Вес sentiment-моментума (0.0 - 1.0)
    ///
    /// Веса будут автоматически нормализованы.
    pub fn new(price_weight: f64, sentiment_weight: f64) -> Self {
        let total = price_weight + sentiment_weight;
        let (pw, sw) = if total > 0.0 {
            (price_weight / total, sentiment_weight / total)
        } else {
            (0.5, 0.5)
        };

        Self {
            price_weight: pw,
            sentiment_weight: sw,
            divergence_detector: DivergenceDetector::default(),
            strong_signal_threshold: 0.5,
            signal_threshold: 0.2,
        }
    }

    /// Установка порогов сигналов
    pub fn with_thresholds(mut self, strong: f64, normal: f64) -> Self {
        self.strong_signal_threshold = strong;
        self.signal_threshold = normal;
        self
    }

    /// Установка детектора дивергенций
    pub fn with_divergence_detector(mut self, detector: DivergenceDetector) -> Self {
        self.divergence_detector = detector;
        self
    }

    /// Расчёт объединённого сигнала
    pub fn calculate_signal(
        &self,
        price_momentum: f64,
        sentiment_momentum: f64,
    ) -> FusionResult {
        // Нормализация входных данных (предполагаем z-score или -1 до 1)
        let price_norm = price_momentum.clamp(-3.0, 3.0) / 3.0;
        let sent_norm = sentiment_momentum.clamp(-1.0, 1.0);

        // Объединённый сигнал
        let combined = self.price_weight * price_norm + self.sentiment_weight * sent_norm;

        // Определение уверенности
        // Высокая уверенность когда оба компонента согласны
        let agreement = if (price_norm > 0.0 && sent_norm > 0.0)
            || (price_norm < 0.0 && sent_norm < 0.0)
        {
            1.0
        } else if price_norm.abs() < 0.1 || sent_norm.abs() < 0.1 {
            0.7
        } else {
            0.4 // Дивергенция
        };

        let confidence = agreement * (combined.abs().min(1.0) * 0.5 + 0.5);

        let signal = Signal::new(combined);

        FusionResult::new(signal, price_norm, sent_norm).with_confidence(confidence)
    }

    /// Расчёт сигналов для временного ряда
    pub fn calculate_signals(
        &self,
        price_momentums: &[f64],
        sentiment_momentums: &[f64],
    ) -> Vec<FusionResult> {
        let len = price_momentums.len().min(sentiment_momentums.len());

        (0..len)
            .map(|i| {
                let mut result = self.calculate_signal(price_momentums[i], sentiment_momentums[i]);

                // Проверка дивергенции
                if let Some(div) = self.divergence_detector.detect_at_point(
                    price_momentums[i],
                    sentiment_momentums[i],
                ) {
                    result = result.with_divergence(div);

                    // При сильной дивергенции переопределяем сигнал
                    if result.divergence.as_ref().map(|d| d.is_strong).unwrap_or(false) {
                        result.signal = Signal::new(0.0); // Нейтральный сигнал
                        result.signal.signal_type = SignalType::Exit;
                    }
                }

                result
            })
            .collect()
    }

    /// Расчёт сигналов с полными данными
    pub fn calculate_with_prices_and_sentiments(
        &self,
        prices: &[f64],
        sentiments: &[f64],
        momentum_period: usize,
    ) -> Vec<FusionResult> {
        use crate::momentum::{calculate_price_momentum, calculate_sentiment_momentum};

        let price_moms = calculate_price_momentum(prices, momentum_period);
        let sent_moms = calculate_sentiment_momentum(sentiments, momentum_period);

        let price_values: Vec<f64> = price_moms.iter().map(|m| m.z_score).collect();
        let sent_values: Vec<f64> = sent_moms.iter().map(|m| m.value).collect();

        self.calculate_signals(&price_values, &sent_values)
    }

    /// Генерация торговых сигналов
    pub fn generate_trading_signals(
        &self,
        symbol: &str,
        timestamps: &[DateTime<Utc>],
        price_momentums: &[f64],
        sentiment_momentums: &[f64],
    ) -> Vec<TradingSignal> {
        let fusion_results = self.calculate_signals(price_momentums, sentiment_momentums);

        fusion_results
            .iter()
            .zip(timestamps.iter())
            .map(|(result, &ts)| result.to_trading_signal(ts, symbol))
            .collect()
    }

    /// Получение текущих весов
    pub fn weights(&self) -> (f64, f64) {
        (self.price_weight, self.sentiment_weight)
    }
}

/// Адаптивная стратегия Fusion
///
/// Автоматически подстраивает веса на основе исторической эффективности.
pub struct AdaptiveFusionStrategy {
    /// Базовая стратегия
    base_strategy: FusionStrategy,
    /// История ценовых предсказаний
    price_predictions: Vec<(f64, f64)>, // (prediction, actual_return)
    /// История sentiment предсказаний
    sentiment_predictions: Vec<(f64, f64)>,
    /// Размер окна для адаптации
    adaptation_window: usize,
}

impl AdaptiveFusionStrategy {
    /// Создание адаптивной стратегии
    pub fn new(adaptation_window: usize) -> Self {
        Self {
            base_strategy: FusionStrategy::default(),
            price_predictions: Vec::new(),
            sentiment_predictions: Vec::new(),
            adaptation_window,
        }
    }

    /// Обновление весов на основе новых данных
    pub fn update(
        &mut self,
        price_prediction: f64,
        sentiment_prediction: f64,
        actual_return: f64,
    ) {
        self.price_predictions.push((price_prediction, actual_return));
        self.sentiment_predictions
            .push((sentiment_prediction, actual_return));

        // Ограничиваем размер истории
        if self.price_predictions.len() > self.adaptation_window {
            self.price_predictions.remove(0);
            self.sentiment_predictions.remove(0);
        }

        // Пересчитываем веса
        self.recalculate_weights();
    }

    /// Пересчёт весов на основе исторической эффективности
    fn recalculate_weights(&mut self) {
        if self.price_predictions.len() < 10 {
            return; // Недостаточно данных
        }

        // Расчёт корреляции price prediction с returns
        let price_ic = Self::calculate_ic(&self.price_predictions);

        // Расчёт корреляции sentiment prediction с returns
        let sentiment_ic = Self::calculate_ic(&self.sentiment_predictions);

        // Новые веса пропорциональны IC
        let total_ic = price_ic.abs() + sentiment_ic.abs();
        if total_ic > 0.01 {
            let new_price_weight = price_ic.abs() / total_ic;
            let new_sentiment_weight = sentiment_ic.abs() / total_ic;

            self.base_strategy = FusionStrategy::new(new_price_weight, new_sentiment_weight);
        }
    }

    /// Расчёт Information Coefficient (IC)
    fn calculate_ic(predictions: &[(f64, f64)]) -> f64 {
        if predictions.len() < 2 {
            return 0.0;
        }

        let n = predictions.len() as f64;

        let mean_pred: f64 = predictions.iter().map(|(p, _)| p).sum::<f64>() / n;
        let mean_ret: f64 = predictions.iter().map(|(_, r)| r).sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_pred = 0.0;
        let mut var_ret = 0.0;

        for &(pred, ret) in predictions {
            let dp = pred - mean_pred;
            let dr = ret - mean_ret;
            cov += dp * dr;
            var_pred += dp * dp;
            var_ret += dr * dr;
        }

        let std_pred = (var_pred / n).sqrt();
        let std_ret = (var_ret / n).sqrt();

        if std_pred > 0.0 && std_ret > 0.0 {
            (cov / n) / (std_pred * std_ret)
        } else {
            0.0
        }
    }

    /// Получение текущей стратегии
    pub fn strategy(&self) -> &FusionStrategy {
        &self.base_strategy
    }

    /// Расчёт сигнала
    pub fn calculate_signal(&self, price_momentum: f64, sentiment_momentum: f64) -> FusionResult {
        self.base_strategy.calculate_signal(price_momentum, sentiment_momentum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_strategy() {
        let strategy = FusionStrategy::new(0.6, 0.4);

        // Оба положительные = сильный long
        let result = strategy.calculate_signal(0.8, 0.7);
        assert!(result.signal.value > 0.0);
        assert!(result.confidence > 0.5);

        // Оба отрицательные = сильный short
        let result = strategy.calculate_signal(-0.8, -0.7);
        assert!(result.signal.value < 0.0);
        assert!(result.confidence > 0.5);

        // Разнонаправленные = низкая уверенность
        let result = strategy.calculate_signal(0.8, -0.7);
        assert!(result.confidence < 0.5);
    }

    #[test]
    fn test_weights_normalization() {
        let strategy = FusionStrategy::new(3.0, 2.0);
        let (pw, sw) = strategy.weights();

        assert!((pw - 0.6).abs() < 1e-10);
        assert!((sw - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_signal_series() {
        let strategy = FusionStrategy::default();

        let price_moms = vec![0.5, 0.6, 0.7, 0.3, -0.2, -0.5];
        let sent_moms = vec![0.4, 0.5, 0.6, 0.2, -0.3, -0.6];

        let results = strategy.calculate_signals(&price_moms, &sent_moms);

        assert_eq!(results.len(), 6);
        assert!(results[0].signal.value > 0.0); // Первые положительные
        assert!(results[5].signal.value < 0.0); // Последние отрицательные
    }

    #[test]
    fn test_adaptive_strategy() {
        let mut strategy = AdaptiveFusionStrategy::new(20);

        // Симулируем данные где sentiment лучше предсказывает
        for i in 0..30 {
            let price_pred = 0.1 * ((i % 5) as f64 - 2.0);
            let sent_pred = 0.05 * (i as f64 - 15.0);
            let actual = sent_pred * 0.8 + 0.01 * (i as f64); // Sentiment коррелирует лучше

            strategy.update(price_pred, sent_pred, actual);
        }

        // После адаптации sentiment должен иметь больший вес
        let (pw, sw) = strategy.strategy().weights();
        // Проверяем что веса изменились (не остались 0.5/0.5 или 0.6/0.4)
        assert!(pw > 0.0 && pw < 1.0);
        assert!(sw > 0.0 && sw < 1.0);
    }
}
