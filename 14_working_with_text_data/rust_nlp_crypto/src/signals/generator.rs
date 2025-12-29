//! Генератор торговых сигналов
//!
//! Объединяет анализ настроений с техническими индикаторами
//! для генерации торговых сигналов

use crate::api::{TechnicalIndicators, Trend};
use crate::models::{Announcement, Kline, Polarity, SignalAction, TradingSignal};
use crate::sentiment::{AggregatedSentiment, SentimentAnalyzer};
use chrono::Utc;

/// Конфигурация генератора сигналов
#[derive(Debug, Clone)]
pub struct SignalConfig {
    /// Минимальная уверенность для генерации сигнала
    pub min_confidence: f64,
    /// Минимальное количество текстов для анализа
    pub min_texts: usize,
    /// Вес sentiment в общем сигнале (0.0 - 1.0)
    pub sentiment_weight: f64,
    /// Вес технического анализа в общем сигнале
    pub technical_weight: f64,
    /// Порог для сильного сигнала
    pub strong_signal_threshold: f64,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            min_texts: 3,
            sentiment_weight: 0.6,
            technical_weight: 0.4,
            strong_signal_threshold: 0.7,
        }
    }
}

/// Генератор торговых сигналов
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    /// Анализатор настроений
    analyzer: SentimentAnalyzer,
    /// Конфигурация
    config: SignalConfig,
}

impl SignalGenerator {
    /// Создать новый генератор
    pub fn new() -> Self {
        Self {
            analyzer: SentimentAnalyzer::new(),
            config: SignalConfig::default(),
        }
    }

    /// Установить конфигурацию
    pub fn with_config(mut self, config: SignalConfig) -> Self {
        self.config = config;
        self
    }

    /// Генерировать сигнал на основе анонсов
    pub fn generate_from_announcements(
        &self,
        symbol: &str,
        announcements: &[Announcement],
    ) -> Option<TradingSignal> {
        // Фильтруем анонсы по символу
        let relevant: Vec<_> = announcements
            .iter()
            .filter(|a| a.symbols.iter().any(|s| s == symbol) || a.title.to_uppercase().contains(symbol))
            .collect();

        if relevant.len() < self.config.min_texts {
            return None;
        }

        // Анализируем тексты
        let texts: Vec<String> = relevant
            .iter()
            .map(|a| format!("{} {}", a.title, a.description))
            .collect();

        self.generate_from_texts(symbol, &texts)
    }

    /// Генерировать сигнал на основе текстов
    pub fn generate_from_texts(&self, symbol: &str, texts: &[String]) -> Option<TradingSignal> {
        if texts.len() < self.config.min_texts {
            return None;
        }

        // Анализируем все тексты
        let results: Vec<_> = texts.iter().map(|t| self.analyzer.analyze(t)).collect();

        // Агрегируем результаты
        let aggregated = self.analyzer.aggregate(&results);

        if aggregated.average_confidence < self.config.min_confidence {
            return None;
        }

        // Генерируем сигнал
        let action = self.determine_action(&aggregated, None);
        let reasons = self.generate_reasons(&aggregated);

        Some(TradingSignal {
            symbol: symbol.to_string(),
            timestamp: Utc::now(),
            action,
            strength: aggregated.average_score.abs(),
            confidence: aggregated.average_confidence,
            sentiment_score: aggregated.average_score,
            texts_analyzed: aggregated.total_count,
            reasons,
        })
    }

    /// Генерировать сигнал с учётом технических данных
    pub fn generate_with_technicals(
        &self,
        symbol: &str,
        texts: &[String],
        klines: &[Kline],
    ) -> Option<TradingSignal> {
        // Сначала получаем базовый сигнал от sentiment
        let mut signal = self.generate_from_texts(symbol, texts)?;

        // Добавляем технический анализ
        if klines.len() >= 20 {
            let trend = TechnicalIndicators::trend(klines, 7, 20);
            let rsi_values = TechnicalIndicators::rsi(klines, 14);

            let technical_score = self.calculate_technical_score(trend, &rsi_values);

            // Комбинируем sentiment и technical
            let combined_score = signal.sentiment_score * self.config.sentiment_weight
                + technical_score * self.config.technical_weight;

            // Обновляем сигнал
            signal.strength = combined_score.abs();
            signal.action = SignalAction::from_score(combined_score, signal.confidence);

            // Добавляем причины от технического анализа
            if let Some(t) = trend {
                signal.reasons.push(format!("Technical trend: {:?}", t));
            }
            if let Some(rsi) = rsi_values.last() {
                signal.reasons.push(format!("RSI: {:.1}", rsi));
            }
        }

        Some(signal)
    }

    /// Определить действие на основе агрегированного sentiment
    fn determine_action(
        &self,
        aggregated: &AggregatedSentiment,
        technical_score: Option<f64>,
    ) -> SignalAction {
        let score = aggregated.average_score;
        let confidence = aggregated.average_confidence;

        // Комбинируем с техническим анализом если есть
        let final_score = match technical_score {
            Some(tech) => {
                score * self.config.sentiment_weight + tech * self.config.technical_weight
            }
            None => score,
        };

        SignalAction::from_score(final_score, confidence)
    }

    /// Рассчитать техническую оценку
    fn calculate_technical_score(&self, trend: Option<Trend>, rsi: &[f64]) -> f64 {
        let mut score = 0.0;

        // Оценка тренда
        if let Some(t) = trend {
            score += match t {
                Trend::StrongBullish => 0.8,
                Trend::Bullish => 0.4,
                Trend::Sideways => 0.0,
                Trend::Bearish => -0.4,
                Trend::StrongBearish => -0.8,
            };
        }

        // Оценка RSI
        if let Some(&rsi_value) = rsi.last() {
            if rsi_value > 70.0 {
                // Перекупленность - медвежий сигнал
                score -= 0.3;
            } else if rsi_value < 30.0 {
                // Перепроданность - бычий сигнал
                score += 0.3;
            }
        }

        score.clamp(-1.0, 1.0)
    }

    /// Генерировать причины для сигнала
    fn generate_reasons(&self, aggregated: &AggregatedSentiment) -> Vec<String> {
        let mut reasons = Vec::new();

        // Общее настроение
        reasons.push(format!(
            "Sentiment score: {:.2} ({:?})",
            aggregated.average_score, aggregated.overall_polarity
        ));

        // Распределение
        let total = aggregated.total_count as f64;
        if total > 0.0 {
            let positive_pct = (aggregated.positive_count as f64 / total) * 100.0;
            let negative_pct = (aggregated.negative_count as f64 / total) * 100.0;
            reasons.push(format!(
                "Distribution: {:.0}% positive, {:.0}% negative",
                positive_pct, negative_pct
            ));
        }

        // Топ ключевые слова
        if !aggregated.top_keywords.is_empty() {
            let keywords: Vec<String> = aggregated
                .top_keywords
                .iter()
                .take(5)
                .map(|(word, score)| {
                    let sign = if *score > 0.0 { "+" } else { "" };
                    format!("{}({}{})", word, sign, format!("{:.1}", score))
                })
                .collect();
            reasons.push(format!("Key words: {}", keywords.join(", ")));
        }

        reasons
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Форматирование торгового сигнала
impl std::fmt::Display for TradingSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let action_emoji = match self.action {
            SignalAction::StrongBuy => "🟢🟢",
            SignalAction::Buy => "🟢",
            SignalAction::Hold => "⚪",
            SignalAction::Sell => "🔴",
            SignalAction::StrongSell => "🔴🔴",
        };

        writeln!(f, "═══════════════════════════════════════")?;
        writeln!(f, "  Trading Signal: {} {}", self.symbol, action_emoji)?;
        writeln!(f, "═══════════════════════════════════════")?;
        writeln!(f, "  Action: {:?}", self.action)?;
        writeln!(f, "  Strength: {:.2}", self.strength)?;
        writeln!(f, "  Confidence: {:.0}%", self.confidence * 100.0)?;
        writeln!(f, "  Sentiment: {:.2}", self.sentiment_score)?;
        writeln!(f, "  Texts analyzed: {}", self.texts_analyzed)?;
        writeln!(f, "  Time: {}", self.timestamp.format("%Y-%m-%d %H:%M:%S UTC"))?;
        writeln!(f, "───────────────────────────────────────")?;
        writeln!(f, "  Reasons:")?;
        for reason in &self.reasons {
            writeln!(f, "    • {}", reason)?;
        }
        writeln!(f, "═══════════════════════════════════════")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let generator = SignalGenerator::new();

        let texts = vec![
            "BTC is mooning! Very bullish!".to_string(),
            "Bitcoin looking strong today".to_string(),
            "Great news for Bitcoin holders".to_string(),
            "BTC to the moon!".to_string(),
        ];

        let signal = generator.generate_from_texts("BTC", &texts);

        assert!(signal.is_some());
        let signal = signal.unwrap();
        assert_eq!(signal.symbol, "BTC");
        assert!(signal.sentiment_score > 0.0);
    }

    #[test]
    fn test_min_texts_threshold() {
        let generator = SignalGenerator::new();

        // Только 2 текста, меньше минимума (3)
        let texts = vec![
            "BTC is good".to_string(),
            "Bitcoin rising".to_string(),
        ];

        let signal = generator.generate_from_texts("BTC", &texts);
        assert!(signal.is_none());
    }

    #[test]
    fn test_signal_display() {
        let signal = TradingSignal {
            symbol: "ETHUSDT".to_string(),
            timestamp: Utc::now(),
            action: SignalAction::Buy,
            strength: 0.65,
            confidence: 0.75,
            sentiment_score: 0.55,
            texts_analyzed: 10,
            reasons: vec!["Test reason".to_string()],
        };

        let display = format!("{}", signal);
        assert!(display.contains("ETHUSDT"));
        assert!(display.contains("Buy"));
    }
}
