//! Cross-Sectional Momentum
//!
//! Этот модуль реализует cross-sectional momentum (relative momentum),
//! который сравнивает активы друг с другом.

use crate::data::{PriceSeries, Signal, Signals};
use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Конфигурация для cross-sectional momentum
#[derive(Debug, Clone)]
pub struct CrossSectionalMomentumConfig {
    /// Период для расчёта моментума (в свечах)
    pub lookback: usize,
    /// Количество топ активов для лонга
    pub top_n: usize,
    /// Количество худших активов для шорта (0 = long-only)
    pub bottom_n: usize,
    /// Использовать percentile ranking
    pub use_percentile: bool,
    /// Порог percentile для лонга (например, 0.75 = топ 25%)
    pub long_percentile: f64,
    /// Порог percentile для шорта (например, 0.25 = худшие 25%)
    pub short_percentile: f64,
}

impl Default for CrossSectionalMomentumConfig {
    fn default() -> Self {
        Self {
            lookback: 30,
            top_n: 3,
            bottom_n: 0, // Long-only для крипто
            use_percentile: false,
            long_percentile: 0.75,
            short_percentile: 0.25,
        }
    }
}

impl CrossSectionalMomentumConfig {
    /// Конфигурация для криптовалют (long-only)
    pub fn crypto_long_only() -> Self {
        Self {
            lookback: 30,
            top_n: 3,
            bottom_n: 0,
            use_percentile: false,
            long_percentile: 0.75,
            short_percentile: 0.25,
        }
    }

    /// Конфигурация для long-short стратегии
    pub fn long_short(top_n: usize, bottom_n: usize) -> Self {
        Self {
            lookback: 30,
            top_n,
            bottom_n,
            use_percentile: false,
            long_percentile: 0.75,
            short_percentile: 0.25,
        }
    }
}

/// Cross-sectional momentum калькулятор
#[derive(Debug)]
pub struct CrossSectionalMomentum {
    config: CrossSectionalMomentumConfig,
}

/// Результат ранжирования активов
#[derive(Debug, Clone)]
pub struct RankedAsset {
    /// Символ актива
    pub symbol: String,
    /// Значение моментума
    pub momentum: f64,
    /// Ранг (1 = лучший)
    pub rank: usize,
    /// Percentile (0-1)
    pub percentile: f64,
}

impl CrossSectionalMomentum {
    /// Создать новый калькулятор
    pub fn new(config: CrossSectionalMomentumConfig) -> Self {
        Self { config }
    }

    /// Рассчитать моментум для каждого актива
    pub fn calculate_momentum(
        &self,
        price_data: &HashMap<String, PriceSeries>,
    ) -> Result<HashMap<String, f64>> {
        let mut momentum_map = HashMap::new();

        for (symbol, series) in price_data {
            let closes = series.closes();
            if closes.len() > self.config.lookback {
                let current = closes[closes.len() - 1];
                let past = closes[closes.len() - 1 - self.config.lookback];
                let momentum = (current - past) / past;
                momentum_map.insert(symbol.clone(), momentum);
            }
        }

        Ok(momentum_map)
    }

    /// Ранжировать активы по моментуму
    pub fn rank_assets(
        &self,
        price_data: &HashMap<String, PriceSeries>,
    ) -> Result<Vec<RankedAsset>> {
        let momentum_map = self.calculate_momentum(price_data)?;

        let mut assets: Vec<_> = momentum_map
            .into_iter()
            .map(|(symbol, momentum)| (symbol, momentum))
            .collect();

        // Сортируем по убыванию моментума (лучшие первые)
        assets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let n = assets.len();
        let ranked: Vec<RankedAsset> = assets
            .into_iter()
            .enumerate()
            .map(|(i, (symbol, momentum))| RankedAsset {
                symbol,
                momentum,
                rank: i + 1,
                percentile: 1.0 - (i as f64 / n as f64),
            })
            .collect();

        Ok(ranked)
    }

    /// Сгенерировать сигналы на основе ранжирования
    pub fn generate_signals(
        &self,
        price_data: &HashMap<String, PriceSeries>,
        timestamp: DateTime<Utc>,
    ) -> Result<Signals> {
        let ranked = self.rank_assets(price_data)?;
        let mut signals = Signals::new(timestamp);

        let n = ranked.len();

        for asset in &ranked {
            let signal = if self.config.use_percentile {
                // Используем percentile
                if asset.percentile >= self.config.long_percentile {
                    Signal::Long
                } else if asset.percentile <= self.config.short_percentile {
                    if self.config.bottom_n > 0 {
                        Signal::Short
                    } else {
                        Signal::Neutral
                    }
                } else {
                    Signal::Neutral
                }
            } else {
                // Используем top_n / bottom_n
                if asset.rank <= self.config.top_n {
                    Signal::Long
                } else if self.config.bottom_n > 0 && asset.rank > n - self.config.bottom_n {
                    Signal::Short
                } else {
                    Signal::Neutral
                }
            };

            signals.set(&asset.symbol, signal);
        }

        Ok(signals)
    }

    /// Получить топ активов по моментуму
    pub fn top_assets(
        &self,
        price_data: &HashMap<String, PriceSeries>,
        n: usize,
    ) -> Result<Vec<RankedAsset>> {
        let ranked = self.rank_assets(price_data)?;
        Ok(ranked.into_iter().take(n).collect())
    }

    /// Получить худшие активы по моментуму
    pub fn bottom_assets(
        &self,
        price_data: &HashMap<String, PriceSeries>,
        n: usize,
    ) -> Result<Vec<RankedAsset>> {
        let ranked = self.rank_assets(price_data)?;
        let len = ranked.len();
        Ok(ranked.into_iter().skip(len.saturating_sub(n)).collect())
    }
}

/// Ранжировать значения и вернуть ранги
pub fn rank_values(values: &[f64]) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();

    // Сортируем по убыванию
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Присваиваем ранги
    let mut ranks = vec![0; values.len()];
    for (rank, (idx, _)) in indexed.into_iter().enumerate() {
        ranks[idx] = rank + 1;
    }

    ranks
}

/// Преобразовать ранги в percentiles
pub fn ranks_to_percentiles(ranks: &[usize]) -> Vec<f64> {
    let n = ranks.len() as f64;
    ranks.iter().map(|&r| 1.0 - (r as f64 - 1.0) / n).collect()
}

/// Z-score нормализация
pub fn zscore_normalize(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return vec![0.0; values.len()];
    }

    values.iter().map(|x| (x - mean) / std_dev).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Candle;

    fn create_test_series(symbol: &str, prices: Vec<f64>) -> PriceSeries {
        let mut series = PriceSeries::new(symbol.to_string(), "D".to_string());
        for (i, price) in prices.iter().enumerate() {
            let candle = Candle::new(
                Utc::now() + chrono::Duration::days(i as i64),
                *price,
                *price,
                *price,
                *price,
                1000.0,
            );
            series.push(candle);
        }
        series
    }

    #[test]
    fn test_rank_values() {
        let values = vec![10.0, 30.0, 20.0, 40.0];
        let ranks = rank_values(&values);

        assert_eq!(ranks[0], 4); // 10.0 = худший (ранг 4)
        assert_eq!(ranks[1], 2); // 30.0 = второй (ранг 2)
        assert_eq!(ranks[2], 3); // 20.0 = третий (ранг 3)
        assert_eq!(ranks[3], 1); // 40.0 = лучший (ранг 1)
    }

    #[test]
    fn test_zscore_normalize() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let zscores = zscore_normalize(&values);

        // Среднее z-score должно быть ~0
        let mean: f64 = zscores.iter().sum::<f64>() / zscores.len() as f64;
        assert!(mean.abs() < 1e-10);

        // Стандартное отклонение должно быть ~1
        let variance: f64 = zscores.iter().map(|x| x.powi(2)).sum::<f64>() / zscores.len() as f64;
        assert!((variance - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_cross_sectional_momentum() {
        // Создаём тестовые данные
        let mut price_data = HashMap::new();

        // BTC: рост 20%
        price_data.insert(
            "BTCUSDT".to_string(),
            create_test_series("BTCUSDT", vec![100.0, 105.0, 110.0, 115.0, 120.0]),
        );

        // ETH: рост 10%
        price_data.insert(
            "ETHUSDT".to_string(),
            create_test_series("ETHUSDT", vec![100.0, 102.0, 105.0, 108.0, 110.0]),
        );

        // SOL: падение 10%
        price_data.insert(
            "SOLUSDT".to_string(),
            create_test_series("SOLUSDT", vec![100.0, 98.0, 95.0, 92.0, 90.0]),
        );

        let config = CrossSectionalMomentumConfig {
            lookback: 3,
            top_n: 2,
            bottom_n: 0,
            use_percentile: false,
            long_percentile: 0.66,
            short_percentile: 0.33,
        };

        let calc = CrossSectionalMomentum::new(config);
        let ranked = calc.rank_assets(&price_data).unwrap();

        // BTC должен быть первым (лучший моментум)
        assert_eq!(ranked[0].symbol, "BTCUSDT");
        // SOL должен быть последним (худший моментум)
        assert_eq!(ranked[2].symbol, "SOLUSDT");

        // Проверяем сигналы
        let signals = calc.generate_signals(&price_data, Utc::now()).unwrap();
        assert_eq!(signals.get("BTCUSDT"), Signal::Long);
        assert_eq!(signals.get("ETHUSDT"), Signal::Long);
        assert_eq!(signals.get("SOLUSDT"), Signal::Neutral);
    }
}
