//! Risk-specific feature extraction
//!
//! Extracts features specifically designed for risk detection

use super::indicators::*;
use crate::data::OHLCVSeries;

/// Collection of risk-related features
#[derive(Debug, Clone)]
pub struct RiskFeatures {
    /// Price returns
    pub returns: Vec<f64>,
    /// Log returns
    pub log_returns: Vec<f64>,
    /// Rolling volatility (standard deviation of returns)
    pub volatility: Vec<f64>,
    /// Volume change percentage
    pub volume_change: Vec<f64>,
    /// ATR (Average True Range)
    pub atr: Vec<f64>,
    /// Bollinger Band width
    pub bb_width: Vec<f64>,
    /// RSI
    pub rsi: Vec<f64>,
    /// Rolling maximum drawdown
    pub max_drawdown: Vec<f64>,
    /// Price momentum (rate of change)
    pub momentum: Vec<f64>,
    /// VWAP deviation
    pub vwap_deviation: Vec<f64>,
    /// Volatility percentile rank
    pub volatility_percentile: Vec<f64>,
}

impl RiskFeatures {
    /// Extract all risk features from OHLCV data
    pub fn from_ohlcv(data: &OHLCVSeries) -> Self {
        let closes = data.closes();
        let volumes = data.volumes();
        let returns = data.returns();
        let log_returns = data.log_returns();

        // Calculate various features
        let volatility = rolling_std(&returns, 20);

        let volume_change: Vec<f64> = if volumes.len() > 1 {
            volumes
                .windows(2)
                .map(|w| {
                    if w[0] > 0.0 {
                        (w[1] - w[0]) / w[0] * 100.0
                    } else {
                        0.0
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        let atr_values = atr(data, 14);
        let bb_width = bollinger_width(&closes, 20, 2.0);
        let rsi_values = rsi(&closes, 14);
        let max_dd = rolling_max_drawdown(&closes, 20);
        let momentum = roc(&closes, 10);
        let vwap_dev = vwap_deviation(&closes, &volumes);
        let vol_percentile = rolling_percentile_rank(&volatility, 60);

        Self {
            returns,
            log_returns,
            volatility,
            volume_change,
            atr: atr_values,
            bb_width,
            rsi: rsi_values,
            max_drawdown: max_dd,
            momentum,
            vwap_deviation: vwap_dev,
            volatility_percentile: vol_percentile,
        }
    }

    /// Get latest feature values as a vector
    pub fn latest_vector(&self) -> Vec<f64> {
        vec![
            *self.returns.last().unwrap_or(&0.0),
            *self.volatility.last().unwrap_or(&0.0),
            *self.volume_change.last().unwrap_or(&0.0),
            *self.atr.last().unwrap_or(&0.0),
            *self.bb_width.last().unwrap_or(&0.0),
            *self.rsi.last().unwrap_or(&50.0),
            *self.max_drawdown.last().unwrap_or(&0.0),
            *self.momentum.last().unwrap_or(&0.0),
            *self.vwap_deviation.last().unwrap_or(&0.0),
            *self.volatility_percentile.last().unwrap_or(&0.5),
        ]
    }

    /// Get feature matrix for all time steps
    pub fn to_matrix(&self) -> Vec<Vec<f64>> {
        let min_len = self
            .returns
            .len()
            .min(self.volatility.len())
            .min(self.volume_change.len())
            .min(self.atr.len())
            .min(self.bb_width.len())
            .min(self.rsi.len())
            .min(self.max_drawdown.len())
            .min(self.momentum.len());

        (0..min_len)
            .map(|i| {
                vec![
                    *self.returns.get(i).unwrap_or(&0.0),
                    *self.volatility.get(i).unwrap_or(&0.0),
                    *self.volume_change.get(i).unwrap_or(&0.0),
                    *self.atr.get(i).unwrap_or(&0.0),
                    *self.bb_width.get(i).unwrap_or(&0.0),
                    *self.rsi.get(i).unwrap_or(&50.0),
                    *self.max_drawdown.get(i).unwrap_or(&0.0),
                    *self.momentum.get(i).unwrap_or(&0.0),
                ]
            })
            .collect()
    }

    /// Calculate composite risk score based on features
    pub fn composite_risk_score(&self) -> f64 {
        let latest = self.latest_vector();

        // Weighted combination of normalized features
        let volatility_score = latest[1].abs() / 5.0; // Normalize by expected max
        let volume_score = latest[2].abs() / 100.0;
        let drawdown_score = latest[6] / 0.2; // Normalize by 20% max drawdown
        let rsi_extreme = ((latest[5] - 50.0).abs() / 50.0).powi(2);

        // Combine with weights
        let score = 0.3 * volatility_score
            + 0.2 * volume_score
            + 0.3 * drawdown_score
            + 0.2 * rsi_extreme;

        score.clamp(0.0, 1.0)
    }
}

/// Crypto-specific risk features
#[derive(Debug, Clone)]
pub struct CryptoRiskFeatures {
    /// Base risk features
    pub base: RiskFeatures,
    /// 24h price change
    pub change_24h: f64,
    /// Volume ratio vs 7-day average
    pub volume_ratio: f64,
    /// Intraday range ratio
    pub range_ratio: f64,
}

impl CryptoRiskFeatures {
    /// Extract crypto-specific features
    pub fn from_ohlcv(data: &OHLCVSeries) -> Self {
        let base = RiskFeatures::from_ohlcv(data);

        // 24h change (assuming hourly data, 24 candles)
        let change_24h = if data.len() >= 24 {
            let start = data.data[data.len() - 24].close;
            let end = data.data.last().map(|c| c.close).unwrap_or(start);
            (end - start) / start * 100.0
        } else {
            0.0
        };

        // Volume ratio vs 7-day average (168 hours)
        let volumes = data.volumes();
        let volume_ratio = if volumes.len() >= 168 {
            let avg_7d: f64 = volumes[(volumes.len() - 168)..].iter().sum::<f64>() / 168.0;
            let current = *volumes.last().unwrap_or(&0.0);
            if avg_7d > 0.0 {
                current / avg_7d
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Intraday range ratio
        let range_ratio = if let Some(candle) = data.data.last() {
            let range = candle.high - candle.low;
            let price = candle.close;
            if price > 0.0 {
                range / price * 100.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        Self {
            base,
            change_24h,
            volume_ratio,
            range_ratio,
        }
    }

    /// Calculate crypto-specific risk score
    pub fn crypto_risk_score(&self) -> f64 {
        let base_score = self.base.composite_risk_score();

        // Add crypto-specific factors
        let change_factor = (self.change_24h.abs() / 10.0).min(1.0); // 10% = max
        let volume_factor = if self.volume_ratio > 3.0 {
            ((self.volume_ratio - 1.0) / 5.0).min(1.0)
        } else {
            0.0
        };
        let range_factor = (self.range_ratio / 5.0).min(1.0); // 5% range = max

        // Weighted combination
        let score =
            0.5 * base_score + 0.2 * change_factor + 0.15 * volume_factor + 0.15 * range_factor;

        score.clamp(0.0, 1.0)
    }
}

/// Multi-asset risk features for portfolio analysis
#[derive(Debug, Clone)]
pub struct MultiAssetRiskFeatures {
    /// Individual asset features
    pub assets: Vec<(String, RiskFeatures)>,
    /// Cross-asset correlations
    pub correlations: Vec<(String, String, f64)>,
    /// Portfolio-level volatility
    pub portfolio_volatility: f64,
}

impl MultiAssetRiskFeatures {
    /// Create from multiple OHLCV series
    pub fn from_multiple(data: &[(String, OHLCVSeries)]) -> Self {
        let assets: Vec<(String, RiskFeatures)> = data
            .iter()
            .map(|(name, series)| (name.clone(), RiskFeatures::from_ohlcv(series)))
            .collect();

        // Calculate correlations between all pairs
        let mut correlations = Vec::new();
        for i in 0..assets.len() {
            for j in (i + 1)..assets.len() {
                let corr = rolling_correlation(&assets[i].1.returns, &assets[j].1.returns, 30);
                let latest_corr = corr.last().copied().unwrap_or(0.0);
                correlations.push((assets[i].0.clone(), assets[j].0.clone(), latest_corr));
            }
        }

        // Simple portfolio volatility (equal weighted)
        let n = assets.len() as f64;
        let portfolio_volatility = if n > 0.0 {
            assets
                .iter()
                .map(|(_, f)| f.volatility.last().copied().unwrap_or(0.0))
                .sum::<f64>()
                / n
        } else {
            0.0
        };

        Self {
            assets,
            correlations,
            portfolio_volatility,
        }
    }

    /// Get average correlation (market stress indicator)
    pub fn average_correlation(&self) -> f64 {
        if self.correlations.is_empty() {
            return 0.0;
        }

        self.correlations.iter().map(|(_, _, c)| c).sum::<f64>()
            / self.correlations.len() as f64
    }

    /// Calculate systemic risk score
    pub fn systemic_risk_score(&self) -> f64 {
        // High correlation + high volatility = systemic risk
        let avg_corr = self.average_correlation();
        let vol = self.portfolio_volatility;

        // Correlation above 0.7 is concerning
        let corr_factor = ((avg_corr - 0.5) / 0.5).max(0.0).min(1.0);

        // Volatility factor
        let vol_factor = (vol / 5.0).min(1.0); // 5% = max

        (corr_factor + vol_factor) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_series() -> OHLCVSeries {
        use crate::data::OHLCV;

        let data: Vec<OHLCV> = (0..100)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.1).sin() * 5.0;
                OHLCV::new(Utc::now(), base, base + 1.0, base - 1.0, base + 0.5, 1000.0)
            })
            .collect();

        OHLCVSeries::with_data("TEST".into(), "1h".into(), data)
    }

    #[test]
    fn test_risk_features_extraction() {
        let series = create_test_series();
        let features = RiskFeatures::from_ohlcv(&series);

        assert!(!features.returns.is_empty());
        assert!(!features.volatility.is_empty());
        assert!(!features.rsi.is_empty());
    }

    #[test]
    fn test_composite_risk_score() {
        let series = create_test_series();
        let features = RiskFeatures::from_ohlcv(&series);
        let score = features.composite_risk_score();

        assert!(score >= 0.0 && score <= 1.0);
    }
}
