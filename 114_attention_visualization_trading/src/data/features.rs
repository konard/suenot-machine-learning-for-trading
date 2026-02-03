//! Feature engineering for financial time series.

use super::Candle;

/// Feature engineering utilities.
pub struct FeatureEngineering;

impl FeatureEngineering {
    /// Compute log returns.
    pub fn compute_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![0.0; prices.len()];
        }

        let mut returns = vec![0.0];
        for i in 1..prices.len() {
            if prices[i - 1] > 0.0 {
                returns.push((prices[i] / prices[i - 1]).ln());
            } else {
                returns.push(0.0);
            }
        }
        returns
    }

    /// Compute rolling volatility.
    pub fn compute_volatility(returns: &[f64], window: usize) -> Vec<f64> {
        let mut volatility = vec![0.0; returns.len()];

        for i in window..returns.len() {
            let slice = &returns[i - window..i];
            let mean: f64 = slice.iter().sum::<f64>() / window as f64;
            let variance: f64 =
                slice.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / window as f64;
            volatility[i] = variance.sqrt();
        }

        volatility
    }

    /// Compute RSI (Relative Strength Index).
    pub fn compute_rsi(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period + 1 {
            return vec![50.0; prices.len()];
        }

        let mut rsi = vec![50.0; prices.len()];
        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;

        // Initial average
        for i in 1..=period {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                avg_gain += change;
            } else {
                avg_loss -= change;
            }
        }
        avg_gain /= period as f64;
        avg_loss /= period as f64;

        if avg_loss > 0.0 {
            let rs = avg_gain / avg_loss;
            rsi[period] = 100.0 - (100.0 / (1.0 + rs));
        }

        // Subsequent RSI values
        for i in (period + 1)..prices.len() {
            let change = prices[i] - prices[i - 1];
            let (gain, loss) = if change > 0.0 {
                (change, 0.0)
            } else {
                (0.0, -change)
            };

            avg_gain = (avg_gain * (period - 1) as f64 + gain) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + loss) / period as f64;

            if avg_loss > 0.0 {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            } else {
                rsi[i] = 100.0;
            }
        }

        rsi
    }

    /// Normalize features using z-score.
    pub fn normalize_zscore(values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return vec![];
        }

        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 =
            values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt().max(1e-9);

        values.iter().map(|&v| (v - mean) / std).collect()
    }

    /// Prepare features from candles.
    pub fn prepare_features(candles: &[Candle]) -> Vec<Vec<f64>> {
        if candles.is_empty() {
            return vec![];
        }

        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        let returns = Self::compute_returns(&closes);
        let volatility = Self::compute_volatility(&returns, 20);
        let rsi = Self::compute_rsi(&closes, 14);

        // Normalize
        let norm_open = Self::normalize_zscore(&candles.iter().map(|c| c.open).collect::<Vec<_>>());
        let norm_high = Self::normalize_zscore(&candles.iter().map(|c| c.high).collect::<Vec<_>>());
        let norm_low = Self::normalize_zscore(&candles.iter().map(|c| c.low).collect::<Vec<_>>());
        let norm_close = Self::normalize_zscore(&closes);
        let norm_volume = Self::normalize_zscore(&volumes);
        let norm_returns = Self::normalize_zscore(&returns);
        let norm_volatility = Self::normalize_zscore(&volatility);
        let norm_rsi = Self::normalize_zscore(&rsi);

        // Combine into feature vectors
        (0..candles.len())
            .map(|i| {
                vec![
                    norm_open[i],
                    norm_high[i],
                    norm_low[i],
                    norm_close[i],
                    norm_volume[i],
                    norm_returns[i],
                    norm_volatility[i],
                    norm_rsi[i],
                ]
            })
            .collect()
    }

    /// Create sequences for model input.
    pub fn create_sequences(
        features: &[Vec<f64>],
        seq_len: usize,
    ) -> (Vec<Vec<Vec<f64>>>, Vec<usize>) {
        if features.len() < seq_len {
            return (vec![], vec![]);
        }

        let n_sequences = features.len() - seq_len;
        let mut sequences = Vec::with_capacity(n_sequences);
        let mut indices = Vec::with_capacity(n_sequences);

        for i in 0..n_sequences {
            sequences.push(features[i..i + seq_len].to_vec());
            indices.push(i + seq_len);
        }

        (sequences, indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_returns() {
        let prices = vec![100.0, 101.0, 99.0, 102.0];
        let returns = FeatureEngineering::compute_returns(&prices);

        assert_eq!(returns.len(), prices.len());
        assert_eq!(returns[0], 0.0);
        assert!((returns[1] - (101.0_f64 / 100.0).ln()).abs() < 1e-9);
    }

    #[test]
    fn test_compute_rsi() {
        // Steadily increasing prices should give RSI > 50
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let rsi = FeatureEngineering::compute_rsi(&prices, 14);

        assert!(rsi[29] > 50.0);
    }

    #[test]
    fn test_normalize() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = FeatureEngineering::normalize_zscore(&values);

        // Mean should be approximately 0
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(mean.abs() < 1e-9);
    }
}
