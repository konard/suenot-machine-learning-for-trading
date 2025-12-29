//! Индикаторы объёма
//!
//! - OBV (On-Balance Volume)
//! - VWAP (Volume Weighted Average Price)
//! - MFI (Money Flow Index)
//! - A/D Line (Accumulation/Distribution)
//! - CMF (Chaikin Money Flow)

use super::trend;

/// OBV (On-Balance Volume) - Балансовый объём
///
/// Если Close > Close_prev: OBV = OBV_prev + Volume
/// Если Close < Close_prev: OBV = OBV_prev - Volume
/// Если Close = Close_prev: OBV = OBV_prev
pub fn obv(closes: &[f64], volumes: &[f64]) -> Vec<f64> {
    if closes.len() != volumes.len() || closes.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(closes.len());
    result.push(volumes[0]); // Первое значение = первый объём

    for i in 1..closes.len() {
        let prev_obv = result[i - 1];
        let obv = if closes[i] > closes[i - 1] {
            prev_obv + volumes[i]
        } else if closes[i] < closes[i - 1] {
            prev_obv - volumes[i]
        } else {
            prev_obv
        };
        result.push(obv);
    }

    result
}

/// VWAP (Volume Weighted Average Price) - Средневзвешенная цена по объёму
///
/// VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
///
/// Обычно рассчитывается внутридневно
pub fn vwap(highs: &[f64], lows: &[f64], closes: &[f64], volumes: &[f64]) -> Vec<f64> {
    let n = closes.len();
    if n != highs.len() || n != lows.len() || n != volumes.len() {
        return vec![];
    }

    let mut cum_tp_vol = 0.0;
    let mut cum_vol = 0.0;
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let tp = (highs[i] + lows[i] + closes[i]) / 3.0;
        cum_tp_vol += tp * volumes[i];
        cum_vol += volumes[i];

        if cum_vol > 0.0 {
            result.push(cum_tp_vol / cum_vol);
        } else {
            result.push(closes[i]);
        }
    }

    result
}

/// Скользящий VWAP с периодом
pub fn vwap_rolling(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[f64],
    period: usize,
) -> Vec<f64> {
    let n = closes.len();
    if n != highs.len() || n != lows.len() || n != volumes.len() || n < period {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..n {
        let start = i + 1 - period;

        let mut tp_vol_sum = 0.0;
        let mut vol_sum = 0.0;

        for j in start..=i {
            let tp = (highs[j] + lows[j] + closes[j]) / 3.0;
            tp_vol_sum += tp * volumes[j];
            vol_sum += volumes[j];
        }

        if vol_sum > 0.0 {
            result.push(tp_vol_sum / vol_sum);
        } else {
            result.push(closes[i]);
        }
    }

    result
}

/// Volume SMA - Простая скользящая средняя объёма
pub fn volume_sma(volumes: &[f64], period: usize) -> Vec<f64> {
    trend::sma(volumes, period)
}

/// Volume EMA - Экспоненциальная скользящая средняя объёма
pub fn volume_ema(volumes: &[f64], period: usize) -> Vec<f64> {
    trend::ema(volumes, period)
}

/// MFI (Money Flow Index) - Индекс денежного потока
///
/// Похож на RSI, но учитывает объём
/// MFI = 100 - (100 / (1 + Money Flow Ratio))
pub fn mfi(highs: &[f64], lows: &[f64], closes: &[f64], volumes: &[f64], period: usize) -> Vec<f64> {
    let n = closes.len();
    if n != highs.len() || n != lows.len() || n != volumes.len() || n < period + 1 {
        return vec![f64::NAN; n];
    }

    // Typical Price
    let tp: Vec<f64> = highs
        .iter()
        .zip(lows.iter())
        .zip(closes.iter())
        .map(|((h, l), c)| (h + l + c) / 3.0)
        .collect();

    // Raw Money Flow = TP * Volume
    let raw_mf: Vec<f64> = tp.iter().zip(volumes.iter()).map(|(t, v)| t * v).collect();

    let mut result = vec![f64::NAN; period];

    for i in period..n {
        let mut positive_mf = 0.0;
        let mut negative_mf = 0.0;

        for j in (i + 1 - period)..=i {
            if j > 0 {
                if tp[j] > tp[j - 1] {
                    positive_mf += raw_mf[j];
                } else if tp[j] < tp[j - 1] {
                    negative_mf += raw_mf[j];
                }
            }
        }

        let mfi = if negative_mf == 0.0 {
            100.0
        } else {
            let money_ratio = positive_mf / negative_mf;
            100.0 - (100.0 / (1.0 + money_ratio))
        };

        result.push(mfi);
    }

    result
}

/// A/D Line (Accumulation/Distribution) - Линия накопления/распределения
///
/// A/D = ((Close - Low) - (High - Close)) / (High - Low) * Volume
pub fn ad_line(highs: &[f64], lows: &[f64], closes: &[f64], volumes: &[f64]) -> Vec<f64> {
    let n = closes.len();
    if n != highs.len() || n != lows.len() || n != volumes.len() {
        return vec![];
    }

    let mut result = Vec::with_capacity(n);
    let mut cum_ad = 0.0;

    for i in 0..n {
        let range = highs[i] - lows[i];
        let clv = if range > 0.0 {
            ((closes[i] - lows[i]) - (highs[i] - closes[i])) / range
        } else {
            0.0
        };

        cum_ad += clv * volumes[i];
        result.push(cum_ad);
    }

    result
}

/// CMF (Chaikin Money Flow) - Денежный поток Чайкина
///
/// CMF = Sum(A/D Volume, period) / Sum(Volume, period)
pub fn cmf(highs: &[f64], lows: &[f64], closes: &[f64], volumes: &[f64], period: usize) -> Vec<f64> {
    let n = closes.len();
    if n != highs.len() || n != lows.len() || n != volumes.len() || n < period {
        return vec![f64::NAN; n];
    }

    // Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    let mfm: Vec<f64> = (0..n)
        .map(|i| {
            let range = highs[i] - lows[i];
            if range > 0.0 {
                ((closes[i] - lows[i]) - (highs[i] - closes[i])) / range
            } else {
                0.0
            }
        })
        .collect();

    // Money Flow Volume = MFM * Volume
    let mfv: Vec<f64> = mfm.iter().zip(volumes.iter()).map(|(m, v)| m * v).collect();

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..n {
        let start = i + 1 - period;

        let mfv_sum: f64 = mfv[start..=i].iter().sum();
        let vol_sum: f64 = volumes[start..=i].iter().sum();

        if vol_sum > 0.0 {
            result.push(mfv_sum / vol_sum);
        } else {
            result.push(0.0);
        }
    }

    result
}

/// Chaikin Oscillator
///
/// CO = EMA(A/D, 3) - EMA(A/D, 10)
pub fn chaikin_oscillator(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[f64],
) -> Vec<f64> {
    let ad = ad_line(highs, lows, closes, volumes);
    let ema_fast = trend::ema(&ad, 3);
    let ema_slow = trend::ema(&ad, 10);

    ema_fast
        .iter()
        .zip(ema_slow.iter())
        .map(|(f, s)| f - s)
        .collect()
}

/// Force Index
///
/// FI = (Close - Close_prev) * Volume
pub fn force_index(closes: &[f64], volumes: &[f64], period: usize) -> Vec<f64> {
    if closes.len() != volumes.len() || closes.len() < 2 {
        return vec![];
    }

    let mut raw_fi = vec![0.0];

    for i in 1..closes.len() {
        raw_fi.push((closes[i] - closes[i - 1]) * volumes[i]);
    }

    trend::ema(&raw_fi, period)
}

/// Ease of Movement (EMV)
///
/// EMV = ((High + Low) / 2 - (High_prev + Low_prev) / 2) / (Volume / (High - Low))
pub fn emv(highs: &[f64], lows: &[f64], volumes: &[f64], period: usize) -> Vec<f64> {
    let n = highs.len();
    if n != lows.len() || n != volumes.len() || n < 2 {
        return vec![];
    }

    let mut raw_emv = vec![0.0];

    for i in 1..n {
        let distance = ((highs[i] + lows[i]) / 2.0) - ((highs[i - 1] + lows[i - 1]) / 2.0);
        let box_ratio = if highs[i] - lows[i] > 0.0 {
            volumes[i] / (highs[i] - lows[i])
        } else {
            0.0
        };

        let emv = if box_ratio > 0.0 {
            distance / box_ratio
        } else {
            0.0
        };
        raw_emv.push(emv);
    }

    trend::sma(&raw_emv, period)
}

/// Volume Price Trend (VPT)
///
/// VPT = VPT_prev + Volume * ((Close - Close_prev) / Close_prev)
pub fn vpt(closes: &[f64], volumes: &[f64]) -> Vec<f64> {
    if closes.len() != volumes.len() || closes.len() < 2 {
        return vec![];
    }

    let mut result = vec![0.0];

    for i in 1..closes.len() {
        let prev_close = closes[i - 1];
        let pct_change = if prev_close > 0.0 {
            (closes[i] - prev_close) / prev_close
        } else {
            0.0
        };

        result.push(result[i - 1] + volumes[i] * pct_change);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obv() {
        let closes = vec![10.0, 11.0, 10.5, 11.5, 11.0];
        let volumes = vec![100.0, 150.0, 120.0, 200.0, 80.0];

        let result = obv(&closes, &volumes);

        assert_eq!(result[0], 100.0);
        assert_eq!(result[1], 250.0); // up, +150
        assert_eq!(result[2], 130.0); // down, -120
        assert_eq!(result[3], 330.0); // up, +200
        assert_eq!(result[4], 250.0); // down, -80
    }

    #[test]
    fn test_vwap() {
        let highs = vec![11.0, 12.0, 11.5];
        let lows = vec![9.0, 10.0, 9.5];
        let closes = vec![10.0, 11.0, 10.5];
        let volumes = vec![100.0, 100.0, 100.0];

        let result = vwap(&highs, &lows, &closes, &volumes);

        assert_eq!(result.len(), 3);
        // Первый VWAP = TP1 = (11+9+10)/3 = 10
        assert_eq!(result[0], 10.0);
    }

    #[test]
    fn test_mfi_range() {
        let highs = vec![
            11.0, 12.0, 11.5, 12.5, 13.0, 12.5, 11.5, 12.0, 12.5, 13.0, 12.0, 11.0, 10.5, 11.0,
            12.0,
        ];
        let lows = vec![
            9.0, 10.0, 9.5, 10.5, 11.0, 10.5, 9.5, 10.0, 10.5, 11.0, 10.0, 9.0, 8.5, 9.0, 10.0,
        ];
        let closes = vec![
            10.0, 11.0, 10.5, 12.0, 12.5, 11.0, 10.0, 11.5, 12.0, 12.5, 11.0, 9.5, 9.0, 10.0, 11.5,
        ];
        let volumes = vec![
            100.0, 150.0, 120.0, 200.0, 180.0, 90.0, 110.0, 160.0, 140.0, 200.0, 100.0, 80.0, 70.0,
            130.0, 170.0,
        ];

        let result = mfi(&highs, &lows, &closes, &volumes, 14);

        for val in result.iter() {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 100.0, "MFI out of range: {}", val);
            }
        }
    }

    #[test]
    fn test_ad_line() {
        let highs = vec![12.0, 13.0, 12.5];
        let lows = vec![10.0, 11.0, 10.5];
        let closes = vec![11.5, 12.5, 11.0]; // Close near high, near high, near low
        let volumes = vec![100.0, 100.0, 100.0];

        let result = ad_line(&highs, &lows, &closes, &volumes);

        assert_eq!(result.len(), 3);
        // Первая свеча: CLV = ((11.5-10)-(12-11.5))/(12-10) = (1.5-0.5)/2 = 0.5
        // A/D = 0.5 * 100 = 50
        assert!((result[0] - 50.0).abs() < 0.01);
    }
}
