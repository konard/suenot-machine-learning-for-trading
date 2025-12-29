//! Формульные альфа-факторы
//!
//! Реализация некоторых альфа-факторов из статьи WorldQuant "101 Formulaic Alphas"
//! (Kakushadze, 2016)
//!
//! Формулы адаптированы для работы с одним активом (вместо кросс-секционных данных).

use super::utils;

/// Alpha #001
///
/// rank(Ts_ArgMax(SignedPower((returns < 0 ? stddev(returns, 20) : close), 2), 5)) - 0.5
///
/// Упрощённая версия: Ранжирование позиции максимума волатильности
pub fn alpha_001(closes: &[f64], period: usize) -> Vec<f64> {
    let rets = utils::returns(closes);
    let std = utils::rolling(&rets, 20, utils::std_dev);

    let n = closes.len();
    if n < period + 20 {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; n];

    // Для каждого окна находим индекс максимума
    for i in (period + 19)..n {
        let window = &std[(i + 1 - period)..=i];
        let max_idx = window
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // Нормализуем индекс в диапазон [-0.5, 0.5]
        result[i] = (max_idx as f64 / period as f64) - 0.5;
    }

    result
}

/// Alpha #002
///
/// -1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6)
///
/// Отрицательная корреляция между изменением объёма и внутридневной доходностью
pub fn alpha_002(
    opens: &[f64],
    closes: &[f64],
    volumes: &[f64],
    period: usize,
) -> Vec<f64> {
    let n = closes.len();
    if n != opens.len() || n != volumes.len() || n < period + 2 {
        return vec![f64::NAN; n];
    }

    // log(volume)
    let log_vol: Vec<f64> = volumes.iter().map(|v| if *v > 0.0 { v.ln() } else { 0.0 }).collect();

    // delta(log(volume), 2)
    let delta_log_vol = utils::diff(&log_vol, 2);

    // (close - open) / open
    let intraday_ret: Vec<f64> = opens
        .iter()
        .zip(closes.iter())
        .map(|(o, c)| if *o > 0.0 { (c - o) / o } else { 0.0 })
        .collect();

    let mut result = vec![f64::NAN; period + 1];

    for i in (period + 1)..n {
        let start = i + 1 - period;

        let x = &delta_log_vol[start..=i];
        let y = &intraday_ret[start..=i];

        let corr = utils::correlation(x, y);
        result.push(-1.0 * corr);
    }

    result
}

/// Alpha #003
///
/// -1 * correlation(rank(open), rank(volume), 10)
///
/// Отрицательная корреляция между ценой открытия и объёмом
pub fn alpha_003(opens: &[f64], volumes: &[f64], period: usize) -> Vec<f64> {
    let n = opens.len();
    if n != volumes.len() || n < period {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..n {
        let start = i + 1 - period;

        let x = &opens[start..=i];
        let y = &volumes[start..=i];

        let corr = utils::correlation(x, y);
        result.push(-1.0 * corr);
    }

    result
}

/// Alpha #004
///
/// -1 * Ts_Rank(rank(low), 9)
///
/// Ранг минимума за последние 9 периодов
pub fn alpha_004(lows: &[f64], period: usize) -> Vec<f64> {
    let n = lows.len();
    if n < period {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..n {
        let window = &lows[(i + 1 - period)..=i];

        // Находим ранг текущего значения в окне
        let current = lows[i];
        let rank = window.iter().filter(|&&x| x < current).count() as f64 + 1.0;

        // Нормализуем ранг
        result.push(-1.0 * rank / period as f64);
    }

    result
}

/// Alpha #005
///
/// rank(open - (sum(vwap, 10) / 10)) * (-1 * abs(rank(close - vwap)))
///
/// Комбинация отклонения от VWAP
pub fn alpha_005(
    opens: &[f64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[f64],
) -> Vec<f64> {
    let vwap = super::volume::vwap(highs, lows, closes, volumes);
    let vwap_sma = super::trend::sma(&vwap, 10);

    let n = closes.len();
    if n < 10 {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; 9];

    for i in 9..n {
        let open_vwap_diff = opens[i] - vwap_sma[i];
        let close_vwap_diff = (closes[i] - vwap[i]).abs();

        // Упрощённая версия без кросс-секционного ранжирования
        result.push(open_vwap_diff.signum() * (-1.0) * close_vwap_diff);
    }

    result
}

/// Alpha #006
///
/// -1 * correlation(open, volume, 10)
///
/// Отрицательная корреляция между ценой открытия и объёмом
pub fn alpha_006(opens: &[f64], volumes: &[f64], period: usize) -> Vec<f64> {
    alpha_003(opens, volumes, period) // Эквивалентно alpha_003
}

/// Alpha #007
///
/// (adv20 < volume) ? (-1 * ts_rank(abs(delta(close, 7)), 60) * sign(delta(close, 7))) : -1
///
/// Сигнал на основе объёма выше среднего
pub fn alpha_007(closes: &[f64], volumes: &[f64]) -> Vec<f64> {
    let n = closes.len();
    if n < 60 {
        return vec![f64::NAN; n];
    }

    let adv20 = super::trend::sma(volumes, 20);
    let delta7 = utils::diff(closes, 7);

    let mut result = vec![f64::NAN; 59];

    for i in 59..n {
        if volumes[i] > adv20[i] {
            // Объём выше среднего
            let abs_delta = delta7[i].abs();
            let sign = delta7[i].signum();

            // ts_rank
            let window = &delta7[(i + 1 - 60)..=i];
            let rank = window
                .iter()
                .filter(|&&x| x.abs() < abs_delta)
                .count() as f64;
            let ts_rank = rank / 60.0;

            result.push(-1.0 * ts_rank * sign);
        } else {
            result.push(-1.0);
        }
    }

    result
}

/// Alpha #008
///
/// -1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))
///
/// Изменение суммы произведения открытия и доходности
pub fn alpha_008(opens: &[f64], closes: &[f64]) -> Vec<f64> {
    let n = opens.len();
    if n < 15 {
        return vec![f64::NAN; n];
    }

    let rets = utils::returns(closes);
    let open_sum = super::trend::sma(opens, 5);
    let ret_sum = super::trend::sma(&rets, 5);

    // product = open_sum * ret_sum
    let product: Vec<f64> = open_sum
        .iter()
        .zip(ret_sum.iter())
        .map(|(o, r)| o * r)
        .collect();

    // delta(product, 10)
    let delta = utils::diff(&product, 10);

    // Инвертируем
    delta.iter().map(|d| -1.0 * d).collect()
}

/// Alpha #009
///
/// (0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) :
/// ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))
///
/// Условный сигнал на основе направления движения
pub fn alpha_009(closes: &[f64]) -> Vec<f64> {
    let n = closes.len();
    if n < 6 {
        return vec![f64::NAN; n];
    }

    let delta1 = utils::diff(closes, 1);

    let ts_min = utils::rolling(&delta1, 5, utils::min);
    let ts_max = utils::rolling(&delta1, 5, utils::max);

    let mut result = vec![f64::NAN; 5];

    for i in 5..n {
        let d = delta1[i];
        let min_d = ts_min[i];
        let max_d = ts_max[i];

        if min_d > 0.0 {
            result.push(d);
        } else if max_d < 0.0 {
            result.push(d);
        } else {
            result.push(-1.0 * d);
        }
    }

    result
}

/// Alpha #010
///
/// rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) :
///       ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
///
/// Ранжированная версия Alpha #009 с периодом 4
pub fn alpha_010(closes: &[f64]) -> Vec<f64> {
    let n = closes.len();
    if n < 5 {
        return vec![f64::NAN; n];
    }

    let delta1 = utils::diff(closes, 1);

    let ts_min = utils::rolling(&delta1, 4, utils::min);
    let ts_max = utils::rolling(&delta1, 4, utils::max);

    let mut result = vec![f64::NAN; 4];

    for i in 4..n {
        let d = delta1[i];
        let min_d = ts_min[i];
        let max_d = ts_max[i];

        if min_d > 0.0 {
            result.push(d);
        } else if max_d < 0.0 {
            result.push(d);
        } else {
            result.push(-1.0 * d);
        }
    }

    result
}

/// Alpha #012
///
/// sign(delta(volume, 1)) * (-1 * delta(close, 1))
///
/// Обратный сигнал: продавать при росте объёма с ростом цены
pub fn alpha_012(closes: &[f64], volumes: &[f64]) -> Vec<f64> {
    let n = closes.len();
    if n != volumes.len() || n < 2 {
        return vec![];
    }

    let delta_vol = utils::diff(volumes, 1);
    let delta_close = utils::diff(closes, 1);

    delta_vol
        .iter()
        .zip(delta_close.iter())
        .map(|(dv, dc)| dv.signum() * (-1.0 * dc))
        .collect()
}

/// Alpha #013
///
/// -1 * rank(covariance(rank(close), rank(volume), 5))
///
/// Отрицательная ковариация между ценой и объёмом
pub fn alpha_013(closes: &[f64], volumes: &[f64], period: usize) -> Vec<f64> {
    let n = closes.len();
    if n != volumes.len() || n < period {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..n {
        let start = i + 1 - period;

        let x = &closes[start..=i];
        let y = &volumes[start..=i];

        let cov = utils::covariance(x, y);
        result.push(-1.0 * cov);
    }

    result
}

/// Alpha #014
///
/// (-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)
///
/// Комбинация изменения доходности и корреляции
pub fn alpha_014(opens: &[f64], closes: &[f64], volumes: &[f64]) -> Vec<f64> {
    let n = closes.len();
    if n != opens.len() || n != volumes.len() || n < 13 {
        return vec![f64::NAN; n];
    }

    let rets = utils::returns(closes);
    let delta_rets = utils::diff(&rets, 3);

    let mut result = vec![f64::NAN; 12];

    for i in 12..n {
        let start = i + 1 - 10;

        let corr = utils::correlation(&opens[start..=i], &volumes[start..=i]);
        let dr = delta_rets[i];

        result.push((-1.0 * dr.signum()) * corr);
    }

    result
}

/// Alpha #015
///
/// -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)
///
/// Сумма отрицательной корреляции high-volume
pub fn alpha_015(highs: &[f64], volumes: &[f64]) -> Vec<f64> {
    let n = highs.len();
    if n != volumes.len() || n < 5 {
        return vec![f64::NAN; n];
    }

    // Сначала рассчитываем корреляции
    let mut corrs = vec![f64::NAN; 2];

    for i in 2..n {
        let start = i + 1 - 3;
        let corr = utils::correlation(&highs[start..=i], &volumes[start..=i]);
        corrs.push(-1.0 * corr);
    }

    // Затем сумму
    super::trend::sma(&corrs, 3)
        .iter()
        .map(|x| x * 3.0) // SMA * period = sum
        .collect()
}

/// Alpha #016
///
/// -1 * rank(covariance(rank(high), rank(volume), 5))
///
/// Отрицательная ковариация high-volume
pub fn alpha_016(highs: &[f64], volumes: &[f64], period: usize) -> Vec<f64> {
    let n = highs.len();
    if n != volumes.len() || n < period {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..n {
        let start = i + 1 - period;
        let cov = utils::covariance(&highs[start..=i], &volumes[start..=i]);
        result.push(-1.0 * cov);
    }

    result
}

/// Alpha #017
///
/// ((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *
/// rank(ts_rank((volume / adv20), 5))
///
/// Комплексный фактор с ускорением цены и относительным объёмом
pub fn alpha_017(closes: &[f64], volumes: &[f64]) -> Vec<f64> {
    let n = closes.len();
    if n < 25 {
        return vec![f64::NAN; n];
    }

    let adv20 = super::trend::sma(volumes, 20);

    // volume / adv20
    let vol_ratio: Vec<f64> = volumes
        .iter()
        .zip(adv20.iter())
        .map(|(v, a)| if *a > 0.0 { v / a } else { 1.0 })
        .collect();

    // delta(delta(close, 1), 1) = acceleration
    let delta1 = utils::diff(closes, 1);
    let delta2 = utils::diff(&delta1, 1);

    let mut result = vec![f64::NAN; 24];

    for i in 24..n {
        // ts_rank(close, 10)
        let close_window = &closes[(i + 1 - 10)..=i];
        let close_rank = close_window
            .iter()
            .filter(|&&x| x < closes[i])
            .count() as f64
            / 10.0;

        // ts_rank(vol_ratio, 5)
        let vol_window = &vol_ratio[(i + 1 - 5)..=i];
        let vol_rank = vol_window
            .iter()
            .filter(|&&x| x < vol_ratio[i])
            .count() as f64
            / 5.0;

        let accel = delta2[i].signum();

        result.push((-1.0 * close_rank) * accel * vol_rank);
    }

    result
}

/// Momentum Factor - простой моментум-фактор
///
/// returns(n) = (close - close_n) / close_n
pub fn momentum_factor(closes: &[f64], period: usize) -> Vec<f64> {
    super::momentum::roc(closes, period)
        .iter()
        .map(|x| x / 100.0) // Конвертируем проценты в доли
        .collect()
}

/// Mean Reversion Factor - фактор возврата к среднему
///
/// (close - SMA(close, n)) / std(close, n)
pub fn mean_reversion_factor(closes: &[f64], period: usize) -> Vec<f64> {
    let sma = super::trend::sma(closes, period);
    let std = utils::rolling(closes, period, utils::std_dev);

    sma.iter()
        .zip(std.iter())
        .zip(closes.iter())
        .map(|((s, st), c)| {
            if *st > 0.0 && !st.is_nan() {
                -1.0 * (c - s) / st // Отрицательный знак для mean reversion
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Volume Spike Factor - фактор всплеска объёма
///
/// (volume - SMA(volume, n)) / std(volume, n)
pub fn volume_spike_factor(volumes: &[f64], period: usize) -> Vec<f64> {
    let sma = super::trend::sma(volumes, period);
    let std = utils::rolling(volumes, period, utils::std_dev);

    sma.iter()
        .zip(std.iter())
        .zip(volumes.iter())
        .map(|((s, st), v)| {
            if *st > 0.0 && !st.is_nan() {
                (v - s) / st
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Volatility Factor - фактор волатильности
///
/// ATR / close * 100
pub fn volatility_factor(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period: usize,
) -> Vec<f64> {
    let atr = super::volatility::atr(highs, lows, closes, period);

    atr.iter()
        .zip(closes.iter())
        .map(|(a, c)| if *c > 0.0 { (a / c) * 100.0 } else { f64::NAN })
        .collect()
}

/// Price Strength Factor - фактор силы цены
///
/// (close - low) / (high - low)
pub fn price_strength_factor(highs: &[f64], lows: &[f64], closes: &[f64]) -> Vec<f64> {
    highs
        .iter()
        .zip(lows.iter())
        .zip(closes.iter())
        .map(|((h, l), c)| {
            let range = h - l;
            if range > 0.0 {
                (c - l) / range
            } else {
                0.5
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = 100;
        let closes: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0 + (i as f64 * 0.01))
            .collect();
        let opens: Vec<f64> = closes.iter().map(|c| c - 0.5).collect();
        let highs: Vec<f64> = closes.iter().map(|c| c + 1.0).collect();
        let lows: Vec<f64> = closes.iter().map(|c| c - 1.0).collect();
        let volumes: Vec<f64> = (0..n).map(|i| 1000.0 + (i as f64 * 0.2).sin() * 500.0).collect();

        (opens, highs, lows, closes, volumes)
    }

    #[test]
    fn test_alpha_003() {
        let (opens, _, _, _, volumes) = sample_data();
        let result = alpha_003(&opens, &volumes, 10);

        assert_eq!(result.len(), opens.len());

        // Значения должны быть в диапазоне [-1, 1]
        for val in result.iter().skip(9) {
            if !val.is_nan() {
                assert!(*val >= -1.0 && *val <= 1.0);
            }
        }
    }

    #[test]
    fn test_alpha_009() {
        let (_, _, _, closes, _) = sample_data();
        let result = alpha_009(&closes);

        assert_eq!(result.len(), closes.len());
    }

    #[test]
    fn test_alpha_012() {
        let (_, _, _, closes, volumes) = sample_data();
        let result = alpha_012(&closes, &volumes);

        assert_eq!(result.len(), closes.len());
    }

    #[test]
    fn test_momentum_factor() {
        let closes = vec![100.0, 110.0, 105.0, 115.0, 120.0];
        let result = momentum_factor(&closes, 1);

        assert_eq!(result.len(), closes.len());
        assert!(result[0].is_nan());
        assert!((result[1] - 0.1).abs() < 0.0001); // 10% рост
    }

    #[test]
    fn test_mean_reversion_factor() {
        let closes: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 5.0)
            .collect();

        let result = mean_reversion_factor(&closes, 10);

        assert_eq!(result.len(), closes.len());

        // Должны быть отрицательные значения (цена выше среднего -> продавать)
        // и положительные (цена ниже среднего -> покупать)
    }

    #[test]
    fn test_price_strength_factor() {
        let highs = vec![110.0, 120.0, 115.0];
        let lows = vec![90.0, 100.0, 95.0];
        let closes = vec![100.0, 110.0, 105.0]; // Середина диапазона

        let result = price_strength_factor(&highs, &lows, &closes);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 0.5); // (100-90)/(110-90) = 0.5
        assert_eq!(result[1], 0.5); // (110-100)/(120-100) = 0.5
        assert_eq!(result[2], 0.5); // (105-95)/(115-95) = 0.5
    }
}
