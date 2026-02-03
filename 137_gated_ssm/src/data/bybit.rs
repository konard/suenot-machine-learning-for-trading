//! Bybit API client for fetching cryptocurrency kline data.

use serde::Deserialize;

/// A single kline (candlestick) from Bybit.
#[derive(Debug, Clone)]
pub struct BybitKline {
    pub open_time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Deserialize)]
struct BybitResponse {
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<serde_json::Value>>,
}

/// Fetch kline data from Bybit public API.
///
/// # Arguments
/// * `symbol` - Trading pair (e.g., "BTCUSDT", "ETHUSDT")
/// * `interval` - Kline interval ("1", "5", "15", "60", "D", "W")
/// * `limit` - Number of klines (max 1000)
///
/// # Returns
/// Vector of klines sorted by time ascending.
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<BybitKline>, Box<dyn std::error::Error>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let resp: BybitResponse = reqwest::get(&url).await?.json().await?;

    let mut klines: Vec<BybitKline> = resp
        .result
        .list
        .iter()
        .filter_map(|item| {
            Some(BybitKline {
                open_time: item.first()?.as_str()?.parse().ok()?,
                open: item.get(1)?.as_str()?.parse().ok()?,
                high: item.get(2)?.as_str()?.parse().ok()?,
                low: item.get(3)?.as_str()?.parse().ok()?,
                close: item.get(4)?.as_str()?.parse().ok()?,
                volume: item.get(5)?.as_str()?.parse().ok()?,
            })
        })
        .collect();

    // Sort by time ascending
    klines.sort_by_key(|k| k.open_time);

    tracing::info!("Fetched {} klines for {} from Bybit", klines.len(), symbol);
    Ok(klines)
}
