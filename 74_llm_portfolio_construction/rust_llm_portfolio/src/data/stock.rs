//! Stock Market Data Client
//!
//! This module provides functionality to fetch stock market data
//! using the Yahoo Finance API.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Stock OHLCV (Open, High, Low, Close, Volume) data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockOHLCV {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
    pub adj_close: f64,
}

/// Stock information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockInfo {
    pub symbol: String,
    pub name: String,
    pub sector: Option<String>,
    pub industry: Option<String>,
    pub market_cap: Option<f64>,
    pub pe_ratio: Option<f64>,
    pub dividend_yield: Option<f64>,
}

/// Yahoo Finance API response structures
#[derive(Debug, Deserialize)]
struct YahooResponse {
    chart: ChartResult,
}

#[derive(Debug, Deserialize)]
struct ChartResult {
    result: Option<Vec<ChartData>>,
    error: Option<YahooError>,
}

#[derive(Debug, Deserialize)]
struct ChartData {
    meta: ChartMeta,
    timestamp: Option<Vec<i64>>,
    indicators: Indicators,
}

#[derive(Debug, Deserialize)]
struct ChartMeta {
    symbol: String,
    #[serde(rename = "regularMarketPrice")]
    regular_market_price: Option<f64>,
    #[serde(rename = "previousClose")]
    previous_close: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct Indicators {
    quote: Vec<Quote>,
    adjclose: Option<Vec<AdjClose>>,
}

#[derive(Debug, Deserialize)]
struct Quote {
    open: Vec<Option<f64>>,
    high: Vec<Option<f64>>,
    low: Vec<Option<f64>>,
    close: Vec<Option<f64>>,
    volume: Vec<Option<u64>>,
}

#[derive(Debug, Deserialize)]
struct AdjClose {
    adjclose: Vec<Option<f64>>,
}

#[derive(Debug, Deserialize)]
struct YahooError {
    code: String,
    description: String,
}

/// Stock data client using Yahoo Finance
pub struct StockClient {
    client: Client,
    base_url: String,
}

impl StockClient {
    /// Create a new stock client
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://query1.finance.yahoo.com/v8/finance/chart".to_string(),
        }
    }

    /// Fetch historical OHLCV data for a stock
    ///
    /// # Arguments
    /// * `symbol` - Stock ticker symbol (e.g., "AAPL", "MSFT")
    /// * `period` - Time period: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
    /// * `interval` - Data interval: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
    pub async fn get_ohlcv(
        &self,
        symbol: &str,
        period: &str,
        interval: &str,
    ) -> Result<Vec<StockOHLCV>, Box<dyn std::error::Error>> {
        let url = format!(
            "{}/{}?range={}&interval={}",
            self.base_url, symbol, period, interval
        );

        let response: YahooResponse = self
            .client
            .get(&url)
            .header("User-Agent", "Mozilla/5.0")
            .send()
            .await?
            .json()
            .await?;

        if let Some(error) = response.chart.error {
            return Err(format!("Yahoo Finance error: {} - {}", error.code, error.description).into());
        }

        let result = response
            .chart
            .result
            .ok_or("No data returned")?
            .into_iter()
            .next()
            .ok_or("Empty result")?;

        let timestamps = result.timestamp.ok_or("No timestamps")?;
        let quote = result.indicators.quote.into_iter().next().ok_or("No quote data")?;
        let adjclose = result.indicators.adjclose.and_then(|a| a.into_iter().next());

        let mut ohlcv_data = Vec::new();
        for i in 0..timestamps.len() {
            if let (Some(open), Some(high), Some(low), Some(close), Some(volume)) = (
                quote.open.get(i).and_then(|v| *v),
                quote.high.get(i).and_then(|v| *v),
                quote.low.get(i).and_then(|v| *v),
                quote.close.get(i).and_then(|v| *v),
                quote.volume.get(i).and_then(|v| *v),
            ) {
                let adj_close = adjclose
                    .as_ref()
                    .and_then(|a| a.adjclose.get(i).and_then(|v| *v))
                    .unwrap_or(close);

                ohlcv_data.push(StockOHLCV {
                    timestamp: timestamps[i],
                    open,
                    high,
                    low,
                    close,
                    volume,
                    adj_close,
                });
            }
        }

        Ok(ohlcv_data)
    }

    /// Fetch current price for a stock
    pub async fn get_current_price(&self, symbol: &str) -> Result<f64, Box<dyn std::error::Error>> {
        let url = format!("{}/{}?range=1d&interval=1d", self.base_url, symbol);

        let response: YahooResponse = self
            .client
            .get(&url)
            .header("User-Agent", "Mozilla/5.0")
            .send()
            .await?
            .json()
            .await?;

        let result = response
            .chart
            .result
            .ok_or("No data returned")?
            .into_iter()
            .next()
            .ok_or("Empty result")?;

        result
            .meta
            .regular_market_price
            .ok_or_else(|| "No price data".into())
    }

    /// Fetch data for multiple stocks
    pub async fn get_multiple_ohlcv(
        &self,
        symbols: &[&str],
        period: &str,
        interval: &str,
    ) -> HashMap<String, Vec<StockOHLCV>> {
        let mut results = HashMap::new();

        for symbol in symbols {
            match self.get_ohlcv(symbol, period, interval).await {
                Ok(data) => {
                    results.insert(symbol.to_string(), data);
                }
                Err(e) => {
                    eprintln!("Error fetching {}: {}", symbol, e);
                }
            }
        }

        results
    }
}

impl Default for StockClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Stock portfolio data fetcher
pub struct StockPortfolioDataFetcher {
    client: StockClient,
}

impl StockPortfolioDataFetcher {
    /// Create a new portfolio data fetcher
    pub fn new(client: StockClient) -> Self {
        Self { client }
    }

    /// Fetch portfolio data for multiple stocks
    pub async fn fetch_portfolio_data(
        &self,
        symbols: &[&str],
        period: &str,
    ) -> Result<PortfolioData, Box<dyn std::error::Error>> {
        let ohlcv_data = self.client.get_multiple_ohlcv(symbols, period, "1d").await;

        let mut prices = HashMap::new();
        let mut returns = HashMap::new();

        for (symbol, data) in &ohlcv_data {
            if data.len() < 2 {
                continue;
            }

            let price_series: Vec<f64> = data.iter().map(|d| d.adj_close).collect();
            let return_series: Vec<f64> = price_series
                .windows(2)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect();

            prices.insert(symbol.clone(), price_series);
            returns.insert(symbol.clone(), return_series);
        }

        Ok(PortfolioData {
            ohlcv: ohlcv_data,
            prices,
            returns,
        })
    }
}

/// Portfolio data container
#[derive(Debug)]
pub struct PortfolioData {
    pub ohlcv: HashMap<String, Vec<StockOHLCV>>,
    pub prices: HashMap<String, Vec<f64>>,
    pub returns: HashMap<String, Vec<f64>>,
}

impl PortfolioData {
    /// Calculate correlation matrix between assets
    pub fn correlation_matrix(&self) -> HashMap<(String, String), f64> {
        let mut correlations = HashMap::new();
        let symbols: Vec<_> = self.returns.keys().collect();

        for (i, sym1) in symbols.iter().enumerate() {
            for sym2 in symbols.iter().skip(i) {
                if let (Some(ret1), Some(ret2)) = (self.returns.get(*sym1), self.returns.get(*sym2))
                {
                    let corr = calculate_correlation(ret1, ret2);
                    correlations.insert(((*sym1).clone(), (*sym2).clone()), corr);
                    if sym1 != sym2 {
                        correlations.insert(((*sym2).clone(), (*sym1).clone()), corr);
                    }
                }
            }
        }

        correlations
    }

    /// Calculate volatility for each asset
    pub fn volatilities(&self) -> HashMap<String, f64> {
        self.returns
            .iter()
            .map(|(symbol, returns)| {
                let vol = calculate_std(returns) * (252.0_f64).sqrt(); // Annualized
                (symbol.clone(), vol)
            })
            .collect()
    }
}

fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let mean_x: f64 = x.iter().take(n).sum::<f64>() / n as f64;
    let mean_y: f64 = y.iter().take(n).sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

fn calculate_std(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = calculate_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.001);

        let y_neg = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_neg = calculate_correlation(&x, &y_neg);
        assert!((corr_neg - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_std() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = calculate_std(&values);
        assert!((std - 2.0).abs() < 0.001);
    }
}
