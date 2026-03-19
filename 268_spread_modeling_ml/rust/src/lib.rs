use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

// ============================================================================
// Spread Calculation Functions
// ============================================================================

/// Compute the absolute (quoted) bid-ask spread.
pub fn absolute_spread(best_bid: f64, best_ask: f64) -> f64 {
    best_ask - best_bid
}

/// Compute the relative (proportional) bid-ask spread normalized by midpoint.
pub fn relative_spread(best_bid: f64, best_ask: f64) -> f64 {
    let mid = (best_bid + best_ask) / 2.0;
    if mid == 0.0 {
        return 0.0;
    }
    (best_ask - best_bid) / mid
}

/// Compute the effective spread from a trade price, midpoint, and direction.
/// direction: +1.0 for buy, -1.0 for sell.
pub fn effective_spread(trade_price: f64, midpoint: f64, direction: f64) -> f64 {
    2.0 * direction * (trade_price - midpoint)
}

/// Compute the midpoint price from best bid and ask.
pub fn midpoint(best_bid: f64, best_ask: f64) -> f64 {
    (best_bid + best_ask) / 2.0
}

// ============================================================================
// Roll Spread Estimator
// ============================================================================

/// Estimate the effective spread using the Roll (1984) model.
///
/// The Roll estimator uses the first-order autocovariance of price changes:
///   Cov(dP_t, dP_{t-1}) = -c^2
/// where c is the half-spread. The estimated spread is 2*sqrt(-Cov).
///
/// If the autocovariance is positive (trending market), returns 0.0.
pub fn roll_spread_estimator(prices: &[f64]) -> f64 {
    if prices.len() < 3 {
        return 0.0;
    }

    let n = prices.len() - 1;
    let mut changes = Vec::with_capacity(n);
    for i in 1..prices.len() {
        changes.push(prices[i] - prices[i - 1]);
    }

    // Compute mean of changes
    let mean: f64 = changes.iter().sum::<f64>() / changes.len() as f64;

    // Compute first-order autocovariance
    let mut cov_sum = 0.0;
    let mut count = 0;
    for i in 1..changes.len() {
        cov_sum += (changes[i] - mean) * (changes[i - 1] - mean);
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }

    let autocovariance = cov_sum / count as f64;

    // Roll estimator: spread = 2 * sqrt(-autocovariance)
    if autocovariance >= 0.0 {
        0.0 // Positive autocovariance indicates trending; Roll model undefined
    } else {
        2.0 * (-autocovariance).sqrt()
    }
}

// ============================================================================
// Linear Regression for Spread Prediction
// ============================================================================

/// Simple linear regression model using the normal equation.
#[derive(Debug, Clone)]
pub struct LinearRegression {
    /// Coefficients including bias term (first element is intercept).
    pub coefficients: Option<Vec<f64>>,
}

impl LinearRegression {
    pub fn new() -> Self {
        Self { coefficients: None }
    }

    /// Fit the model using the normal equation: beta = (X^T X)^{-1} X^T y.
    /// features: matrix of shape (n_samples, n_features)
    /// targets: vector of shape (n_samples,)
    pub fn fit(&mut self, features: &[Vec<f64>], targets: &[f64]) -> Result<()> {
        let n = features.len();
        if n == 0 || targets.len() != n {
            return Err(anyhow!("Invalid input dimensions"));
        }
        let p = features[0].len();

        // Build X matrix with bias column (n x (p+1))
        let mut x = Array2::<f64>::zeros((n, p + 1));
        for i in 0..n {
            x[[i, 0]] = 1.0; // bias
            for j in 0..p {
                x[[i, j + 1]] = features[i][j];
            }
        }

        let y = Array1::from_vec(targets.to_vec());

        // X^T X
        let xt = x.t();
        let xtx = xt.dot(&x);

        // X^T y
        let xty = xt.dot(&y);

        // Solve using simple approach: invert (X^T X) via LU-like manual method
        // For small matrices, we use a direct approach
        let dim = p + 1;
        let coeffs = solve_linear_system(&xtx, &xty, dim)?;

        self.coefficients = Some(coeffs);
        Ok(())
    }

    /// Predict spread values for new features.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        let coeffs = self
            .coefficients
            .as_ref()
            .ok_or_else(|| anyhow!("Model not fitted"))?;

        let mut predictions = Vec::with_capacity(features.len());
        for row in features {
            let mut pred = coeffs[0]; // intercept
            for (j, val) in row.iter().enumerate() {
                pred += coeffs[j + 1] * val;
            }
            predictions.push(pred);
        }
        Ok(predictions)
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>, dim: usize) -> Result<Vec<f64>> {
    // Build augmented matrix
    let mut aug = vec![vec![0.0f64; dim + 1]; dim];
    for i in 0..dim {
        for j in 0..dim {
            aug[i][j] = a[[i, j]];
        }
        aug[i][dim] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..dim {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..dim {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        if aug[col][col].abs() < 1e-12 {
            return Err(anyhow!("Singular matrix"));
        }

        for row in (col + 1)..dim {
            let factor = aug[row][col] / aug[col][col];
            for j in col..=dim {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; dim];
    for i in (0..dim).rev() {
        x[i] = aug[i][dim];
        for j in (i + 1)..dim {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Ok(x)
}

// ============================================================================
// Spread Regime Classifier
// ============================================================================

/// Spread regime labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpreadRegime {
    Tight,
    Wide,
}

/// Classifier that labels spreads as Tight or Wide based on a threshold.
#[derive(Debug, Clone)]
pub struct SpreadRegimeClassifier {
    pub threshold: f64,
}

impl SpreadRegimeClassifier {
    /// Create a new classifier with a given threshold.
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Create a classifier using the median of historical spreads as threshold.
    pub fn from_median(spreads: &[f64]) -> Self {
        let mut sorted = spreads.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        Self { threshold: median }
    }

    /// Classify a single spread value.
    pub fn classify(&self, spread: f64) -> SpreadRegime {
        if spread <= self.threshold {
            SpreadRegime::Tight
        } else {
            SpreadRegime::Wide
        }
    }

    /// Classify a series of spreads.
    pub fn classify_series(&self, spreads: &[f64]) -> Vec<SpreadRegime> {
        spreads.iter().map(|&s| self.classify(s)).collect()
    }

    /// Compute regime transition probabilities from a series of spreads.
    /// Returns (P(Tight->Wide), P(Wide->Tight)).
    pub fn transition_probabilities(&self, spreads: &[f64]) -> (f64, f64) {
        let regimes = self.classify_series(spreads);
        let mut tight_to_wide = 0u64;
        let mut tight_count = 0u64;
        let mut wide_to_tight = 0u64;
        let mut wide_count = 0u64;

        for i in 0..regimes.len() - 1 {
            match regimes[i] {
                SpreadRegime::Tight => {
                    tight_count += 1;
                    if regimes[i + 1] == SpreadRegime::Wide {
                        tight_to_wide += 1;
                    }
                }
                SpreadRegime::Wide => {
                    wide_count += 1;
                    if regimes[i + 1] == SpreadRegime::Tight {
                        wide_to_tight += 1;
                    }
                }
            }
        }

        let p_tw = if tight_count > 0 {
            tight_to_wide as f64 / tight_count as f64
        } else {
            0.0
        };
        let p_wt = if wide_count > 0 {
            wide_to_tight as f64 / wide_count as f64
        } else {
            0.0
        };

        (p_tw, p_wt)
    }
}

// ============================================================================
// Transaction Cost Estimator
// ============================================================================

/// Estimates total transaction costs including spread and market impact.
#[derive(Debug, Clone)]
pub struct TransactionCostEstimator {
    /// Market impact coefficient (Kyle's lambda).
    pub impact_coefficient: f64,
}

impl TransactionCostEstimator {
    pub fn new(impact_coefficient: f64) -> Self {
        Self { impact_coefficient }
    }

    /// Estimate the half-spread cost for immediate execution.
    pub fn half_spread_cost(&self, spread: f64) -> f64 {
        spread / 2.0
    }

    /// Estimate market impact using square-root model:
    ///   impact = k * volatility * sqrt(order_size / avg_volume)
    pub fn market_impact(
        &self,
        volatility: f64,
        order_size: f64,
        avg_volume: f64,
    ) -> f64 {
        if avg_volume <= 0.0 {
            return 0.0;
        }
        self.impact_coefficient * volatility * (order_size / avg_volume).sqrt()
    }

    /// Estimate total transaction cost (one-way) in price units.
    pub fn total_cost(
        &self,
        spread: f64,
        volatility: f64,
        order_size: f64,
        avg_volume: f64,
    ) -> f64 {
        self.half_spread_cost(spread) + self.market_impact(volatility, order_size, avg_volume)
    }

    /// Estimate total transaction cost as a fraction of the midpoint price.
    pub fn total_cost_bps(
        &self,
        spread: f64,
        midpoint: f64,
        volatility: f64,
        order_size: f64,
        avg_volume: f64,
    ) -> f64 {
        if midpoint <= 0.0 {
            return 0.0;
        }
        let cost = self.total_cost(spread, volatility, order_size, avg_volume);
        (cost / midpoint) * 10_000.0 // Convert to basis points
    }
}

// ============================================================================
// Bybit Orderbook Integration
// ============================================================================

/// A single level in the order book (price, quantity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Parsed order book data.
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub symbol: String,
}

impl OrderBook {
    /// Get the best bid price (highest bid).
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get the best ask price (lowest ask).
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Compute the quoted spread.
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(absolute_spread(bid, ask)),
            _ => None,
        }
    }

    /// Compute the relative spread.
    pub fn relative_spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(relative_spread(bid, ask)),
            _ => None,
        }
    }

    /// Compute the midpoint.
    pub fn midpoint(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(midpoint(bid, ask)),
            _ => None,
        }
    }

    /// Compute total bid depth (sum of bid quantities).
    pub fn bid_depth(&self) -> f64 {
        self.bids.iter().map(|l| l.quantity).sum()
    }

    /// Compute total ask depth (sum of ask quantities).
    pub fn ask_depth(&self) -> f64 {
        self.asks.iter().map(|l| l.quantity).sum()
    }

    /// Compute order imbalance: (bid_depth - ask_depth) / (bid_depth + ask_depth).
    pub fn order_imbalance(&self) -> f64 {
        let bd = self.bid_depth();
        let ad = self.ask_depth();
        let total = bd + ad;
        if total == 0.0 {
            return 0.0;
        }
        (bd - ad) / total
    }
}

/// Bybit API response structures.
#[derive(Debug, Deserialize)]
struct BybitResponse {
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    #[serde(rename = "b")]
    bids: Vec<Vec<String>>,
    #[serde(rename = "a")]
    asks: Vec<Vec<String>>,
    #[serde(rename = "s")]
    symbol: String,
}

/// Fetch the order book from Bybit's v5 API.
///
/// Uses the spot market category and returns up to `limit` levels on each side.
pub fn fetch_bybit_orderbook(symbol: &str, limit: u32) -> Result<OrderBook> {
    let url = format!(
        "https://api.bybit.com/v5/market/orderbook?category=spot&symbol={}&limit={}",
        symbol, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    let bids = resp
        .result
        .bids
        .iter()
        .map(|level| {
            Ok(OrderBookLevel {
                price: level[0].parse::<f64>()?,
                quantity: level[1].parse::<f64>()?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let asks = resp
        .result
        .asks
        .iter()
        .map(|level| {
            Ok(OrderBookLevel {
                price: level[0].parse::<f64>()?,
                quantity: level[1].parse::<f64>()?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(OrderBook {
        bids,
        asks,
        symbol: resp.result.symbol,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_absolute_spread() {
        let spread = absolute_spread(100.0, 100.5);
        assert!((spread - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_relative_spread() {
        let spread = relative_spread(99.0, 101.0);
        // Midpoint = 100, spread = 2, relative = 2/100 = 0.02
        assert!((spread - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_effective_spread() {
        // Buy at 100.3, midpoint at 100.0 => effective = 2 * 1.0 * 0.3 = 0.6
        let eff = effective_spread(100.3, 100.0, 1.0);
        assert!((eff - 0.6).abs() < 1e-10);

        // Sell at 99.8, midpoint at 100.0 => effective = 2 * (-1.0) * (-0.2) = 0.4
        let eff_sell = effective_spread(99.8, 100.0, -1.0);
        assert!((eff_sell - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_roll_spread_estimator() {
        // Create a price series with bid-ask bounce (alternating up/down)
        // This should produce negative autocovariance
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let roll = roll_spread_estimator(&prices);
        assert!(roll > 0.0, "Roll spread should be positive for bouncing prices");

        // Perfectly linear trending prices: constant changes => zero autocovariance
        // after mean subtraction, so Roll should be 0 (or very close)
        let trending: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.1).collect();
        let roll_trend = roll_spread_estimator(&trending);
        assert!(
            roll_trend < 1e-6,
            "Roll spread should be ~0 for linear trending prices, got {}",
            roll_trend
        );
    }

    #[test]
    fn test_linear_regression() {
        // Simple case: y = 2 + 3*x
        let features: Vec<Vec<f64>> = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ];
        let targets: Vec<f64> = vec![5.0, 8.0, 11.0, 14.0, 17.0];

        let mut model = LinearRegression::new();
        model.fit(&features, &targets).unwrap();

        let coeffs = model.coefficients.as_ref().unwrap();
        assert!(
            (coeffs[0] - 2.0).abs() < 1e-6,
            "Intercept should be ~2.0, got {}",
            coeffs[0]
        );
        assert!(
            (coeffs[1] - 3.0).abs() < 1e-6,
            "Slope should be ~3.0, got {}",
            coeffs[1]
        );

        let preds = model.predict(&vec![vec![6.0]]).unwrap();
        assert!(
            (preds[0] - 20.0).abs() < 1e-6,
            "Prediction for x=6 should be ~20.0"
        );
    }

    #[test]
    fn test_spread_regime_classifier() {
        let spreads = vec![0.1, 0.2, 0.5, 0.3, 0.8, 0.15, 0.9, 0.25];
        let classifier = SpreadRegimeClassifier::from_median(&spreads);

        // Median of sorted [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.8, 0.9] = (0.25+0.3)/2 = 0.275
        assert!((classifier.threshold - 0.275).abs() < 1e-10);

        assert_eq!(classifier.classify(0.1), SpreadRegime::Tight);
        assert_eq!(classifier.classify(0.5), SpreadRegime::Wide);
        assert_eq!(classifier.classify(0.275), SpreadRegime::Tight); // <= threshold
    }

    #[test]
    fn test_transition_probabilities() {
        // Alternating tight/wide: threshold = 0.5
        let classifier = SpreadRegimeClassifier::new(0.5);
        let spreads = vec![0.1, 0.8, 0.1, 0.8, 0.1, 0.8];
        let (p_tw, p_wt) = classifier.transition_probabilities(&spreads);

        // Every Tight is followed by Wide, and every Wide by Tight
        assert!((p_tw - 1.0).abs() < 1e-10);
        assert!((p_wt - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_transaction_cost_estimator() {
        let estimator = TransactionCostEstimator::new(0.1);

        let half_spread = estimator.half_spread_cost(2.0);
        assert!((half_spread - 1.0).abs() < 1e-10);

        // impact = 0.1 * 0.02 * sqrt(100/1000) = 0.1 * 0.02 * 0.3162 = 0.000632
        let impact = estimator.market_impact(0.02, 100.0, 1000.0);
        let expected = 0.1 * 0.02 * (100.0f64 / 1000.0).sqrt();
        assert!(
            (impact - expected).abs() < 1e-10,
            "Impact mismatch: {} vs {}",
            impact,
            expected
        );

        let total = estimator.total_cost(2.0, 0.02, 100.0, 1000.0);
        assert!((total - (1.0 + expected)).abs() < 1e-10);
    }

    #[test]
    fn test_orderbook_methods() {
        let ob = OrderBook {
            bids: vec![
                OrderBookLevel { price: 100.0, quantity: 5.0 },
                OrderBookLevel { price: 99.5, quantity: 10.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 100.5, quantity: 3.0 },
                OrderBookLevel { price: 101.0, quantity: 7.0 },
            ],
            symbol: "BTCUSDT".to_string(),
        };

        assert_eq!(ob.best_bid(), Some(100.0));
        assert_eq!(ob.best_ask(), Some(100.5));
        assert!((ob.spread().unwrap() - 0.5).abs() < 1e-10);
        assert!((ob.midpoint().unwrap() - 100.25).abs() < 1e-10);
        assert!((ob.bid_depth() - 15.0).abs() < 1e-10);
        assert!((ob.ask_depth() - 10.0).abs() < 1e-10);

        // imbalance = (15-10)/(15+10) = 5/25 = 0.2
        assert!((ob.order_imbalance() - 0.2).abs() < 1e-10);
    }
}
