//! Example: Time Series Encoding
//!
//! This example demonstrates temporal encodings specifically designed
//! for financial time series data, including calendar and market session features.

use positional_encoding::{
    CalendarEncoding, MarketSessionEncoding, MultiScaleTemporalEncoding,
    TimeSeriesSinusoidalEncoding, MarketType, PositionalEncoding,
};
use chrono::{DateTime, Utc, Datelike, Timelike};

fn main() {
    println!("Time Series Encoding Example");
    println!("============================\n");

    // 1. Time Series Sinusoidal Encoding
    println!("1. Multi-Scale Sinusoidal Encoding");
    println!("-----------------------------------");

    let ts_encoding = TimeSeriesSinusoidalEncoding::new(32);

    // Encode one week of hourly data
    let hourly_positions: Vec<usize> = (0..168).collect();
    let encoded = ts_encoding.encode(&hourly_positions);

    println!("Scales: 1h, 24h (daily), 168h (weekly), 720h (monthly)");
    println!("Dimension: {}", ts_encoding.dim());
    println!("One week of hourly data: {} positions", hourly_positions.len());
    println!("Output shape: {:?}", encoded.shape());

    // Show how different scales capture patterns
    println!("\nPattern capture:");
    println!("  Hour 0 (Monday 00:00): captures start of week");
    println!("  Hour 24 (Tuesday 00:00): 1 day elapsed, weekly pattern shifted");
    println!("  Hour 168 (Next Monday): weekly cycle complete");

    // 2. Calendar Encoding
    println!("\n2. Calendar Encoding");
    println!("--------------------");

    let calendar = CalendarEncoding::new();

    // Test timestamps for different scenarios
    let test_cases = vec![
        (1704067200, "2024-01-01 00:00 UTC (New Year, Monday)"),
        (1704110400, "2024-01-01 12:00 UTC (Monday noon)"),
        (1704499200, "2024-01-06 00:00 UTC (Saturday)"),
        (1706745600, "2024-02-01 00:00 UTC (Feb 1, Thursday)"),
    ];

    println!("Features: hour_sin/cos, dow_sin/cos, dom_sin/cos, month_sin/cos,");
    println!("          minute_sin/cos, week_sin/cos, is_weekend, month_start/end, quarter");
    println!("\nDimension: {}", calendar.dim());

    for (ts, desc) in &test_cases {
        let encoded = calendar.encode_timestamp(*ts);

        // Extract key features
        let is_weekend = encoded[12];
        let is_month_start = encoded[13];
        let is_month_end = encoded[14];
        let quarter = encoded[15];

        println!("\n{}", desc);
        println!("  Weekend: {:.0}, Month Start: {:.0}, Month End: {:.0}, Quarter: {:.2}",
            is_weekend, is_month_start, is_month_end, quarter);
    }

    // 3. Market Session Encoding
    println!("\n3. Market Session Encoding");
    println!("--------------------------");

    // Crypto market (24/7)
    println!("\n3a. Cryptocurrency (24/7)");
    let crypto_enc = MarketSessionEncoding::new(MarketType::Crypto);
    println!("Dimension: {}", crypto_enc.dim());
    println!("Features: is_asia, is_europe, is_americas, overlaps, hour_cycle, weekend, funding, midnight");

    let crypto_hours = vec![
        (1704070800, "01:00 UTC - Asia session"),
        (1704085200, "05:00 UTC - Asia session"),
        (1704099600, "10:00 UTC - Europe session"),
        (1704114000, "14:00 UTC - Europe/Americas overlap"),
        (1704128400, "18:00 UTC - Americas session"),
        (1704142800, "22:00 UTC - Americas session"),
    ];

    for (ts, desc) in &crypto_hours {
        let enc = crypto_enc.encode_timestamp(*ts);
        println!("{}: Asia={:.0}, Europe={:.0}, Americas={:.0}",
            desc, enc[0], enc[1], enc[2]);
    }

    // Stock market (9:30 AM - 4 PM EST)
    println!("\n3b. Stock Market (US)");
    let stock_enc = MarketSessionEncoding::new(MarketType::Stock);
    println!("Dimension: {}", stock_enc.dim());
    println!("Features: premarket, regular, afterhours, closed, session_progress, open/close_proximity, weekend");

    let stock_hours = vec![
        (1704099600, "10:00 UTC (5:00 EST) - Pre-market"),
        (1704114000, "14:00 UTC (9:00 EST) - Just before open"),
        (1704117600, "15:00 UTC (10:00 EST) - Regular session"),
        (1704139200, "21:00 UTC (16:00 EST) - Close"),
        (1704146400, "23:00 UTC (18:00 EST) - After hours"),
    ];

    for (ts, desc) in &stock_hours {
        let enc = stock_enc.encode_timestamp(*ts);
        println!("{}: Pre={:.0}, Reg={:.0}, After={:.0}, Closed={:.0}",
            desc, enc[0], enc[1], enc[2], enc[3]);
    }

    // Forex market
    println!("\n3c. Forex Market");
    let forex_enc = MarketSessionEncoding::new(MarketType::Forex);
    println!("Dimension: {}", forex_enc.dim());
    println!("Features: sydney, tokyo, london, ny, london_ny_overlap, tokyo_london_overlap, active_count");

    let forex_hours = vec![
        (1704070800, "01:00 UTC - Tokyo/Sydney"),
        (1704099600, "10:00 UTC - London"),
        (1704114000, "14:00 UTC - London/NY overlap (highest volume)"),
        (1704135600, "20:00 UTC - NY"),
    ];

    for (ts, desc) in &forex_hours {
        let enc = forex_enc.encode_timestamp(*ts);
        println!("{}: Sydney={:.0}, Tokyo={:.0}, London={:.0}, NY={:.0}",
            desc, enc[0], enc[1], enc[2], enc[3]);
    }

    // 4. Combined Multi-Scale Encoding
    println!("\n4. Multi-Scale Temporal Encoding");
    println!("---------------------------------");

    let multi_scale = MultiScaleTemporalEncoding::new(MarketType::Crypto);
    println!("Combines Calendar + Market Session encoding");
    println!("Total dimension: {}", multi_scale.dim());

    // Show how it encodes different market conditions
    let scenarios = vec![
        (1704117600, "Monday morning (Europe) - High activity expected"),
        (1704499200, "Saturday midnight - Low crypto activity"),
        (1704031200, "Month end Friday - Portfolio rebalancing"),
    ];

    for (ts, desc) in &scenarios {
        let enc = multi_scale.encode_timestamp(*ts);
        let dt = DateTime::from_timestamp(*ts, 0).unwrap();
        println!("\n{}", desc);
        println!("  Timestamp: {} ({})", ts, dt.format("%Y-%m-%d %H:%M UTC"));
        println!("  Encoding vector length: {}", enc.len());
        println!("  First 5 values: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
            enc[0], enc[1], enc[2], enc[3], enc[4]);
    }

    // 5. Practical Application
    println!("\n5. Practical Application");
    println!("------------------------");

    println!("\nRecommended encoding combinations by use case:");
    println!();
    println!("Intraday Trading (1-min to 1-hour):");
    println!("  - TimeSeriesSinusoidal (multi-scale patterns)");
    println!("  - MarketSession (session awareness)");
    println!("  - RoPE (sequence position)");
    println!();
    println!("Swing Trading (4-hour to daily):");
    println!("  - Calendar (day of week effects)");
    println!("  - TimeSeriesSinusoidal (weekly/monthly patterns)");
    println!("  - Sinusoidal (sequence position)");
    println!();
    println!("Long-term Investment (daily to monthly):");
    println!("  - Calendar (month/quarter effects)");
    println!("  - Learned (let model discover patterns)");
    println!();
    println!("Crypto 24/7:");
    println!("  - MarketSession(Crypto) - regional sessions");
    println!("  - MultiScaleTemporalEncoding");
    println!("  - RoPE (handles variable sequence lengths)");

    println!("\nExample completed!");
}
