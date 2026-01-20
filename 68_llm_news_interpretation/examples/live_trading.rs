//! Live trading example (paper trading mode)
//!
//! This example demonstrates how to:
//! - Connect to Bybit exchange
//! - Fetch real-time market data
//! - Process news and generate signals
//! - Execute paper trades
//!
//! Run with: cargo run --example live_trading
//!
//! WARNING: This is for educational purposes only. Do NOT use with real funds
//! without thorough testing and understanding of the risks involved.

use llm_news_interpretation::{
    BybitClient, BybitConfig, MarketData,
    NewsItem, NewsSource, SentimentAnalyzer, EventClassifier,
    NewsStrategy, StrategyConfig, RiskManager, RiskConfig,
    Metrics, MetricsRecorder, Position, PositionSide, PositionSizer,
};
use chrono::Utc;
use std::collections::HashMap;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== LLM News Interpretation - Live Trading Example ===\n");
    println!("⚠️  PAPER TRADING MODE - No real orders will be placed\n");

    // Configuration
    let symbols = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()];
    let initial_capital = 10000.0;
    let use_testnet = true;

    // Initialize components
    let bybit_config = if use_testnet {
        BybitConfig::testnet()
    } else {
        BybitConfig::default()
    };

    let client = BybitClient::new(bybit_config)?;
    let sentiment_analyzer = SentimentAnalyzer::new();
    let event_classifier = EventClassifier::new();
    let metrics_recorder = MetricsRecorder::new();
    let position_sizer = PositionSizer::new(initial_capital)
        .with_risk_per_trade(0.02)
        .with_max_position(0.1);

    let mut risk_manager = RiskManager::new(RiskConfig {
        max_risk_per_trade: 0.02,
        max_total_risk: 0.10,
        max_drawdown: 0.15,
        stop_loss_pct: 0.03,
        take_profit_pct: 0.06,
        use_trailing_stop: true,
        trailing_stop_distance: 0.02,
    });
    risk_manager.init_equity(initial_capital);

    let strategy_config = StrategyConfig {
        symbols: symbols.clone(),
        min_signal_strength: 0.3,
        min_confidence: 0.6,
        max_position_size: 0.1,
        trade_cooldown_secs: 300,
        allow_shorts: false,
        max_positions: 2,
    };

    let mut strategy = NewsStrategy::new(strategy_config);

    // Paper trading state
    let mut paper_capital = initial_capital;
    let mut paper_positions: HashMap<String, PaperPosition> = HashMap::new();

    println!("Configuration:");
    println!("  Symbols: {:?}", symbols);
    println!("  Initial Capital: ${:.2}", initial_capital);
    println!("  Max Risk Per Trade: 2%");
    println!("  Use Testnet: {}", use_testnet);
    println!();

    // Fetch initial market data
    println!("Fetching current market data...\n");
    println!("{:-<60}", "");
    println!("{:<12} {:>12} {:>12} {:>12}", "Symbol", "Price", "24h Change", "Volume");
    println!("{:-<60}", "");

    for symbol in &symbols {
        match client.get_ticker(symbol).await {
            Ok(ticker) => {
                println!("{:<12} {:>12.2} {:>11.2}% {:>12.0}",
                         ticker.symbol,
                         ticker.last_price,
                         ticker.change_24h,
                         ticker.volume_24h);
            }
            Err(e) => {
                println!("{:<12} Error: {}", symbol, e);
            }
        }
    }
    println!("{:-<60}", "");

    // Simulate news stream and trading loop
    println!("\nStarting paper trading loop (press Ctrl+C to stop)...\n");

    let simulated_news = get_simulated_news_stream();
    let mut news_index = 0;

    loop {
        // Get "new" news item
        if news_index < simulated_news.len() {
            let news = &simulated_news[news_index];
            news_index += 1;

            println!("📰 New News: {}", news.title);
            println!("   Source: {:?}", news.source);

            // Analyze the news
            let full_text = format!("{} {}", news.title, news.content);
            let sentiment = sentiment_analyzer.analyze_rules(&full_text);
            let event = event_classifier.classify(&full_text);

            println!("   Sentiment: {:.2} ({})", sentiment.score, sentiment.label);
            println!("   Event: {:?} | Impact: {:?}", event.event_type, event.impact);

            metrics_recorder.record_news_processed(1);

            // Check if signal is strong enough
            let signal_strength = sentiment.score.abs() * sentiment.confidence;

            if signal_strength > 0.3 {
                metrics_recorder.record_signal(sentiment.confidence);

                // Determine affected symbol
                let affected_symbol = news.symbols.first()
                    .map(|s| if s.ends_with("USDT") { s.clone() } else { format!("{}USDT", s) })
                    .unwrap_or_else(|| "BTCUSDT".to_string());

                // Get current price
                if let Ok(ticker) = client.get_ticker(&affected_symbol).await {
                    let current_price = ticker.last_price;

                    // Check risk
                    let risk_check = risk_manager.can_trade();
                    if !risk_check.is_allowed() {
                        println!("   ⚠️ Trade blocked by risk manager");
                        continue;
                    }

                    // Trading logic
                    if let Some(position) = paper_positions.get(&affected_symbol) {
                        // Check exit conditions
                        let should_exit = (position.side == PositionSide::Long && sentiment.is_bearish()) ||
                                          position.is_stop_loss_hit(current_price) ||
                                          position.is_take_profit_hit(current_price);

                        if should_exit {
                            // Close position
                            let pnl = position.calculate_pnl(current_price);
                            paper_capital += position.notional + pnl;

                            println!("   📤 CLOSE {} @ ${:.2}", affected_symbol, current_price);
                            println!("   💰 P&L: ${:.2}", pnl);

                            metrics_recorder.record_trade(pnl);
                            paper_positions.remove(&affected_symbol);
                        }
                    } else if sentiment.is_bullish() && paper_positions.len() < 2 {
                        // Open new position
                        let stop_loss = risk_manager.calculate_stop_loss(current_price, true);
                        let size_info = position_sizer.calculate_size(
                            current_price,
                            stop_loss,
                            sentiment.confidence,
                        );

                        if size_info.notional > 0.0 && size_info.notional <= paper_capital * 0.1 {
                            let position = PaperPosition {
                                symbol: affected_symbol.clone(),
                                side: PositionSide::Long,
                                entry_price: current_price,
                                size: size_info.size,
                                notional: size_info.notional,
                                stop_loss,
                                take_profit: risk_manager.calculate_take_profit(current_price, true),
                            };

                            paper_capital -= size_info.notional;
                            paper_positions.insert(affected_symbol.clone(), position);

                            println!("   📥 BUY {} @ ${:.2}", affected_symbol, current_price);
                            println!("   📊 Size: {:.6} | Notional: ${:.2}", size_info.size, size_info.notional);
                            println!("   🛡️ SL: ${:.2} | TP: ${:.2}", stop_loss, risk_manager.calculate_take_profit(current_price, true));
                        }
                    }
                }
            }

            println!();
        }

        // Update position values and check stops
        for (symbol, position) in paper_positions.iter() {
            if let Ok(ticker) = client.get_ticker(symbol).await {
                let unrealized_pnl = position.calculate_pnl(ticker.last_price);
                let pnl_pct = (unrealized_pnl / position.notional) * 100.0;
                println!("📊 {} Position: ${:.2} ({:+.2}%)",
                         symbol, position.notional + unrealized_pnl, pnl_pct);
            }
        }

        // Portfolio summary
        let total_position_value: f64 = paper_positions.values()
            .map(|p| p.notional)
            .sum();
        let total_value = paper_capital + total_position_value;

        println!("\n💼 Portfolio: Cash ${:.2} | Positions ${:.2} | Total ${:.2}",
                 paper_capital, total_position_value, total_value);
        println!("   Return: {:+.2}%\n", (total_value / initial_capital - 1.0) * 100.0);

        risk_manager.update_equity(total_value);

        // Wait before next iteration
        if news_index >= simulated_news.len() {
            println!("No more news items. Exiting...");
            break;
        }

        sleep(Duration::from_secs(5)).await;
    }

    // Final summary
    let metrics = metrics_recorder.snapshot();
    println!("\n=== Final Results ===");
    println!("News Processed: {}", metrics.news_processed);
    println!("Signals Generated: {}", metrics.signals_generated);
    println!("Total Trades: {}", metrics.total_trades);
    println!("Total P&L: ${:.2}", metrics.total_pnl);

    Ok(())
}

/// Paper trading position
#[derive(Debug, Clone)]
struct PaperPosition {
    symbol: String,
    side: PositionSide,
    entry_price: f64,
    size: f64,
    notional: f64,
    stop_loss: f64,
    take_profit: f64,
}

impl PaperPosition {
    fn calculate_pnl(&self, current_price: f64) -> f64 {
        match self.side {
            PositionSide::Long => self.size * (current_price - self.entry_price),
            PositionSide::Short => self.size * (self.entry_price - current_price),
        }
    }

    fn is_stop_loss_hit(&self, current_price: f64) -> bool {
        match self.side {
            PositionSide::Long => current_price <= self.stop_loss,
            PositionSide::Short => current_price >= self.stop_loss,
        }
    }

    fn is_take_profit_hit(&self, current_price: f64) -> bool {
        match self.side {
            PositionSide::Long => current_price >= self.take_profit,
            PositionSide::Short => current_price <= self.take_profit,
        }
    }
}

/// Generate simulated news stream for demo
fn get_simulated_news_stream() -> Vec<NewsItem> {
    vec![
        NewsItem {
            id: "1".to_string(),
            title: "Bitcoin ETF sees record inflows".to_string(),
            content: "Spot Bitcoin ETFs recorded their highest daily inflows since launch, with institutional investors showing strong demand.".to_string(),
            source: NewsSource::Bloomberg,
            url: None,
            published_at: Utc::now(),
            symbols: vec!["BTC".to_string()],
            relevance_score: 0.9,
        },
        NewsItem {
            id: "2".to_string(),
            title: "Ethereum staking rewards increase".to_string(),
            content: "The Ethereum network has seen an increase in staking rewards following the latest protocol update.".to_string(),
            source: NewsSource::CoinDesk,
            url: None,
            published_at: Utc::now(),
            symbols: vec!["ETH".to_string()],
            relevance_score: 0.85,
        },
        NewsItem {
            id: "3".to_string(),
            title: "Major exchange reports technical issues".to_string(),
            content: "Users report intermittent service disruptions on major cryptocurrency exchange during high volatility period.".to_string(),
            source: NewsSource::Twitter,
            url: None,
            published_at: Utc::now(),
            symbols: vec!["BTC".to_string()],
            relevance_score: 0.7,
        },
        NewsItem {
            id: "4".to_string(),
            title: "Bitcoin mining difficulty reaches all-time high".to_string(),
            content: "The Bitcoin network difficulty has adjusted to a new record, reflecting strong miner participation.".to_string(),
            source: NewsSource::CoinTelegraph,
            url: None,
            published_at: Utc::now(),
            symbols: vec!["BTC".to_string()],
            relevance_score: 0.75,
        },
        NewsItem {
            id: "5".to_string(),
            title: "Regulatory clarity improves in major market".to_string(),
            content: "New cryptocurrency regulations provide clearer framework for institutional participation.".to_string(),
            source: NewsSource::Reuters,
            url: None,
            published_at: Utc::now(),
            symbols: vec!["BTC".to_string(), "ETH".to_string()],
            relevance_score: 0.9,
        },
    ]
}
