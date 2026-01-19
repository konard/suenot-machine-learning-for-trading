//! Integration tests for LLM News Interpretation

use llm_news_interpretation::{
    // Analysis
    SentimentAnalyzer, SentimentResult, EventClassifier, EventType,
    SignalAggregator, AggregatedSignal,
    // News
    NewsItem, NewsSource, NewsPreprocessor, NewsCollector,
    // Strategy
    NewsStrategy, StrategyConfig, RiskManager, RiskConfig, RiskLevel,
    Position, PositionSide, PositionSizer,
    // Data
    OHLCV, Ticker, OrderBook,
    // Utils
    Metrics, MetricsRecorder, AppConfig,
};
use chrono::Utc;

mod sentiment_analysis {
    use super::*;

    #[test]
    fn test_sentiment_analyzer_bullish_news() {
        let analyzer = SentimentAnalyzer::new();
        let text = "Bitcoin shows strong bullish momentum with prices surging to new all-time highs";
        let result = analyzer.analyze_rules(text);

        assert!(result.score > 0.0, "Bullish news should have positive sentiment");
        assert!(result.is_bullish(), "Should be classified as bullish");
    }

    #[test]
    fn test_sentiment_analyzer_bearish_news() {
        let analyzer = SentimentAnalyzer::new();
        let text = "Major cryptocurrency exchange hacked, causing market panic and crash";
        let result = analyzer.analyze_rules(text);

        assert!(result.score < 0.0, "Bearish news should have negative sentiment");
        assert!(result.is_bearish(), "Should be classified as bearish");
    }

    #[test]
    fn test_sentiment_analyzer_neutral_news() {
        let analyzer = SentimentAnalyzer::new();
        let text = "Bitcoin price remains stable around current levels";
        let result = analyzer.analyze_rules(text);

        // Neutral text should have score close to 0
        assert!(result.score.abs() < 0.3, "Neutral news should have score near zero");
    }

    #[test]
    fn test_sentiment_weighted_score() {
        let result = SentimentResult::new(0.8, 0.6);
        let weighted = result.weighted_score();

        assert!((weighted - 0.48).abs() < 0.001, "Weighted score should be score * confidence");
    }
}

mod event_classification {
    use super::*;

    #[test]
    fn test_classify_regulatory_event() {
        let classifier = EventClassifier::new();
        let result = classifier.classify("SEC approves Bitcoin ETF application after years of delay");

        assert_eq!(result.event_type, EventType::Regulatory);
    }

    #[test]
    fn test_classify_security_event() {
        let classifier = EventClassifier::new();
        let result = classifier.classify("DeFi protocol suffers major hack, $50M stolen from users");

        assert_eq!(result.event_type, EventType::Security);
    }

    #[test]
    fn test_classify_technical_event() {
        let classifier = EventClassifier::new();
        let result = classifier.classify("Ethereum completes major protocol upgrade to improve performance");

        assert_eq!(result.event_type, EventType::Technical);
    }

    #[test]
    fn test_event_type_impact() {
        use llm_news_interpretation::data::market::MarketDataError;

        // Security events should have high impact
        assert_eq!(EventType::Security.typical_impact(), llm_news_interpretation::analysis::events::MarketImpact::High);

        // Sentiment events should have low impact
        assert_eq!(EventType::Sentiment.typical_impact(), llm_news_interpretation::analysis::events::MarketImpact::Low);
    }
}

mod news_processing {
    use super::*;

    fn create_test_news() -> NewsItem {
        NewsItem {
            id: "test-1".to_string(),
            title: "Bitcoin ETF Sees Record Inflows".to_string(),
            content: "Institutional investors continue to pour money into Bitcoin ETFs.".to_string(),
            source: NewsSource::Bloomberg,
            url: Some("https://example.com".to_string()),
            published_at: Utc::now(),
            symbols: vec!["BTC".to_string()],
            relevance_score: 0.9,
        }
    }

    #[test]
    fn test_news_item_creation() {
        let news = create_test_news();
        assert_eq!(news.id, "test-1");
        assert_eq!(news.source, NewsSource::Bloomberg);
        assert!(news.is_fresh(3600)); // Should be fresh within 1 hour
    }

    #[test]
    fn test_news_source_reliability() {
        assert!(NewsSource::Bloomberg.reliability_score() > NewsSource::Twitter.reliability_score());
        assert!(NewsSource::Reuters.reliability_score() > NewsSource::Reddit.reliability_score());
    }

    #[test]
    fn test_news_preprocessor() {
        let preprocessor = NewsPreprocessor::new();
        let text = "  Bitcoin   price SURGES  to $50,000!!! ";
        let cleaned = preprocessor.clean_text(text);

        assert!(!cleaned.contains("  ")); // Should remove extra spaces
    }

    #[test]
    fn test_entity_extraction() {
        let preprocessor = NewsPreprocessor::new();
        let text = "BTC and ETH prices rise as SOL reaches new highs";
        let entities = preprocessor.extract_entities(text);

        assert!(entities.contains(&"BTC".to_string()));
        assert!(entities.contains(&"ETH".to_string()));
        assert!(entities.contains(&"SOL".to_string()));
    }
}

mod strategy {
    use super::*;

    #[test]
    fn test_strategy_config_defaults() {
        let config = StrategyConfig::default();

        assert!(!config.symbols.is_empty());
        assert!(config.max_position_size > 0.0);
        assert!(config.max_position_size <= 1.0);
    }

    #[test]
    fn test_news_strategy_creation() {
        let config = StrategyConfig {
            symbols: vec!["BTCUSDT".to_string()],
            min_signal_strength: 0.3,
            min_confidence: 0.6,
            ..Default::default()
        };
        let strategy = NewsStrategy::new(config);

        assert_eq!(strategy.active_position_count(), 0);
        assert!(!strategy.has_position("BTCUSDT"));
    }
}

mod risk_management {
    use super::*;

    #[test]
    fn test_risk_manager_creation() {
        let mut manager = RiskManager::new(RiskConfig::default());
        manager.init_equity(10000.0);

        assert_eq!(manager.assess_risk(), RiskLevel::Low);
        assert!(manager.can_trade().is_allowed());
    }

    #[test]
    fn test_drawdown_calculation() {
        let mut manager = RiskManager::new(RiskConfig::default());
        manager.init_equity(10000.0);
        manager.update_equity(9000.0);

        let drawdown = manager.current_drawdown();
        assert!((drawdown - 0.1).abs() < 0.001, "Drawdown should be 10%");
    }

    #[test]
    fn test_stop_loss_calculation() {
        let manager = RiskManager::new(RiskConfig {
            stop_loss_pct: 0.05,
            ..Default::default()
        });

        let stop_long = manager.calculate_stop_loss(100.0, true);
        let stop_short = manager.calculate_stop_loss(100.0, false);

        assert!((stop_long - 95.0).abs() < 0.001);
        assert!((stop_short - 105.0).abs() < 0.001);
    }

    #[test]
    fn test_critical_risk_level() {
        let mut manager = RiskManager::new(RiskConfig {
            max_drawdown: 0.10,
            ..Default::default()
        });
        manager.init_equity(10000.0);
        manager.update_equity(8900.0); // 11% drawdown

        assert_eq!(manager.assess_risk(), RiskLevel::Critical);
        assert!(!manager.can_trade().is_allowed());
    }
}

mod position_management {
    use super::*;

    #[test]
    fn test_position_creation() {
        let position = Position::new(
            "pos-1".to_string(),
            "BTCUSDT".to_string(),
            PositionSide::Long,
            50000.0,
            0.1,
            1.0,
        );

        assert_eq!(position.symbol, "BTCUSDT");
        assert_eq!(position.entry_price, 50000.0);
        assert_eq!(position.notional, 5000.0);
    }

    #[test]
    fn test_position_pnl_calculation() {
        let position = Position::new(
            "pos-1".to_string(),
            "BTCUSDT".to_string(),
            PositionSide::Long,
            50000.0,
            0.1, // 0.1 BTC
            1.0,
        );

        let pnl = position.unrealized_pnl(55000.0);
        assert!((pnl - 500.0).abs() < 0.001); // 0.1 * 5000 = 500
    }

    #[test]
    fn test_position_sizer() {
        let sizer = PositionSizer::new(10000.0)
            .with_risk_per_trade(0.02)
            .with_max_position(0.25);

        let size = sizer.calculate_size(100.0, 95.0, 1.0);

        // Risk = $200, stop distance = $5, size = 40 units
        assert!((size.size - 40.0).abs() < 0.01);
        assert!((size.notional - 4000.0).abs() < 0.01);
    }
}

mod market_data {
    use super::*;

    #[test]
    fn test_ohlcv_calculations() {
        let candle = OHLCV::new(Utc::now(), 100.0, 110.0, 90.0, 105.0, 1000.0);

        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
        assert_eq!(candle.range(), 20.0);
        assert_eq!(candle.body_size(), 5.0);
    }

    #[test]
    fn test_ticker_spread() {
        let ticker = Ticker {
            symbol: "BTCUSDT".to_string(),
            last_price: 50000.0,
            bid_price: 49990.0,
            ask_price: 50010.0,
            volume_24h: 1000000.0,
            change_24h: 2.5,
            timestamp: Utc::now(),
        };

        assert_eq!(ticker.spread(), 20.0);
        assert_eq!(ticker.mid_price(), 50000.0);
    }

    #[test]
    fn test_orderbook_imbalance() {
        let book = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![(50000.0, 10.0), (49990.0, 5.0)],
            asks: vec![(50010.0, 5.0), (50020.0, 5.0)],
            timestamp: Utc::now(),
        };

        // Bid vol = 15, Ask vol = 10, Imbalance = 15/25 = 0.6
        assert!((book.imbalance() - 0.6).abs() < 0.001);
    }
}

mod metrics {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let mut metrics = Metrics::new();

        metrics.record_trade(100.0);
        metrics.record_trade(-50.0);
        metrics.record_trade(75.0);

        assert_eq!(metrics.total_trades, 3);
        assert_eq!(metrics.winning_trades, 2);
        assert_eq!(metrics.losing_trades, 1);
        assert!((metrics.total_pnl - 125.0).abs() < 0.01);
    }

    #[test]
    fn test_metrics_recorder_thread_safety() {
        let recorder = MetricsRecorder::new();

        recorder.record_trade(100.0);
        recorder.record_news_processed(10);
        recorder.record_signal(0.8);

        let snapshot = recorder.snapshot();
        assert_eq!(snapshot.total_trades, 1);
        assert_eq!(snapshot.news_processed, 10);
        assert_eq!(snapshot.signals_generated, 1);
    }

    #[test]
    fn test_expectancy_calculation() {
        let mut metrics = Metrics::new();
        metrics.record_trade(100.0);
        metrics.record_trade(-50.0);

        let expectancy = metrics.expectancy();
        assert!((expectancy - 25.0).abs() < 0.01);
    }
}

mod configuration {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();

        assert_eq!(config.llm.provider, "openai");
        assert!(config.trading.testnet);
        assert!(config.trading.paper_trading);
    }

    #[test]
    fn test_llm_settings_api_key() {
        let settings = llm_news_interpretation::utils::config::LlmSettings {
            api_key: Some("test-key".to_string()),
            ..Default::default()
        };

        assert_eq!(settings.get_api_key(), Some("test-key".to_string()));
    }
}

mod end_to_end {
    use super::*;

    #[test]
    fn test_news_to_signal_pipeline() {
        // Create news item
        let news = NewsItem {
            id: "e2e-1".to_string(),
            title: "Major Bank Launches Bitcoin Custody Service".to_string(),
            content: "One of the largest banks announces institutional Bitcoin custody, signaling strong adoption.".to_string(),
            source: NewsSource::Bloomberg,
            url: None,
            published_at: Utc::now(),
            symbols: vec!["BTC".to_string()],
            relevance_score: 0.95,
        };

        // Analyze sentiment
        let sentiment_analyzer = SentimentAnalyzer::new();
        let sentiment = sentiment_analyzer.analyze_rules(&format!("{} {}", news.title, news.content));

        // Classify event
        let event_classifier = EventClassifier::new();
        let event = event_classifier.classify(&news.content);

        // Verify analysis
        assert!(sentiment.score > 0.0, "Should detect positive sentiment");
        assert_eq!(event.event_type, EventType::Adoption, "Should classify as adoption event");

        // Check signal strength
        let signal_strength = sentiment.score.abs() * sentiment.confidence;
        assert!(signal_strength > 0.2, "Should generate actionable signal");
    }

    #[test]
    fn test_complete_trading_flow() {
        // Initialize components
        let mut risk_manager = RiskManager::new(RiskConfig::default());
        risk_manager.init_equity(10000.0);

        let position_sizer = PositionSizer::new(10000.0)
            .with_risk_per_trade(0.02);

        let metrics = MetricsRecorder::new();

        // Simulate bullish signal
        let entry_price = 50000.0;
        let stop_loss = risk_manager.calculate_stop_loss(entry_price, true);

        // Check if we can trade
        assert!(risk_manager.can_trade().is_allowed());

        // Calculate position size
        let size = position_sizer.calculate_size(entry_price, stop_loss, 0.8);
        assert!(size.size > 0.0);
        assert!(size.notional <= 2500.0); // Max 25% of portfolio

        // Create position
        let position = Position::new(
            "flow-1".to_string(),
            "BTCUSDT".to_string(),
            PositionSide::Long,
            entry_price,
            size.size,
            1.0,
        )
        .with_stop_loss(stop_loss)
        .with_take_profit(risk_manager.calculate_take_profit(entry_price, true));

        // Simulate price move and close
        let exit_price = 52000.0;
        let pnl = position.unrealized_pnl(exit_price);

        metrics.record_trade(pnl);
        metrics.record_signal(0.8);
        metrics.record_news_processed(1);

        // Verify metrics
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_trades, 1);
        assert!(snapshot.total_pnl > 0.0);
    }
}
