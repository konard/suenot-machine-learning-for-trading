use t5_pretraining_finance::*;

fn main() {
    println!("=== Chapter 286: T5 Pretraining for Finance ===\n");

    // --------------------------------------------------------
    // 1. Initialize the T5 model and tokenizer
    // --------------------------------------------------------
    println!("1. Initializing T5-inspired model...");
    let config = T5Config {
        d_model: 64,
        num_heads: 4,
        num_encoder_layers: 2,
        num_decoder_layers: 2,
        d_ff: 128,
        vocab_size: 1000,
        max_seq_len: 128,
        dropout_rate: 0.1,
    };

    let model = T5Model::new(config);
    let tokenizer = SimpleTokenizer::new(1000);

    println!("   Model initialized with {} encoder layers, {} decoder layers",
        model.config.num_encoder_layers, model.config.num_decoder_layers);
    println!("   d_model={}, num_heads={}, d_ff={}\n",
        model.config.d_model, model.config.num_heads, model.config.d_ff);

    // --------------------------------------------------------
    // 2. Demonstrate span corruption for pretraining
    // --------------------------------------------------------
    println!("2. Demonstrating span corruption pretraining objective...");
    let sample_text = "Bitcoin price surges as institutional investors increase crypto allocations amid favorable regulatory environment";
    let tokens = tokenizer.tokenize(sample_text);
    println!("   Original tokens ({} tokens): {:?}", tokens.len(), &tokens);

    let corruptor = SpanCorruptor::new(0.15, 2);
    let (corrupted, target) = corruptor.corrupt(&tokens, 990);
    println!("   Corrupted input ({} tokens): {:?}", corrupted.len(), corrupted);
    println!("   Target output  ({} tokens): {:?}\n", target.len(), target);

    // --------------------------------------------------------
    // 3. Financial sentiment analysis pipeline
    // --------------------------------------------------------
    println!("3. Financial Sentiment Analysis Pipeline\n");
    println!("   Text-to-text format: 'classify financial sentiment: <text>' -> 'positive/negative/neutral'\n");

    let headlines = vec![
        "Bitcoin hits all-time high as ETF inflows surge to record levels",
        "Crypto exchange faces regulatory crackdown and potential shutdown",
        "Market trades sideways with low volume and no clear direction",
        "Major bank announces Bitcoin custody services for institutional clients",
        "Hackers steal millions from DeFi protocol in flash loan attack",
        "Federal Reserve maintains interest rates unchanged as expected",
        "Ethereum upgrade successfully completed boosting network efficiency",
        "Global recession fears intensify as manufacturing data disappoints",
        "Stablecoin market cap reaches new milestone signaling growing adoption",
        "SEC delays decision on cryptocurrency ETF applications again",
    ];

    println!("   {:<70} | {:>10} | {:>6} | Signal", "Headline", "Sentiment", "Conf");
    println!("   {}", "-".repeat(110));

    let strategy = SentimentStrategy::new(model, tokenizer);

    for headline in &headlines {
        let result = strategy.analyze(headline);
        let sentiment_str = match result.sentiment {
            Sentiment::Positive => "Positive",
            Sentiment::Negative => "Negative",
            Sentiment::Neutral  => "Neutral",
        };
        let signal_str = match result.signal {
            TradingSignal::Buy  => "BUY",
            TradingSignal::Sell => "SELL",
            TradingSignal::Hold => "HOLD",
        };
        let display = if headline.len() > 67 {
            format!("{}...", &headline[..67])
        } else {
            headline.to_string()
        };
        println!("   {:<70} | {:>10} | {:.4} | {}",
            display, sentiment_str, result.confidence, signal_str);
    }

    // --------------------------------------------------------
    // 4. Fetch Bybit market data
    // --------------------------------------------------------
    println!("\n4. Fetching BTCUSDT market data from Bybit...");
    let client = BybitClient::new();

    let candles = match client.fetch_klines("BTCUSDT", "60", 20) {
        Ok(c) => {
            println!("   Successfully fetched {} candles", c.len());
            if let Some(latest) = c.first() {
                println!("   Latest candle: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Volume={:.2}",
                    latest.open, latest.high, latest.low, latest.close, latest.volume);
            }
            c
        }
        Err(e) => {
            println!("   Could not fetch live data: {}. Using simulated data.", e);
            generate_simulated_candles(20)
        }
    };

    // --------------------------------------------------------
    // 5. Backtest sentiment-based strategy
    // --------------------------------------------------------
    println!("\n5. Backtesting sentiment-based trading strategy...\n");

    let backtest_headlines: Vec<&str> = vec![
        "Bitcoin demand surges from institutional investors",
        "Regulatory concerns weigh on crypto markets",
        "Trading volumes remain stable across exchanges",
        "New partnership boosts blockchain adoption in finance",
        "Security breach reported at major exchange",
        "Positive earnings from crypto-related stocks",
        "Central bank digital currency trials show promise",
        "Market manipulation concerns arise in altcoin sector",
        "Bitcoin mining difficulty reaches record high levels",
        "Whale movements detected suggesting potential volatility",
        "Crypto lending platform reports strong quarterly growth",
        "Geopolitical tensions drive safe haven asset demand",
        "DeFi total value locked exceeds previous all time high",
        "Regulatory clarity expected from upcoming policy meeting",
        "Network congestion causes transaction fee spike concern",
        "Institutional grade custody solutions gain traction",
        "Bear market fears as technical indicators turn negative",
        "Layer two scaling solutions see massive adoption increase",
        "Cross border payment volumes grow via blockchain rails",
        "Market sentiment shifts bullish on positive macro data",
    ];

    let n = candles.len().min(backtest_headlines.len());
    let result = strategy.backtest(&candles[..n], &backtest_headlines[..n]);

    println!("   Backtest Results:");
    println!("   -----------------");
    println!("   Total trades:    {}", result.total_trades);
    println!("   Winning trades:  {}", result.winning_trades);
    println!("   Losing trades:   {}", result.losing_trades);
    println!("   Win rate:        {:.2}%", result.win_rate * 100.0);
    println!("   Total PnL:       {:.4}", result.total_pnl);
    println!("   Max drawdown:    {:.4}", result.max_drawdown);
    println!("   Sharpe ratio:    {:.4}", result.sharpe_ratio);

    // --------------------------------------------------------
    // 6. Summary
    // --------------------------------------------------------
    println!("\n6. Summary");
    println!("   --------");
    println!("   - T5 text-to-text paradigm enables unified financial NLP");
    println!("   - Span corruption pretraining teaches financial language understanding");
    println!("   - Sentiment signals can be mapped directly to trading actions");
    println!("   - Combining NLP signals with price data creates alpha potential");
    println!("   - Risk management is essential for noisy sentiment-based signals");
    println!("\n=== End of Chapter 286 Demo ===");
}

/// Generate simulated candle data for when the API is unavailable.
fn generate_simulated_candles(n: usize) -> Vec<Candle> {
    let mut rng = rand::thread_rng();
    let mut price = 50000.0;
    let mut candles = Vec::with_capacity(n);

    for i in 0..n {
        let change = (rng.gen::<f64>() - 0.5) * 1000.0;
        let open = price;
        let close = price + change;
        let high = open.max(close) + rng.gen::<f64>() * 200.0;
        let low = open.min(close) - rng.gen::<f64>() * 200.0;
        let volume = 100.0 + rng.gen::<f64>() * 500.0;

        candles.push(Candle {
            timestamp: 1700000000 + (i as u64) * 3600,
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    candles
}

use rand::Rng;
