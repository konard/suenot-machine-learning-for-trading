# Crypto Time Series Analysis (Rust)

Библиотека на Rust для анализа временных рядов криптовалют с использованием данных биржи Bybit.

## Структура проекта

```
rust_examples/
├── Cargo.toml
├── src/
│   ├── lib.rs                 # Главный модуль библиотеки
│   ├── api/                   # API модуль
│   │   ├── mod.rs
│   │   ├── bybit.rs          # Клиент Bybit API
│   │   └── storage.rs        # Сохранение/загрузка CSV
│   ├── analysis/             # Анализ временных рядов
│   │   ├── mod.rs
│   │   ├── statistics.rs     # Базовые статистики
│   │   ├── stationarity.rs   # Тесты стационарности (ADF, KPSS)
│   │   ├── autocorrelation.rs # ACF, PACF, Ljung-Box
│   │   └── decomposition.rs  # Декомпозиция рядов
│   ├── models/               # Модели прогнозирования
│   │   ├── mod.rs
│   │   ├── arima.rs          # ARIMA модели
│   │   └── garch.rs          # GARCH модели волатильности
│   ├── trading/              # Торговые стратегии
│   │   ├── mod.rs
│   │   ├── cointegration.rs  # Тесты коинтеграции
│   │   ├── pairs.rs          # Парная торговля
│   │   ├── signals.rs        # Торговые сигналы
│   │   └── backtest.rs       # Бэктестинг
│   └── bin/                  # Примеры использования
│       ├── fetch_data.rs
│       ├── analyze_stationarity.rs
│       ├── arima_forecast.rs
│       ├── volatility_analysis.rs
│       ├── cointegration_test.rs
│       └── pairs_trading.rs
```

## Установка

```bash
cd rust_examples
cargo build --release
```

## Примеры использования

### 1. Загрузка данных с Bybit

```bash
# Загрузить 30 дней часовых свечей BTC
cargo run --bin fetch_data -- --symbol BTCUSDT --interval 1h --days 30

# Загрузить несколько пар
cargo run --bin fetch_data -- --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 4h --days 60
```

### 2. Анализ стационарности

```bash
cargo run --bin analyze_stationarity -- --file data/BTCUSDT_1h.csv

# Анализ доходностей вместо цен
cargo run --bin analyze_stationarity -- --file data/BTCUSDT_1h.csv --returns

# С логарифмическими доходностями
cargo run --bin analyze_stationarity -- --file data/BTCUSDT_1h.csv --log-returns
```

### 3. Прогнозирование ARIMA

```bash
# Автоматический выбор порядка модели
cargo run --bin arima_forecast -- --file data/BTCUSDT_1h.csv --auto --horizon 24

# Заданный порядок ARIMA(2,1,1)
cargo run --bin arima_forecast -- --file data/BTCUSDT_1h.csv -p 2 -d 1 -q 1 -H 12
```

### 4. Анализ волатильности (GARCH)

```bash
cargo run --bin volatility_analysis -- --file data/BTCUSDT_1h.csv

# GARCH(2,1) модель
cargo run --bin volatility_analysis -- --file data/BTCUSDT_1h.csv -p 2 -q 1
```

### 5. Тест на коинтеграцию

```bash
cargo run --bin cointegration_test -- \
    --file1 data/BTCUSDT_1h.csv \
    --file2 data/ETHUSDT_1h.csv
```

### 6. Парная торговля (бэктест)

```bash
# Базовый бэктест
cargo run --bin pairs_trading -- \
    --file1 data/BTCUSDT_1h.csv \
    --file2 data/ETHUSDT_1h.csv

# С настройками
cargo run --bin pairs_trading -- \
    --file1 data/BTCUSDT_1h.csv \
    --file2 data/ETHUSDT_1h.csv \
    --entry 2.0 \
    --exit 0.5 \
    --stop-loss 4.0 \
    --capital 10000

# С walk-forward анализом
cargo run --bin pairs_trading -- \
    --file1 data/BTCUSDT_1h.csv \
    --file2 data/ETHUSDT_1h.csv \
    --walk-forward
```

## Использование как библиотеки

```rust
use crypto_time_series::api::{BybitClient, Interval};
use crypto_time_series::analysis::{adf_test, acf, pacf};
use crypto_time_series::models::{ArimaModel, ArimaParams, GarchModel, GarchParams};
use crypto_time_series::trading::{
    engle_granger_test,
    PairsTradingStrategy,
    PairsTradingParams,
    run_backtest,
    BacktestParams,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Загрузка данных
    let client = BybitClient::new();
    let candles = client.get_klines("BTCUSDT", Interval::Hour1, 1000).await?;

    let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

    // Тест стационарности
    let adf = adf_test(&returns, None);
    println!("ADF: {:.4}, p-value: {:.4}", adf.statistic, adf.p_value);

    // ARIMA прогноз
    let params = ArimaParams::new(1, 0, 1);
    if let Some(model) = ArimaModel::fit(&returns, params) {
        let forecast = model.forecast(&returns, 10);
        println!("Forecast: {:?}", forecast);
    }

    // GARCH волатильность
    let garch_params = GarchParams::garch11();
    if let Some(garch) = GarchModel::fit(&returns, garch_params) {
        let vol_forecast = garch.forecast_volatility(&returns, 10);
        println!("Volatility forecast: {:?}", vol_forecast);
    }

    Ok(())
}
```

## Модули

### `api` - Работа с Bybit API

- `BybitClient` - HTTP клиент для Bybit
- `Interval` - Интервалы свечей (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
- `save_candles` / `load_candles` - Сохранение/загрузка CSV

### `analysis` - Анализ временных рядов

- **Статистики**: mean, variance, std_dev, correlation, moving_average, ema
- **Стационарность**: adf_test, kpss_test, rolling_stationarity_check
- **Автокорреляция**: acf, pacf, ljung_box_test
- **Декомпозиция**: decompose (аддитивная/мультипликативная)

### `models` - Модели прогнозирования

- **ARIMA**: ArimaModel, ArimaParams, auto_arima
- **GARCH**: GarchModel, GarchParams, arch_test

### `trading` - Торговые стратегии

- **Коинтеграция**: engle_granger_test, johansen_test, find_cointegrated_pairs
- **Парная торговля**: PairsTradingStrategy, PairsTradingParams
- **Сигналы**: Signal, Position, Trade, TradeStats
- **Бэктест**: run_backtest, WalkForwardAnalysis

## Лицензия

MIT
