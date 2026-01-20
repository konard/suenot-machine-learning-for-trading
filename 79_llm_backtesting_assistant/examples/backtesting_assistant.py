"""
LLM Backtesting Assistant - Python Implementation
Analyzes trading strategy backtest results using LLMs

This example demonstrates:
1. Fetching data from Bybit and stock markets
2. Running a simple backtest
3. Calculating performance metrics
4. Getting LLM-powered analysis
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class AssetClass(Enum):
    """Asset class types"""
    EQUITY = "equity"
    CRYPTOCURRENCY = "cryptocurrency"
    FOREX = "forex"
    FUTURES = "futures"


class TradeDirection(Enum):
    """Trade direction"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """Individual trade record"""
    id: str
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: TradeDirection
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float


@dataclass
class PerformanceMetrics:
    """Strategy performance metrics"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration_hours: float
    turnover: float


@dataclass
class BacktestResults:
    """Complete backtest results"""
    strategy_id: str
    strategy_type: str
    asset_class: AssetClass
    start_date: datetime
    end_date: datetime
    metrics: PerformanceMetrics
    trades: List[Trade]
    equity_curve: pd.Series
    regime_performance: Optional[Dict[str, float]] = None


@dataclass
class Recommendation:
    """Strategy improvement recommendation"""
    priority: str  # critical, high, medium, low
    category: str
    description: str
    expected_impact: str
    implementation_steps: List[str]


@dataclass
class AnalysisReport:
    """LLM-generated analysis report"""
    report_id: str
    generated_at: datetime
    strategy_id: str
    summary: str
    grade: str  # A, B, C, D, F
    strengths: List[str]
    concerns: List[str]
    recommendations: List[Recommendation]
    metric_explanations: Dict[str, str]


# =============================================================================
# Metrics Calculator
# =============================================================================

class MetricsCalculator:
    """Calculate performance metrics from trade data"""

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.Series,
        periods_per_year: int = 252
    ) -> PerformanceMetrics:
        """Calculate all performance metrics"""
        if len(trades) == 0:
            return self._empty_metrics()

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        if len(returns) == 0:
            return self._empty_metrics()

        # Total and annual return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        n_periods = len(equity_curve)
        n_years = n_periods / periods_per_year
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / periods_per_year
        sharpe_ratio = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std() \
            if returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.001
        sortino_ratio = np.sqrt(periods_per_year) * excess_returns.mean() / downside_std \
            if downside_std > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_drawdown = drawdowns.min()

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Cap profit factor at reasonable value for display
        if profit_factor == float('inf'):
            profit_factor = 10.0

        # Average trade duration
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
        avg_duration = np.mean(durations) if durations else 0

        # Turnover (based on number of trades per year)
        days_in_backtest = (equity_curve.index[-1] - equity_curve.index[0]).days if len(equity_curve) > 1 else 1
        turnover = len(trades) / max(1, days_in_backtest / 365)

        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_duration_hours=avg_duration,
            turnover=turnover
        )

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for no trades"""
        return PerformanceMetrics(
            total_return=0, annual_return=0, sharpe_ratio=0,
            sortino_ratio=0, calmar_ratio=0, max_drawdown=0,
            win_rate=0, profit_factor=0, total_trades=0,
            avg_trade_duration_hours=0, turnover=0
        )


# =============================================================================
# Data Fetchers
# =============================================================================

class BybitDataFetcher:
    """Fetch historical data from Bybit for backtesting"""

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        try:
            import requests
            self.session = requests.Session()
        except ImportError:
            logger.warning("requests library not installed. Install with: pip install requests")
            self.session = None

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        category: str = "spot"
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Bybit"""
        if self.session is None:
            raise RuntimeError("requests library not installed")

        endpoint = f"{self.BASE_URL}/v5/market/kline"

        all_data = []
        current_start = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)

        while current_start < end_ts:
            params = {
                "category": category,
                "symbol": symbol,
                "interval": interval,
                "start": current_start,
                "end": end_ts,
                "limit": 1000
            }

            try:
                response = self.session.get(endpoint, params=params, timeout=30)
                data = response.json()

                if data["retCode"] != 0:
                    logger.error(f"Bybit API error: {data['retMsg']}")
                    break

                klines = data["result"]["list"]
                if not klines:
                    break

                all_data.extend(klines)

                # Move to next batch (Bybit returns data in descending order)
                last_ts = int(klines[-1][0])
                if last_ts <= current_start:
                    break
                current_start = last_ts + 1

            except Exception as e:
                logger.error(f"Error fetching Bybit data: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df = df.set_index("timestamp").sort_index()
        df = df.astype(float)

        return df

    def get_funding_history(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Fetch funding rate history for perpetuals"""
        if self.session is None:
            raise RuntimeError("requests library not installed")

        endpoint = f"{self.BASE_URL}/v5/market/funding/history"

        params = {
            "category": "linear",
            "symbol": symbol,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
            "limit": 200
        }

        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            data = response.json()

            if data["retCode"] != 0:
                logger.error(f"Bybit API error: {data['retMsg']}")
                return pd.DataFrame()

            records = data["result"]["list"]
            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df["fundingRateTimestamp"] = pd.to_datetime(
                df["fundingRateTimestamp"].astype(int), unit="ms"
            )
            df = df.set_index("fundingRateTimestamp").sort_index()
            return df

        except Exception as e:
            logger.error(f"Error fetching funding history: {e}")
            return pd.DataFrame()


class StockDataFetcher:
    """Fetch stock market data using yfinance"""

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            logger.warning("yfinance not installed. Install with: pip install yfinance")
            self.yf = None

    def get_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch stock data using yfinance"""
        if self.yf is None:
            raise RuntimeError("yfinance library not installed")

        try:
            ticker = self.yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            data.columns = [c.lower() for c in data.columns]
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            return pd.DataFrame()


# =============================================================================
# LLM Client
# =============================================================================

class LlmClient:
    """Client for LLM API calls"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def analyze(self, system_prompt: str, user_prompt: str) -> str:
        """Send analysis request to LLM"""
        if not self.api_key:
            return self._mock_analysis(user_prompt)

        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }

            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

            return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            logger.warning(f"LLM API call failed: {e}. Using mock analysis.")
            return self._mock_analysis(user_prompt)

    def _mock_analysis(self, user_prompt: str) -> str:
        """Generate mock analysis when API is not available"""
        return """## Strategy Analysis Report

### Overall Assessment
Your strategy shows moderate performance with room for improvement. The risk-adjusted returns are acceptable but could be enhanced with better risk management.

### Grade: B-

### Strengths:
- Positive total return over the backtest period
- Win rate above 50% indicates consistent edge
- Profit factor greater than 1.0 shows positive expectancy

### Concerns:
- Maximum drawdown may be higher than optimal for risk-averse investors
- Sharpe ratio could be improved with volatility-adjusted position sizing
- Consider the impact of transaction costs on net returns

### Recommendations:
1. **High Priority - Risk Management**: Implement ATR-based position sizing to reduce drawdowns during volatile periods
2. **Medium Priority - Entry Optimization**: Add confirmation filters to improve win rate
3. **Medium Priority - Exit Strategy**: Consider trailing stops to lock in profits on winning trades
4. **Low Priority - Diversification**: Test the strategy across multiple assets to reduce correlation risk

### Metric Explanations:
- **Sharpe Ratio**: Measures risk-adjusted return. Values above 1.0 are generally considered good.
- **Maximum Drawdown**: The largest peak-to-trough decline. Lower is better for capital preservation.
- **Win Rate**: Percentage of winning trades. Context matters - trend followers may have lower win rates but larger wins.
- **Profit Factor**: Gross profit divided by gross loss. Values above 1.5 indicate a healthy edge.

*Note: This is a mock analysis. Connect to an LLM API for personalized insights.*"""


# =============================================================================
# Backtesting Assistant
# =============================================================================

class BacktestingAssistant:
    """LLM-powered backtesting analysis assistant"""

    SYSTEM_PROMPT = """You are an expert quantitative analyst specializing in trading strategy evaluation.
Your task is to analyze backtest results and provide actionable insights.

When analyzing results:
1. Consider the strategy type and its expected behavior
2. Compare metrics against industry benchmarks
3. Identify potential risks and failure modes
4. Suggest specific, implementable improvements

Benchmark reference values:
- Sharpe Ratio: > 1.0 good, > 2.0 excellent
- Max Drawdown: < 15% conservative, < 25% moderate
- Win Rate: context-dependent (50%+ for frequency trading)
- Profit Factor: > 1.5 good, > 2.0 excellent

For cryptocurrency strategies, adjust expectations:
- Sharpe > 1.5 is good (due to higher volatility)
- Max Drawdown < 40% is acceptable
- Consider funding rates and liquidation risks

Provide your analysis in a structured format with:
1. Overall assessment (2-3 sentences)
2. Performance grade (A/B/C/D/F)
3. Key strengths (bullet points)
4. Areas of concern (bullet points)
5. Specific recommendations with priority levels
6. Brief explanation of key metrics"""

    def __init__(self, llm_client: Optional[LlmClient] = None, config: Optional[Dict] = None):
        self.llm_client = llm_client or LlmClient()
        self.metrics_calculator = MetricsCalculator()
        self.config = config or {
            "min_trades": 30,
            "benchmark_sharpe": 1.0
        }

    def analyze(self, results: BacktestResults) -> AnalysisReport:
        """Analyze backtest results and generate report"""
        # Prepare the user prompt with metrics
        user_prompt = self._build_user_prompt(results)

        # Get LLM analysis
        llm_response = self.llm_client.analyze(self.SYSTEM_PROMPT, user_prompt)

        # Parse response into structured report
        return self._parse_response(results, llm_response)

    def _build_user_prompt(self, results: BacktestResults) -> str:
        """Build the user prompt with backtest data"""
        metrics = results.metrics

        prompt = f"""Analyze the following backtest results for a {results.strategy_type} strategy
trading {results.asset_class.value} from {results.start_date.date()} to {results.end_date.date()}:

PERFORMANCE METRICS:
- Total Return: {metrics.total_return:.2%}
- Annual Return: {metrics.annual_return:.2%}
- Sharpe Ratio: {metrics.sharpe_ratio:.2f}
- Sortino Ratio: {metrics.sortino_ratio:.2f}
- Calmar Ratio: {metrics.calmar_ratio:.2f}
- Maximum Drawdown: {metrics.max_drawdown:.2%}
- Win Rate: {metrics.win_rate:.2%}
- Profit Factor: {metrics.profit_factor:.2f}
- Total Trades: {metrics.total_trades}
- Average Trade Duration: {metrics.avg_trade_duration_hours:.1f} hours
- Annual Turnover: {metrics.turnover:.1f}x

"""

        if results.regime_performance:
            prompt += "REGIME PERFORMANCE:\n"
            for regime, perf in results.regime_performance.items():
                prompt += f"- {regime}: {perf:.2%}\n"
            prompt += "\n"

        prompt += """Please provide your analysis with:
1. Overall assessment
2. Performance grade (A/B/C/D/F)
3. Key strengths
4. Areas of concern
5. Specific recommendations with priority (critical/high/medium/low)
6. Brief metric explanations"""

        return prompt

    def _parse_response(self, results: BacktestResults, llm_response: str) -> AnalysisReport:
        """Parse LLM response into structured report"""
        lines = llm_response.split('\n')

        # Extract grade
        grade = "C"  # default
        for line in lines:
            if "grade" in line.lower():
                for g in ["A", "B", "C", "D", "F"]:
                    if g in line:
                        grade = g
                        break

        return AnalysisReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            strategy_id=results.strategy_id,
            summary=llm_response[:500] + "..." if len(llm_response) > 500 else llm_response,
            grade=grade,
            strengths=self._extract_section(llm_response, "strength"),
            concerns=self._extract_section(llm_response, "concern"),
            recommendations=self._extract_recommendations(llm_response),
            metric_explanations=self._extract_explanations(results.metrics, llm_response)
        )

    def _extract_section(self, text: str, keyword: str) -> List[str]:
        """Extract bullet points from a section"""
        items = []
        in_section = False
        for line in text.split('\n'):
            if keyword in line.lower():
                in_section = True
                continue
            if in_section and line.strip().startswith(('-', '*', '•')):
                items.append(line.strip().lstrip('-*• '))
            elif in_section and line.strip() and not line.strip().startswith(('-', '*', '•')):
                if any(x in line.lower() for x in ['recommendation', 'concern', 'strength', 'metric', '##']):
                    in_section = False
        return items[:5]

    def _extract_recommendations(self, text: str) -> List[Recommendation]:
        """Extract recommendations from response"""
        recommendations = []
        items = self._extract_section(text, "recommendation")

        for item in items:
            priority = "medium"
            for p in ["critical", "high", "medium", "low"]:
                if p in item.lower():
                    priority = p
                    break

            recommendations.append(Recommendation(
                priority=priority,
                category="general",
                description=item,
                expected_impact="Improvement in risk-adjusted returns",
                implementation_steps=["Implement the suggested change", "Backtest the modification", "Compare results"]
            ))

        return recommendations

    def _extract_explanations(self, metrics: PerformanceMetrics, text: str) -> Dict[str, str]:
        """Extract metric explanations"""
        explanations = {}
        metric_names = ["sharpe", "sortino", "drawdown", "win rate", "profit factor"]
        for name in metric_names:
            for line in text.split('\n'):
                if name in line.lower() and ':' in line:
                    explanations[name] = line.strip()
                    break
        return explanations


# =============================================================================
# Backtesting Engine
# =============================================================================

class SimpleBacktester:
    """Simple backtesting engine for demonstration"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital

    def run_momentum_strategy(
        self,
        data: pd.DataFrame,
        fast_period: int = 20,
        slow_period: int = 50,
        position_size_pct: float = 0.1
    ) -> Tuple[List[Trade], pd.Series]:
        """Run a simple moving average crossover strategy"""
        if len(data) < slow_period:
            return [], pd.Series([self.initial_capital])

        # Calculate moving averages
        data = data.copy()
        data["sma_fast"] = data["close"].rolling(window=fast_period).mean()
        data["sma_slow"] = data["close"].rolling(window=slow_period).mean()

        # Generate signals
        data["signal"] = 0
        data.loc[data["sma_fast"] > data["sma_slow"], "signal"] = 1
        data.loc[data["sma_fast"] < data["sma_slow"], "signal"] = -1

        # Simulate trades
        trades = []
        position = 0
        entry_price = 0.0
        entry_time = None
        capital = self.initial_capital
        equity = []

        for timestamp, row in data.iterrows():
            if pd.isna(row["signal"]) or pd.isna(row["sma_fast"]) or pd.isna(row["sma_slow"]):
                equity.append(capital)
                continue

            current_signal = int(row["signal"])

            # Entry
            if position == 0 and current_signal != 0:
                position = current_signal
                entry_price = row["close"]
                entry_time = timestamp

            # Exit on signal change
            elif position != 0 and current_signal != position:
                exit_price = row["close"]
                trade_return = position * (exit_price - entry_price) / entry_price
                pnl = trade_return * capital * position_size_pct
                capital += pnl

                trades.append(Trade(
                    id=f"trade_{len(trades)}",
                    entry_time=entry_time,
                    exit_time=timestamp,
                    symbol=data.index.name or "UNKNOWN",
                    direction=TradeDirection.LONG if position == 1 else TradeDirection.SHORT,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=capital * position_size_pct / entry_price,
                    pnl=pnl,
                    return_pct=trade_return
                ))

                # Reset or open new position
                if current_signal != 0:
                    position = current_signal
                    entry_price = row["close"]
                    entry_time = timestamp
                else:
                    position = 0
                    entry_price = 0.0
                    entry_time = None

            equity.append(capital)

        # Create equity curve
        equity_curve = pd.Series(equity, index=data.index[:len(equity)])

        return trades, equity_curve


# =============================================================================
# Main Example
# =============================================================================

def print_metrics_summary(metrics: PerformanceMetrics):
    """Print a formatted metrics summary"""
    print("\n" + "=" * 60)
    print("BACKTEST METRICS SUMMARY")
    print("=" * 60)
    print(f"Total Return:       {metrics.total_return:>12.2%}")
    print(f"Annual Return:      {metrics.annual_return:>12.2%}")
    print(f"Sharpe Ratio:       {metrics.sharpe_ratio:>12.2f}")
    print(f"Sortino Ratio:      {metrics.sortino_ratio:>12.2f}")
    print(f"Calmar Ratio:       {metrics.calmar_ratio:>12.2f}")
    print(f"Max Drawdown:       {metrics.max_drawdown:>12.2%}")
    print(f"Win Rate:           {metrics.win_rate:>12.2%}")
    print(f"Profit Factor:      {metrics.profit_factor:>12.2f}")
    print(f"Total Trades:       {metrics.total_trades:>12}")
    print(f"Avg Trade Duration: {metrics.avg_trade_duration_hours:>12.1f} hours")
    print("=" * 60)


def example_with_sample_data():
    """Example using generated sample data"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Backtesting Assistant with Sample Data")
    print("=" * 60)

    # Generate sample OHLCV data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="h")
    n = len(dates)

    # Generate realistic price movements
    returns = np.random.normal(0.0001, 0.02, n)
    price = 50000 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        "open": price * (1 + np.random.uniform(-0.005, 0.005, n)),
        "high": price * (1 + np.abs(np.random.normal(0, 0.01, n))),
        "low": price * (1 - np.abs(np.random.normal(0, 0.01, n))),
        "close": price,
        "volume": np.random.uniform(100, 1000, n)
    }, index=dates)

    data.index.name = "BTCUSDT"

    print(f"Generated {len(data)} hourly candles")
    print(f"Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")

    # Run backtest
    backtester = SimpleBacktester(initial_capital=100000)
    trades, equity_curve = backtester.run_momentum_strategy(
        data,
        fast_period=20,
        slow_period=50,
        position_size_pct=0.1
    )

    print(f"Executed {len(trades)} trades")

    # Calculate metrics
    calculator = MetricsCalculator()
    metrics = calculator.calculate_metrics(trades, equity_curve, periods_per_year=8760)

    # Create results object
    results = BacktestResults(
        strategy_id="btc_momentum_sample",
        strategy_type="momentum following with moving average crossover",
        asset_class=AssetClass.CRYPTOCURRENCY,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        metrics=metrics,
        trades=trades,
        equity_curve=equity_curve,
        regime_performance={
            "bull_market": 0.15,
            "bear_market": -0.05,
            "high_volatility": 0.08,
            "low_volatility": 0.12
        }
    )

    print_metrics_summary(metrics)

    # Analyze with assistant
    print("\n" + "=" * 60)
    print("LLM ANALYSIS")
    print("=" * 60)

    assistant = BacktestingAssistant()
    report = assistant.analyze(results)

    print(f"\nReport ID: {report.report_id}")
    print(f"Grade: {report.grade}")
    print(f"\nSummary:\n{report.summary}")

    if report.recommendations:
        print("\nTop Recommendations:")
        for i, rec in enumerate(report.recommendations[:3], 1):
            print(f"  {i}. [{rec.priority.upper()}] {rec.description[:80]}...")

    return results


def example_with_bybit_data():
    """Example using real Bybit data"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Backtesting Assistant with Bybit Data")
    print("=" * 60)

    try:
        fetcher = BybitDataFetcher()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days

        print(f"Fetching BTCUSDT data from {start_date.date()} to {end_date.date()}...")

        data = fetcher.get_klines(
            symbol="BTCUSDT",
            interval="60",  # 1 hour
            start_time=start_date,
            end_time=end_date,
            category="spot"
        )

        if data.empty:
            print("Could not fetch Bybit data. Check your internet connection.")
            return None

        print(f"Fetched {len(data)} candles")
        data.index.name = "BTCUSDT"

        # Run backtest
        backtester = SimpleBacktester(initial_capital=100000)
        trades, equity_curve = backtester.run_momentum_strategy(data)

        print(f"Executed {len(trades)} trades")

        # Calculate metrics
        calculator = MetricsCalculator()
        metrics = calculator.calculate_metrics(trades, equity_curve, periods_per_year=8760)

        # Create results
        results = BacktestResults(
            strategy_id="btc_momentum_bybit",
            strategy_type="momentum following",
            asset_class=AssetClass.CRYPTOCURRENCY,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve
        )

        print_metrics_summary(metrics)

        # Analyze
        assistant = BacktestingAssistant()
        report = assistant.analyze(results)

        print(f"\nGrade: {report.grade}")

        return results

    except Exception as e:
        print(f"Error: {e}")
        return None


def example_with_stock_data():
    """Example using stock market data"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Backtesting Assistant with Stock Data")
    print("=" * 60)

    try:
        fetcher = StockDataFetcher()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Last year

        print(f"Fetching AAPL data from {start_date.date()} to {end_date.date()}...")

        data = fetcher.get_data("AAPL", start_date, end_date)

        if data.empty:
            print("Could not fetch stock data. Install yfinance: pip install yfinance")
            return None

        print(f"Fetched {len(data)} daily candles")
        data.index.name = "AAPL"

        # Run backtest
        backtester = SimpleBacktester(initial_capital=100000)
        trades, equity_curve = backtester.run_momentum_strategy(
            data,
            fast_period=10,
            slow_period=30
        )

        print(f"Executed {len(trades)} trades")

        # Calculate metrics (252 trading days per year)
        calculator = MetricsCalculator()
        metrics = calculator.calculate_metrics(trades, equity_curve, periods_per_year=252)

        # Create results
        results = BacktestResults(
            strategy_id="aapl_momentum_daily",
            strategy_type="momentum following",
            asset_class=AssetClass.EQUITY,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve
        )

        print_metrics_summary(metrics)

        # Analyze
        assistant = BacktestingAssistant()
        report = assistant.analyze(results)

        print(f"\nGrade: {report.grade}")

        return results

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("LLM BACKTESTING ASSISTANT - EXAMPLES")
    print("=" * 60)

    # Run example with sample data (always works)
    results_sample = example_with_sample_data()

    # Try with Bybit data
    print("\n")
    results_bybit = example_with_bybit_data()

    # Try with stock data
    print("\n")
    results_stock = example_with_stock_data()

    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETED")
    print("=" * 60)
    print("\nTo use with a real LLM, set the OPENAI_API_KEY environment variable")
    print("or pass an api_key to LlmClient(api_key='your-key')")
