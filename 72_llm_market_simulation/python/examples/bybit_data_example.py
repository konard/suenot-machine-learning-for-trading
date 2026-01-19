#!/usr/bin/env python3
"""
Bybit Data Integration Example

Demonstrates fetching real cryptocurrency data from Bybit exchange
and using it to initialize a market simulation with realistic prices.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data import BybitDataFetcher
from market import OrderBook, MarketEnvironment
from agents import ValueInvestorAgent, MomentumTraderAgent, MarketMakerAgent
from simulation import SimulationEngine, calculate_performance_metrics


def main():
    """Fetch Bybit data and run simulation"""
    print("=" * 60)
    print("LLM Market Simulation - Bybit Data Integration")
    print("=" * 60)

    # Initialize Bybit data fetcher
    fetcher = BybitDataFetcher()

    # Fetch data for BTCUSDT
    symbol = "BTCUSDT"
    print(f"\nFetching data for {symbol}...")

    try:
        # Get current ticker info
        ticker = fetcher.get_ticker(symbol)
        current_price = ticker.get("last_price", 0)

        print(f"\nCurrent Market Data:")
        print(f"  Symbol: {symbol}")
        print(f"  Last Price: ${current_price:,.2f}")
        print(f"  Bid: ${ticker.get('bid_price', 0):,.2f}")
        print(f"  Ask: ${ticker.get('ask_price', 0):,.2f}")
        print(f"  24h High: ${ticker.get('high_24h', 0):,.2f}")
        print(f"  24h Low: ${ticker.get('low_24h', 0):,.2f}")
        print(f"  24h Change: {ticker.get('price_change_24h', 0):.2f}%")
        print(f"  24h Volume: ${ticker.get('turnover_24h', 0):,.0f}")

        # Get historical prices
        print(f"\nFetching 30-day price history...")
        historical_prices = fetcher.fetch_price_history(symbol, days=30, interval="D")
        print(f"  Retrieved {len(historical_prices)} daily prices")

        if historical_prices:
            price_change = (historical_prices[-1] / historical_prices[0] - 1) * 100
            volatility = np.std(np.diff(historical_prices) / historical_prices[:-1]) * np.sqrt(365) * 100
            print(f"  30-day return: {price_change:.2f}%")
            print(f"  Annualized volatility: {volatility:.2f}%")

        # Get order book
        print(f"\nFetching order book...")
        orderbook = fetcher.get_orderbook(symbol, limit=10)
        print(f"  Top 5 bids:")
        for i, (price, qty) in enumerate(orderbook.bids[:5]):
            print(f"    {i+1}. ${price:,.2f} x {qty:.4f}")
        print(f"  Top 5 asks:")
        for i, (price, qty) in enumerate(orderbook.asks[:5]):
            print(f"    {i+1}. ${price:,.2f} x {qty:.4f}")

        if orderbook.bids and orderbook.asks:
            spread = orderbook.asks[0][0] - orderbook.bids[0][0]
            spread_pct = spread / orderbook.bids[0][0] * 100
            print(f"  Spread: ${spread:.2f} ({spread_pct:.4f}%)")

        # Get recent trades
        print(f"\nFetching recent trades...")
        trades = fetcher.get_recent_trades(symbol, limit=10)
        print(f"  Last 5 trades:")
        for trade in trades[:5]:
            side = "BUY " if trade["side"] == "buy" else "SELL"
            print(f"    {side} ${trade['price']:,.2f} x {trade['quantity']:.4f}")

    except Exception as e:
        print(f"\nError fetching Bybit data: {e}")
        print("Using default simulation parameters instead...")
        current_price = 50000.0
        historical_prices = []

    # Now run a simulation initialized with real market data
    print("\n" + "=" * 60)
    print("RUNNING SIMULATION WITH MARKET DATA")
    print("=" * 60)

    # Use current price scaled down for simulation (for easier numbers)
    sim_price = 100.0  # Normalized for simulation
    scale_factor = current_price / sim_price if current_price > 0 else 1

    # Estimate fundamental value (use 20-day moving average if available)
    if len(historical_prices) >= 20:
        fundamental = np.mean(historical_prices[-20:]) / scale_factor
    else:
        fundamental = sim_price

    print(f"\nSimulation Parameters:")
    print(f"  Initial Price: ${sim_price:.2f} (scaled from ${current_price:,.2f})")
    print(f"  Fundamental Value: ${fundamental:.2f}")
    print(f"  Scale Factor: {scale_factor:,.2f}")

    # Create simulation
    engine = SimulationEngine(
        initial_price=sim_price,
        fundamental_value=fundamental,
        volatility=0.03  # Crypto is more volatile
    )

    # Add agents
    initial_cash = 100000.0
    initial_shares = 50

    # Value investors
    for i in range(2):
        engine.add_agent(ValueInvestorAgent(
            agent_id=f"value_{i+1}",
            initial_cash=initial_cash,
            initial_shares=initial_shares,
            fundamental_value=fundamental,
            discount_threshold=0.05
        ))

    # Momentum traders (more aggressive for crypto)
    for i in range(3):
        engine.add_agent(MomentumTraderAgent(
            agent_id=f"momentum_{i+1}",
            initial_cash=initial_cash,
            initial_shares=initial_shares,
            short_window=3,  # Faster signals for crypto
            long_window=10
        ))

    # Market makers
    for i in range(2):
        engine.add_agent(MarketMakerAgent(
            agent_id=f"mm_{i+1}",
            initial_cash=initial_cash * 2,
            initial_shares=initial_shares * 2,
            base_spread=0.003  # Wider spread for crypto volatility
        ))

    # Run simulation
    num_steps = 200
    print(f"\nRunning {num_steps} step simulation...")
    result = engine.run(num_steps=num_steps, verbose=True)

    # Results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    print(f"\nPrice Evolution (Scaled):")
    print(f"  Start: ${result.price_history[0]:.2f} (${result.price_history[0] * scale_factor:,.2f} real)")
    print(f"  End: ${result.price_history[-1]:.2f} (${result.price_history[-1] * scale_factor:,.2f} real)")

    metrics = calculate_performance_metrics(result.price_history, result.fundamental_history)
    print(f"\nMetrics:")
    print(f"  Return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"  Volatility: {metrics.get('volatility_pct', 0):.2f}%")
    print(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")

    print(f"\nTotal Trades: {result.total_trades}")

    print("\nAgent Final Values:")
    for agent_id, agent_result in result.agent_results.items():
        final_val = agent_result.get("final_value", 0)
        # Scale back to real dollars
        real_val = final_val * scale_factor
        print(f"  {agent_id}: ${final_val:,.0f} (${real_val:,.0f} in real terms)")

    print("\nDone!")


if __name__ == "__main__":
    main()
