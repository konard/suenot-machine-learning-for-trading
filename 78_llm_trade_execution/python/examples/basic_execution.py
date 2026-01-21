#!/usr/bin/env python3
"""
Basic execution example demonstrating TWAP and VWAP strategies.

This example shows how to use the execution engine with traditional
execution algorithms without LLM integration.
"""

import asyncio
from llm_trade_execution import (
    ExecutionEngine,
    ExecutionConfig,
    ParentOrder,
    Side,
    TwapStrategy,
    VwapStrategy,
)


async def main():
    print("=== Basic Execution Example ===\n")

    # Create execution engine with default configuration
    config = ExecutionConfig(
        min_slice_interval_ms=100,  # Fast for demo
        max_slice_interval_ms=500,
        verbose=True,
    )

    engine = ExecutionEngine(config)

    # Example 1: TWAP Execution
    print("--- TWAP Execution ---")

    order = ParentOrder(
        symbol="BTCUSDT",
        side=Side.BUY,
        total_quantity=0.5,  # 0.5 BTC
        time_horizon=10,  # 10 seconds (fast for demo)
        urgency=0.5,
    )

    strategy = TwapStrategy(slice_interval_secs=2)

    result = await engine.execute(order, strategy)

    print("TWAP Execution Result:")
    print(f"  Symbol: {result.symbol}")
    print(f"  Side: {result.side.value}")
    print(f"  Total Quantity: {result.total_quantity:.4f}")
    print(f"  Filled Quantity: {result.filled_quantity:.4f}")
    print(f"  Child Orders: {result.child_order_count}")
    print(f"  Average Price: {result.average_price:.2f}")
    print(f"  Arrival Price: {result.arrival_price:.2f}")
    print(f"  Implementation Shortfall: {result.implementation_shortfall:.2f} bps")
    print(f"  VWAP Slippage: {result.vwap_slippage:.2f} bps")
    print(f"  Duration: {result.duration_secs} seconds")
    print()

    # Example 2: VWAP Execution
    print("--- VWAP Execution ---")

    engine2 = ExecutionEngine(config)

    order2 = ParentOrder(
        symbol="ETHUSDT",
        side=Side.SELL,
        total_quantity=2.0,  # 2 ETH
        time_horizon=10,
        urgency=0.3,
    )

    vwap_strategy = VwapStrategy(num_periods=5)

    result2 = await engine2.execute(order2, vwap_strategy)

    print("VWAP Execution Result:")
    print(f"  Symbol: {result2.symbol}")
    print(f"  Side: {result2.side.value}")
    print(f"  Total Quantity: {result2.total_quantity:.4f}")
    print(f"  Filled Quantity: {result2.filled_quantity:.4f}")
    print(f"  Child Orders: {result2.child_order_count}")
    print(f"  Average Price: {result2.average_price:.2f}")
    print(f"  Implementation Shortfall: {result2.implementation_shortfall:.2f} bps")
    print()

    # Example 3: Compare TWAP vs VWAP
    print("--- Strategy Comparison ---")
    print(f"TWAP IS: {result.implementation_shortfall:.2f} bps")
    print(f"VWAP IS: {result2.implementation_shortfall:.2f} bps")

    if abs(result.implementation_shortfall) < abs(result2.implementation_shortfall):
        print("TWAP performed better in this simulation.")
    else:
        print("VWAP performed better in this simulation.")

    print("\nNote: Results use simulated market data. Real execution will vary.")


if __name__ == "__main__":
    asyncio.run(main())
