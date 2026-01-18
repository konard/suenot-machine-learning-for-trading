#!/usr/bin/env python3
"""
Time Series Encoding Example

This example demonstrates temporal encodings specifically designed
for financial time series data, including calendar and market session features.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from datetime import datetime, timedelta
from positional_encoding import (
    TimeSeriesSinusoidalEncoding,
    CalendarEncoding,
    MarketSessionEncoding,
    MultiScaleTemporalEncoding,
)


def main():
    print("Time Series Encoding Example")
    print("=" * 50)

    d_model = 64
    batch_size = 2
    seq_len = 24  # 24 hours

    # 1. Time Series Sinusoidal Encoding
    print("\n1. Multi-Scale Sinusoidal Encoding")
    print("-" * 40)

    ts_encoding = TimeSeriesSinusoidalEncoding(d_model, max_len=1000)
    x = torch.zeros(batch_size, seq_len, d_model)
    encoded = ts_encoding(x)

    print(f"Scales: 1h, 24h (daily), 168h (weekly), 720h (monthly)")
    print(f"Dimension: {d_model}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {encoded.shape}")

    # 2. Calendar Encoding
    print("\n2. Calendar Encoding")
    print("-" * 40)

    calendar = CalendarEncoding(d_model)

    # Generate timestamps for a week
    base_time = datetime(2024, 1, 1, 0, 0)  # Monday
    timestamps = torch.tensor([
        int((base_time + timedelta(hours=i)).timestamp())
        for i in range(168)  # One week
    ]).unsqueeze(0)  # (1, 168)

    cal_encoded = calendar(timestamps)

    print(f"Calendar encoding dimension: {d_model}")
    print(f"Timestamps shape: {timestamps.shape}")
    print(f"Encoded shape: {cal_encoded.shape}")

    # Show features for specific times
    print("\nCalendar features for different times:")
    for idx, name in [(0, "Monday 00:00"), (24, "Tuesday 00:00"),
                       (120, "Saturday 00:00"), (144, "Sunday 00:00")]:
        dt = base_time + timedelta(hours=idx)
        is_weekend = 1 if dt.weekday() >= 5 else 0
        print(f"  {name}: weekend={is_weekend}, encoded[0,:4]={cal_encoded[0, idx, :4].numpy().round(3)}")

    # 3. Market Session Encoding
    print("\n3. Market Session Encoding")
    print("-" * 40)

    # Crypto market (24/7)
    print("\n3a. Cryptocurrency (24/7)")
    session_crypto = MarketSessionEncoding(d_model, market_type='crypto')

    # Generate hourly timestamps for a day
    day_timestamps = torch.tensor([
        int((base_time + timedelta(hours=i)).timestamp())
        for i in range(24)
    ]).unsqueeze(0)

    session_encoded = session_crypto(day_timestamps)
    print(f"Session encoding dimension: {d_model}")

    print("\nSession activity by hour (crypto):")
    for hour in [0, 6, 12, 18]:
        dt = base_time + timedelta(hours=hour)
        # Determine session based on UTC hour
        if hour < 8:
            session = "Asia"
        elif hour < 16:
            session = "Europe"
        else:
            session = "Americas"
        print(f"  {hour:02d}:00 UTC - {session} session")

    # Stock market
    print("\n3b. Stock Market (US)")
    session_stock = MarketSessionEncoding(d_model, market_type='stock')
    stock_encoded = session_stock(day_timestamps)

    print("Session activity by hour (stock, EST = UTC-5):")
    for hour in [9, 14, 17, 21]:  # UTC hours
        dt = base_time + timedelta(hours=hour)
        est_hour = (hour - 5) % 24
        if 4 <= est_hour < 9.5:
            session = "Pre-market"
        elif 9.5 <= est_hour < 16:
            session = "Regular"
        elif 16 <= est_hour < 20:
            session = "After-hours"
        else:
            session = "Closed"
        print(f"  {hour:02d}:00 UTC ({est_hour:.0f}:00 EST) - {session}")

    # 4. Multi-Scale Temporal Encoding
    print("\n4. Multi-Scale Temporal Encoding")
    print("-" * 40)

    multi_scale = MultiScaleTemporalEncoding(d_model, market_type='crypto')
    multi_encoded = multi_scale(day_timestamps)

    print(f"Combined calendar + session encoding")
    print(f"Total dimension: {d_model}")
    print(f"Encoded shape: {multi_encoded.shape}")

    # 5. Practical Application
    print("\n5. Practical Application")
    print("-" * 40)

    print("\nRecommended encoding combinations by use case:")
    print()
    print("Intraday Trading (1-min to 1-hour):")
    print("  - TimeSeriesSinusoidal (multi-scale patterns)")
    print("  - MarketSession (session awareness)")
    print("  - RoPE (sequence position)")
    print()
    print("Swing Trading (4-hour to daily):")
    print("  - Calendar (day of week effects)")
    print("  - TimeSeriesSinusoidal (weekly/monthly patterns)")
    print("  - Sinusoidal (sequence position)")
    print()
    print("Long-term Investment (daily to monthly):")
    print("  - Calendar (month/quarter effects)")
    print("  - Learned (let model discover patterns)")
    print()
    print("Crypto 24/7:")
    print("  - MarketSession('crypto') - regional sessions")
    print("  - MultiScaleTemporalEncoding")
    print("  - RoPE (handles variable sequence lengths)")


if __name__ == "__main__":
    main()
