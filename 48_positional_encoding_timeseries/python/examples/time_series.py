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

    ts_encoding = TimeSeriesSinusoidalEncoding(d_model)
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
    # Create datetime list for a week
    dt_list = [base_time + timedelta(hours=i) for i in range(168)]

    # Extract discrete calendar features
    dayofweek = torch.tensor([dt.weekday() for dt in dt_list]).unsqueeze(0)
    month = torch.tensor([dt.month - 1 for dt in dt_list]).unsqueeze(0)  # 0-indexed
    quarter = torch.tensor([(dt.month - 1) // 3 for dt in dt_list]).unsqueeze(0)
    hour = torch.tensor([dt.hour for dt in dt_list]).unsqueeze(0)
    session = torch.zeros_like(hour)  # placeholder for session

    cal_encoded = calendar(dayofweek, month, quarter, hour, session)

    print(f"Calendar encoding dimension: {d_model}")
    print(f"Hour tensor shape: {hour.shape}")
    print(f"Encoded shape: {cal_encoded.shape}")

    # Show features for specific times
    print("\nCalendar features for different times:")
    for idx, name in [(0, "Monday 00:00"), (24, "Tuesday 00:00"),
                       (120, "Saturday 00:00"), (144, "Sunday 00:00")]:
        dt = base_time + timedelta(hours=idx)
        is_weekend = 1 if dt.weekday() >= 5 else 0
        print(f"  {name}: weekend={is_weekend}, encoded[0,:4]={cal_encoded[0, idx, :4].detach().numpy().round(3)}")

    # 3. Market Session Encoding
    print("\n3. Market Session Encoding")
    print("-" * 40)

    # Crypto market (24/7)
    print("\n3a. Cryptocurrency (24/7)")
    session_crypto = MarketSessionEncoding(d_model, market_type='crypto')

    # Generate hours for a day (MarketSessionEncoding for crypto expects hour tensor)
    hours_24 = torch.arange(24).unsqueeze(0)

    session_encoded = session_crypto(hours_24)
    print(f"Session encoding dimension: {d_model}")

    print("\nSession activity by hour (crypto):")
    for h in [0, 6, 12, 18]:
        # Determine session based on UTC hour
        if h < 8:
            sess_name = "Asia"
        elif h < 16:
            sess_name = "Europe"
        else:
            sess_name = "Americas"
        print(f"  {h:02d}:00 UTC - {sess_name} session")

    # Stock market
    print("\n3b. Stock Market (US)")
    session_stock = MarketSessionEncoding(d_model, market_type='stock')
    # For stock market, we need session and time_in_session tensors
    est_hours = (hours_24 - 5) % 24  # Convert UTC to EST

    # Compute session indices: 0=pre, 1=regular, 2=after, 3=closed
    session_idx = torch.full_like(hours_24, 3)  # default closed
    session_idx[(est_hours >= 4) & (est_hours < 9)] = 0   # pre-market
    session_idx[(est_hours >= 9) & (est_hours < 16)] = 1  # regular
    session_idx[(est_hours >= 16) & (est_hours < 20)] = 2 # after-hours

    # Time in session (minutes since session start)
    time_in_session = torch.zeros_like(hours_24)
    # For regular session (9:30-16:00), compute time in session
    regular_mask = session_idx == 1
    time_in_session[regular_mask] = ((est_hours[regular_mask] - 9) * 60).clamp(min=0, max=99)

    stock_encoded = session_stock(hours_24, session_idx, time_in_session)

    print("Session activity by hour (stock, EST = UTC-5):")
    for h in [9, 14, 17, 21]:  # UTC hours
        est_h = (h - 5) % 24
        # Match the session_idx logic above (integer hour boundaries)
        if 4 <= est_h < 9:
            sess_name = "Pre-market"
        elif 9 <= est_h < 16:
            sess_name = "Regular"
        elif 16 <= est_h < 20:
            sess_name = "After-hours"
        else:
            sess_name = "Closed"
        print(f"  {h:02d}:00 UTC ({est_h:.0f}:00 EST) - {sess_name}")

    # 4. Multi-Scale Temporal Encoding
    print("\n4. Multi-Scale Temporal Encoding")
    print("-" * 40)

    multi_scale = MultiScaleTemporalEncoding(d_model)
    # MultiScaleTemporalEncoding expects a dict of scale tensors
    ts_dict = {
        'minute': torch.zeros_like(hours_24),
        'hour': hours_24,
        'day': torch.tensor([dt.day - 1 for dt in dt_list[:24]]).unsqueeze(0),
        'week': torch.tensor([dt.weekday() for dt in dt_list[:24]]).unsqueeze(0),
        'month': torch.tensor([dt.month - 1 for dt in dt_list[:24]]).unsqueeze(0),
    }
    multi_encoded = multi_scale(ts_dict)

    print(f"Combined multi-scale encoding")
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
