#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive tests for utils.py module.

Tests cover:
- format_time: Time formatting utility function
- MultipleTimeSeriesCV: Time series cross-validation splitter
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import format_time, MultipleTimeSeriesCV


# =============================================================================
# Tests for format_time function
# =============================================================================

class TestFormatTime:
    """Tests for the format_time function."""

    def test_zero_seconds(self):
        """Test formatting of zero seconds."""
        assert format_time(0) == "00:00:00"

    def test_one_second(self):
        """Test formatting of one second."""
        assert format_time(1) == "00:00:01"

    def test_one_minute(self):
        """Test formatting of exactly one minute (60 seconds)."""
        assert format_time(60) == "00:01:00"

    def test_one_hour(self):
        """Test formatting of exactly one hour (3600 seconds)."""
        assert format_time(3600) == "01:00:00"

    def test_mixed_time(self):
        """Test formatting of a mixed time value."""
        # 1 hour, 30 minutes, 45 seconds = 3600 + 1800 + 45 = 5445
        assert format_time(5445) == "01:30:45"

    def test_multiple_hours(self):
        """Test formatting of times with multiple hours."""
        # 12 hours = 43200 seconds
        assert format_time(43200) == "12:00:00"

    def test_large_hour_value(self):
        """Test formatting of very large hour values (>99 hours)."""
        # 100 hours = 360000 seconds
        result = format_time(360000)
        assert result == "100:00:00"

    def test_24_hours(self):
        """Test formatting of exactly 24 hours."""
        # 24 hours = 86400 seconds
        assert format_time(86400) == "24:00:00"

    def test_floating_point_seconds(self):
        """Test formatting with floating point seconds (rounds down)."""
        # 90.5 seconds = 1 minute, 30.5 seconds -> should show 01:30
        result = format_time(90.5)
        assert result == "00:01:30"

    def test_floating_point_rounds_correctly(self):
        """Test that floating point seconds are formatted correctly.

        Note: The function uses f-string formatting with .0f which rounds
        to the nearest integer rather than truncating.
        """
        # 90.9 seconds = 1 minute 30.9 seconds -> rounds to 31
        result = format_time(90.9)
        assert result == "00:01:31"

    def test_small_fractional_seconds(self):
        """Test formatting of small fractional seconds.

        Note: The function rounds to the nearest integer, so 0.5 rounds
        to 0 (banker's rounding) and 0.9 rounds to 1.
        """
        assert format_time(0.4) == "00:00:00"
        assert format_time(0.9) == "00:00:01"

    def test_59_seconds(self):
        """Test boundary at 59 seconds."""
        assert format_time(59) == "00:00:59"

    def test_59_minutes(self):
        """Test boundary at 59 minutes."""
        # 59 minutes, 59 seconds = 3599 seconds
        assert format_time(3599) == "00:59:59"

    @pytest.mark.parametrize("seconds,expected", [
        (0, "00:00:00"),
        (1, "00:00:01"),
        (59, "00:00:59"),
        (60, "00:01:00"),
        (61, "00:01:01"),
        (3599, "00:59:59"),
        (3600, "01:00:00"),
        (3661, "01:01:01"),
        (7322, "02:02:02"),
        (86399, "23:59:59"),
        (86400, "24:00:00"),
    ])
    def test_parametrized_time_values(self, seconds, expected):
        """Parametrized test for various time values."""
        assert format_time(seconds) == expected

    def test_negative_time_handling(self):
        """Test behavior with negative time values.

        Note: The function doesn't explicitly handle negative values,
        so this documents the current behavior.
        """
        # This tests current behavior - negative values produce negative components
        result = format_time(-1)
        # -1 / 60 = -1 remainder -1, so s=-1, m=-1
        # The exact output depends on how divmod handles negatives
        assert isinstance(result, str)


# =============================================================================
# Tests for MultipleTimeSeriesCV class
# =============================================================================

class TestMultipleTimeSeriesCV:
    """Tests for the MultipleTimeSeriesCV cross-validation splitter."""

    @pytest.fixture
    def sample_data(self):
        """Create sample multi-indexed DataFrame for testing."""
        # Create date range
        dates = pd.date_range('2020-01-01', periods=252, freq='B')  # ~1 year of trading days
        symbols = ['AAPL', 'GOOGL', 'MSFT']

        # Create MultiIndex
        idx = pd.MultiIndex.from_product(
            [symbols, dates],
            names=['symbol', 'date']
        )

        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame(
            {
                'feature1': np.random.randn(len(idx)),
                'feature2': np.random.randn(len(idx)),
                'target': np.random.randn(len(idx))
            },
            index=idx
        )
        return data

    @pytest.fixture
    def small_data(self):
        """Create smaller dataset for detailed testing."""
        dates = pd.date_range('2020-01-01', periods=50, freq='B')
        symbols = ['AAPL', 'GOOGL']

        idx = pd.MultiIndex.from_product(
            [symbols, dates],
            names=['symbol', 'date']
        )

        np.random.seed(42)
        data = pd.DataFrame(
            {'feature': np.random.randn(len(idx))},
            index=idx
        )
        return data

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        cv = MultipleTimeSeriesCV()
        assert cv.n_splits == 3
        assert cv.train_length == 126
        assert cv.test_length == 21
        assert cv.lookahead is None
        assert cv.shuffle is False
        assert cv.date_idx == 'date'

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        cv = MultipleTimeSeriesCV(
            n_splits=5,
            train_period_length=100,
            test_period_length=10,
            lookahead=5,
            date_idx='trade_date',
            shuffle=True
        )
        assert cv.n_splits == 5
        assert cv.train_length == 100
        assert cv.test_length == 10
        assert cv.lookahead == 5
        assert cv.shuffle is True
        assert cv.date_idx == 'trade_date'

    def test_get_n_splits(self, sample_data):
        """Test get_n_splits returns correct number of splits."""
        cv = MultipleTimeSeriesCV(n_splits=5, lookahead=1)
        assert cv.get_n_splits(sample_data, None) == 5

    def test_split_returns_correct_number_of_folds(self, sample_data):
        """Test that split generates correct number of train/test folds."""
        cv = MultipleTimeSeriesCV(
            n_splits=3,
            train_period_length=50,
            test_period_length=10,
            lookahead=1
        )

        folds = list(cv.split(sample_data))
        assert len(folds) == 3

    def test_split_returns_numpy_arrays(self, sample_data):
        """Test that split returns numpy arrays for indices."""
        cv = MultipleTimeSeriesCV(
            n_splits=2,
            train_period_length=50,
            test_period_length=10,
            lookahead=1
        )

        for train_idx, test_idx in cv.split(sample_data):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)

    def test_train_test_indices_non_empty(self, sample_data):
        """Test that train and test indices are non-empty."""
        cv = MultipleTimeSeriesCV(
            n_splits=2,
            train_period_length=50,
            test_period_length=10,
            lookahead=1
        )

        for train_idx, test_idx in cv.split(sample_data):
            assert len(train_idx) > 0, "Train indices should not be empty"
            assert len(test_idx) > 0, "Test indices should not be empty"

    def test_train_test_no_overlap(self, sample_data):
        """Test that train and test indices do not overlap."""
        cv = MultipleTimeSeriesCV(
            n_splits=3,
            train_period_length=50,
            test_period_length=10,
            lookahead=1
        )

        for train_idx, test_idx in cv.split(sample_data):
            train_set = set(train_idx)
            test_set = set(test_idx)
            overlap = train_set.intersection(test_set)
            assert len(overlap) == 0, f"Train and test should not overlap: {overlap}"

    def test_train_dates_before_test_dates(self, sample_data):
        """Test that training dates come before test dates."""
        cv = MultipleTimeSeriesCV(
            n_splits=2,
            train_period_length=50,
            test_period_length=10,
            lookahead=1
        )

        dates = sample_data.reset_index()['date']

        for train_idx, test_idx in cv.split(sample_data):
            max_train_date = dates.iloc[train_idx].max()
            min_test_date = dates.iloc[test_idx].min()
            assert max_train_date < min_test_date, \
                f"Max train date {max_train_date} should be before min test date {min_test_date}"

    def test_lookahead_gap_between_train_and_test(self, sample_data):
        """Test that lookahead creates proper gap between train and test."""
        lookahead = 5
        cv = MultipleTimeSeriesCV(
            n_splits=2,
            train_period_length=50,
            test_period_length=10,
            lookahead=lookahead
        )

        dates = sample_data.reset_index()['date']
        unique_dates = sorted(dates.unique())

        for train_idx, test_idx in cv.split(sample_data):
            max_train_date = dates.iloc[train_idx].max()
            min_test_date = dates.iloc[test_idx].min()

            # Find the position of these dates
            train_pos = unique_dates.index(max_train_date)
            test_pos = unique_dates.index(min_test_date)

            # There should be a gap due to lookahead
            gap = test_pos - train_pos
            assert gap >= lookahead, \
                f"Gap ({gap}) should be at least lookahead ({lookahead})"

    def test_multiple_symbols_included(self, sample_data):
        """Test that splits include data from multiple symbols."""
        cv = MultipleTimeSeriesCV(
            n_splits=2,
            train_period_length=50,
            test_period_length=10,
            lookahead=1
        )

        symbols = sample_data.reset_index()['symbol']

        for train_idx, test_idx in cv.split(sample_data):
            train_symbols = symbols.iloc[train_idx].unique()
            test_symbols = symbols.iloc[test_idx].unique()

            # Both train and test should have multiple symbols
            assert len(train_symbols) > 1, "Train should include multiple symbols"
            assert len(test_symbols) > 1, "Test should include multiple symbols"

    def test_consistent_results_without_shuffle(self, sample_data):
        """Test that results are consistent when shuffle is False."""
        cv = MultipleTimeSeriesCV(
            n_splits=2,
            train_period_length=50,
            test_period_length=10,
            lookahead=1,
            shuffle=False
        )

        # Run split twice
        folds1 = [(train.copy(), test.copy()) for train, test in cv.split(sample_data)]
        folds2 = [(train.copy(), test.copy()) for train, test in cv.split(sample_data)]

        # Compare results
        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

    def test_train_length_respected(self, sample_data):
        """Test that training period length is approximately correct."""
        train_length = 50
        cv = MultipleTimeSeriesCV(
            n_splits=2,
            train_period_length=train_length,
            test_period_length=10,
            lookahead=1
        )

        dates = sample_data.reset_index()['date']
        n_symbols = len(sample_data.index.get_level_values('symbol').unique())

        for train_idx, _ in cv.split(sample_data):
            train_dates = dates.iloc[train_idx].unique()
            # The number of unique dates should be approximately train_length
            # (might be slightly less due to boundary conditions)
            assert len(train_dates) <= train_length + 1
            assert len(train_dates) >= train_length - 5  # Allow some tolerance

    def test_test_length_respected(self, sample_data):
        """Test that test period length is approximately correct."""
        test_length = 10
        cv = MultipleTimeSeriesCV(
            n_splits=2,
            train_period_length=50,
            test_period_length=test_length,
            lookahead=1
        )

        dates = sample_data.reset_index()['date']

        for _, test_idx in cv.split(sample_data):
            test_dates = dates.iloc[test_idx].unique()
            # The number of unique dates should be approximately test_length
            assert len(test_dates) <= test_length + 1
            assert len(test_dates) >= test_length - 1  # Allow some tolerance

    def test_folds_are_chronologically_ordered(self, sample_data):
        """Test that later folds use more recent data."""
        cv = MultipleTimeSeriesCV(
            n_splits=3,
            train_period_length=40,
            test_period_length=10,
            lookahead=1
        )

        dates = sample_data.reset_index()['date']

        prev_test_end = None
        for _, test_idx in cv.split(sample_data):
            test_end = dates.iloc[test_idx].max()

            if prev_test_end is not None:
                # Each subsequent fold should have earlier test data
                # (because splits are generated from end to beginning)
                assert test_end < prev_test_end, \
                    "Test periods should move backwards in time"
            prev_test_end = test_end

    def test_custom_date_column_name(self):
        """Test using a custom date column name."""
        dates = pd.date_range('2020-01-01', periods=100, freq='B')
        symbols = ['A', 'B']

        idx = pd.MultiIndex.from_product(
            [symbols, dates],
            names=['symbol', 'trade_date']  # Custom date column name
        )

        data = pd.DataFrame({'feature': np.random.randn(len(idx))}, index=idx)

        cv = MultipleTimeSeriesCV(
            n_splits=2,
            train_period_length=30,
            test_period_length=10,
            lookahead=1,
            date_idx='trade_date'  # Use custom name
        )

        folds = list(cv.split(data))
        assert len(folds) == 2

        for train_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_single_split(self, sample_data):
        """Test with n_splits=1."""
        cv = MultipleTimeSeriesCV(
            n_splits=1,
            train_period_length=100,
            test_period_length=20,
            lookahead=1
        )

        folds = list(cv.split(sample_data))
        assert len(folds) == 1

    def test_many_splits(self, sample_data):
        """Test with many splits."""
        cv = MultipleTimeSeriesCV(
            n_splits=10,
            train_period_length=20,
            test_period_length=5,
            lookahead=1
        )

        folds = list(cv.split(sample_data))
        assert len(folds) == 10

    def test_sklearn_interface_compatibility(self, sample_data):
        """Test that the interface is compatible with sklearn expectations."""
        cv = MultipleTimeSeriesCV(
            n_splits=3,
            train_period_length=50,
            test_period_length=10,
            lookahead=1
        )

        # Test that it can be iterated like sklearn splitters
        X = sample_data[['feature1', 'feature2']]
        y = sample_data['target']

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            # Verify indices can be used to slice data
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            assert len(X_train) == len(y_train)
            assert len(X_test) == len(y_test)

    def test_large_lookahead(self, sample_data):
        """Test with a larger lookahead value."""
        cv = MultipleTimeSeriesCV(
            n_splits=2,
            train_period_length=50,
            test_period_length=10,
            lookahead=20  # Large lookahead
        )

        dates = sample_data.reset_index()['date']
        unique_dates = sorted(dates.unique())

        for train_idx, test_idx in cv.split(sample_data):
            max_train_date = dates.iloc[train_idx].max()
            min_test_date = dates.iloc[test_idx].min()

            train_pos = unique_dates.index(max_train_date)
            test_pos = unique_dates.index(min_test_date)

            gap = test_pos - train_pos
            assert gap >= 20, f"Gap ({gap}) should be at least lookahead (20)"


class TestMultipleTimeSeriesCVEdgeCases:
    """Edge case tests for MultipleTimeSeriesCV."""

    def test_single_symbol(self):
        """Test with only a single symbol."""
        dates = pd.date_range('2020-01-01', periods=200, freq='B')

        idx = pd.MultiIndex.from_product(
            [['AAPL'], dates],
            names=['symbol', 'date']
        )

        data = pd.DataFrame({'feature': np.random.randn(len(idx))}, index=idx)

        cv = MultipleTimeSeriesCV(
            n_splits=2,
            train_period_length=50,
            test_period_length=10,
            lookahead=1
        )

        folds = list(cv.split(data))
        assert len(folds) == 2

        for train_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_equal_train_test_length(self):
        """Test when train and test periods are equal length."""
        dates = pd.date_range('2020-01-01', periods=200, freq='B')
        symbols = ['A', 'B']

        idx = pd.MultiIndex.from_product([symbols, dates], names=['symbol', 'date'])
        data = pd.DataFrame({'feature': np.random.randn(len(idx))}, index=idx)

        cv = MultipleTimeSeriesCV(
            n_splits=2,
            train_period_length=30,
            test_period_length=30,  # Same as train
            lookahead=1
        )

        folds = list(cv.split(data))
        assert len(folds) == 2

    def test_minimum_lookahead(self):
        """Test with lookahead=1 (minimum practical value)."""
        dates = pd.date_range('2020-01-01', periods=100, freq='B')
        symbols = ['A', 'B']

        idx = pd.MultiIndex.from_product([symbols, dates], names=['symbol', 'date'])
        data = pd.DataFrame({'feature': np.random.randn(len(idx))}, index=idx)

        cv = MultipleTimeSeriesCV(
            n_splits=2,
            train_period_length=30,
            test_period_length=10,
            lookahead=1
        )

        folds = list(cv.split(data))
        assert len(folds) == 2


class TestMultipleTimeSeriesCVIntegration:
    """Integration tests for MultipleTimeSeriesCV with typical ML workflows."""

    def test_with_model_training_simulation(self):
        """Simulate a typical ML cross-validation workflow."""
        # Create realistic trading data
        np.random.seed(42)
        dates = pd.date_range('2019-01-01', periods=504, freq='B')  # ~2 years
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

        idx = pd.MultiIndex.from_product([symbols, dates], names=['symbol', 'date'])

        # Features and target
        n = len(idx)
        data = pd.DataFrame({
            'returns': np.random.randn(n) * 0.02,
            'volume': np.random.exponential(1000000, n),
            'volatility': np.random.exponential(0.02, n),
            'momentum': np.random.randn(n) * 0.1,
            'target': np.random.randn(n) * 0.02
        }, index=idx)

        # Setup CV with realistic parameters
        cv = MultipleTimeSeriesCV(
            n_splits=5,
            train_period_length=252,  # ~1 year
            test_period_length=21,     # ~1 month
            lookahead=5
        )

        X = data.drop('target', axis=1)
        y = data['target']

        # Simulate cross-validation
        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Verify data integrity
            assert not X_train.empty
            assert not X_test.empty
            assert len(X_train) == len(y_train)
            assert len(X_test) == len(y_test)

            # Simulate model prediction (just mean for testing)
            predictions = np.full(len(y_test), y_train.mean())
            mse = ((y_test - predictions) ** 2).mean()
            fold_results.append(mse)

        assert len(fold_results) == 5
        # All MSE values should be reasonable
        assert all(mse >= 0 for mse in fold_results)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
