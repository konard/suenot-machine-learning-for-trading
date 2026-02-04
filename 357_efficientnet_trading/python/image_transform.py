"""
Image Transformation Module for EfficientNet Trading

This module provides utilities for converting time series data into
2D image representations suitable for EfficientNet models.

Transformations include:
- Gramian Angular Summation Field (GASF)
- Gramian Angular Difference Field (GADF)
- Markov Transition Field (MTF)
- Recurrence Plot (RP)
- Spectrogram (STFT-based)
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class GAFTransformer:
    """
    Gramian Angular Field Transformer.

    Converts time series to polar coordinates and creates
    angular field images (GASF and GADF).

    Example:
        >>> transformer = GAFTransformer(image_size=224)
        >>> prices = np.array([100, 105, 103, 110, 108])
        >>> gasf = transformer.transform(prices, field_type='summation')
        >>> gadf = transformer.transform(prices, field_type='difference')
    """

    def __init__(self, image_size: int = 224):
        """
        Initialize GAF transformer.

        Args:
            image_size: Output image size (will be image_size x image_size)
        """
        self.image_size = image_size

    def transform(
        self,
        series: np.ndarray,
        field_type: str = "summation",
    ) -> np.ndarray:
        """
        Transform time series to Gramian Angular Field.

        Args:
            series: 1D array of values (e.g., close prices)
            field_type: 'summation' for GASF, 'difference' for GADF

        Returns:
            2D array of shape (image_size, image_size)
        """
        # Normalize to [-1, 1]
        min_val = series.min()
        max_val = series.max()

        if max_val - min_val < 1e-10:
            # Avoid division by zero for constant series
            scaled = np.zeros_like(series)
        else:
            scaled = 2 * (series - min_val) / (max_val - min_val) - 1

        # Clip to avoid numerical issues with arccos
        scaled = np.clip(scaled, -1, 1)

        # Convert to angular representation
        phi = np.arccos(scaled)

        # Create Gramian matrix
        if field_type == "summation":
            # GASF: cos(phi_i + phi_j)
            gaf = np.cos(np.add.outer(phi, phi))
        else:
            # GADF: sin(phi_i - phi_j)
            gaf = np.sin(np.subtract.outer(phi, phi))

        # Resize to target image size
        gaf_resized = self._resize(gaf, self.image_size)

        return gaf_resized

    def transform_gasf(self, series: np.ndarray) -> np.ndarray:
        """Transform to Gramian Angular Summation Field."""
        return self.transform(series, field_type="summation")

    def transform_gadf(self, series: np.ndarray) -> np.ndarray:
        """Transform to Gramian Angular Difference Field."""
        return self.transform(series, field_type="difference")

    def _resize(self, image: np.ndarray, size: int) -> np.ndarray:
        """Resize image using bilinear interpolation."""
        h, w = image.shape
        if h == size and w == size:
            return image

        # Simple bilinear interpolation
        new_image = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                # Map to original coordinates
                orig_i = i * (h - 1) / (size - 1) if size > 1 else 0
                orig_j = j * (w - 1) / (size - 1) if size > 1 else 0

                # Bilinear interpolation
                i0, i1 = int(orig_i), min(int(orig_i) + 1, h - 1)
                j0, j1 = int(orig_j), min(int(orig_j) + 1, w - 1)

                di = orig_i - i0
                dj = orig_j - j0

                new_image[i, j] = (
                    image[i0, j0] * (1 - di) * (1 - dj) +
                    image[i1, j0] * di * (1 - dj) +
                    image[i0, j1] * (1 - di) * dj +
                    image[i1, j1] * di * dj
                )

        return new_image


class MTFTransformer:
    """
    Markov Transition Field Transformer.

    Encodes transition dynamics between quantile bins.

    Example:
        >>> transformer = MTFTransformer(n_bins=8, image_size=224)
        >>> prices = np.array([100, 105, 103, 110, 108])
        >>> mtf = transformer.transform(prices)
    """

    def __init__(self, n_bins: int = 8, image_size: int = 224):
        """
        Initialize MTF transformer.

        Args:
            n_bins: Number of quantile bins
            image_size: Output image size
        """
        self.n_bins = n_bins
        self.image_size = image_size

    def transform(self, series: np.ndarray) -> np.ndarray:
        """
        Transform time series to Markov Transition Field.

        Args:
            series: 1D array of values

        Returns:
            2D array of shape (image_size, image_size)
        """
        n = len(series)

        # Discretize into quantile bins
        bins = np.percentile(series, np.linspace(0, 100, self.n_bins + 1))
        bins[-1] += 1e-10  # Ensure max value is included
        digitized = np.digitize(series, bins[:-1]) - 1
        digitized = np.clip(digitized, 0, self.n_bins - 1)

        # Build transition matrix
        trans_matrix = np.zeros((self.n_bins, self.n_bins))
        for i in range(len(digitized) - 1):
            trans_matrix[digitized[i], digitized[i + 1]] += 1

        # Normalize rows (handle rows with zero sum)
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        trans_matrix = trans_matrix / row_sums

        # Create MTF
        mtf = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mtf[i, j] = trans_matrix[digitized[i], digitized[j]]

        # Resize to target image size
        mtf_resized = self._resize(mtf, self.image_size)

        return mtf_resized

    def _resize(self, image: np.ndarray, size: int) -> np.ndarray:
        """Resize image using bilinear interpolation."""
        h, w = image.shape
        if h == size and w == size:
            return image

        new_image = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                orig_i = i * (h - 1) / (size - 1) if size > 1 else 0
                orig_j = j * (w - 1) / (size - 1) if size > 1 else 0

                i0, i1 = int(orig_i), min(int(orig_i) + 1, h - 1)
                j0, j1 = int(orig_j), min(int(orig_j) + 1, w - 1)

                di = orig_i - i0
                dj = orig_j - j0

                new_image[i, j] = (
                    image[i0, j0] * (1 - di) * (1 - dj) +
                    image[i1, j0] * di * (1 - dj) +
                    image[i0, j1] * (1 - di) * dj +
                    image[i1, j1] * di * dj
                )

        return new_image


class RecurrencePlotTransformer:
    """
    Recurrence Plot Transformer.

    Shows recurring patterns in phase space.

    Example:
        >>> transformer = RecurrencePlotTransformer(threshold=0.1, image_size=224)
        >>> prices = np.array([100, 105, 103, 110, 108])
        >>> rp = transformer.transform(prices)
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        percentage: float = 10.0,
        image_size: int = 224,
    ):
        """
        Initialize Recurrence Plot transformer.

        Args:
            threshold: Fixed distance threshold (if None, use percentage)
            percentage: Percentage of max distance for threshold
            image_size: Output image size
        """
        self.threshold = threshold
        self.percentage = percentage
        self.image_size = image_size

    def transform(self, series: np.ndarray) -> np.ndarray:
        """
        Transform time series to Recurrence Plot.

        Args:
            series: 1D array of values

        Returns:
            2D array of shape (image_size, image_size)
        """
        n = len(series)

        # Compute distance matrix
        distance_matrix = np.abs(np.subtract.outer(series, series))

        # Determine threshold
        if self.threshold is None:
            threshold = self.percentage / 100 * distance_matrix.max()
        else:
            threshold = self.threshold

        # Create recurrence plot (1 if distance < threshold, else 0)
        rp = (distance_matrix < threshold).astype(float)

        # Resize to target image size
        rp_resized = self._resize(rp, self.image_size)

        return rp_resized

    def _resize(self, image: np.ndarray, size: int) -> np.ndarray:
        """Resize using nearest neighbor for binary images."""
        h, w = image.shape
        if h == size and w == size:
            return image

        new_image = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                orig_i = int(i * h / size)
                orig_j = int(j * w / size)
                new_image[i, j] = image[orig_i, orig_j]

        return new_image


class SpectrogramTransformer:
    """
    Spectrogram Transformer using Short-Time Fourier Transform.

    Creates time-frequency representation of the signal.

    Example:
        >>> transformer = SpectrogramTransformer(n_fft=64, hop_length=16, image_size=224)
        >>> prices = np.array([...])  # 256 prices
        >>> spec = transformer.transform(prices)
    """

    def __init__(
        self,
        n_fft: int = 64,
        hop_length: int = 16,
        image_size: int = 224,
    ):
        """
        Initialize Spectrogram transformer.

        Args:
            n_fft: FFT window size
            hop_length: Number of samples between STFT columns
            image_size: Output image size
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.image_size = image_size

    def transform(self, series: np.ndarray) -> np.ndarray:
        """
        Transform time series to Spectrogram.

        Args:
            series: 1D array of values

        Returns:
            2D array of shape (image_size, image_size)
        """
        # Normalize the series
        series = (series - series.mean()) / (series.std() + 1e-10)

        # Pad if necessary
        if len(series) < self.n_fft:
            series = np.pad(series, (0, self.n_fft - len(series)))

        # Compute STFT manually
        n_frames = 1 + (len(series) - self.n_fft) // self.hop_length
        n_frames = max(1, n_frames)

        # Hann window
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.n_fft) / self.n_fft))

        spectrogram = np.zeros((self.n_fft // 2 + 1, n_frames))

        for i in range(n_frames):
            start = i * self.hop_length
            frame = series[start:start + self.n_fft] * window
            fft = np.fft.rfft(frame)
            spectrogram[:, i] = np.abs(fft)

        # Convert to log scale
        spectrogram = np.log1p(spectrogram)

        # Normalize to [0, 1]
        if spectrogram.max() > spectrogram.min():
            spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

        # Resize to target image size
        spec_resized = self._resize(spectrogram, self.image_size)

        return spec_resized

    def _resize(self, image: np.ndarray, size: int) -> np.ndarray:
        """Resize image using bilinear interpolation."""
        h, w = image.shape
        if h == size and w == size:
            return image

        new_image = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                orig_i = i * (h - 1) / (size - 1) if size > 1 else 0
                orig_j = j * (w - 1) / (size - 1) if size > 1 else 0

                i0, i1 = int(orig_i), min(int(orig_i) + 1, h - 1)
                j0, j1 = int(orig_j), min(int(orig_j) + 1, w - 1)

                di = orig_i - i0
                dj = orig_j - j0

                new_image[i, j] = (
                    image[i0, j0] * (1 - di) * (1 - dj) +
                    image[i1, j0] * di * (1 - dj) +
                    image[i0, j1] * (1 - di) * dj +
                    image[i1, j1] * di * dj
                )

        return new_image


class ImageStackGenerator:
    """
    Generate multi-channel image stacks for EfficientNet.

    Combines multiple timeframes and transformation types
    into RGB (or multi-channel) images.

    Example:
        >>> generator = ImageStackGenerator(image_size=224)
        >>> multi_tf_data = {
        ...     '1m': df_1m['close'].values,
        ...     '5m': df_5m['close'].values,
        ...     '15m': df_15m['close'].values,
        ... }
        >>> image = generator.create_timeframe_stack(multi_tf_data)
        >>> print(image.shape)  # (224, 224, 3)
    """

    def __init__(
        self,
        image_size: int = 224,
        gaf_transformer: Optional[GAFTransformer] = None,
        mtf_transformer: Optional[MTFTransformer] = None,
    ):
        """
        Initialize Image Stack Generator.

        Args:
            image_size: Output image size
            gaf_transformer: Custom GAF transformer (optional)
            mtf_transformer: Custom MTF transformer (optional)
        """
        self.image_size = image_size
        self.gaf = gaf_transformer or GAFTransformer(image_size)
        self.mtf = mtf_transformer or MTFTransformer(image_size=image_size)

    def create_timeframe_stack(
        self,
        timeframe_data: Dict[str, np.ndarray],
        transform_type: str = "gasf",
    ) -> np.ndarray:
        """
        Create RGB image from multiple timeframes.

        Args:
            timeframe_data: Dictionary mapping timeframe to price data
                           e.g., {'1m': prices_1m, '5m': prices_5m, '15m': prices_15m}
            transform_type: 'gasf', 'gadf', or 'mtf'

        Returns:
            3D array of shape (image_size, image_size, 3)
        """
        channels = []

        for tf_name, prices in timeframe_data.items():
            if transform_type == "gasf":
                channel = self.gaf.transform_gasf(prices)
            elif transform_type == "gadf":
                channel = self.gaf.transform_gadf(prices)
            elif transform_type == "mtf":
                channel = self.mtf.transform(prices)
            else:
                raise ValueError(f"Unknown transform type: {transform_type}")

            # Normalize to [0, 1]
            channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-10)
            channels.append(channel)

        # Stack channels
        if len(channels) == 1:
            # Grayscale to RGB
            image = np.stack([channels[0]] * 3, axis=-1)
        elif len(channels) == 2:
            # Two channels + average
            image = np.stack([channels[0], channels[1], (channels[0] + channels[1]) / 2], axis=-1)
        else:
            # Use first three channels
            image = np.stack(channels[:3], axis=-1)

        return image

    def create_multi_transform_stack(
        self,
        prices: np.ndarray,
    ) -> np.ndarray:
        """
        Create RGB image from different transformations of same data.

        Channels:
        - Red: GASF (angular summation)
        - Green: GADF (angular difference)
        - Blue: MTF (transition dynamics)

        Args:
            prices: 1D array of price values

        Returns:
            3D array of shape (image_size, image_size, 3)
        """
        gasf = self.gaf.transform_gasf(prices)
        gadf = self.gaf.transform_gadf(prices)
        mtf = self.mtf.transform(prices)

        # Normalize each channel to [0, 1]
        gasf = (gasf - gasf.min()) / (gasf.max() - gasf.min() + 1e-10)
        gadf = (gadf - gadf.min()) / (gadf.max() - gadf.min() + 1e-10)
        mtf = (mtf - mtf.min()) / (mtf.max() - mtf.min() + 1e-10)

        image = np.stack([gasf, gadf, mtf], axis=-1)

        return image

    def create_batch(
        self,
        windows: np.ndarray,
        stack_type: str = "multi_transform",
    ) -> np.ndarray:
        """
        Create batch of images from price windows.

        Args:
            windows: 2D array of shape (n_samples, lookback)
            stack_type: 'multi_transform' or 'single_gasf'

        Returns:
            4D array of shape (n_samples, image_size, image_size, 3)
        """
        images = []

        for window in windows:
            if stack_type == "multi_transform":
                image = self.create_multi_transform_stack(window)
            else:
                gasf = self.gaf.transform_gasf(window)
                gasf = (gasf - gasf.min()) / (gasf.max() - gasf.min() + 1e-10)
                image = np.stack([gasf] * 3, axis=-1)

            images.append(image)

        return np.array(images)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.

    Args:
        image: Input image

    Returns:
        Normalized image
    """
    min_val = image.min()
    max_val = image.max()

    if max_val - min_val < 1e-10:
        return np.zeros_like(image)

    return (image - min_val) / (max_val - min_val)


def create_combined_image(
    prices_1m: np.ndarray,
    prices_5m: np.ndarray,
    prices_15m: np.ndarray,
    image_size: int = 224,
) -> np.ndarray:
    """
    Convenience function to create multi-timeframe GASF image.

    Args:
        prices_1m: 1-minute close prices
        prices_5m: 5-minute close prices
        prices_15m: 15-minute close prices
        image_size: Output image size

    Returns:
        3D array of shape (image_size, image_size, 3)
    """
    generator = ImageStackGenerator(image_size=image_size)
    data = {
        "1m": prices_1m,
        "5m": prices_5m,
        "15m": prices_15m,
    }
    return generator.create_timeframe_stack(data, transform_type="gasf")


if __name__ == "__main__":
    # Example usage
    print("Testing Image Transformers...")

    # Generate sample price data
    np.random.seed(42)
    n_samples = 120
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

    print(f"Sample price range: {prices.min():.2f} - {prices.max():.2f}")

    # Test GAF
    print("\nTesting GAF Transformer...")
    gaf = GAFTransformer(image_size=64)  # Smaller for demo
    gasf = gaf.transform_gasf(prices)
    gadf = gaf.transform_gadf(prices)
    print(f"  GASF shape: {gasf.shape}, range: [{gasf.min():.3f}, {gasf.max():.3f}]")
    print(f"  GADF shape: {gadf.shape}, range: [{gadf.min():.3f}, {gadf.max():.3f}]")

    # Test MTF
    print("\nTesting MTF Transformer...")
    mtf = MTFTransformer(n_bins=8, image_size=64)
    mtf_image = mtf.transform(prices)
    print(f"  MTF shape: {mtf_image.shape}, range: [{mtf_image.min():.3f}, {mtf_image.max():.3f}]")

    # Test Recurrence Plot
    print("\nTesting Recurrence Plot Transformer...")
    rp = RecurrencePlotTransformer(percentage=10, image_size=64)
    rp_image = rp.transform(prices)
    print(f"  RP shape: {rp_image.shape}, range: [{rp_image.min():.3f}, {rp_image.max():.3f}]")

    # Test Spectrogram
    print("\nTesting Spectrogram Transformer...")
    spec = SpectrogramTransformer(n_fft=32, hop_length=8, image_size=64)
    spec_image = spec.transform(prices)
    print(f"  Spec shape: {spec_image.shape}, range: [{spec_image.min():.3f}, {spec_image.max():.3f}]")

    # Test Image Stack Generator
    print("\nTesting Image Stack Generator...")
    generator = ImageStackGenerator(image_size=64)

    # Multi-transform stack
    multi_image = generator.create_multi_transform_stack(prices)
    print(f"  Multi-transform shape: {multi_image.shape}")

    # Batch creation
    windows = np.array([prices[i:i+60] for i in range(0, 60, 10)])
    print(f"  Windows shape: {windows.shape}")
    batch = generator.create_batch(windows)
    print(f"  Batch shape: {batch.shape}")

    print("\nAll tests passed!")
