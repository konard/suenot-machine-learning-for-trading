"""
Unit tests for VAE Volatility Surface Python implementation.

Run with:
    cd 240_vae_volatility_surface
    python -m pytest python/test_vae.py -v
"""

import numpy as np
import pytest

from .vol_surface import ArbitrageChecker, VolSurface, compute_butterfly, compute_skew
from .surface_generator import SurfaceGenerator, default_maturity_days, default_moneyness_grid
from .model import VAEConfig, VAEModel
from .bybit_client import BybitClient


class TestVolSurface:
    def test_creation(self) -> None:
        moneyness = [0.9, 0.95, 1.0, 1.05, 1.1]
        maturities = [0.25, 0.5, 1.0]
        ivols = np.full((3, 5), 0.25)
        surface = VolSurface(moneyness, maturities, ivols)
        assert surface.grid_size == 15
        assert surface.get(0, 2) == 0.25

    def test_atm_vol(self) -> None:
        moneyness = [0.9, 0.95, 1.0, 1.05, 1.1]
        maturities = [0.25]
        ivols = np.full((1, 5), 0.20)
        ivols[0, 2] = 0.30
        surface = VolSurface(moneyness, maturities, ivols)
        assert abs(surface.atm_vol(0) - 0.30) < 1e-10

    def test_flatten(self) -> None:
        moneyness = [0.9, 1.0, 1.1]
        maturities = [0.25, 0.5]
        ivols = np.array([[0.2, 0.25, 0.3], [0.22, 0.27, 0.32]])
        surface = VolSurface(moneyness, maturities, ivols)
        flat = surface.flatten()
        assert len(flat) == 6
        assert flat[0] == 0.2


class TestSurfaceGenerator:
    def test_sabr_iv_atm(self) -> None:
        iv = SurfaceGenerator.sabr_iv(0.3, 0.5, -0.3, 0.4, 1.0, 1.0, 0.5)
        assert 0.0 < iv < 2.0

    def test_sabr_iv_skew(self) -> None:
        iv_put = SurfaceGenerator.sabr_iv(0.3, 0.5, -0.5, 0.4, 1.0, 0.9, 0.5)
        iv_call = SurfaceGenerator.sabr_iv(0.3, 0.5, -0.5, 0.4, 1.0, 1.1, 0.5)
        assert iv_put > iv_call, f"Expected negative skew: put_iv={iv_put} > call_iv={iv_call}"

    def test_batch_generation(self) -> None:
        gen = SurfaceGenerator(seed=42)
        moneyness = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
        maturities = np.array([0.25, 0.5, 1.0])
        batch = gen.generate_batch(10, moneyness, maturities)
        assert len(batch) == 10
        for surface in batch:
            assert surface.grid_size == 15
            assert np.all(surface.ivols > 0.0)


class TestArbitrageChecker:
    def test_clean_surface(self) -> None:
        gen = SurfaceGenerator(seed=42)
        moneyness = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
        maturities = np.array([0.25, 0.5, 1.0])
        surface = gen.generate_surface(moneyness, maturities)
        cal_violations, _ = ArbitrageChecker.check_calendar(surface)
        assert cal_violations == 0, "SABR surface should have no calendar arbitrage"


class TestVAEModel:
    def _make_config(self, input_dim: int = 15) -> VAEConfig:
        return VAEConfig(input_dim=input_dim, hidden_dim=8, latent_dim=4, beta=1.0)

    def test_creation(self) -> None:
        config = self._make_config()
        vae = VAEModel(config)
        assert vae.latent_dim == 4
        assert abs(vae.beta - 1.0) < 1e-10

    def test_encode_decode(self) -> None:
        config = self._make_config()
        vae = VAEModel(config)
        x = np.full(15, 0.25)
        mu, logvar = vae.encode(x)
        assert mu.shape[-1] == 4
        assert logvar.shape[-1] == 4

        z = mu  # use mu directly for deterministic test
        recon = vae.decode(z)
        assert recon.shape[-1] == 15
        assert np.all(recon >= 0.01)
        assert np.all(recon <= 2.0)

    def test_training(self) -> None:
        gen = SurfaceGenerator(seed=42)
        moneyness = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
        maturities = np.array([0.25, 0.5, 1.0])
        surfaces = gen.generate_batch(50, moneyness, maturities)
        data = [s.flatten() for s in surfaces]

        config = self._make_config()
        vae = VAEModel(config)

        recon_pre, mu_pre, lv_pre = vae.forward(data[0])
        loss_pre, _, _ = vae.loss(data[0], recon_pre, mu_pre, lv_pre)

        loss_after, _, _ = vae.train(data, epochs=20, lr=0.01)
        assert loss_after < loss_pre, f"Training should reduce loss: before={loss_pre}, after={loss_after}"

    def test_sample(self) -> None:
        config = self._make_config()
        vae = VAEModel(config)
        sample = vae.sample()
        assert sample.shape[-1] == 15
        assert np.all(sample >= 0.01)
        assert np.all(sample <= 2.0)


class TestBybitClient:
    def test_parse_symbol(self) -> None:
        result = BybitClient.parse_symbol("BTC-28MAR25-90000-C")
        assert result is not None
        expiry, strike, is_call = result
        assert expiry == "28MAR25"
        assert abs(strike - 90000.0) < 1e-10
        assert is_call is True

        result = BybitClient.parse_symbol("BTC-28MAR25-90000-P")
        assert result is not None
        _, _, is_call = result
        assert is_call is False

    def test_parse_symbol_invalid(self) -> None:
        assert BybitClient.parse_symbol("invalid") is None
        assert BybitClient.parse_symbol("BTC-28MAR25") is None


class TestMetrics:
    def test_skew_and_butterfly(self) -> None:
        moneyness = [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]
        maturities = [0.25]
        ivols = np.array([[0.35, 0.30, 0.27, 0.25, 0.26, 0.28, 0.30]])
        surface = VolSurface(moneyness, maturities, ivols)

        skew = compute_skew(surface, 0)
        assert skew > 0.0, f"Expected positive skew, got {skew}"

        bfly = compute_butterfly(surface, 0)
        assert bfly > 0.0, f"Expected positive butterfly, got {bfly}"
