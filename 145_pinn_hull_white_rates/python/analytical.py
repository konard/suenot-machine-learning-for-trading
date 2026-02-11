"""
Analytical Hull-White Pricing Formulas

Closed-form solutions for the Hull-White (extended Vasicek) model:
- Zero-coupon bond prices
- Yield curves
- Forward rates
- Bond options (Jamshidian formula)
- Rate distribution
"""

import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from typing import Optional, Tuple, List


class HullWhiteAnalytical:
    """
    Analytical Hull-White model for interest rate pricing.

    The short rate dynamics under the risk-neutral measure:
        dr(t) = [theta(t) - a * r(t)] dt + sigma * dW(t)

    Parameters:
        a: float - Mean reversion speed (> 0)
        sigma: float - Short rate volatility (> 0)
        market_times: Optional[np.ndarray] - Market yield curve times
        market_rates: Optional[np.ndarray] - Market zero rates at those times
    """

    def __init__(
        self,
        a: float = 0.1,
        sigma: float = 0.01,
        market_times: Optional[np.ndarray] = None,
        market_rates: Optional[np.ndarray] = None,
    ):
        self.a = a
        self.sigma = sigma

        if market_times is not None and market_rates is not None:
            self._setup_market_curve(market_times, market_rates)
        else:
            # Use flat curve as default
            self._use_flat_curve(0.03)

    def _use_flat_curve(self, flat_rate: float):
        """Set up a flat yield curve."""
        self._flat_rate = flat_rate
        self._market_spline = None

    def _setup_market_curve(self, times: np.ndarray, rates: np.ndarray):
        """Set up market yield curve interpolation."""
        self._flat_rate = None
        # Ensure time 0 is included
        if times[0] > 0:
            times = np.concatenate([[0.0], times])
            rates = np.concatenate([[rates[0]], rates])
        self._market_times = times
        self._market_rates = rates
        self._market_spline = CubicSpline(times, rates)

    def market_zero_rate(self, t: float) -> float:
        """Get the market zero rate for maturity t."""
        if self._market_spline is not None:
            return float(self._market_spline(t))
        return self._flat_rate

    def market_discount_factor(self, t: float) -> float:
        """Market discount factor P_M(0, t)."""
        if t <= 0:
            return 1.0
        r = self.market_zero_rate(t)
        return np.exp(-r * t)

    def market_forward_rate(self, t: float, dt: float = 1e-4) -> float:
        """
        Instantaneous forward rate f_M(0, t).

        f(0, t) = -d/dt [ln P_M(0,t)] = r(t) + t * r'(t)
        """
        if t <= dt:
            return self.market_zero_rate(dt)

        r_t = self.market_zero_rate(t)
        r_t_plus = self.market_zero_rate(t + dt)
        r_t_minus = self.market_zero_rate(t - dt)

        # f(0,t) = -d/dt [ln P(0,t)] = -d/dt [-r(t)*t] = r(t) + t * r'(t)
        dr_dt = (r_t_plus - r_t_minus) / (2 * dt)
        return r_t + t * dr_dt

    def theta(self, t: float, dt: float = 1e-4) -> float:
        """
        Time-dependent drift theta(t) that fits the initial term structure.

        theta(t) = df_M(0,t)/dt + a * f_M(0,t) + (sigma^2 / (2a)) * (1 - exp(-2at))
        """
        f_t = self.market_forward_rate(t)
        f_t_plus = self.market_forward_rate(t + dt)
        f_t_minus = self.market_forward_rate(max(0, t - dt))

        df_dt = (f_t_plus - f_t_minus) / (2 * dt) if t > dt else (f_t_plus - f_t) / dt

        return df_dt + self.a * f_t + (self.sigma**2 / (2 * self.a)) * (1 - np.exp(-2 * self.a * t))

    def B(self, t: float, T: float) -> float:
        """B(t, T) function in the affine bond price formula."""
        tau = T - t
        if abs(self.a) < 1e-10:
            return tau
        return (1 - np.exp(-self.a * tau)) / self.a

    def A(self, t: float, T: float) -> float:
        """A(t, T) function in the affine bond price formula (log of)."""
        B_val = self.B(t, T)
        P0_T = self.market_discount_factor(T)
        P0_t = self.market_discount_factor(t)
        f0_t = self.market_forward_rate(t)

        ln_A = np.log(P0_T / P0_t) + B_val * f0_t \
            - (self.sigma**2 / (4 * self.a)) * (1 - np.exp(-2 * self.a * t)) * B_val**2

        return ln_A

    def bond_price(self, r: float, t: float, T: float) -> float:
        """
        Analytical Hull-White zero-coupon bond price.

        P(r, t; T) = A(t,T) * exp(-B(t,T) * r)

        Args:
            r: Current short rate
            t: Current time
            T: Maturity

        Returns:
            Zero-coupon bond price
        """
        if T <= t:
            return 1.0

        B_val = self.B(t, T)
        ln_A = self.A(t, T)

        return np.exp(ln_A - B_val * r)

    def bond_price_batch(
        self, r: np.ndarray, t: np.ndarray, T: np.ndarray
    ) -> np.ndarray:
        """Vectorized bond pricing."""
        results = np.zeros(len(r))
        for i in range(len(r)):
            results[i] = self.bond_price(r[i], t[i], T[i])
        return results

    def yield_from_price(self, price: float, t: float, T: float) -> float:
        """Convert bond price to continuously compounded yield."""
        tau = T - t
        if tau <= 0 or price <= 0:
            return 0.0
        return -np.log(price) / tau

    def yield_curve(
        self, r: float, t: float = 0.0, maturities: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute yield curve at a given rate and time.

        Args:
            r: Current short rate
            t: Current time
            maturities: Array of maturities (default: standard set)

        Returns:
            (maturities, yields) tuple
        """
        if maturities is None:
            maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])

        yields = np.array([
            self.yield_from_price(self.bond_price(r, t, T), t, T)
            for T in maturities
        ])
        return maturities, yields

    def forward_rate_curve(
        self, r: float, t: float = 0.0, maturities: Optional[np.ndarray] = None,
        dT: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute instantaneous forward rate curve.

        f(t, T) = -d/dT ln P(r, t; T)
        """
        if maturities is None:
            maturities = np.linspace(t + 0.1, 30.0, 100)

        forwards = np.array([
            -(np.log(self.bond_price(r, t, T + dT)) - np.log(self.bond_price(r, t, T))) / dT
            for T in maturities
        ])
        return maturities, forwards

    def rate_mean(self, r0: float, t: float, s: float) -> float:
        """
        Mean of r(s) given r(t) = r0.

        E[r(s)|r(t)] = r0 * exp(-a*(s-t)) + integral of theta(u)*exp(-a*(s-u)) du
        """
        tau = s - t
        # Numerical integration for theta integral
        n_steps = max(int(tau * 100), 10)
        u_vals = np.linspace(t, s, n_steps)
        du = tau / n_steps

        integral = 0.0
        for u in u_vals:
            integral += self.theta(u) * np.exp(-self.a * (s - u)) * du

        return r0 * np.exp(-self.a * tau) + integral

    def rate_variance(self, t: float, s: float) -> float:
        """
        Variance of r(s) given r(t).

        Var[r(s)|r(t)] = (sigma^2 / (2a)) * (1 - exp(-2a*(s-t)))
        """
        tau = s - t
        return (self.sigma**2 / (2 * self.a)) * (1 - np.exp(-2 * self.a * tau))

    def rate_std(self, t: float, s: float) -> float:
        """Standard deviation of r(s) given r(t)."""
        return np.sqrt(self.rate_variance(t, s))

    def simulate_rate_path(
        self,
        r0: float,
        T: float,
        n_steps: int = 252,
        n_paths: int = 1,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate short rate paths using Euler-Maruyama scheme.

        Args:
            r0: Initial short rate
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of simulated paths
            seed: Random seed

        Returns:
            (times, paths) where paths has shape (n_paths, n_steps+1)
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = r0

        for i in range(n_steps):
            t_i = times[i]
            r_i = paths[:, i]
            theta_i = self.theta(t_i)
            dW = np.random.randn(n_paths) * sqrt_dt
            paths[:, i + 1] = r_i + (theta_i - self.a * r_i) * dt + self.sigma * dW

        return times, paths

    def bond_option_price(
        self,
        r: float,
        t: float,
        T_option: float,
        T_bond: float,
        K: float,
        is_call: bool = True,
    ) -> float:
        """
        Price a European option on a zero-coupon bond.

        The option expires at T_option on a bond maturing at T_bond > T_option.
        Under Hull-White, this has an analytical solution.

        Args:
            r: Current short rate
            t: Current time
            T_option: Option expiry
            T_bond: Bond maturity
            K: Strike price
            is_call: True for call, False for put

        Returns:
            Option price
        """
        if T_option <= t or T_bond <= T_option:
            return 0.0

        P_t_To = self.bond_price(r, t, T_option)
        P_t_Tb = self.bond_price(r, t, T_bond)

        B_To_Tb = self.B(T_option, T_bond)

        sigma_p = self.sigma * B_To_Tb * np.sqrt(
            (1 - np.exp(-2 * self.a * (T_option - t))) / (2 * self.a)
        )

        if sigma_p < 1e-12:
            if is_call:
                return max(P_t_Tb - K * P_t_To, 0.0)
            else:
                return max(K * P_t_To - P_t_Tb, 0.0)

        d1 = (np.log(P_t_Tb / (K * P_t_To)) / sigma_p) + sigma_p / 2
        d2 = d1 - sigma_p

        if is_call:
            return P_t_Tb * norm.cdf(d1) - K * P_t_To * norm.cdf(d2)
        else:
            return K * P_t_To * norm.cdf(-d2) - P_t_Tb * norm.cdf(-d1)


class FlatCurveHullWhite(HullWhiteAnalytical):
    """Hull-White model with a flat initial yield curve (convenient for testing)."""

    def __init__(self, a: float = 0.1, sigma: float = 0.01, flat_rate: float = 0.03):
        super().__init__(a=a, sigma=sigma)
        self._use_flat_curve(flat_rate)
        self.flat_rate = flat_rate

    def theta(self, t: float, dt: float = 1e-4) -> float:
        """For a flat curve, theta is simplified."""
        return self.a * self.flat_rate + (self.sigma**2 / (2 * self.a)) * (
            1 - np.exp(-2 * self.a * t)
        )

    def bond_price(self, r: float, t: float, T: float) -> float:
        """Simplified bond price for flat curve."""
        if T <= t:
            return 1.0
        tau = T - t
        B_val = self.B(t, T)
        r_inf = self.flat_rate - self.sigma**2 / (2 * self.a**2)
        ln_A = r_inf * (B_val - tau) - (self.sigma**2 / (4 * self.a)) * B_val**2
        return np.exp(ln_A - B_val * r)


if __name__ == "__main__":
    # Test with flat curve
    hw = FlatCurveHullWhite(a=0.1, sigma=0.01, flat_rate=0.03)

    print("=== Hull-White Analytical Pricing ===\n")

    # Bond prices
    r0 = 0.03
    print(f"Short rate: r0 = {r0}")
    print(f"Parameters: a = {hw.a}, sigma = {hw.sigma}\n")

    print("Zero-coupon bond prices:")
    for T in [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]:
        price = hw.bond_price(r0, 0.0, T)
        yld = hw.yield_from_price(price, 0.0, T)
        print(f"  T={T:5.1f}  P={price:.6f}  y={yld:.4%}")

    print("\nRate distribution at T=1:")
    mu = hw.rate_mean(r0, 0.0, 1.0)
    std = hw.rate_std(0.0, 1.0)
    print(f"  E[r(1)] = {mu:.4%}")
    print(f"  Std[r(1)] = {std:.4%}")

    # Bond option
    print("\nBond option prices:")
    for K in [0.94, 0.96, 0.98]:
        call = hw.bond_option_price(r0, 0.0, 1.0, 2.0, K, is_call=True)
        put = hw.bond_option_price(r0, 0.0, 1.0, 2.0, K, is_call=False)
        print(f"  K={K:.2f}  Call={call:.6f}  Put={put:.6f}")

    # Simulate paths
    print("\nSimulating 5 rate paths:")
    times, paths = hw.simulate_rate_path(r0, T=5.0, n_steps=1260, n_paths=5, seed=42)
    for i in range(5):
        print(f"  Path {i}: r(5) = {paths[i, -1]:.4%}")
