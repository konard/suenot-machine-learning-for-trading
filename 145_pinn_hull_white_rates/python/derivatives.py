"""
Interest Rate Derivatives Pricing under Hull-White

Implements pricing for:
- Caplets and Caps
- Floorlets and Floors
- European Swaptions (Jamshidian decomposition)
- Bond options

All formulas are analytical under the Hull-White model.
"""

import numpy as np
from scipy.stats import norm
from typing import List, Optional, Tuple
from analytical import HullWhiteAnalytical


class HullWhiteDerivativesPricer:
    """
    Prices interest rate derivatives under the Hull-White model.

    Parameters:
        a: Mean reversion speed
        sigma: Short rate volatility
        market_maturities: Yield curve maturities
        market_rates: Yield curve zero rates
    """

    def __init__(
        self,
        a: float = 0.1,
        sigma: float = 0.01,
        market_maturities: Optional[np.ndarray] = None,
        market_rates: Optional[np.ndarray] = None,
    ):
        self.a = a
        self.sigma = sigma

        if market_maturities is not None and market_rates is not None:
            self.hw = HullWhiteAnalytical(
                a=a, sigma=sigma,
                market_times=market_maturities,
                market_rates=market_rates,
            )
        else:
            self.hw = HullWhiteAnalytical(a=a, sigma=sigma)

    def _sigma_p(self, t: float, T_option: float, T_bond: float) -> float:
        """
        Compute the bond option volatility parameter sigma_p.

        sigma_p = (sigma/a) * (1 - exp(-a*(T_bond - T_option)))
                  * sqrt((1 - exp(-2*a*(T_option - t))) / (2*a))
        """
        B = (1 - np.exp(-self.a * (T_bond - T_option))) / self.a
        var_term = (1 - np.exp(-2 * self.a * (T_option - t))) / (2 * self.a)
        return self.sigma * B * np.sqrt(max(var_term, 0))

    # ═══════════════════════════════════════════════════════════════════════════
    # BOND OPTIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def bond_option(
        self,
        r: float,
        t: float,
        T_option: float,
        T_bond: float,
        K: float,
        is_call: bool = True,
        notional: float = 1.0,
    ) -> float:
        """
        Price a European option on a zero-coupon bond.

        Uses the exact Hull-White formula.

        Args:
            r: Current short rate
            t: Current time
            T_option: Option expiry
            T_bond: Bond maturity (must be > T_option)
            K: Strike price
            is_call: True for call, False for put
            notional: Notional amount

        Returns:
            Option price
        """
        if T_option <= t or T_bond <= T_option:
            return 0.0

        P_t_To = self.hw.bond_price(r, t, T_option)
        P_t_Tb = self.hw.bond_price(r, t, T_bond)

        sigma_p = self._sigma_p(t, T_option, T_bond)

        if sigma_p < 1e-12:
            intrinsic = P_t_Tb - K * P_t_To if is_call else K * P_t_To - P_t_Tb
            return notional * max(intrinsic, 0.0)

        d1 = (np.log(P_t_Tb / (K * P_t_To)) + 0.5 * sigma_p**2) / sigma_p
        d2 = d1 - sigma_p

        if is_call:
            price = P_t_Tb * norm.cdf(d1) - K * P_t_To * norm.cdf(d2)
        else:
            price = K * P_t_To * norm.cdf(-d2) - P_t_Tb * norm.cdf(-d1)

        return notional * price

    # ═══════════════════════════════════════════════════════════════════════════
    # CAPLETS AND CAPS
    # ═══════════════════════════════════════════════════════════════════════════

    def caplet(
        self,
        r: float,
        t: float,
        T_reset: float,
        T_pay: float,
        K: float,
        notional: float = 1.0,
    ) -> float:
        """
        Price a caplet under Hull-White.

        A caplet pays max(L(T_reset, T_pay) - K, 0) * delta * N at T_pay,
        where L is the LIBOR rate and delta = T_pay - T_reset.

        This is equivalent to a put on a zero-coupon bond:
        Caplet = (1 + K*delta) * Put(P(T_reset, T_pay), X)
        where X = 1 / (1 + K*delta)

        Args:
            r: Current short rate
            t: Current time
            T_reset: Rate reset date
            T_pay: Payment date
            K: Cap strike rate
            notional: Notional amount

        Returns:
            Caplet price
        """
        delta = T_pay - T_reset
        if delta <= 0 or T_reset <= t:
            return 0.0

        X = 1.0 / (1.0 + K * delta)
        put_price = self.bond_option(
            r, t, T_reset, T_pay, X, is_call=False, notional=1.0
        )
        return notional * (1.0 + K * delta) * put_price

    def cap(
        self,
        r: float,
        t: float,
        reset_dates: np.ndarray,
        K: float,
        notional: float = 1.0,
    ) -> Tuple[float, List[float]]:
        """
        Price an interest rate cap as a sum of caplets.

        Args:
            r: Current short rate
            t: Current time
            reset_dates: Array of reset dates [T_1, T_2, ..., T_n]
                         Payment dates are [T_2, T_3, ..., T_{n+1}]
            K: Cap strike rate
            notional: Notional amount

        Returns:
            (total_cap_price, list_of_caplet_prices)
        """
        caplet_prices = []
        for i in range(len(reset_dates) - 1):
            T_reset = reset_dates[i]
            T_pay = reset_dates[i + 1]
            if T_reset > t:
                cp = self.caplet(r, t, T_reset, T_pay, K, notional)
            else:
                cp = 0.0
            caplet_prices.append(cp)

        return sum(caplet_prices), caplet_prices

    # ═══════════════════════════════════════════════════════════════════════════
    # FLOORLETS AND FLOORS
    # ═══════════════════════════════════════════════════════════════════════════

    def floorlet(
        self,
        r: float,
        t: float,
        T_reset: float,
        T_pay: float,
        K: float,
        notional: float = 1.0,
    ) -> float:
        """
        Price a floorlet under Hull-White.

        A floorlet pays max(K - L(T_reset, T_pay), 0) * delta * N at T_pay.

        Equivalent to a call on a zero-coupon bond:
        Floorlet = (1 + K*delta) * Call(P(T_reset, T_pay), X)

        Args:
            r: Current short rate
            t: Current time
            T_reset: Rate reset date
            T_pay: Payment date
            K: Floor strike rate
            notional: Notional amount

        Returns:
            Floorlet price
        """
        delta = T_pay - T_reset
        if delta <= 0 or T_reset <= t:
            return 0.0

        X = 1.0 / (1.0 + K * delta)
        call_price = self.bond_option(
            r, t, T_reset, T_pay, X, is_call=True, notional=1.0
        )
        return notional * (1.0 + K * delta) * call_price

    def floor(
        self,
        r: float,
        t: float,
        reset_dates: np.ndarray,
        K: float,
        notional: float = 1.0,
    ) -> Tuple[float, List[float]]:
        """
        Price an interest rate floor as a sum of floorlets.

        Args:
            r: Current short rate
            t: Current time
            reset_dates: Array of reset/payment dates
            K: Floor strike rate
            notional: Notional amount

        Returns:
            (total_floor_price, list_of_floorlet_prices)
        """
        floorlet_prices = []
        for i in range(len(reset_dates) - 1):
            T_reset = reset_dates[i]
            T_pay = reset_dates[i + 1]
            if T_reset > t:
                fp = self.floorlet(r, t, T_reset, T_pay, K, notional)
            else:
                fp = 0.0
            floorlet_prices.append(fp)

        return sum(floorlet_prices), floorlet_prices

    # ═══════════════════════════════════════════════════════════════════════════
    # SWAPTIONS (JAMSHIDIAN DECOMPOSITION)
    # ═══════════════════════════════════════════════════════════════════════════

    def _find_critical_rate(
        self, T_option: float, payment_dates: np.ndarray,
        fixed_rate: float, tol: float = 1e-10,
    ) -> float:
        """
        Find the critical rate r* at which the swap has zero value.

        At r*, the sum of discounted fixed payments equals the floating leg.
        This is the key step in Jamshidian's decomposition.
        """
        delta = np.diff(payment_dates)

        def swap_value(r_star):
            # P(r*, T_option; T_i) for all payment dates
            bond_prices = np.array([
                self.hw.bond_price(r_star, T_option, T_i)
                for T_i in payment_dates[1:]
            ])
            # Fixed leg present value
            fixed_pv = np.sum(fixed_rate * delta * bond_prices)
            # Add back final notional
            fixed_pv += bond_prices[-1]
            # Floating leg = P(r*, T_option; T_0) = 1 (at-the-money)
            floating_pv = self.hw.bond_price(r_star, T_option, payment_dates[0])

            return floating_pv - fixed_pv

        # Bisection search
        r_low, r_high = -0.1, 0.3
        for _ in range(100):
            r_mid = (r_low + r_high) / 2
            val = swap_value(r_mid)
            if abs(val) < tol:
                return r_mid
            if val > 0:
                r_low = r_mid
            else:
                r_high = r_mid

        return (r_low + r_high) / 2

    def swaption(
        self,
        r: float,
        t: float,
        T_option: float,
        payment_dates: np.ndarray,
        fixed_rate: float,
        is_payer: bool = True,
        notional: float = 1.0,
    ) -> float:
        """
        Price a European swaption using Jamshidian's decomposition.

        A payer swaption gives the right to enter a swap paying fixed, receiving floating.
        A receiver swaption gives the right to receive fixed, pay floating.

        Under Hull-White, since bond prices are monotonically decreasing in r,
        we can decompose the swaption into a portfolio of bond options.

        Args:
            r: Current short rate
            t: Current time
            T_option: Swaption expiry
            payment_dates: Swap payment dates [T_0, T_1, ..., T_n]
            fixed_rate: Fixed swap rate (strike)
            is_payer: True for payer, False for receiver
            notional: Notional amount

        Returns:
            Swaption price
        """
        if T_option <= t:
            return 0.0

        n_periods = len(payment_dates) - 1
        delta = np.diff(payment_dates)

        # Find critical rate r*
        r_star = self._find_critical_rate(T_option, payment_dates, fixed_rate)

        # Strike prices for individual bond options
        strike_prices = np.array([
            self.hw.bond_price(r_star, T_option, payment_dates[i + 1])
            for i in range(n_periods)
        ])

        # Swaption = sum of bond options
        total = 0.0
        for i in range(n_periods):
            T_bond = payment_dates[i + 1]
            coupon = fixed_rate * delta[i]
            if i == n_periods - 1:
                coupon += 1.0  # Add notional at last payment

            # Payer swaption -> put on bonds (value goes up when rates go up)
            # Receiver swaption -> call on bonds
            option_price = self.bond_option(
                r, t, T_option, T_bond, strike_prices[i],
                is_call=(not is_payer), notional=1.0,
            )
            total += coupon * option_price

        return notional * total

    # ═══════════════════════════════════════════════════════════════════════════
    # PUT-CALL PARITY AND COLLAR
    # ═══════════════════════════════════════════════════════════════════════════

    def collar(
        self,
        r: float,
        t: float,
        reset_dates: np.ndarray,
        K_cap: float,
        K_floor: float,
        notional: float = 1.0,
    ) -> float:
        """
        Price an interest rate collar (long cap + short floor).

        A collar provides protection against rates moving outside [K_floor, K_cap].

        Args:
            r: Current short rate
            t: Current time
            reset_dates: Payment schedule
            K_cap: Cap strike
            K_floor: Floor strike
            notional: Notional

        Returns:
            Collar price
        """
        cap_price, _ = self.cap(r, t, reset_dates, K_cap, notional)
        floor_price, _ = self.floor(r, t, reset_dates, K_floor, notional)
        return cap_price - floor_price

    def cap_floor_parity_check(
        self,
        r: float,
        t: float,
        reset_dates: np.ndarray,
        K: float,
        notional: float = 1.0,
    ) -> Dict:
        """
        Verify cap-floor parity: Cap - Floor = Swap (forward).

        Returns pricing of cap, floor, and the implied swap value.
        """
        cap_price, cap_components = self.cap(r, t, reset_dates, K, notional)
        floor_price, floor_components = self.floor(r, t, reset_dates, K, notional)

        # The difference should equal the value of a forward swap
        swap_value = cap_price - floor_price

        # Compute the actual forward swap value for comparison
        delta = np.diff(reset_dates)
        fixed_leg = 0.0
        for i in range(len(delta)):
            T_pay = reset_dates[i + 1]
            P = self.hw.bond_price(r, t, T_pay)
            fixed_leg += K * delta[i] * P

        P_first = self.hw.bond_price(r, t, reset_dates[0])
        P_last = self.hw.bond_price(r, t, reset_dates[-1])
        float_leg = P_first - P_last

        actual_swap = notional * (float_leg - fixed_leg)

        return {
            "cap_price": cap_price,
            "floor_price": floor_price,
            "cap_minus_floor": swap_value,
            "forward_swap_value": actual_swap,
            "parity_error": abs(swap_value - actual_swap),
        }


if __name__ == "__main__":
    from data_loader import load_sample_treasury_curve

    print("=" * 60)
    print("Hull-White Derivatives Pricing")
    print("=" * 60)

    mat, rates = load_sample_treasury_curve()

    pricer = HullWhiteDerivativesPricer(
        a=0.1, sigma=0.01,
        market_maturities=mat,
        market_rates=rates,
    )

    r0 = 0.05
    t = 0.0

    # Bond options
    print("\n--- Bond Options ---")
    for K in [0.90, 0.95, 0.98]:
        call = pricer.bond_option(r0, t, 1.0, 5.0, K, is_call=True, notional=100)
        put = pricer.bond_option(r0, t, 1.0, 5.0, K, is_call=False, notional=100)
        print(f"  K={K:.2f}  Call={call:.4f}  Put={put:.4f}")

    # Caps
    print("\n--- Interest Rate Caps ---")
    reset_dates = np.arange(0.5, 5.5, 0.5)
    for K in [0.03, 0.04, 0.05, 0.06]:
        cap_price, caplets = pricer.cap(r0, t, reset_dates, K, notional=1_000_000)
        print(f"  Strike={K:.2%}  Cap={cap_price:,.2f}")

    # Floors
    print("\n--- Interest Rate Floors ---")
    for K in [0.02, 0.03, 0.04]:
        floor_price, floorlets = pricer.floor(r0, t, reset_dates, K, notional=1_000_000)
        print(f"  Strike={K:.2%}  Floor={floor_price:,.2f}")

    # Cap-Floor parity
    print("\n--- Cap-Floor Parity Check ---")
    parity = pricer.cap_floor_parity_check(r0, t, reset_dates, 0.04, 1_000_000)
    for key, val in parity.items():
        print(f"  {key}: {val:,.6f}")

    # Swaptions
    print("\n--- European Swaptions ---")
    swap_dates = np.arange(1.0, 6.0, 0.5)
    for K in [0.03, 0.04, 0.05]:
        payer = pricer.swaption(r0, t, 1.0, swap_dates, K, is_payer=True, notional=1_000_000)
        receiver = pricer.swaption(r0, t, 1.0, swap_dates, K, is_payer=False, notional=1_000_000)
        print(f"  K={K:.2%}  Payer={payer:,.2f}  Receiver={receiver:,.2f}")

    # Collar
    print("\n--- Interest Rate Collar ---")
    collar = pricer.collar(r0, t, reset_dates, K_cap=0.06, K_floor=0.02, notional=1_000_000)
    print(f"  Collar (cap=6%, floor=2%): {collar:,.2f}")
