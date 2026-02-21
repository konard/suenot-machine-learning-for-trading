import numpy as np

class DynamicHedgingEnv:
    """
    A custom Gym-style environment simulating a continuous Markov Decision Process
    for Dynamic Portfolio Hedging.
    
    The agent holds 1 unit of a risky asset. It must decide to short `h_t` units
    of the asset to minimize PnL variance, while balancing the transaction costs
    of changing `h_t` at each step.
    """
    def __init__(self, n_steps=100, mu=0.00, sigma=0.02, initial_price=100.0, transaction_cost=0.01):
        self.n_steps = n_steps
        self.mu = mu
        self.sigma = sigma
        self.initial_price = initial_price
        self.transaction_cost = transaction_cost # Cost percentage applied to the Delta of the hedge
        self.reset()

    def reset(self):
        """
        Resets the episode, generating a fresh price path using Geometric Brownian Motion.
        """
        self.current_step = 0
        self.current_hedge = 0.0 # Agent starts unhedged
        
        # Simulate price path (GBM)
        self.prices = np.zeros(self.n_steps + 1)
        self.prices[0] = self.initial_price
        
        dt = 1.0 # 1 step
        for i in range(1, self.n_steps + 1):
            # dS = S * (mu * dt + sigma * dW)
            dW = np.random.normal(0, np.sqrt(dt))
            self.prices[i] = self.prices[i-1] * np.exp((self.mu - 0.5 * self.sigma**2)*dt + self.sigma * dW)
            
        return self._get_state()

    def _get_state(self):
        """
        Returns the observable Markov State S_t for the Actor-Critic networks.
        Includes:
        1. Normalized Price
        2. Previous step Return
        3. Current Hedge Ratio Inventory
        4. Time remaining in episode
        """
        if self.current_step == 0:
            step_return = 0.0
        else:
            step_return = np.log(self.prices[self.current_step] / self.prices[self.current_step - 1])
            
        return np.array([
            self.prices[self.current_step] / self.initial_price, # State 0
            step_return,                                       # State 1
            self.current_hedge,                                # State 2
            (self.n_steps - self.current_step) / self.n_steps  # State 3
        ], dtype=np.float32)

    def step(self, action):
        """
        Executes the continuous action a_t (target hedge ratio) and transitions to S_{t+1}.
        Returns: (next_state, reward, done)
        """
        # The agent's action is its target hedge ratio, e.g., -0.6 (shorting 60% of the long position)
        target_hedge = np.clip(action, -1.0, 1.0) # Hedge bounds
        
        # Calculate Transaction Cost for taking the action
        hedge_delta = target_hedge - self.current_hedge
        trade_volume_dollars = np.abs(hedge_delta) * self.prices[self.current_step]
        cost = trade_volume_dollars * self.transaction_cost
        
        # Step forward in time
        self.current_hedge = target_hedge
        self.current_step += 1
        
        # Determine actual price movement
        price_change = self.prices[self.current_step] - self.prices[self.current_step - 1]
        
        # PnL logic:
        # Long position PnL = price_change
        # Short hedge PnL = price_change * current_hedge (Note: hedge is usually negative)
        step_pnl = (price_change + (self.current_hedge * price_change))
        
        # Reward Function: Goal is to maintain Zero PnL Variance (perfectly hedged)
        # while taking the least amount of transaction costs.
        # We severely penalize ANY variance in PnL (Risk Penalty), and subtract the explicit cost.
        risk_penalty = 1.0 * (step_pnl ** 2)
        reward = -risk_penalty - cost
        
        done = self.current_step >= self.n_steps
        
        return self._get_state(), reward, done
