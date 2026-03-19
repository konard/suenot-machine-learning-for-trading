# Chapter 318: Option Framework RL Trading

## 1. Introduction

The **Options Framework** for reinforcement learning provides a principled approach to **hierarchical decision-making** in trading systems. While standard RL operates at a single timescale — selecting primitive actions at each step — real trading involves decisions at multiple temporal resolutions: macro-level regime identification, medium-term strategy selection, and micro-level order execution.

In the Options Framework, an **option** is a temporally extended action that encapsulates a complete sub-policy. A "TrendFollow" option, for example, might persist for dozens of time steps, executing buy/hold decisions while the trend continues, then automatically terminating when the regime shifts. This mirrors how professional traders operate: they commit to a strategy (trend-following, mean-reversion, or staying flat) and execute it over a period, rather than making independent buy/sell decisions at every tick.

The framework was introduced by Sutton, Precup, and Singh in their foundational work on temporal abstraction in RL. It extends standard MDPs to **semi-MDPs** (SMDPs), where actions can take variable amounts of time. For trading, this is natural: a trend-following strategy might last days in a strong trend but terminate within hours during choppy markets.

Key advantages for trading applications:

- **Multi-timescale reasoning**: Macro regime detection drives strategy selection; micro execution handles individual trades
- **Exploration efficiency**: Options enable structured exploration across strategy space rather than random action perturbation
- **Transfer learning**: Options learned in one market can be applied to similar markets
- **Interpretability**: Each option corresponds to a recognizable trading strategy, making the system auditable

This chapter implements the Options Framework with an **Option-Critic** architecture that jointly learns the options themselves (their policies and termination conditions) alongside the policy over options, all applied to cryptocurrency trading with Bybit market data.

## 2. Mathematical Foundation

### 2.1 Semi-Markov Decision Process (Semi-MDP)

A standard MDP is defined by the tuple $(S, A, P, R, \gamma)$. The **semi-MDP** extends this by allowing actions to take variable durations. Formally, transitions are characterized by:

$$P(s', k \mid s, o) = \Pr(s_{t+k} = s', \text{duration} = k \mid s_t = s, o_t = o)$$

where $k$ is the number of primitive time steps the option $o$ takes before terminating in state $s'$.

The discount factor accumulates over the option's duration:

$$\gamma^k \cdot V(s')$$

This means longer-duration options face greater discounting of future rewards, creating a natural trade-off between commitment and flexibility.

### 2.2 Options: The Triple $(I, \pi, \beta)$

An **option** $o$ is formally defined as a triple:

$$o = (I_o, \pi_o, \beta_o)$$

where:

- **Initiation set** $I_o \subseteq S$: The set of states where option $o$ can be started. For trading, a TrendFollow option might only be initiable when momentum indicators exceed a threshold.

- **Intra-option policy** $\pi_o(a \mid s)$: The probability of taking primitive action $a$ in state $s$ while executing option $o$. This is the "inner" policy that drives actual trading decisions within the strategy.

- **Termination condition** $\beta_o(s) \in [0, 1]$: The probability of terminating option $o$ upon entering state $s$. High termination probability in adverse states allows the agent to quickly abandon failing strategies.

### 2.3 Policy Over Options

The **policy over options** $\mu(o \mid s)$ selects which option to execute given the current state. This operates at a higher level than the intra-option policies:

$$\mu(o \mid s) = \begin{cases} \text{option selection policy} & \text{if no option is active or current option terminated} \end{cases}$$

The call-and-return execution model works as follows:
1. In state $s$, if no option is active, select option $o \sim \mu(\cdot \mid s)$
2. Execute action $a \sim \pi_o(\cdot \mid s)$
3. Transition to $s'$, receive reward $r$
4. With probability $\beta_o(s')$, option $o$ terminates; go to step 1
5. Otherwise, continue with option $o$ from step 2

### 2.4 Intra-Option Q-Learning

The option-value function $Q_\Omega(s, o)$ represents the expected return of executing option $o$ in state $s$ and then following the policy over options $\mu$ thereafter:

$$Q_\Omega(s, o) = \sum_a \pi_o(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ r(s, a) + \gamma \, U(o, s') \right]$$

where $U(o, s')$ is the **option-completion function**:

$$U(o, s') = (1 - \beta_o(s')) \, Q_\Omega(s', o) + \beta_o(s') \max_{o'} Q_\Omega(s', o')$$

This captures the fact that in state $s'$, the option either continues (with probability $1 - \beta$) or terminates (with probability $\beta$), in which case a new option is selected greedily.

The SMDP Q-learning update for the option-value function after option $o$ executes for $k$ steps with cumulative reward $R$:

$$Q_\Omega(s, o) \leftarrow Q_\Omega(s, o) + \alpha \left[ R + \gamma^k \max_{o'} Q_\Omega(s', o') - Q_\Omega(s, o) \right]$$

### 2.5 Option-Critic Architecture

The **Option-Critic** architecture (Bacon et al., 2017) learns all components end-to-end using policy gradient methods:

**Intra-option policy gradient** (updates $\pi_o$):

$$\frac{\partial Q_U(s, o, a)}{\partial \theta_o} \propto Q_U(s, o, a) \nabla_{\theta_o} \ln \pi_o(a \mid s; \theta_o)$$

where $Q_U(s, o, a)$ is the intra-option action-value function.

**Termination gradient** (updates $\beta_o$):

$$\frac{\partial}{\partial \vartheta_o} \beta_o(s'; \vartheta_o) \cdot A_\Omega(s', o)$$

where $A_\Omega(s', o) = Q_\Omega(s', o) - V_\Omega(s')$ is the advantage of continuing with option $o$ versus selecting a new option. When the advantage is negative (the current option is worse than average), the gradient increases the termination probability.

### 2.6 Termination Condition Details

The termination function is parameterized using a sigmoid:

$$\beta_o(s; \vartheta) = \sigma(\vartheta_o^T \phi(s))$$

The termination gradient theorem states:

$$-\frac{\partial J}{\partial \vartheta_{o,s}} = -\beta_o(s)(1 - \beta_o(s)) \cdot A_\Omega(s, o)$$

This elegant result means:
- If the current option is better than alternatives ($A_\Omega > 0$), decrease termination probability (keep going)
- If the current option is worse ($A_\Omega < 0$), increase termination probability (switch strategies)

## 3. Applications in Trading

### 3.1 Multi-Timescale Trading

The Options Framework naturally maps to the multi-timescale structure of financial markets:

| Level | Timescale | Options Framework | Trading Application |
|-------|-----------|-------------------|---------------------|
| Macro | Days-Weeks | Policy over options $\mu$ | Regime identification (bull/bear/sideways) |
| Meso | Hours-Days | Intra-option policy $\pi_o$ | Strategy execution (trend-follow, mean-revert) |
| Micro | Minutes | Primitive actions | Order placement (buy/sell/hold) |

**Macro regime detection** operates through the policy over options. When the agent detects a trending market (high momentum, rising volume), it selects the TrendFollow option. In ranging markets with mean-reverting characteristics, it switches to MeanRevert. During uncertain or low-volatility periods, it activates Hold.

**Micro execution** is handled by the intra-option policies. The TrendFollow option's policy predominantly selects Buy actions in uptrends and Sell actions in downtrends. The MeanRevert option does the opposite: Buy on dips and Sell on rallies.

### 3.2 Portfolio Rebalancing

Options can represent complete rebalancing strategies:

- **AggressiveRebalance**: Rapidly moves toward target allocation, terminates when within tolerance
- **GradualRebalance**: Slowly adjusts positions over time to minimize market impact
- **OpportunisticRebalance**: Waits for favorable price movements before rebalancing

The termination condition naturally captures "when to stop rebalancing" — the strategy terminates when the portfolio is sufficiently close to the target allocation or when market conditions change.

### 3.3 Risk Management

Options provide a natural framework for risk-aware trading:

- **RiskOff** option: Activated during high-volatility regimes, reduces position sizes, terminates when volatility subsides
- **Recovery** option: After a drawdown, implements conservative position building with tight stop-losses

The initiation sets encode risk constraints: the RiskOff option can only be initiated when volatility exceeds a threshold, preventing premature de-risking.

## 4. Rust Implementation

The implementation in `rust/src/lib.rs` provides a complete Options Framework for trading:

### Core Components

**MarketState**: Discretizes continuous market features (momentum, volatility, volume) into a finite state space of 45 states (5 momentum bins x 3 volatility bins x 3 volume bins). This enables tabular methods while capturing the essential market structure.

**TradingOption**: Implements the $(I, \pi, \beta)$ triple with three pre-configured strategies:
- `TrendFollow`: Buys in uptrends, sells in downtrends, terminates in neutral markets
- `MeanRevert`: Buys dips, sells rallies, terminates when price returns to mean
- `Hold`: Minimal trading, low termination probability for patience during uncertainty

**SemiMdpEnv**: The trading environment supporting both primitive steps and option execution. It tracks position, PnL, and transaction costs. The `execute_option` method runs an option until termination, accumulating discounted rewards.

**IntraOptionQLearning**: Implements SMDP Q-learning over the option-value function $Q_\Omega(s, o)$ with epsilon-greedy option selection.

**OptionCritic**: The full Option-Critic architecture with:
- $Q_\Omega(s, o)$: Option-level value function
- $Q_U(s, o, a)$: Intra-option action-value function
- Learned termination functions $\beta_o(s)$ via sigmoid parameterization
- Softmax intra-option action selection

### Key Design Decisions

1. **Tabular approach**: With 45 states and 3 options, the Q-tables are small enough for fast learning while capturing meaningful market structure
2. **Sigmoid termination**: The termination function uses logits with sigmoid activation, allowing gradient-based learning of when to switch strategies
3. **Transaction costs**: The environment includes realistic 10bp transaction costs for position changes, penalizing excessive switching
4. **Option duration limit**: A safety cap of 50 steps prevents options from running indefinitely

## 5. Bybit Data Integration

The implementation fetches real market data from the Bybit v5 API:

```
GET /v5/market/kline?category=spot&symbol=BTCUSDT&interval=60&limit=500
```

The `fetch_bybit_klines` function:
1. Queries the Bybit spot market for BTCUSDT hourly candles
2. Parses the response into structured `Kline` objects (open, high, low, close, volume)
3. Reverses the data to chronological order (Bybit returns newest first)
4. Falls back to synthetic data with regime changes if the API is unavailable

The synthetic data generator creates realistic price series with cycling regimes (uptrend, downtrend, sideways) every 100 steps, providing a controlled environment for testing the option framework's regime-switching behavior.

Market features are computed from the raw price/volume data:
- **Momentum**: Window return over the lookback period
- **Volatility**: Standard deviation of log returns
- **Volume ratio**: Current volume relative to the window average

## 6. Key Takeaways

1. **Temporal abstraction matches trading reality**: Traders naturally think in terms of strategies (options) rather than individual trades (primitive actions). The Options Framework formalizes this hierarchy.

2. **Semi-MDPs handle variable-duration strategies**: Unlike fixed-step MDPs, semi-MDPs properly discount rewards over the actual duration of a strategy, creating appropriate incentives for strategy commitment versus flexibility.

3. **The termination condition is critical**: Learning *when to stop* a strategy is as important as learning *what to do*. The termination gradient theorem provides an elegant solution — abandon strategies that underperform the average.

4. **Option-Critic enables end-to-end learning**: Rather than hand-designing options, the Option-Critic architecture learns intra-option policies, termination conditions, and the policy over options simultaneously from market data.

5. **Initiation sets encode domain knowledge**: By restricting which states can initiate each option, we incorporate trading expertise — for example, preventing trend-following in sideways markets or mean-reversion during strong trends.

6. **Multi-timescale decomposition improves exploration**: Instead of exploring the full action space at every step, the agent explores at the strategy level, making learning more sample-efficient in complex market environments.

7. **Transaction costs interact with option duration**: Longer-lasting options reduce trading frequency and transaction costs, while the discount factor penalizes excessive commitment. This natural tension produces balanced trading behavior.

## References

- Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. *Artificial Intelligence*, 112(1-2), 181-211.
- Bacon, P. L., Harb, J., & Precup, D. (2017). The Option-Critic Architecture. *AAAI Conference on Artificial Intelligence*.
- Precup, D. (2000). Temporal abstraction in reinforcement learning. *Ph.D. thesis, University of Massachusetts Amherst*.
- Harb, J., Bacon, P. L., Klissarov, M., & Precup, D. (2018). When is a good time to terminate? *Advances in Neural Information Processing Systems*.
