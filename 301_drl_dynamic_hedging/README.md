# Chapter 165: Deep Reinforcement Learning for Dynamic Portfolio Hedging

Welcome to Chapter 165. Until now, our models have primarily focused on *prediction* (Supervised Learning) or *representation* (Self-Supervised/Contrastive Learning). In this chapter, we enter the domain of **Sequential Decision Making** using Reinforcement Learning (RL).

We implement a **Proximal Policy Optimization (PPO)** agent that learns to continuously dynamically hedge a stock portfolio. Instead of just predicting if a stock will go up or down, the agent learns an optimal continuous control policy: "Given this specific market volatility, my current inventory, and transaction costs, exactly how much of the asset should I short right now to minimize risk without bleeding money in fees?"

## Architecture overview

This project implements an Actor-Critic architecture tailored for continuous mathematical finance:
1.  **The Environment (`environment.py`)**: A custom Gym-style simulator that generates random walks simulating price and volatility, tracking the portfolio's value over time and calculating step-by-step rewards.
2.  **The Agent (`ppo_agent.py`)**: A PyTorch Dual-Network.
    *   **The Actor**: Maps the current market state directly to a Gaussian distribution (Mean $\mu$, StdDev $\sigma$) over the optimal hedge ratio.
    *   **The Critic**: Maps the current market state to the expected total future sum of rewards (Value $V$), acting as a baseline to tell the Actor if a move was surprisingly good or bad.
3.  **The Rust Core (`rust/src/lib.rs`)**: A blazingly fast re-implementation of the trained Actor network's forward pass, allowing the strategy to be deployed in a microsecond-latency trading engine.

## The Mathematical Challenge

Dynamic hedging is notoriously difficult because of **Transaction Costs**. 
If you simply hedge 100% of your portfolio's Delta ($ \Delta $) at every microsecond, your risk drops to zero, but you will instantly go bankrupt paying exchange trading fees. If you rarely hedge, your fees are low, but a sudden market crash will cause massive losses. 
Reinforcement Learning solves this by exploring the trade-off iteratively against the reward function, learning exactly when the risk of holding an unhedged position mathematically outweighs the cost of the trading fee.

## Usage

1. Enter the `python` directory and set up the virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Train the PPO agent:
   ```bash
   python train.py
   ```
   *Watch the terminal output as the agent's reward gradually increases while it learns to balance risk and fees.*
3. Test the Rust Inference engine:
   ```bash
   cd ../rust
   cargo test
   ```
