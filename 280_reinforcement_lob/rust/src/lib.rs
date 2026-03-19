//! Reinforcement Learning for Limit Order Book (LOB) Trading
//!
//! This crate implements:
//! - LOB simulator with bid/ask queues and order matching
//! - Market making agent with DQN-style Q-learning
//! - Inventory penalty reward shaping
//! - Bybit orderbook data fetching

use std::collections::{BTreeMap, HashMap};
use rand::Rng;
use serde::Deserialize;

// ============================================================
// LOB Simulator
// ============================================================

/// A single price level in the order book.
#[derive(Debug, Clone)]
pub struct PriceLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Limit Order Book with bid and ask sides.
#[derive(Debug, Clone)]
pub struct OrderBook {
    /// Bids sorted by price descending (best bid first).
    pub bids: BTreeMap<i64, f64>, // price_ticks -> quantity
    /// Asks sorted by price ascending (best ask first).
    pub asks: BTreeMap<i64, f64>, // price_ticks -> quantity
    /// Tick size in price units.
    pub tick_size: f64,
}

impl OrderBook {
    pub fn new(tick_size: f64) -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            tick_size,
        }
    }

    /// Convert a price to tick representation.
    pub fn price_to_ticks(&self, price: f64) -> i64 {
        (price / self.tick_size).round() as i64
    }

    /// Convert ticks back to price.
    pub fn ticks_to_price(&self, ticks: i64) -> f64 {
        ticks as f64 * self.tick_size
    }

    /// Add a bid (buy) order at the given price.
    pub fn add_bid(&mut self, price: f64, quantity: f64) {
        let ticks = self.price_to_ticks(price);
        let entry = self.bids.entry(ticks).or_insert(0.0);
        *entry += quantity;
    }

    /// Add an ask (sell) order at the given price.
    pub fn add_ask(&mut self, price: f64, quantity: f64) {
        let ticks = self.asks.entry(self.price_to_ticks(price)).or_insert(0.0);
        *ticks += quantity;
    }

    /// Get the best bid price and quantity.
    pub fn best_bid(&self) -> Option<PriceLevel> {
        self.bids.iter().next_back().map(|(&ticks, &qty)| PriceLevel {
            price: self.ticks_to_price(ticks),
            quantity: qty,
        })
    }

    /// Get the best ask price and quantity.
    pub fn best_ask(&self) -> Option<PriceLevel> {
        self.asks.iter().next().map(|(&ticks, &qty)| PriceLevel {
            price: self.ticks_to_price(ticks),
            quantity: qty,
        })
    }

    /// Get the mid-price.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid.price + ask.price) / 2.0),
            _ => None,
        }
    }

    /// Get the spread.
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask.price - bid.price),
            _ => None,
        }
    }

    /// Execute a market buy order. Returns (avg_fill_price, filled_quantity).
    pub fn market_buy(&mut self, mut quantity: f64) -> Option<(f64, f64)> {
        let mut total_cost = 0.0;
        let mut total_filled = 0.0;
        let tick_size = self.tick_size;

        let ask_levels: Vec<(i64, f64)> = self.asks.iter().map(|(&t, &q)| (t, q)).collect();
        for (ticks, ask_qty) in ask_levels {
            if quantity <= 0.0 {
                break;
            }
            let fill = quantity.min(ask_qty);
            let price = ticks as f64 * tick_size;
            total_cost += fill * price;
            total_filled += fill;
            quantity -= fill;
            let remaining = ask_qty - fill;
            if remaining <= 1e-12 {
                self.asks.remove(&ticks);
            } else {
                self.asks.insert(ticks, remaining);
            }
        }

        if total_filled > 0.0 {
            Some((total_cost / total_filled, total_filled))
        } else {
            None
        }
    }

    /// Execute a market sell order. Returns (avg_fill_price, filled_quantity).
    pub fn market_sell(&mut self, mut quantity: f64) -> Option<(f64, f64)> {
        let mut total_revenue = 0.0;
        let mut total_filled = 0.0;
        let tick_size = self.tick_size;

        // Iterate bids from highest to lowest.
        let bid_levels: Vec<(i64, f64)> = self.bids.iter().rev().map(|(&t, &q)| (t, q)).collect();
        for (ticks, bid_qty) in bid_levels {
            if quantity <= 0.0 {
                break;
            }
            let fill = quantity.min(bid_qty);
            let price = ticks as f64 * tick_size;
            total_revenue += fill * price;
            total_filled += fill;
            quantity -= fill;
            let remaining = bid_qty - fill;
            if remaining <= 1e-12 {
                self.bids.remove(&ticks);
            } else {
                self.bids.insert(ticks, remaining);
            }
        }

        if total_filled > 0.0 {
            Some((total_revenue / total_filled, total_filled))
        } else {
            None
        }
    }

    /// Get top N bid levels (descending price).
    pub fn top_bids(&self, n: usize) -> Vec<PriceLevel> {
        self.bids
            .iter()
            .rev()
            .take(n)
            .map(|(&ticks, &qty)| PriceLevel {
                price: self.ticks_to_price(ticks),
                quantity: qty,
            })
            .collect()
    }

    /// Get top N ask levels (ascending price).
    pub fn top_asks(&self, n: usize) -> Vec<PriceLevel> {
        self.asks
            .iter()
            .take(n)
            .map(|(&ticks, &qty)| PriceLevel {
                price: self.ticks_to_price(ticks),
                quantity: qty,
            })
            .collect()
    }

    /// Create from vectors of price levels.
    pub fn from_levels(bids: &[(f64, f64)], asks: &[(f64, f64)], tick_size: f64) -> Self {
        let mut ob = Self::new(tick_size);
        for &(price, qty) in bids {
            ob.add_bid(price, qty);
        }
        for &(price, qty) in asks {
            ob.add_ask(price, qty);
        }
        ob
    }
}

// ============================================================
// LOB Trading Environment
// ============================================================

/// State observed by the RL agent.
#[derive(Debug, Clone)]
pub struct MarketState {
    pub mid_price: f64,
    pub spread: f64,
    pub bid_depth: f64,   // total quantity on bid side (top 5)
    pub ask_depth: f64,   // total quantity on ask side (top 5)
    pub inventory: f64,
    pub time_remaining: f64,
    pub volatility: f64,
}

/// Discretized state key for tabular Q-learning.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct StateKey {
    pub spread_bin: i32,
    pub inventory_bin: i32,
    pub imbalance_bin: i32,
    pub time_bin: i32,
}

impl MarketState {
    /// Discretize the state for tabular Q-learning.
    pub fn to_key(&self) -> StateKey {
        let spread_bin = ((self.spread * 100.0).round() as i32).clamp(0, 10);
        let inventory_bin = (self.inventory.round() as i32).clamp(-5, 5);
        let imbalance = if self.bid_depth + self.ask_depth > 0.0 {
            (self.bid_depth - self.ask_depth) / (self.bid_depth + self.ask_depth)
        } else {
            0.0
        };
        let imbalance_bin = ((imbalance * 5.0).round() as i32).clamp(-5, 5);
        let time_bin = ((self.time_remaining * 10.0).round() as i32).clamp(0, 10);
        StateKey {
            spread_bin,
            inventory_bin,
            imbalance_bin,
            time_bin,
        }
    }
}

/// Action: spread offsets for bid and ask quotes.
/// Each index maps to a multiple of the tick size.
#[derive(Debug, Clone, Copy)]
pub struct SpreadAction {
    pub bid_offset_idx: usize,
    pub ask_offset_idx: usize,
}

/// Market making environment that simulates LOB trading.
pub struct TradingEnvironment {
    pub orderbook: OrderBook,
    pub initial_mid: f64,
    pub volatility: f64,
    pub time_horizon: usize,
    pub current_step: usize,
    pub agent_inventory: f64,
    pub agent_cash: f64,
    pub quote_size: f64,
    /// Spread offset options as multiples of tick_size.
    pub spread_options: Vec<f64>,
    /// Risk aversion for inventory penalty.
    pub risk_aversion: f64,
    /// Random number generator.
    rng: rand::rngs::ThreadRng,
}

impl TradingEnvironment {
    pub fn new(
        initial_mid: f64,
        tick_size: f64,
        volatility: f64,
        time_horizon: usize,
        quote_size: f64,
        risk_aversion: f64,
    ) -> Self {
        let spread_options = vec![0.5, 1.0, 1.5, 2.0, 2.5];
        let mut env = Self {
            orderbook: OrderBook::new(tick_size),
            initial_mid,
            volatility,
            time_horizon,
            current_step: 0,
            agent_inventory: 0.0,
            agent_cash: 0.0,
            quote_size,
            spread_options,
            risk_aversion,
            rng: rand::thread_rng(),
        };
        env.reset_orderbook();
        env
    }

    /// Number of discrete actions = spread_options^2.
    pub fn num_actions(&self) -> usize {
        self.spread_options.len() * self.spread_options.len()
    }

    /// Decode action index into SpreadAction.
    pub fn decode_action(&self, action_idx: usize) -> SpreadAction {
        let n = self.spread_options.len();
        SpreadAction {
            bid_offset_idx: action_idx / n,
            ask_offset_idx: action_idx % n,
        }
    }

    /// Reset the environment for a new episode.
    pub fn reset(&mut self) -> MarketState {
        self.current_step = 0;
        self.agent_inventory = 0.0;
        self.agent_cash = 0.0;
        self.reset_orderbook();
        self.get_state()
    }

    /// Populate the orderbook with synthetic levels.
    fn reset_orderbook(&mut self) {
        self.orderbook = OrderBook::new(self.orderbook.tick_size);
        let mid = self.initial_mid;
        let tick = self.orderbook.tick_size;

        for i in 1..=10 {
            let qty = self.rng.gen_range(0.5..5.0);
            self.orderbook.add_bid(mid - i as f64 * tick, qty);
            let qty = self.rng.gen_range(0.5..5.0);
            self.orderbook.add_ask(mid + i as f64 * tick, qty);
        }
    }

    /// Get the current market state.
    pub fn get_state(&self) -> MarketState {
        let mid = self.orderbook.mid_price().unwrap_or(self.initial_mid);
        let spread = self.orderbook.spread().unwrap_or(self.orderbook.tick_size);
        let bid_depth: f64 = self.orderbook.top_bids(5).iter().map(|l| l.quantity).sum();
        let ask_depth: f64 = self.orderbook.top_asks(5).iter().map(|l| l.quantity).sum();
        let time_remaining = (self.time_horizon - self.current_step) as f64
            / self.time_horizon as f64;

        MarketState {
            mid_price: mid,
            spread,
            bid_depth,
            ask_depth,
            inventory: self.agent_inventory,
            time_remaining,
            volatility: self.volatility,
        }
    }

    /// Step the environment with the given action.
    /// Returns (next_state, reward, done).
    pub fn step(&mut self, action_idx: usize) -> (MarketState, f64, bool) {
        let action = self.decode_action(action_idx);
        let tick = self.orderbook.tick_size;
        let mid = self.orderbook.mid_price().unwrap_or(self.initial_mid);

        // Agent's bid and ask quote prices.
        let bid_offset = self.spread_options[action.bid_offset_idx] * tick;
        let ask_offset = self.spread_options[action.ask_offset_idx] * tick;
        let bid_price = mid - bid_offset;
        let ask_price = mid + ask_offset;

        // Simulate random market activity.
        let mut _step_pnl = 0.0;

        // Random market orders hitting the book.
        let market_order_prob = 0.3;
        if self.rng.gen::<f64>() < market_order_prob {
            // Market buy hits our ask quote.
            let incoming_qty = self.rng.gen_range(0.1..self.quote_size * 2.0);
            if incoming_qty <= self.quote_size {
                // Our ask gets filled.
                _step_pnl += ask_price * self.quote_size.min(incoming_qty);
                self.agent_inventory -= self.quote_size.min(incoming_qty);
                self.agent_cash += ask_price * self.quote_size.min(incoming_qty);
            }
        }

        if self.rng.gen::<f64>() < market_order_prob {
            // Market sell hits our bid quote.
            let incoming_qty = self.rng.gen_range(0.1..self.quote_size * 2.0);
            if incoming_qty <= self.quote_size {
                // Our bid gets filled.
                _step_pnl -= bid_price * self.quote_size.min(incoming_qty);
                self.agent_inventory += self.quote_size.min(incoming_qty);
                self.agent_cash -= bid_price * self.quote_size.min(incoming_qty);
            }
        }

        // Random mid-price movement (Brownian motion).
        let price_change = self.rng.gen::<f64>() * 2.0 * self.volatility * tick
            - self.volatility * tick;
        let new_mid = mid + price_change;

        // Rebuild orderbook around new mid.
        self.orderbook = OrderBook::new(tick);
        for i in 1..=10 {
            let qty = self.rng.gen_range(0.5..5.0);
            self.orderbook.add_bid(new_mid - i as f64 * tick, qty);
            let qty = self.rng.gen_range(0.5..5.0);
            self.orderbook.add_ask(new_mid + i as f64 * tick, qty);
        }

        // Add agent's quotes to the book.
        let agent_bid = new_mid - bid_offset;
        let agent_ask = new_mid + ask_offset;
        self.orderbook.add_bid(agent_bid, self.quote_size);
        self.orderbook.add_ask(agent_ask, self.quote_size);

        self.current_step += 1;
        let done = self.current_step >= self.time_horizon;

        // Mark-to-market PnL change.
        let mtm_pnl = self.agent_inventory * price_change;

        // Inventory penalty.
        let inv_penalty = self.risk_aversion * self.agent_inventory.powi(2);

        // Terminal liquidation penalty.
        let terminal_penalty = if done {
            let spread = self.orderbook.spread().unwrap_or(tick);
            self.agent_inventory.abs() * spread * 0.5
        } else {
            0.0
        };

        let reward = mtm_pnl - inv_penalty - terminal_penalty;
        let state = self.get_state();

        (state, reward, done)
    }

    /// Get the agent's total PnL (cash + mark-to-market inventory value).
    pub fn total_pnl(&self) -> f64 {
        let mid = self.orderbook.mid_price().unwrap_or(self.initial_mid);
        self.agent_cash + self.agent_inventory * mid
    }
}

// ============================================================
// DQN Market Making Agent
// ============================================================

/// Experience tuple for replay buffer.
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: StateKey,
    pub action: usize,
    pub reward: f64,
    pub next_state: StateKey,
    pub done: bool,
}

/// Tabular Q-learning agent for market making.
pub struct DQNMarketMaker {
    /// Q-table: state -> action values.
    pub q_table: HashMap<StateKey, Vec<f64>>,
    /// Number of actions.
    pub num_actions: usize,
    /// Learning rate.
    pub alpha: f64,
    /// Discount factor.
    pub gamma: f64,
    /// Exploration rate.
    pub epsilon: f64,
    /// Minimum exploration rate.
    pub epsilon_min: f64,
    /// Exploration decay rate.
    pub epsilon_decay: f64,
    /// Experience replay buffer.
    pub replay_buffer: Vec<Experience>,
    /// Maximum buffer size.
    pub buffer_size: usize,
    /// Training step counter.
    pub train_steps: usize,
}

impl DQNMarketMaker {
    pub fn new(num_actions: usize) -> Self {
        Self {
            q_table: HashMap::new(),
            num_actions,
            alpha: 0.01,
            gamma: 0.99,
            epsilon: 1.0,
            epsilon_min: 0.01,
            epsilon_decay: 0.995,
            replay_buffer: Vec::new(),
            buffer_size: 10_000,
            train_steps: 0,
        }
    }

    /// Get Q-values for a state, initializing if needed.
    pub fn get_q_values(&mut self, state: &StateKey) -> Vec<f64> {
        self.q_table
            .entry(state.clone())
            .or_insert_with(|| vec![0.0; self.num_actions])
            .clone()
    }

    /// Select action using epsilon-greedy policy.
    pub fn select_action(&mut self, state: &StateKey) -> usize {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..self.num_actions)
        } else {
            let q_values = self.get_q_values(state);
            q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }
    }

    /// Store an experience in the replay buffer.
    pub fn store_experience(&mut self, exp: Experience) {
        if self.replay_buffer.len() >= self.buffer_size {
            self.replay_buffer.remove(0);
        }
        self.replay_buffer.push(exp);
    }

    /// Train on a single experience (online Q-learning update).
    pub fn train_step(&mut self, exp: &Experience) {
        let next_q = if exp.done {
            0.0
        } else {
            let next_values = self.get_q_values(&exp.next_state);
            next_values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        };

        let target = exp.reward + self.gamma * next_q;
        let q_values = self
            .q_table
            .entry(exp.state.clone())
            .or_insert_with(|| vec![0.0; self.num_actions]);
        let current = q_values[exp.action];
        q_values[exp.action] = current + self.alpha * (target - current);

        self.train_steps += 1;
    }

    /// Train on a mini-batch sampled from the replay buffer.
    pub fn train_batch(&mut self, batch_size: usize) {
        if self.replay_buffer.len() < batch_size {
            return;
        }
        let mut rng = rand::thread_rng();
        let experiences: Vec<Experience> = (0..batch_size)
            .map(|_| {
                let idx = rng.gen_range(0..self.replay_buffer.len());
                self.replay_buffer[idx].clone()
            })
            .collect();

        for exp in &experiences {
            self.train_step(exp);
        }
    }

    /// Decay exploration rate.
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }

    /// Get the best action (greedy) for a state.
    pub fn best_action(&mut self, state: &StateKey) -> usize {
        let q_values = self.get_q_values(state);
        q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

// ============================================================
// Fixed-Spread Baseline Agent
// ============================================================

/// Baseline agent that always uses a fixed spread.
pub struct FixedSpreadAgent {
    /// Fixed action index to use.
    pub action_idx: usize,
}

impl FixedSpreadAgent {
    /// Create with the middle spread option.
    pub fn new(num_spread_options: usize) -> Self {
        // Use mid-range spread (index 2 out of 5 = spread of 1.5)
        let mid_idx = num_spread_options / 2;
        let action_idx = mid_idx * num_spread_options + mid_idx;
        Self { action_idx }
    }

    pub fn select_action(&self) -> usize {
        self.action_idx
    }
}

// ============================================================
// Training Loop
// ============================================================

/// Result from a single training episode.
#[derive(Debug, Clone)]
pub struct EpisodeResult {
    pub total_reward: f64,
    pub total_pnl: f64,
    pub final_inventory: f64,
    pub steps: usize,
}

/// Run a single episode of the market making environment.
pub fn run_episode(
    env: &mut TradingEnvironment,
    agent: &mut DQNMarketMaker,
    training: bool,
) -> EpisodeResult {
    let state = env.reset();
    let mut state_key = state.to_key();
    let mut total_reward = 0.0;
    let mut steps = 0;

    loop {
        let action = if training {
            agent.select_action(&state_key)
        } else {
            agent.best_action(&state_key)
        };

        let (next_state, reward, done) = env.step(action);
        let next_key = next_state.to_key();
        total_reward += reward;
        steps += 1;

        if training {
            let exp = Experience {
                state: state_key.clone(),
                action,
                reward,
                next_state: next_key.clone(),
                done,
            };
            agent.store_experience(exp.clone());
            agent.train_step(&exp);

            if steps % 10 == 0 {
                agent.train_batch(32);
            }
        }

        if done {
            break;
        }
        state_key = next_key;
    }

    if training {
        agent.decay_epsilon();
    }

    EpisodeResult {
        total_reward,
        total_pnl: env.total_pnl(),
        final_inventory: env.agent_inventory,
        steps,
    }
}

/// Run a single episode with the fixed-spread baseline.
pub fn run_baseline_episode(
    env: &mut TradingEnvironment,
    agent: &FixedSpreadAgent,
) -> EpisodeResult {
    let _state = env.reset();
    let mut total_reward = 0.0;
    let mut steps = 0;

    loop {
        let action = agent.select_action();
        let (_next_state, reward, done) = env.step(action);
        total_reward += reward;
        steps += 1;

        if done {
            break;
        }
    }

    EpisodeResult {
        total_reward,
        total_pnl: env.total_pnl(),
        final_inventory: env.agent_inventory,
        steps,
    }
}

// ============================================================
// Bybit Orderbook Integration
// ============================================================

#[derive(Debug, Deserialize)]
struct BybitResponse {
    result: BybitOrderbookResult,
}

#[derive(Debug, Deserialize)]
struct BybitOrderbookResult {
    #[serde(rename = "b")]
    bids: Vec<[String; 2]>,
    #[serde(rename = "a")]
    asks: Vec<[String; 2]>,
}

/// Fetch orderbook from Bybit V5 API.
pub async fn fetch_bybit_orderbook(symbol: &str, limit: usize) -> anyhow::Result<OrderBook> {
    let url = format!(
        "https://api.bybit.com/v5/market/orderbook?category=spot&symbol={}&limit={}",
        symbol, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse = client.get(&url).send().await?.json().await?;

    // Determine tick size from data.
    let tick_size = if !resp.result.bids.is_empty() && resp.result.bids.len() >= 2 {
        let p1: f64 = resp.result.bids[0][0].parse()?;
        let p2: f64 = resp.result.bids[1][0].parse()?;
        (p1 - p2).abs().max(0.01)
    } else {
        0.01
    };

    let mut ob = OrderBook::new(tick_size);

    for level in &resp.result.bids {
        let price: f64 = level[0].parse()?;
        let qty: f64 = level[1].parse()?;
        ob.add_bid(price, qty);
    }

    for level in &resp.result.asks {
        let price: f64 = level[0].parse()?;
        let qty: f64 = level[1].parse()?;
        ob.add_ask(price, qty);
    }

    Ok(ob)
}

/// Fetch orderbook using blocking HTTP (for non-async contexts).
pub fn fetch_bybit_orderbook_blocking(symbol: &str, limit: usize) -> anyhow::Result<OrderBook> {
    let url = format!(
        "https://api.bybit.com/v5/market/orderbook?category=spot&symbol={}&limit={}",
        symbol, limit
    );

    let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

    let tick_size = if resp.result.bids.len() >= 2 {
        let p1: f64 = resp.result.bids[0][0].parse()?;
        let p2: f64 = resp.result.bids[1][0].parse()?;
        (p1 - p2).abs().max(0.01)
    } else {
        0.01
    };

    let mut ob = OrderBook::new(tick_size);

    for level in &resp.result.bids {
        let price: f64 = level[0].parse()?;
        let qty: f64 = level[1].parse()?;
        ob.add_bid(price, qty);
    }

    for level in &resp.result.asks {
        let price: f64 = level[0].parse()?;
        let qty: f64 = level[1].parse()?;
        ob.add_ask(price, qty);
    }

    Ok(ob)
}

// ============================================================
// Unit Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orderbook_creation() {
        let ob = OrderBook::from_levels(
            &[(100.0, 10.0), (99.0, 20.0)],
            &[(101.0, 15.0), (102.0, 25.0)],
            1.0,
        );
        let best_bid = ob.best_bid().unwrap();
        let best_ask = ob.best_ask().unwrap();
        assert_eq!(best_bid.price, 100.0);
        assert_eq!(best_bid.quantity, 10.0);
        assert_eq!(best_ask.price, 101.0);
        assert_eq!(best_ask.quantity, 15.0);
    }

    #[test]
    fn test_mid_price_and_spread() {
        let ob = OrderBook::from_levels(
            &[(100.0, 10.0)],
            &[(102.0, 10.0)],
            1.0,
        );
        assert!((ob.mid_price().unwrap() - 101.0).abs() < 1e-9);
        assert!((ob.spread().unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_market_buy() {
        let mut ob = OrderBook::from_levels(
            &[(99.0, 10.0)],
            &[(101.0, 5.0), (102.0, 10.0)],
            1.0,
        );
        let (avg_price, filled) = ob.market_buy(3.0).unwrap();
        assert!((avg_price - 101.0).abs() < 1e-9);
        assert!((filled - 3.0).abs() < 1e-9);
        // Remaining ask at 101 should have 2.0
        let best_ask = ob.best_ask().unwrap();
        assert!((best_ask.quantity - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_market_sell() {
        let mut ob = OrderBook::from_levels(
            &[(99.0, 5.0), (98.0, 10.0)],
            &[(101.0, 10.0)],
            1.0,
        );
        let (avg_price, filled) = ob.market_sell(3.0).unwrap();
        assert!((avg_price - 99.0).abs() < 1e-9);
        assert!((filled - 3.0).abs() < 1e-9);
        let best_bid = ob.best_bid().unwrap();
        assert!((best_bid.quantity - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_market_buy_crossing_levels() {
        let mut ob = OrderBook::from_levels(
            &[(99.0, 10.0)],
            &[(101.0, 2.0), (102.0, 10.0)],
            1.0,
        );
        let (avg_price, filled) = ob.market_buy(5.0).unwrap();
        // 2 @ 101 + 3 @ 102 = 508 / 5 = 101.6
        assert!((avg_price - 101.6).abs() < 1e-9);
        assert!((filled - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_state_discretization() {
        let state = MarketState {
            mid_price: 100.0,
            spread: 0.02,
            bid_depth: 10.0,
            ask_depth: 5.0,
            inventory: 2.0,
            time_remaining: 0.5,
            volatility: 0.01,
        };
        let key = state.to_key();
        assert_eq!(key.inventory_bin, 2);
        assert_eq!(key.time_bin, 5);
    }

    #[test]
    fn test_dqn_agent_action_selection() {
        let mut agent = DQNMarketMaker::new(25);
        let state = StateKey {
            spread_bin: 2,
            inventory_bin: 0,
            imbalance_bin: 0,
            time_bin: 5,
        };
        // With epsilon=1.0, action is random (just check it doesn't crash).
        let action = agent.select_action(&state);
        assert!(action < 25);

        // Set epsilon to 0 for greedy.
        agent.epsilon = 0.0;
        // Set a specific Q-value.
        agent
            .q_table
            .insert(state.clone(), vec![0.0; 25]);
        agent.q_table.get_mut(&state).unwrap()[12] = 1.0;
        let action = agent.select_action(&state);
        assert_eq!(action, 12);
    }

    #[test]
    fn test_q_learning_update() {
        let mut agent = DQNMarketMaker::new(25);
        let state = StateKey {
            spread_bin: 1,
            inventory_bin: 0,
            imbalance_bin: 0,
            time_bin: 5,
        };
        let next_state = StateKey {
            spread_bin: 1,
            inventory_bin: 1,
            imbalance_bin: 0,
            time_bin: 4,
        };
        let exp = Experience {
            state: state.clone(),
            action: 5,
            reward: 1.0,
            next_state,
            done: false,
        };
        agent.train_step(&exp);
        let q_values = agent.get_q_values(&state);
        // After one update: Q = 0 + 0.01 * (1.0 + 0.99*0 - 0) = 0.01
        assert!((q_values[5] - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_environment_step() {
        let mut env = TradingEnvironment::new(
            100.0, 0.01, 1.0, 100, 0.1, 0.001,
        );
        let state = env.reset();
        assert!(state.mid_price > 0.0);
        assert!(state.time_remaining > 0.0);

        let (next_state, _reward, done) = env.step(12);
        assert!(!done);
        assert!(next_state.time_remaining < state.time_remaining);
    }

    #[test]
    fn test_full_episode() {
        let mut env = TradingEnvironment::new(
            100.0, 0.01, 1.0, 50, 0.1, 0.001,
        );
        let mut agent = DQNMarketMaker::new(env.num_actions());
        let result = run_episode(&mut env, &mut agent, true);
        assert_eq!(result.steps, 50);
    }
}
