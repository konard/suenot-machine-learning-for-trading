# Message Passing Neural Networks: A Simple Guide

## What is a Message Passing Neural Network?

Imagine a classroom where students are trying to solve a difficult math problem. Instead of working alone, each student can whisper hints to their neighbors. After several rounds of whispering, every student knows more than they started with because they've combined their own knowledge with what they learned from friends.

**That's exactly how Message Passing Neural Networks (MPNNs) work!**

## The Telephone Game Analogy

Remember the telephone game? You whisper a message to your friend, they whisper to the next person, and so on. By the end, the message has traveled through everyone.

MPNNs are similar, but smarter:
- Each person (node) starts with some information
- They "pass messages" to their connected friends
- Everyone combines what they hear with what they know
- After a few rounds, everyone has learned from the whole group

## Real-Life Example: School Gossip Network

Let's say there's news about a surprise test next week. Here's how it spreads:

```
       [Alice] -------- [Bob]
          |              |
          |              |
       [Carol] ------- [Dave] -------- [Emma]
```

1. **Round 1**: Alice tells Bob and Carol
2. **Round 2**: Bob tells Dave, Carol tells Dave (Dave now heard it twice!)
3. **Round 3**: Dave tells Emma

Notice that:
- Dave is more "connected" - he hears from multiple sources
- Emma only has one connection - she might miss news
- The structure of friendships matters!

## Why Use This for Trading?

### The Market is Like a Social Network

In cryptocurrency markets:
- **Bitcoin (BTC)** is like the popular kid - when something happens to them, everyone notices
- **Ethereum (ETH)** is Bitcoin's best friend - they often move together
- **Altcoins** are like followers - they react to what the big coins do

```
         [BTC] ============ [ETH]
          /|\                /|\
         / | \              / | \
        /  |  \            /  |  \
    [SOL] [ADA] [DOT]  [UNI] [AAVE] [LINK]
```

When BTC goes up:
1. ETH usually follows quickly (strong connection)
2. Other coins react later (weaker connections)
3. Some coins might not react at all (no connection)

### The "Whisper" in Trading

In MPNNs for trading, the "messages" being passed are:
- Price changes
- Volume spikes
- Momentum signals
- Risk warnings

## How Does It Work? Step by Step

### Step 1: Build the Network

First, we need to figure out who is "friends" with whom. For crypto:

```
Connection Strength = How much do these coins move together?

BTC <---> ETH:  Very Strong (0.9)
BTC <---> DOGE: Medium (0.5)
ETH <---> UNI:  Strong (0.7)  # UNI runs on Ethereum
SOL <---> DOGE: Weak (0.2)
```

### Step 2: Gather Information

Each coin starts with its own data:
- Today's price change: +5%
- Volume: High
- Momentum: Bullish
- Volatility: Medium

### Step 3: Pass Messages

**Round 1:**
- BTC tells ETH: "I'm going up 5%!"
- ETH tells BTC: "Me too, up 3%!"
- They both update their understanding

**Round 2:**
- BTC now knows: "I'm up, ETH is up... this might be a real trend"
- ETH now knows: "BTC is leading, I should pay attention"

**Round 3:**
- The signal spreads to altcoins
- Coins update their predictions based on the whole network

### Step 4: Make Decisions

After all the message passing, each coin has a "score":
- High score = Strong buy signal
- Low score = Strong sell signal
- Middle score = No clear signal

## A Day in the Life of an MPNN Trader

### Morning: News Breaks

Breaking news: "Major company announces Bitcoin investment"

### The Network Reacts

```
9:00 AM - BTC spikes 5%
  |
  v
9:01 AM - MPNN detects: "BTC signal spreading to ETH"
  |
  v
9:02 AM - MPNN predicts: "Altcoin rally likely in 5-10 minutes"
  |
  v
9:03 AM - Buy signals generated for selected altcoins
  |
  v
9:10 AM - Altcoins start rallying (as predicted!)
  |
  v
9:15 AM - MPNN detects weakening signals, prepares to exit
```

## The Three Key Parts

### 1. Message Function (The Whisper)

What information do we share?

Like choosing what to tell your friend:
- "The test is on Monday" (specific info)
- "I'm worried about the test" (sentiment)
- "I'm going to study all weekend" (action)

In trading:
- "BTC is up 5%" (price info)
- "Volume is 3x normal" (market activity)
- "RSI is overbought" (technical signal)

### 2. Aggregation Function (The Listening)

How do we combine messages from multiple friends?

Options:
- **Sum**: Add up everything you hear (total influence)
- **Average**: Take the middle ground (balanced view)
- **Maximum**: Only remember the most important thing (focus on extremes)

Example:
```
You hear from 3 friends:
- "BTC up 5%"
- "BTC up 4%"
- "BTC up 6%"

Sum: Total bullish pressure = 15%
Average: Average signal = 5%
Maximum: Strongest signal = 6%
```

### 3. Update Function (The Learning)

How do we update our own opinion?

Like updating your study plan after talking to friends:
- Old plan: Study for 2 hours
- Friend's advice: "The test is really hard!"
- New plan: Study for 4 hours

In trading:
- Old prediction: Slightly bullish
- Network messages: Strong bullish from neighbors
- New prediction: Very bullish

## Why Graphs Beat Simple Methods

### Traditional Approach: Everyone for Themselves

```
BTC: "I'm up, so I'm bullish on myself"
ETH: "I'm up, so I'm bullish on myself"
SOL: "I'm down, so I'm bearish on myself"
```

Problem: Misses that BTC and ETH rising might help SOL!

### MPNN Approach: Network Intelligence

```
BTC: "I'm up + ETH is up + the whole market feels bullish"
ETH: "I'm up + BTC leading + DeFi sector strong"
SOL: "I'm down BUT BTC/ETH rising = opportunity!"
```

The MPNN catches that SOL might be a buying opportunity!

## Real Trading Example

### Scenario: Market Crash Recovery

Day 1: Market crashes -20%
```
All nodes: [FEAR FEAR FEAR]
```

Day 3: BTC shows first green candle
```
BTC: "I'm recovering"
     |
     v
MPNN: "Signal sent to ETH"
     |
     v
ETH: "BTC recovering + I'm stabilizing = good sign"
     |
     v
MPNN: "Positive signals spreading through network"
     |
     v
Altcoins: "Leaders recovering + fear decreasing = BUY signals"
```

The MPNN detects the recovery spreading through the network before each individual asset clearly shows it!

## Key Takeaways

1. **Markets are networks** - Crypto assets are connected, not independent
2. **Information spreads** - What happens to BTC affects everything
3. **Structure matters** - Some coins are more connected than others
4. **Timing is key** - MPNNs can detect signals before they fully propagate
5. **More than correlation** - MPNNs learn complex, multi-hop relationships

## Summary: The MPNN Recipe

1. **Ingredients**:
   - Crypto assets (nodes)
   - Connections (edges)
   - Market data (features)

2. **Process**:
   - Build the network of relationships
   - Let each asset share information
   - Combine messages from neighbors
   - Update predictions
   - Repeat several times

3. **Result**:
   - Each asset gets a signal informed by the whole network
   - Signals spread naturally through connections
   - Early detection of market-wide moves

## Next Steps

Ready to see this in action? Check out the Rust examples in the `rust/` folder:
- `basic_mpnn.rs` - Simple example with fake data
- `bybit_signals.rs` - Real crypto data from Bybit
- `backtest.rs` - Test the strategy on historical data

Remember: The network is smarter than any single node!
