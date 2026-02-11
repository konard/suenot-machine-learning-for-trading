# Chapter 146: PINN CIR Model Explained Simply

## Imagine a Room with a Smart Thermostat...

Let's understand the CIR model and physics-informed neural networks through a simple story about temperature control.

---

## The Thermostat Problem

### Your room has a thermostat

You set your thermostat to **72 degrees** (that's your "theta" -- the comfortable temperature you want).

```
Outside: It's winter, 20 degrees.
Your thermostat setting: 72 degrees.
Current room temperature: 65 degrees (a bit chilly).

What happens?
  -> The heater kicks in, pushing temperature UP toward 72.
  -> The closer you get to 72, the LESS the heater works.
  -> Eventually, the room stabilizes near 72.
```

This is **mean-reversion** -- the temperature (interest rate) always pulls back toward the thermostat setting (theta).

### But the temperature wobbles!

Even with the thermostat, the temperature is not perfectly steady. The door opens, wind blows, the sun comes through a window. These random fluctuations are the "noise" in our model.

```
With a REGULAR thermostat (Vasicek model):
  Temperature fluctuates randomly around 72.
  Problem: Temperature could theoretically drop to -50 degrees!
  That's... not possible for a room temperature.

With a SMART thermostat (CIR model):
  Temperature fluctuates, BUT the fluctuations get SMALLER
  as the temperature approaches zero (absolute zero).
  Temperature can NEVER go below zero!
```

---

## Why Can't Temperature Go Below Zero?

### The key insight: sqrt(r)

In the CIR model, the amount of randomness is proportional to **the square root of the current value**.

```
If temperature is 100 degrees:
  Randomness = sigma * sqrt(100) = sigma * 10 --> BIG wobbles

If temperature is 4 degrees:
  Randomness = sigma * sqrt(4) = sigma * 2 --> small wobbles

If temperature is 0.01 degrees:
  Randomness = sigma * sqrt(0.01) = sigma * 0.1 --> TINY wobbles

If temperature is 0 degrees:
  Randomness = sigma * sqrt(0) = sigma * 0 = 0 --> NO wobbles at all!
```

When the temperature reaches zero, there's ZERO randomness, so it can never go negative. It's like a ball at the bottom of a bowl -- it can roll around, but gravity always brings it back.

### The Feller Condition: Will it actually reach zero?

There's a special rule called the "Feller condition":

```
2 * kappa * theta >= sigma^2

In thermostat terms:
  2 * (heater strength) * (target temperature) >= (wind strength)^2

If the heater is strong enough compared to the wind:
  -> Temperature NEVER reaches zero. You always stay warm!

If the wind is too strong:
  -> Temperature CAN reach zero temporarily. Brrr!
```

---

## What Are Interest Rates?

### The price of borrowing money

When you borrow money from a bank, you pay **interest**. The interest rate is like the "price" of borrowing:

```
You borrow $1,000 at 5% interest for 1 year.
You pay back: $1,000 + $50 = $1,050.

The "5%" is the interest rate.
```

### Interest rates change over time

Just like temperature, interest rates go up and down:

```
January:  5.0%  (economy is slow, rates go down)
March:    4.5%
June:     4.2%  (central bank cuts rates)
September: 4.8% (economy improves, rates go up)
December:  5.1% (inflation worries)
```

They tend to **mean-revert** -- when rates are too high, the central bank cuts them; when too low, they raise them. Just like a thermostat!

---

## What Are Bonds?

### A bond is an IOU

A bond is a promise to pay you money in the future:

```
You buy a bond for $95 today.
In 1 year, you get $100 back.
Your profit: $5 (about 5.26% return).
```

### Bond prices and interest rates are opposites

```
If interest rates GO UP:
  -> New bonds pay MORE interest
  -> Your OLD bond (paying less) is worth LESS
  -> Bond price DROPS

If interest rates GO DOWN:
  -> New bonds pay LESS interest
  -> Your OLD bond (paying more) is worth MORE
  -> Bond price RISES
```

This is why knowing future interest rates is so valuable -- it tells you what bonds are worth!

---

## The Math Problem: Pricing Bonds

### The CIR equation

Given the CIR model for interest rates, mathematicians derived an equation (a PDE) that tells us the "fair" price of a bond:

```
The CIR Bond Pricing PDE (don't worry about the details!):

dP/dt + kappa*(theta - r)*dP/dr + 0.5*sigma^2*r*d^2P/dr^2 - r*P = 0

Translation into plain English:
"The bond price changes over time in a way that balances:
 - How rates pull toward the average (mean-reversion)
 - How rates randomly fluctuate (diffusion, proportional to sqrt(r))
 - How the current rate eats into the bond value (discounting)"
```

### The good news: there's an exact answer!

Unlike many financial problems, the CIR model has a **beautiful closed-form solution**:

```
Bond Price = A(time) * e^(-B(time) * current_rate)

where A and B are specific formulas.
```

So why do we need a neural network? Read on!

---

## Enter the Neural Network

### Why use AI when we have a formula?

Good question! Here are some reasons:

```
1. SPEED: Once trained, the neural network is INSTANT
   - Formula: requires computing exp(), log(), sqrt() each time
   - Neural network: just multiply some matrices (super fast on GPUs)

2. FLEXIBILITY: What if the model is more complicated?
   - Real-world rates don't follow CIR exactly
   - Neural networks can handle messier equations

3. DERIVATIVES: The neural network gives us "sensitivities" for free
   - How does the bond price change if rates move? (automatic differentiation)

4. LEARNING FROM DATA: We can combine the math AND real market data
   - The formula assumes CIR is perfectly correct
   - The neural network can also learn from actual bond prices
```

---

## How the PINN Works

### What is "Physics-Informed"?

A normal neural network just learns from data (examples of inputs and outputs).

A **Physics-Informed** Neural Network also learns from the **equation itself**:

```
Normal neural network:
  "Here are 10,000 bond prices. Learn the pattern."

Physics-informed neural network:
  "Here are the LAWS OF FINANCE (the PDE).
   Also, here are some bond prices.
   Learn BOTH the laws and the data."
```

It's like teaching a student:
- **Normal approach**: Give them 100 exam answers to memorize
- **Physics-informed approach**: Teach them the formula AND give them practice problems

The physics-informed student can solve problems they've never seen before!

### The training process (simplified)

```
Step 1: Pick random points (interest rate, time)
        "What should the bond price be at r=3%, t=2 years?"

Step 2: Ask the neural network for its answer
        "I think the price is 0.85"

Step 3: Check if the answer obeys the CIR equation
        "Does dP/dt + kappa*(theta-r)*dP/dr + ... = 0?"
        "Hmm, it equals 0.003 instead of 0. That's an error!"

Step 4: Also check boundary conditions
        "At maturity (t=T), the bond pays $1. Does the network say 1?"
        "It says 0.98. That's wrong too!"

Step 5: Adjust the network to reduce ALL errors
        "Tweak the weights so the PDE error AND the boundary error get smaller"

Repeat 5,000 times.
```

---

## Real-World Application: Crypto Funding Rates

### What are funding rates?

On crypto exchanges like Bybit, there are "perpetual futures" -- contracts that never expire. To keep their price close to the real price, traders pay each other a "funding rate" every 8 hours:

```
Funding rate = 0.01% every 8 hours

If you're LONG (betting price goes up):
  You PAY 0.01% of your position to short sellers

If you're SHORT (betting price goes down):
  You RECEIVE 0.01% from long buyers
```

### Funding rates behave like the CIR model!

```
1. They mean-revert: When funding is too high, it drops back
2. They have level-dependent volatility: Higher rates = more volatile
3. They're bounded: The absolute value is always non-negative

This is exactly what the CIR model describes!
```

### A simple trading strategy

```
Step 1: Calibrate CIR to past funding rates
        "kappa=50, theta=0.01%, sigma=0.005"

Step 2: When funding rate is WAY above normal:
        "Rate is 0.1% but CIR says it should be 0.01%"
        -> Go SHORT (collect the high funding rate, expect it to drop)

Step 3: When funding rate is WAY below normal:
        "Rate is 0.001% but CIR says it should be 0.01%"
        -> Go LONG (pay small funding, expect rate to normalize)
```

---

## Credit Risk: Who Will Default?

### Another use of CIR: predicting defaults

Companies can go bankrupt (default on their debt). The probability of default changes over time -- just like interest rates!

```
CIR for default intensity:
  d(lambda) = kappa*(theta - lambda)*dt + sigma*sqrt(lambda)*dW

lambda = "how likely is this company to default RIGHT NOW?"

If lambda is high: Company is in trouble!
If lambda is low:  Company is healthy.

lambda always pulls back to theta (the long-run default risk).
lambda can never go BELOW zero (you can't have negative default risk).
```

The EXACT SAME math as bond pricing! The CIR PINN we built can also compute:

```
Survival probability: "What's the chance this company survives 5 years?"
CDS spread: "How much should default insurance cost?"
```

---

## Summary: The Big Picture

```
1. Interest rates are like room temperature
   -> They wander around but pull back to a "thermostat setting" (theta)
   -> CIR model: the wandering gets smaller as rates approach zero

2. Bond prices depend on interest rates
   -> The CIR PDE tells us the exact relationship
   -> We can solve it analytically (formula) OR with a neural network

3. PINN = Neural network that knows physics
   -> It learns the CIR equation, not just data points
   -> Faster inference, works with incomplete data, gives sensitivities

4. Real-world applications
   -> Crypto funding rates (Bybit) follow CIR-like dynamics
   -> Credit risk (default probability) uses the SAME math
   -> Trading strategies based on mean-reversion
```

---

## The Key Formulas (at a glance)

```
CIR Model:
  dr = kappa*(theta - r)*dt + sigma*sqrt(r)*dW
  "Rate changes = pull toward average + random wobble (smaller near zero)"

Feller Condition:
  2*kappa*theta >= sigma^2
  "Heater must be stronger than the wind"

Bond Price:
  P = A(tau) * exp(-B(tau) * r)
  "Price depends on time-to-maturity and current rate"

PINN Loss:
  L = PDE_error + boundary_error + data_error
  "Network must satisfy the equation AND match known values"
```

That's the CIR PINN in a nutshell!
