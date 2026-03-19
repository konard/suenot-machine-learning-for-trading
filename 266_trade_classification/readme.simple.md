# Trade Classification: Who Started the Trade?

## What Is Trade Classification?

Imagine you are at a lemonade stand. Two people are involved in every sale: the person selling lemonade and the person buying it. But who made the deal happen? Did the buyer walk up and say "I want lemonade!" or did the seller call out "Come buy my lemonade!"?

In the stock market and crypto exchanges, something similar happens with every trade. A trade happens when a buyer and a seller agree on a price. But we want to know: **who was more eager to make the trade happen?** Was it the buyer rushing in, or the seller rushing out?

This is called **trade classification** -- figuring out who started (or "initiated") each trade.

## Why Does It Matter?

Think of it like a game of tug-of-war. If more trades are started by buyers, the price tends to go up (buyers are pulling the rope their way). If more trades are started by sellers, the price tends to go down.

Knowing who started each trade helps us:
- **Predict price movements**: If buyers are dominating, prices might keep going up
- **Understand market health**: A market where both sides are active is usually healthy
- **Build smarter trading robots**: Robots can make better decisions with this information

## How Do We Figure It Out?

### Method 1: The Price Direction Trick (Tick Test)

This is the simplest method. It is like watching a bouncing ball:
- If the price went **UP** from the last trade, a buyer probably started it (they were willing to pay more)
- If the price went **DOWN**, a seller probably started it (they were willing to sell for less)
- If the price **stayed the same**, we just guess it is the same as the last one

It is like watching which way a seesaw is tilting!

### Method 2: The Middle Price Rule (Quote Rule)

Imagine a store has a sign that says "We buy for $9, we sell for $11." The middle price is $10.

- If someone buys at $10.50 (above the middle), they were probably eager to buy
- If someone buys at $9.50 (below the middle), the seller was probably eager to sell

The middle of the buy-and-sell prices tells us a lot!

### Method 3: Lee-Ready (The Best of Both)

A scientist named Lee and another named Ready said: "Why not use BOTH methods?" Their algorithm:
1. First, check the middle price rule
2. If that does not give a clear answer, use the price direction trick

It is like asking two friends for advice and going with whichever one has a clearer answer!

### Method 4: Bulk Volume Classification (For Big Batches)

Sometimes we do not have information about individual trades -- just summaries (like "1000 shares traded in the last minute"). BVC is clever: it looks at how the price moved during that batch and uses math to guess what fraction were buys vs sells.

It is like estimating how many blue and red marbles are in a jar by looking at the overall color!

### Method 5: Machine Learning Ensemble

Our smartest method uses a computer brain (machine learning) that looks at LOTS of clues at once:
- Where was the trade price compared to the middle?
- How big was the trade?
- Which direction was the price moving?
- Were there more buy orders or sell orders waiting?

Then it combines its guess with all the other methods, like a team of detectives pooling their clues!

## A Fun Analogy

Imagine you are a detective at an auction house. You cannot see who raises their paddle first, but you CAN see:
- The final price of each item (did it go up or down?)
- The auctioneer's starting price (the "middle" price)
- How quickly the bidding happened
- How many bidders are in the room

Using all these clues, you can figure out whether the buyer or seller was more motivated for each sale. That is exactly what trade classification does!

## Try It Yourself

The code in this chapter:
1. Creates pretend trades and classifies them with all five methods
2. Fetches REAL trades from Bybit (a crypto exchange) and tests how well each method works
3. Shows which method is most accurate by comparing against the exchange's own labels

It is like a science experiment where we test our detective methods against the answer key!
