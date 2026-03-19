# LIT Transformer LOB -- Explained Simply

Imagine a translator that reads both the price list AND customer requests to predict what sells next.

## What is a Limit Order Book?

Think of a farmer's market where every seller has a sign showing their prices and how much they have. On one side, sellers say "I'll sell 5 apples for $2 each." On the other side, buyers say "I'll buy 3 apples for $1.50 each." All these signs together make up the "order book" -- it is a big list of who wants to buy and sell, and at what price.

## What does LIT do?

LIT is like a really smart helper who watches two things at the same time:

1. **The price signs** -- All the buy and sell signs at the market (the order book). The helper notices things like "Wow, there are tons of people wanting to buy at $1.90, but almost nobody selling at $2.00."

2. **What people actually bought** -- The helper also watches the cash register and sees every sale: "Someone just bought 100 apples really fast! And then 50 more right after!"

## How does it work?

The helper has three special skills:

### Skill 1: Reading the Price Signs
The helper looks at all the buy and sell signs and figures out which ones are important. Signs that are close to the current price matter more than signs far away. If there is a huge pile of buy orders right below the current price, that is a strong signal!

### Skill 2: Watching Recent Sales
The helper remembers the last bunch of sales and looks for patterns. Did someone just buy a LOT really quickly? Are sales speeding up or slowing down? This is like watching a movie of recent shopping activity.

### Skill 3: Putting It All Together
Here is the magic part. The helper combines what they learned from the signs AND the sales. For example: "There is a huge wall of buy orders at $1.90 AND someone just bought 200 apples aggressively. The price is probably going UP!"

## Why is this cool?

Most old-fashioned helpers just mashed all the numbers together into one big soup. LIT keeps things organized -- it reads the price signs carefully, watches the sales carefully, and THEN combines them. It is like the difference between:

- **Old way**: Dumping all your ingredients into a bowl at once
- **LIT way**: Preparing each ingredient perfectly, then combining them like a chef

## The prediction

After combining everything, LIT makes a simple prediction with three choices:
- **Price goes UP** (thumbs up)
- **Price goes DOWN** (thumbs down)
- **Price stays the SAME** (shrug)

And it does this incredibly fast -- thousands of times per second -- because it is written in Rust, which is like a super-fast race car for computers!
