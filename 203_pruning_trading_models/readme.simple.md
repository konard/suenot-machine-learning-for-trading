# Pruning Trading Models - Simple Explanation

Imagine a tree with hundreds of branches. Some branches have lots of fruit, others have none. Pruning is like cutting off the empty branches so the tree can focus its energy on growing the best fruit!

## What is a Neural Network?

Think of a neural network like a giant web of connected strings. Each string has a different thickness. Thick strings are important -- they carry a lot of information. Thin strings barely do anything at all.

When we build a brain for a trading computer, we start with LOTS of strings because we don't know which ones will be useful yet. After training (teaching the computer), we discover that many strings are so thin they're practically invisible!

## What is Pruning?

Pruning means cutting away the parts that don't matter. Just like a gardener trims a bush:

- **Before pruning**: A big, bushy plant with branches going everywhere. Some branches have beautiful flowers, but many are just wasting the plant's energy.
- **After pruning**: A neat, focused plant where every branch has flowers. It's smaller but actually healthier!

For our trading computer:
- **Before pruning**: A big, slow brain with millions of connections. Many connections do almost nothing.
- **After pruning**: A small, fast brain with only the important connections. It makes decisions just as well, but MUCH faster!

## Why Does Speed Matter in Trading?

Imagine you and your friend both see a $1 bill on the ground. Whoever picks it up first gets to keep it! In trading, computers race to buy and sell things. The faster computer wins.

A pruned model is like a runner who took off their heavy backpack -- they can run much faster without losing any strength!

## How Do We Decide What to Cut?

It's like cleaning your room:

1. **Look at each toy** (each connection in the brain)
2. **Ask: "Do I play with this?"** (Is this connection important?)
3. **If you never play with it, donate it!** (Remove the connection)
4. **Check if your room still feels right** (Make sure the model still works well)

We measure importance by looking at how big each connection is. Big connections = important. Tiny connections = can be removed.

## The Step-by-Step Process

1. **Train the big model**: Teach the computer with lots of data
2. **Look at all the connections**: Find the ones that are very small
3. **Cut the small ones**: Remove them carefully
4. **Check the results**: Does the computer still make good predictions?
5. **Repeat**: Cut a little more, check again, cut a little more...

It's like slowly removing Jenga blocks -- you keep going until the tower starts to wobble, then you stop!

## A Real Example

Say our trading computer has 1,000,000 connections:

| How Much We Cut | Connections Left | Still Works Well? |
|:---:|:---:|:---:|
| Cut 50% | 500,000 | Yes! Almost the same! |
| Cut 75% | 250,000 | Yes! Still pretty good! |
| Cut 90% | 100,000 | Yes! A tiny bit less accurate |
| Cut 95% | 50,000 | Getting wobbly... |
| Cut 99% | 10,000 | Too much! It forgot things! |

## Why This Is Cool

- **Faster decisions**: The small brain thinks quicker
- **Less memory**: It fits in tiny computers
- **Better focus**: It only remembers the important stuff, like a student who studies the key topics instead of trying to memorize the entire textbook!

## The Fun Part

Scientists discovered something amazing: when you start with a big brain and prune it, the small brain works BETTER than if you had started with a small brain from the beginning. It's like the big brain figured out which connections matter, and then we kept only those!

This is called the "Lottery Ticket" idea -- inside every big brain, there's a winning small brain hiding, just waiting to be found!
