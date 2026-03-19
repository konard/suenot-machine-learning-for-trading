# LOBFrame Benchmark Trading - Explained Simply

## What is this about?

Imagine a school competition where all students solve the same test so we can fairly compare them. Some students might be great at math, others at reading, but unless they all take the same test under the same conditions, we can't really say who did better. That is exactly what LOBFrame does, but for computer programs that try to predict stock prices!

## The Order Book - Like a Line at a Store

Think of a market like a store where people line up to buy and sell things. On one side, you have buyers saying "I want to buy for this price." On the other side, sellers say "I will sell for this price." All these offers get written down in a big list called the **order book**.

The order book looks like two lines facing each other:
- **Buyers** (bids): "I will pay $99... I will pay $98... I will pay $97..."
- **Sellers** (asks): "I will sell for $101... I will sell for $102... I will sell for $103..."

The gap between the best buyer ($99) and best seller ($101) is called the **spread**. The middle point ($100) is called the **mid-price**.

## The Prediction Game

The big question is: **will the price go up, down, or stay the same?**

This is like trying to predict who will win a game. You look at the current situation (the order book) and make your best guess.

Different computer programs (called **models**) try to answer this question:
- **LSTM** - This model reads the order book like a story, remembering what happened before to guess what happens next
- **CNN** - This model looks at the order book like a picture, finding patterns and shapes in the data
- **Transformer** - This model can look at everything at once and figure out which parts are most important

## Why Do We Need a Benchmark?

Here is the problem: if each scientist tests their model on different data with different rules, we cannot compare them fairly!

It is like if one student takes a math test with a calculator and another takes it without one. Even if the first student scores higher, we cannot say they are better at math.

LOBFrame says: "Everyone plays by the same rules!"
- Same data, split the same way
- Same scoring system
- Same preparation steps

## How Do We Keep Score?

LOBFrame uses three ways to grade the models:

1. **Accuracy** - How many predictions were correct? Simple, like counting how many quiz answers you got right.

2. **F1 Score** - This is smarter. It checks if the model is good at finding ALL the ups AND not calling things "up" when they are not. Think of it like a detective who needs to catch all the bad guys (recall) but also never arrest innocent people (precision).

3. **MCC (Matthews Correlation Coefficient)** - This is the fairest score. Even if 90% of the time the price stays flat, a model that just always says "flat" would score high on accuracy but score zero on MCC. MCC catches cheaters!

## Making the Data Fair

Before we give data to the models, we need to make it fair. This is called **normalization**.

Think of it this way: if you are comparing how tall kids are in two different classes, but one class measures in centimeters and the other in inches, the numbers look very different even though the kids might be the same height! We need to convert everything to the same scale.

LOBFrame uses **z-score normalization**: it takes each number, subtracts the average, and divides by how spread out the numbers are. After this, all features are on the same scale.

Important rule: we only look at the "practice" data (training set) to figure out the scale. We never peek at the "test" data - that would be cheating, like reading the test answers before the exam!

## Real Trading with Bybit

LOBFrame does not just work with old data - it can also get fresh, live data from a real cryptocurrency exchange called **Bybit**. This means you can test your models on what is happening in the market right now!

It is like practicing for a sports competition with recorded games, but also being able to play practice matches against real teams.

## What Did We Learn?

1. **Fair tests matter** - You cannot compare models without a standard benchmark, just like you cannot compare students without the same test.

2. **One score is not enough** - Accuracy can be misleading. Using multiple scores (accuracy, F1, MCC) gives us a fuller picture.

3. **Preparation matters** - How you prepare the data (normalization) is just as important as the model itself.

4. **Start with basics** - Before trying fancy new models, make sure they can beat the simple ones (LSTM, CNN) on the standard benchmark.

5. **Practice and real life should match** - Using the same rules for research data and live trading data means your models will actually work in the real world.

Think of LOBFrame as the "official rulebook" for the LOB prediction competition. Everyone follows the same rules, uses the same scoring, and that is how we find out which models are truly the best!
