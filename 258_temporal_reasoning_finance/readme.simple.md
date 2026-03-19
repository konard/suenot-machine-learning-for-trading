# Chapter 258: Temporal Reasoning in Finance - Simple Explanation

## What is Temporal Reasoning?

Imagine you are reading a detective story. The detective needs to figure out the order of events: "The suspect left the house at 3 PM. The robbery happened at 4 PM. The suspect was seen at the mall at 5 PM." By understanding WHEN things happened, the detective can figure out what is possible and what is not.

Temporal reasoning in finance works the same way! Computers read financial news, reports, and announcements, and they figure out WHEN things happened or will happen. Then they use this timeline to make smart trading decisions.

## Why Does Time Matter in Finance?

Think about a school cafeteria. If you know that pizza day is every Friday, you can plan ahead and bring extra money on Fridays. But if someone tells you "pizza day is moving to Wednesday starting next month," you now have new time information that changes your plan.

Financial markets work just like this:
- **Earnings reports** come out on specific dates - traders prepare for them
- **Interest rate decisions** happen on scheduled days - everyone watches the calendar
- **Token unlocks** in crypto release new coins at specific times - prices often drop

Knowing WHEN something will happen gives you a huge advantage, just like knowing pizza day lets you plan your lunch money!

## The Time Detective: Finding Dates in Text

Imagine a computer reading this sentence: "Apple will report Q3 earnings on August 1, after which analysts expect guidance for the holiday quarter."

The time detective (our computer program) finds:
- "Q3" = the third quarter of the year (July-September)
- "August 1" = a specific date
- "after which" = something happens AFTER August 1
- "holiday quarter" = Q4, October-December

Now the computer knows the ORDER of events: earnings report first, then guidance, then the holiday season. This timeline helps predict what will happen to the stock price!

## Putting Events in Order: Allen's Rules

Think of time like blocks you put on a table. Two blocks (events) can relate to each other in different ways:

- **Before**: "Lunch is BEFORE recess" - lunch ends, then recess starts
- **After**: "Recess is AFTER lunch" - the opposite view
- **During**: "It rained DURING the game" - rain happened while the game was playing
- **Overlaps**: "The movie OVERLAPS with dinner time" - part of the movie is during dinner
- **Meets**: "School MEETS soccer practice" - school ends exactly when practice starts

A mathematician named James Allen figured out there are exactly 13 ways two events can relate in time. Computers use these rules to build a complete picture of how all events connect!

In finance, this helps answer questions like: "Did the CEO sell shares BEFORE or AFTER they knew about the bad news?"

## The Time Memory: Temporal Knowledge Graphs

Imagine a giant notebook where you write down facts about the world, but each fact has a "valid from" and "valid until" date:

- "Tim Cook is CEO of Apple (since August 2011)"
- "Interest rate is 5.25% (from July 2023 to September 2024)"
- "Bitcoin trend is bullish (from January to March 2024)"

This notebook is called a **temporal knowledge graph**. It is like a regular encyclopedia, but it remembers that facts CHANGE over time. The interest rate is not always 5.25% - it was different before and will be different later.

This is super useful because you can ask: "What was happening in the economy WHEN Bitcoin was going up?" and the notebook can tell you!

## How Computers Score Time Signals

The computer creates special scores from the time information it finds:

### Freshness Score
How recent is the news? Yesterday's earnings report matters more than last year's. The computer gives higher scores to newer events, like how yesterday's weather forecast is more useful than last month's.

### Future-Looking Score
Is the document talking about the past or the future? A document that says "we expect growth next quarter" is more useful for trading than one that says "we grew last quarter." The computer counts how many "future" words versus "past" words there are.

### Time Density Score
How many time references are in the document? An earnings report full of dates ("Q1 revenue was... Q2 guidance is... by year-end we expect...") is more useful for temporal trading than a vague opinion piece with no dates.

## Real-World Examples

### The Earnings Calendar Trader
Imagine you know that:
1. Company XYZ reports earnings on February 15
2. Their biggest customer reports on February 10
3. The industry conference is on February 20

A temporal reasoning system connects these dots: the customer's report on Feb 10 might hint at XYZ's results on Feb 15. And the conference on Feb 20 might reveal what happens next quarter. Smart traders use this timeline to position themselves!

### The Crypto Unlock Tracker
In cryptocurrency, tokens are often "locked up" and released on a schedule. Imagine a game where new playing cards are added every month. When lots of new cards flood in, each card becomes less special (the price drops).

A temporal reasoning system tracks ALL upcoming token unlocks across hundreds of projects and alerts traders: "Warning! $500 million in tokens unlock next Tuesday for Project ABC."

## Why This Matters

- **For traders**: It is like having a calendar that automatically finds every important financial date and tells you what to expect
- **For analysts**: It is like having a research assistant who can instantly build a timeline of everything that happened to a company
- **For risk managers**: It is like having an alarm system that warns you about upcoming events that could affect your investments
- **For everyone**: It helps make sense of the overwhelming amount of financial news by organizing it on a timeline

## Try It Yourself

The Rust code in this chapter lets you:
1. Parse financial text and extract temporal expressions (dates, quarters, deadlines)
2. Build timelines of financial events
3. Compute temporal features (freshness, future-looking ratio)
4. Backtest a temporal trading strategy using real Bybit cryptocurrency data

Run the example to see temporal reasoning in action with live market data!
