# Chapter 241: FinBERT Sentiment - Simple Explanation

## What is Sentiment Analysis?

Imagine you are a detective reading through a huge pile of letters. Some letters say nice things like "The company is doing great!" and "Profits are soaring!" Other letters say worrying things like "Sales are declining" and "The CEO just resigned." Your job is to figure out whether, overall, people are feeling happy or worried about a company.

Sentiment analysis does exactly this, but with a computer! It reads financial news, reports, and social media posts, and decides whether each one sounds positive, negative, or neutral.

## Why Regular Computers Get Confused

Regular word-counting programs get confused by financial language. Imagine if someone said "The company's risk management is excellent." A simple program might see the word "risk" and think it is bad news, but actually the sentence is saying something positive!

Or consider "The stock did not decline." A simple program might see "decline" and think it is negative, but the sentence actually means something good happened.

## What Makes FinBERT Special

FinBERT is like a student who has studied at a finance school. It has read thousands of financial documents and learned the special language of Wall Street.

Think of it this way:
- A **regular computer** is like someone who learned English from a dictionary — they know what words mean individually, but get confused by context
- **FinBERT** is like a financial analyst who has worked on Wall Street for years — they understand that "restructuring" usually means bad news, while "beat expectations" is great news

FinBERT learned by reading a huge library of financial texts, like a student cramming for an exam. After all that studying, it can read a new headline and immediately tell you whether it sounds bullish or bearish.

## How It Scores Feelings

FinBERT gives each piece of text three scores, like votes:
- **Positive vote**: "This sounds like good news" (0% to 100%)
- **Negative vote**: "This sounds like bad news" (0% to 100%)
- **Neutral vote**: "This is just facts, no strong feeling" (0% to 100%)

For example, for the headline "Company reports record profits":
- Positive: 92%
- Negative: 3%
- Neutral: 5%

We combine these into a single "mood score" from -1 to +1:
- **+1**: Extremely positive (everyone is cheering!)
- **0**: Neutral (nobody cares)
- **-1**: Extremely negative (everyone is worried!)

## Using Moods to Trade

Imagine you are at a school election. If everyone is saying nice things about a candidate, that candidate will probably win. Similarly, if everyone is saying positive things about a stock, its price might go up.

Our trading strategy works like this:
1. **Read** all the latest news about a stock or cryptocurrency
2. **Score** each piece of news (is it positive, negative, or neutral?)
3. **Average** all the scores to get an overall mood
4. **Trade** based on the mood:
   - If the mood is very positive → buy
   - If the mood is very negative → sell
   - If the mood is mixed → wait

## Why This Matters

- **For traders**: It is like having a team of analysts reading every news article in the world and summarizing their opinions in one number
- **For crypto markets**: Crypto prices are especially affected by mood — a single tweet can move Bitcoin! FinBERT helps track the overall mood across thousands of social media posts
- **For everyone**: It helps make better investment decisions by turning messy human opinions into clear signals

## Try It Yourself

Our Rust program demonstrates sentiment-based trading:
1. Analyzes financial texts using keyword-based sentiment (like a simpler version of FinBERT)
2. Aggregates sentiment scores over time (computes the overall mood)
3. Connects to a real crypto exchange (Bybit) for price data
4. Generates trading signals based on sentiment thresholds
5. Tracks how the strategy would have performed

It is like building a robot that reads the news and decides whether to buy or sell based on what it reads!
