# Chapter 253: Document Classification in Finance - Simple Explanation

## What is Document Classification?

Imagine you work in a library and hundreds of new books arrive every day. Your job is to put each book on the right shelf: fiction goes here, science goes there, history goes in that corner. You read the title, flip through a few pages, and decide where it belongs.

Document classification in finance works the same way! Instead of books, we have financial documents like news articles, company reports, and analyst opinions. Instead of shelves, we have categories like "good news", "bad news", "about mergers", or "about earnings". And instead of a librarian, we train a computer to sort them automatically.

## How Does the Computer Read?

A computer cannot read words like you do. So we need to translate words into numbers. Here is the simplest way:

Imagine you have a checklist of every word you know. For each document, you go through the checklist and count how many times each word appears. A document about earnings might have the word "profit" 5 times and "loss" 0 times. A document about a company failing might have "profit" 0 times and "loss" 7 times.

These number lists become the computer's "fingerprint" for each document. Documents with similar fingerprints probably belong to the same category!

## Smart Word Counting: TF-IDF

Not all words are equally important. The word "the" appears in every document, so it tells us nothing. But the word "acquisition" is special - it mostly shows up in documents about one company buying another.

**TF-IDF** is a smart way of counting words. It says: "If a word appears a lot in THIS document but rarely in OTHER documents, it must be really important for understanding what THIS document is about." It is like a teacher who gives more points for answering hard questions that most students got wrong.

## Teaching the Computer to Sort

We teach the computer like training a puppy. We show it many examples:

- "Here is a document about earnings. See, it has words like 'revenue', 'profit', 'guidance'."
- "Here is a document about a merger. See, it has words like 'acquisition', 'deal', 'takeover'."
- "Here is a document about a regulation. See, it has words like 'SEC', 'compliance', 'penalty'."

After seeing thousands of examples, the computer learns patterns. When a new document arrives, it checks which pattern it matches most closely and puts it in the right category - just like our librarian!

## The Simplest Sorter: Naive Bayes

The simplest sorting method is called "Naive Bayes", and it works like this:

Imagine you are trying to guess if someone is a basketball player or a jockey just by looking at their height. Tall people are more likely to be basketball players. Short people are more likely to be jockeys. You combine all the clues (height, weight, shoe size) to make your best guess.

Naive Bayes does the same thing with words. If a document has the word "dividend", that is a clue it might be about earnings. If it also has "quarterly" and "EPS", those are more clues pointing the same way. It multiplies all the clues together to make its best guess.

## Why This Matters in Finance

- **For traders**: Imagine getting a news alert that says "this article is about a company being bought" before anyone else reads it. You could trade on that information faster!
- **For risk managers**: Imagine a computer reading every news article about companies you have invested in and immediately warning you if something bad is happening
- **For compliance teams**: Imagine automatically sorting thousands of regulatory filings into the right categories so the right people review them

## Try It Yourself

Our Rust program shows document classification in action:
1. It creates a vocabulary of financial words (like building a dictionary)
2. It converts documents into number fingerprints using TF-IDF
3. It trains a classifier on example financial documents
4. It connects to Bybit (a crypto exchange) to get real market data
5. It classifies market events and predicts their impact on prices

It is like building a robot librarian that specializes in financial documents!
