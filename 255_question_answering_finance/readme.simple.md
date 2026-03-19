# Chapter 255: Question Answering for Finance - Simple Explanation

## What is Financial Question Answering?

Imagine you have a huge library filled with thousands of company reports, news articles, and financial tables. You want to find out: "How much money did Company X make last year?" Instead of reading through hundreds of pages yourself, you ask a smart robot assistant to find the answer for you.

That is exactly what financial question answering does! It is like having a super-fast librarian who can read all those documents in seconds and give you the exact answer you need.

## Finding the Answer: Extractive QA

Think about a highlighter pen. When your teacher asks a question about a reading passage, you scan through the text and highlight the sentence that contains the answer.

**Extractive QA** works the same way. The computer reads a passage and "highlights" the part that answers your question. For example:

- **Document**: "Apple reported revenue of $94.8 billion for Q3 2024, an increase of 5% year over year."
- **Question**: "What was Apple's revenue in Q3 2024?"
- **Answer** (highlighted): "$94.8 billion"

The computer learns which words to highlight by studying thousands of question-answer examples, just like you get better at finding answers on tests the more you practice!

## Doing Math: Numerical Reasoning

Sometimes the answer is not written directly in the text. You need to do some math first!

Imagine your friend asks: "How much taller did you grow this year?" The doctor's chart shows you were 150 cm last year and 155 cm this year. The answer is not written anywhere — you have to subtract: 155 - 150 = 5 cm.

Financial QA works similarly. If a report says revenue was $100 million last year and $120 million this year, and someone asks "What was the revenue growth?", the computer has to:
1. Find both numbers
2. Calculate: ($120M - $100M) / $100M = 20% growth

This is called **numerical reasoning** — teaching computers to do math with numbers they find in documents.

## Finding the Right Book: Retrieval

Imagine you are in a library with 10,000 books and someone asks: "What is Tesla's market cap?" You cannot read all 10,000 books! Instead, you:

1. First, go to the catalog and find which books are about Tesla
2. Then, look through just those few books for the answer

This two-step process is called **retrieval-augmented QA**:
- Step 1 (the Retriever): Find the right documents
- Step 2 (the Reader): Find the answer in those documents

It is like having two helpers — one who is great at finding the right books, and another who is great at reading them!

## Reading Tables

Financial reports are full of tables with rows and columns of numbers. Reading tables is tricky because you need to understand which row and which column contain the information you need.

Think of it like a multiplication table you use in school. If someone asks "What is 7 times 8?", you find row 7 and column 8, then look at where they meet. Financial table QA works the same way — find the right row (maybe "Revenue") and the right column (maybe "2024") to get the answer.

## Knowing When You Don't Know

A really smart assistant does not just give answers — it also tells you when it is not sure. Imagine a weather app that says "80% chance of rain" versus one that just says "rain". The first one is more helpful because you know how confident it is.

Our QA system does the same thing. It might say: "The answer is $94.8 billion (confidence: 92%)" or "I am not sure, but it might be around $90 billion (confidence: 45%)." This way, you know when to trust the answer and when to double-check it yourself.

## Why This Matters

- **For investors**: Instead of reading 200-page annual reports, ask specific questions and get instant answers
- **For traders**: Quickly extract numbers from earnings calls to make faster trading decisions
- **For analysts**: Automate the tedious parts of research so you can focus on strategy
- **For everyone**: Makes financial information more accessible, even if you are not an expert

## Try It Yourself

Our Rust program builds a mini financial QA system that can:
1. Store financial documents about companies and crypto markets
2. Classify what type of question you are asking
3. Search through documents to find relevant information
4. Extract answers or compute them when math is needed
5. Connect to Bybit exchange for real-time crypto market data

It is like building your own smart financial librarian!
