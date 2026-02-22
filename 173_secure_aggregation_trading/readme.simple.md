# Secure Aggregation: The "Sealed Envelopes" Analogy

Imagine three traders who want to find their group's average profit, but no one wants to reveal their individual profit to the others.

### Method 1: Trusted Intermediary (Standard Server)
They all tell the intermediary: "My profit is 100", "Mine is 200", "Mine is 300". The intermediary calculates the average (200) and announces it.
**Problem**: The intermediary now knows everyone's individual profit. In trading, this is dangerous — the intermediary could steal your strategy.

### Method 2: Secure Aggregation (Masking)
Traders A, B, and C agree secretly:

1. **Exchange of Numbers**:
   - A and B agree on a random number 50 (A adds it, B subtracts it).
   - B and C agree on a random number 30 (B adds it, C subtracts it).
   - C and A agree on a random number 10 (C adds it, A subtracts it).

2. **Blinding the Data**:
   - Trader A sends to the server: $100 + 50 - 10 = 140$.
   - Trader B sends to the server: $200 - 50 + 30 = 180$.
   - Trader C sends to the server: $300 - 30 + 10 = 280$.

3. **Result**:
   - The server sees 140, 180, 280. These numbers are meaningless and do not reveal the real profits.
   - But when the server sums them up: $140 + 180 + 280 = 600$.
   - All the masks (+50 and -50, +30 and -30, +10 and -10) cancelled each other out!
   - The server gets the exact sum: $100 + 200 + 300 = 600$.

In the end, the server learned the total sum (and the average) but never understood how much each individuals earned. This is **Secure Aggregation**. We "blind" the server while still allowing it to compute the collective statistics.
