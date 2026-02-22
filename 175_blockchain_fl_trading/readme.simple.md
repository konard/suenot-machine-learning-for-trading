# Blockchain FL: The "Shared Ledger" Analogy

Imagine 10 banks that want to create a shared AI advisor, but they don't trust each other and don't want to hire a third-party company to manage a central server.

### 1. How it works (Decentralization)
Instead of sending their models to a "boss," the banks record their training results in a **Shared Ledger** (Blockchain) that everyone can see.

1. **Recording**: Bank A trains a model and writes in the ledger: "I did 100 iterations, and here is the encrypted hash of my work."
2. **Verification**: Other banks verify that Bank A actually performed the work.
3. **Merging**: At the end of the day, all banks look at the ledger and combine all entries into a new global model themselves. They don't need a "manager" because the rules for writing to the ledger are the same for everyone.

### 2. Why do it? (Audit and Integrity)
- **Fairness**: If Bank B sends bad data that ruins the global model, it remains in the ledger forever. They can be penalized.
- **Reliability**: If the building with the "main server" burns down, training doesn't stop because every bank has its own copy of the ledger.

In trading, this allows hedge funds to collaborate without fear that a server administrator will steal their alpha signals or manipulate the training results in favor of one participant.
