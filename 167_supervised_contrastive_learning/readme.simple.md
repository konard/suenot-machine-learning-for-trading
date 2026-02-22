# SupCon: The "Library" Analogy

Imagine you are a librarian and you need to arrange books on shelves.

### 1. Standard Training (Classifier)
You are shown a book and told: "This is a detective novel." You memorize the cover. If a book looks like previous detective novels, you put it in the same stack. But you don't really think about how much detective novels *differ* from romances — you just look for common features.

### 2. Self-Supervised Learning
You are given the same book with different covers (black and white vs. color). You realize it's the same book and put them together. You learn to recognize the structure of the text, but you still don't know the genres.

### 3. SupCon (Supervised Contrastive)
You are told: "Here are 5 different detective novels and 5 different romances."
Your task is to put **all detective novels** in one corner and **all romances** in the opposite one.
In doing so, you don't just put them in stacks; you try to make sure the distance between any two detective novels is minimized, while the distance between a detective novel and a romance is maximized.

In trading, SupCon forces the neural network to learn: "All moments before a sharp price increase should look the same in my 'imagination' (latent space), regardless of whether it happened on Apple or Bitcoin, while moments before a fall should look fundamentally different."
This creates a very clear "map" of the market, where "Buy" and "Sell" regions do not overlap.
