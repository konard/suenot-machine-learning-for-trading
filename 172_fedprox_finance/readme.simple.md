# FedProx: The "Rescue Team" Analogy

Imagine a rescue team working in different climate zones: some in the desert, some in the arctic, and some in the forest. They have a Shared Manual (Global Model).

### 1. The FedAvg Problem (Simple Averaging)
Each rescuer trains in their own way. The arctic rescuer might decide that the most important things are wool socks and heating. If they train for too long without communicating with headquarters, they will completely rewrite the Manual for themselves. After averaging, the Shared Manual becomes "arctic-focused," and the desert rescuer might perish from overheating.

### 2. The FedProx Solution (Proximal Control)
FedProx introduces a rule: "You can train however you like, but you must not stray too far from the Shared Manual."

The parameter **$\mu$** (Mu) is the length of the "safety rope":
- If $\mu = 0$, it's FedAvg (no rope, do whatever you want).
- If $\mu$ is large, the rope is short (you must follow the general rules, even if your local conditions are unique).

In trading, this allows the model to learn from different types of markets, but prevents a single "crypto fanatic" (a volatile client) from breaking the model for stable institutional investors. The global model remains stable, absorbing only common, fundamental patterns.
