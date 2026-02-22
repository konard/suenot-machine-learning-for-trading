# Communication-Efficient FL: The "Telegram Economy" Analogy

Imagine you need to send a football match result to a friend in another country, but the cost of the telegram depends on the word count.

### 1. How it works in normal mode (Dense FL)
You send a detailed report: "Player #7 kicked the ball at the 15th minute, the wind speed was 5 m/s, the ball hit the top left corner at a speed of 90 km/h... Result: Goal."
- **Data volume**: Huge.
- **Cost**: High.

### 2. How it works in economical mode (Efficient FL)

**Method A: Quantization (Precision reduction)**
Instead of "92.456 km/h," you just write "Fast." Instead of detailed trajectories, just "High."
- In trading: Instead of sending 0.00045612, we send just "+1" (price goes up).

**Method B: Sparsification (Sending only the important stuff)**
You don't describe every pass. You only send a message when a goal is scored or a red card is shown.
- In trading: We only transmit the neural network weights that changed the most.

**The Result**: Your friend understands the match just as well as with a detailed report, but you spent 95% less on telegrams. The trading model trains just as well but uses 20-30 times less internet traffic.
