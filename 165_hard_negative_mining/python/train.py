import torch
import torch.optim as optim
import math
import numpy as np
from model import DualEncoderModel
from miner import HardNegativeMiner, mining_loss

# Mock Data Generation (Same logic as Ch 164 for consistency)
def synthesize_multimodal_batch(batch_size=32, window_size=128):
    batch_price = torch.zeros(batch_size, 1, window_size)
    batch_text = torch.zeros(batch_size, 8, dtype=torch.long)
    
    for i in range(batch_size):
        x = torch.linspace(0, 5 * math.pi, window_size)
        noise = torch.randn(window_size) * 0.1
        phase_shift = (i / batch_size) * math.pi * 2
        amplitude = 0.5 + (i / batch_size)
        
        if i % 4 == 0:
            # Type 0: Trending up
            shape = x * amplitude + noise
            text_tokens = [1, 5, 6, 7, (i % 5) + 1, 0, 0, 0]
        elif i % 4 == 1:
            # Type 1: Trending down
            shape = -x * amplitude + noise
            text_tokens = [2, 8, 9, (i % 5) + 1, 0, 0, 0, 0]
        elif i % 4 == 2:
            # Type 2: Sine wave
            shape = torch.sin(x + phase_shift) * amplitude + noise
            text_tokens = [3, 4, 3, 4, (i % 5) + 1, 0, 0, 0]
        else:
            # Type 3: Flat / Noise
            shape = torch.randn(window_size) * 0.2 + noise
            text_tokens = [4, 4, 4, (i % 5) + 1, 0, 0, 0, 0]
            
        shape = (shape - shape.mean()) / (shape.std() + 1e-6)
        batch_price[i, 0, :] = shape
        batch_text[i, :] = torch.tensor(text_tokens)
        
    return batch_price, batch_text

def train_with_mining():
    # Parameters
    VOCAB_SIZE = 100
    WINDOW_SIZE = 128
    BATCH_SIZE = 64
    EPOCHS = 10
    STEPS_PER_EPOCH = 50
    TOP_K = 10 # Number of hard negatives to mine
    
    print(f"Initializing Hard Negative Mining Training (Top-{TOP_K})...")
    
    model = DualEncoderModel(vocab_size=VOCAB_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    miner = HardNegativeMiner(top_k=TOP_K)
    
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        
        for step in range(STEPS_PER_EPOCH):
            # 1. Generate Data
            x_price, x_text = synthesize_multimodal_batch(batch_size=BATCH_SIZE, window_size=WINDOW_SIZE)
            
            optimizer.zero_grad()
            
            # 2. Forward Pass
            v_price, v_text = model(x_price, x_text)
            
            # 3. Calculate Mining Loss
            # We treat v_price as anchors and v_text as candidates for hard negatives
            loss = mining_loss(v_price, v_text, miner, model.logit_scale)
            
            # 4. Backward Pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / STEPS_PER_EPOCH
        print(f"Epoch {epoch}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

    # Save Model
    torch.save(model.state_dict(), "hard_mining_contrastive.pth")
    print("Training complete. Model saved to hard_mining_contrastive.pth")

if __name__ == "__main__":
    train_with_mining()
