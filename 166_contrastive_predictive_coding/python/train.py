import torch
import torch.optim as optim
import math
from model import CPCModel
from cpc_loss import info_nce_loss

def synthesize_sequence_batch(batch_size=8, seq_len=20, window_size=64):
    """
    Generates batches of sequences of windows.
    Each sequence represents a 'regime' (e.g., sine with increasing freq).
    """
    batch_x = torch.zeros(batch_size, seq_len, 1, window_size)
    
    for b in range(batch_size):
        # Every batch is slightly different frequency and noise
        freq_base = 0.1 + (b * 0.05)
        for s in range(seq_len):
            t = torch.linspace(s * window_size, (s+1) * window_size, window_size)
            # Add some temporal trend across windows
            freq = freq_base * (1.0 + 0.05 * s) 
            window = torch.sin(t * freq) + torch.randn(window_size) * 0.1
            batch_x[b, s, 0, :] = (window - window.mean()) / (window.std() + 1e-6)
            
    return batch_x

def train_cpc():
    # Parameters
    BATCH_SIZE = 16
    SEQ_LEN = 24
    WINDOW_SIZE = 64
    LATENT_DIM = 64
    CONTEXT_DIM = 128
    PREDICT_STEPS = 4
    EPOCHS = 15
    STEPS_PER_EPOCH = 30
    
    print(f"Initializing CPC Training...")
    print(f"Sequence Length: {SEQ_LEN}, Predict Steps: {PREDICT_STEPS}")
    
    model = CPCModel(latent_dim=LATENT_DIM, context_dim=CONTEXT_DIM, predict_steps=PREDICT_STEPS)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        
        for step in range(STEPS_PER_EPOCH):
            # 1. Generate Sequential Data
            x_seq = synthesize_sequence_batch(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, window_size=WINDOW_SIZE)
            
            optimizer.zero_grad()
            
            # 2. Forward Pass: Get latents and contexts
            z_all, c_all = model(x_seq)
            
            # 3. CPC Loss (InfoNCE)
            loss = info_nce_loss(z_all, c_all, model)
            
            # 4. Backward Pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / STEPS_PER_EPOCH
        print(f"Epoch {epoch}/{EPOCHS} | CPC InfoNCE Loss: {avg_loss:.4f}")

    # Save Model
    torch.save(model.state_dict(), "cpc_trading_model.pth")
    print("Training complete. Model saved to cpc_trading_model.pth")

if __name__ == "__main__":
    train_cpc()
