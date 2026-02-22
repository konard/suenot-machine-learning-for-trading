import torch
import torch.optim as optim
import math
from model import CrossModalCLIPModel, CLIPLoss

# --- Mock Text Data Settings ---
# We will define a very simple vocabulary mapping to simulate "News"
# Let's say we have 4 distinct types of news events:
# 0: <PAD>
# 1: BULLISH EVENT ("company announces record profits")
# 2: BEARISH EVENT ("ceo resigns amid scandal")
# 3: VOLATILITY EVENT ("federal reserve unexpected rate hike")
# 4: CHOPPY FLAT ("markets quiet ahead of holidays")
VOCAB_SIZE = 10
MAX_TEXT_LEN = 8

def get_static_4_item_batch(window_size=128):
    """
    Creates exactly 4 highly distinct static items for an overfit test to guarantee 
    the architecture is capable of contrastive mapping.
    """
    batch_price = torch.zeros((4, 1, window_size))
    batch_text = torch.zeros((4, MAX_TEXT_LEN), dtype=torch.long)
    
    x = torch.linspace(0, 5 * math.pi, window_size)
    
    # 0: Uptrend
    shape0 = x
    text0 = [1, 5, 0, 0, 0, 0, 0, 0]
    batch_price[0, 0, :] = (shape0 - shape0.mean()) / (shape0.std() + 1e-6)
    batch_text[0, :] = torch.tensor(text0)
    
    # 1: Downtrend
    shape1 = -x
    text1 = [2, 8, 0, 0, 0, 0, 0, 0]
    batch_price[1, 0, :] = (shape1 - shape1.mean()) / (shape1.std() + 1e-6)
    batch_text[1, :] = torch.tensor(text1)
    
    # 2: Sine Wave
    shape2 = torch.sin(x * 2)
    text2 = [3, 4, 0, 0, 0, 0, 0, 0]
    batch_price[2, 0, :] = (shape2 - shape2.mean()) / (shape2.std() + 1e-6)
    batch_text[2, :] = torch.tensor(text2)
    
    # 3: Flat
    shape3 = torch.zeros(window_size)
    text3 = [4, 4, 0, 0, 0, 0, 0, 0]
    batch_price[3, 0, :] = (shape3 - shape3.mean()) / (shape3.std() + 1e-6)
    batch_text[3, :] = torch.tensor(text3)
    
    return batch_price, batch_text

def train_clip():
    """
    Training script for Cross-Modal Contrastive Learning (CLIP style).
    """
    print("Starting Cross-Modal (CLIP) Training Loop...")
    
    epochs = 200
    lr = 1e-3
    window_size = 128

    model = CrossModalCLIPModel(vocab_size=VOCAB_SIZE, projection_dim=32)
    criterion = CLIPLoss(margin=0.2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Static batch for overfitting
    x_price_static, x_text_static = get_static_4_item_batch(window_size)

    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Overfit the same 4 items repeatedly
        for _ in range(20):
            optimizer.zero_grad()
            
            # Extract features for positive pairs
            v_price, v_text = model(x_price_static, x_text_static)
            
            # Explicit negative pairs by rolling the 4 items
            x_text_neg = torch.roll(x_text_static, shifts=1, dims=0)
            _, v_text_neg = model(x_price_static, x_text_neg)
            
            # Cosine Embedding Loss
            loss = criterion(v_price, v_text, v_text_neg)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Cosine Embedding Loss: {epoch_loss / 50:.4f}")

    print("Cross-Modal Fast Pre-training completed.")
    torch.save(model.state_dict(), "cross_modal_clip.pth")

if __name__ == "__main__":
    train_clip()
