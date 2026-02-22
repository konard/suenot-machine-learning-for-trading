import torch
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from model import PCLCNNEncoder
from pcl_loss import ProtoNCELoss, info_nce_loss

def simulate_market_batch(batch_size=256, window_size=64):
    """
    Simulates market data with underlying regimes (prototypes).
    """
    # 5 Hidden "Proto-regimes"
    regimes = [0, 1, 2, 3, 4]
    
    x1 = torch.zeros(batch_size, 1, window_size)
    x2 = torch.zeros(batch_size, 1, window_size)
    
    t = torch.linspace(0, 1, window_size)
    
    for i in range(batch_size):
        r = i % 5
        noise = torch.randn(window_size) * 0.05
        
        # Base signal per regime
        if r == 0: base = t # Up
        elif r == 1: base = -t # Down
        elif r == 2: base = torch.sin(t * 10) # Sideways
        elif r == 3: base = torch.exp(t) # Exponential
        else: base = -torch.exp(t) # Crash
            
        # Augmentations (Views)
        x1[i, 0, :] = base + torch.randn_like(base) * 0.1
        x2[i, 0, :] = base + torch.randn_like(base) * 0.1
        
        # Normalization
        x1[i] = (x1[i] - x1[i].mean()) / (x1[i].std() + 1e-6)
        x2[i] = (x2[i] - x2[i].mean()) / (x2[i].std() + 1e-6)
        
    return x1, x2

def train_pcl():
    print("Initializing Prototypical Contrastive Learning (PCL)...")
    
    BATCH_SIZE = 128
    NUM_CLUSTERS = 5
    EPOCHS = 10
    STEPS_PER_EPOCH = 20
    
    model = PCLCNNEncoder()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    proto_criterion = ProtoNCELoss()
    
    for epoch in range(1, EPOCHS + 1):
        model.eval()
        # 1. Clustering Step (E-step)
        # In a real scenario, we'd use a memory bank. Here we generate a fresh clustering batch.
        with torch.no_grad():
            cx, _ = simulate_market_batch(batch_size=500)
            _, z_all = model(cx)
            z_np = z_all.cpu().numpy()
            
            kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init=10)
            cluster_labels = kmeans.fit_predict(z_np)
            prototypes = torch.tensor(kmeans.cluster_centers_).float()
            
            # Estimate concentration (phi) per cluster
            # Simplification: we use a fixed temperature or estimate variance
            concentrations = torch.ones(NUM_CLUSTERS) * 0.1
            
        # 2. Training Step (M-step)
        model.train()
        total_loss = 0
        for step in range(STEPS_PER_EPOCH):
            x1, x2 = simulate_market_batch(batch_size=BATCH_SIZE)
            
            optimizer.zero_grad()
            
            # Forward
            _, z1 = model(x1)
            _, z2 = model(x2)
            
            # InfoNCE (Local)
            loss_inst = info_nce_loss(z1, z2)
            
            # ProtoNCE (Global)
            # Find closest prototype for each sample in the batch
            # Note: For simplicity, we re-assign during training
            with torch.no_grad():
                dist = torch.cdist(z1, prototypes)
                batch_cluster_labels = torch.argmin(dist, dim=1)
                
            loss_proto = proto_criterion(z1, prototypes, batch_cluster_labels, concentrations)
            
            # Total Loss
            loss = loss_inst + loss_proto
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch}/{EPOCHS} | PCL Total Loss: {total_loss/STEPS_PER_EPOCH:.4f}")

    print("Training complete. Model saved to pcl_trading_model.pth")
    torch.save(model.state_dict(), "pcl_trading_model.pth")

if __name__ == "__main__":
    train_pcl()
