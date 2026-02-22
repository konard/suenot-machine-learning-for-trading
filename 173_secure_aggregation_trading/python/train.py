import torch
from model import TradingNN
from secure_agg_core import SecureMasker, SecureAggregator

def simulate_secure_training():
    print("Starting Secure Aggregation Simulation...")
    
    NUM_CLIENTS = 3
    input_dim = 20
    
    clients = []
    # Setup clients and shared keys
    for i in range(NUM_CLIENTS):
        clients.append({
            "id": i,
            "masker": SecureMasker(i),
            "model": TradingNN(input_dim)
        })

    # Peer-to-peer key exchange simulation
    # In reality, this happens via a secure channel
    for i in range(NUM_CLIENTS):
        for j in range(i + 1, NUM_CLIENTS):
            shared_seed = np.random.randint(0, 1000000)
            clients[i]["masker"].set_shared_seed(j, shared_seed)
            clients[j]["masker"].set_shared_seed(i, shared_seed)

    # Simulate one round of training and aggregation
    print("\n--- Round 1: Masking & Submission ---")
    blinded_updates = []
    real_updates = [] # For verification only (server won't see this)

    for c in clients:
        # 1. Update (dummy SGD for simplicity)
        weights = c["model"].state_dict()
        real_updates.append(weights)
        
        # 2. Mask
        blinded = c["masker"].mask_weights(weights)
        blinded_updates.append(blinded)
        
        # Proof of blinding: check first weight of first layer
        first_key = list(blinded.keys())[0]
        original_val = weights[first_key].flatten()[0].item()
        blinded_val = blinded[first_key].flatten()[0].item()
        print(f"Client {c['id']}: Original={original_val:.4f} | Blinded={blinded_val:.4f}")

    # 3. Server Aggregation
    aggregator = SecureAggregator()
    aggregated_weights = aggregator.aggregate(blinded_updates)
    
    # 4. Verification (Comparing with centralized sum)
    print("\n--- Verification ---")
    sum_real = {k: torch.zeros_like(v) for k, v in real_updates[0].items()}
    for w in real_updates:
        for k in w:
            sum_real[k] += w[k]
    avg_real = {k: v / NUM_CLIENTS for k, v in sum_real.items()}
    
    first_key = list(aggregated_weights.keys())[0]
    agg_val = aggregated_weights[first_key].flatten()[0].item()
    real_val = avg_real[first_key].flatten()[0].item()
    
    diff = abs(agg_val - real_val)
    print(f"Aggregated Average: {agg_val:.6f}")
    print(f"Real Centralized Average: {real_val:.6f}")
    print(f"Difference (Mask cancellation error): {diff:.10f}")
    
    if diff < 1e-5:
        print("\nSUCCESS: Masks cancelled out perfectly. Server learned the average without seeing individual weights.")
    else:
        print("\nFAILURE: Mask cancellation failed.")

if __name__ == "__main__":
    import numpy as np
    simulate_secure_training()
