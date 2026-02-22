import torch
import numpy as np
import copy
from model import TradingNN
from fedprox_core import FedProxClient
from collections import OrderedDict

def aggregate_fedavg(client_weights, client_sample_counts):
    """Standard weighted averaging (server-side)."""
    total_samples = sum(client_sample_counts)
    global_dict = OrderedDict()
    for key in client_weights[0].keys():
        weighted_params = [
            weights[key] * (count / total_samples)
            for weights, count in zip(client_weights, client_sample_counts)
        ]
        global_dict[key] = torch.stack(weighted_params, dim=0).sum(dim=0)
    return global_dict

def generate_heterogeneous_data(num_clients=5, samples_per_client=200):
    """
    Simulates high heterogeneity (Non-IID) across clients.
    """
    clients = []
    input_dim = 20
    test_data = torch.randn(100, input_dim)
    test_labels = (test_data.sum(dim=1, keepdim=True) > 0).float()

    for i in range(num_clients):
        # High heterogeneity: different clients have vastly different data shifts
        bias = np.random.uniform(-1.0, 1.0) # Larger bias than FedAvg test
        scale = np.random.uniform(0.1, 3.0) # More varied variance
        
        x = torch.randn(samples_per_client, input_dim) * scale + bias
        y = (x.sum(dim=1, keepdim=True) > 0).float()
        
        client = FedProxClient(i, x, y, lambda: TradingNN(input_dim))
        clients.append(client)
        
    return clients, test_data, test_labels

def evaluate(model, data, labels):
    model.eval()
    with torch.no_grad():
        preds = model(data)
        mse = torch.mean((preds - labels)**2)
    return mse.item()

def run_fedprox_experiment():
    print("Starting FedProx Heterogeneous Simulation...")
    
    NUM_CLIENTS = 10
    ROUNDS = 20
    MU = 0.1 # Proximal intensity
    
    clients, test_x, test_y = generate_heterogeneous_data(NUM_CLIENTS)
    global_model = TradingNN(input_dim=20)
    global_weights = global_model.state_dict()
    
    for r in range(1, ROUNDS + 1):
        round_weights = []
        round_counts = []
        
        # Active Clients (90% selection for each round)
        selected_indices = np.random.choice(NUM_CLIENTS, int(NUM_CLIENTS * 0.9), replace=False)
        
        for idx in selected_indices:
            client = clients[idx]
            # Vary local epochs simulating stragglers/heterogeneous compute
            local_epochs = np.random.randint(5, 15)
            
            weights, count = client.local_train(
                global_weights, 
                mu=MU, 
                epochs=local_epochs, 
                lr=0.01
            )
            round_weights.append(weights)
            round_counts.append(count)
            
        # Global Aggregation
        global_weights = aggregate_fedavg(round_weights, round_counts)
        global_model.load_state_dict(global_weights)
        
        mse = evaluate(global_model, test_x, test_y)
        print(f"Round {r:02d}/{ROUNDS} (mu={MU}) | Global Test MSE: {mse:.4f}")

    print("\nFedProx Simulation Complete.")
    torch.save(global_model.state_dict(), "fedprox_global_model.pth")

if __name__ == "__main__":
    run_fedprox_experiment()
