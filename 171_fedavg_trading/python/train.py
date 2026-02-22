import torch
import numpy as np
from model import TradingNN
from federated_core import FederatedClient, FedAvgAggregator

def generate_non_iid_data(num_clients=5, samples_per_client=200):
    """
    Simulates Non-IID market data across different clients.
    Each client sees a different 'regime' (shift in mean/variance).
    """
    clients = []
    input_dim = 20
    
    # Global test data
    test_data = torch.randn(100, input_dim)
    test_labels = (test_data.sum(dim=1, keepdim=True) > 0).float()

    for i in range(num_clients):
        # Client-specific bias (Non-IID)
        bias = np.random.uniform(-0.5, 0.5)
        scale = np.random.uniform(0.5, 2.0)
        
        x = torch.randn(samples_per_client, input_dim) * scale + bias
        # Target: relationship between features and outcome (sign of sum)
        y = (x.sum(dim=1, keepdim=True) > 0).float()
        
        client = FederatedClient(i, x, y, lambda: TradingNN(input_dim))
        clients.append(client)
        
    return clients, test_data, test_labels

def evaluate(model, data, labels):
    model.eval()
    with torch.no_grad():
        preds = model(data)
        mse = torch.mean((preds - labels)**2)
    return mse.item()

def run_federated_averaging():
    print("Starting Federated Averaging Simulation...")
    
    NUM_CLIENTS = 5
    ROUNDS = 15
    LOCAL_EPOCHS = 10
    
    clients, test_x, test_y = generate_non_iid_data(NUM_CLIENTS)
    aggregator = FedAvgAggregator()
    global_model = TradingNN(input_dim=20)
    
    # Initial weights
    global_weights = global_model.state_dict()
    
    for r in range(1, ROUNDS + 1):
        round_weights = []
        round_counts = []
        
        # All clients participate in this simulation
        for client in clients:
            weights, count = client.local_train(global_weights, epochs=LOCAL_EPOCHS)
            round_weights.append(weights)
            round_counts.append(count)
            
        # Server Aggregation
        global_weights = aggregator.aggregate(round_weights, round_counts)
        global_model.load_state_dict(global_weights)
        
        mse = evaluate(global_model, test_x, test_y)
        print(f"Round {r:02d}/{ROUNDS} | Global Test MSE: {mse:.4f}")

    print("\nFederated Training Complete.")
    torch.save(global_model.state_dict(), "fedavg_global_model.pth")

if __name__ == "__main__":
    run_federated_averaging()
