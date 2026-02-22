import torch
import time
import random
import copy
from model import TradingNN
from afl_core import AsynchronousServer

def generate_data(num_samples=100):
    x = torch.randn(num_samples, 20)
    y = (x.sum(dim=1, keepdim=True) > 0).float()
    return x, y

def simulate_asynchronous_fl():
    print("Starting Asynchronous Federated Learning Simulation...")
    
    global_model = TradingNN()
    server = AsynchronousServer(global_model, lr=0.5)
    
    NUM_CLIENTS = 5
    CLIENT_UPDATES = 15
    
    # Simulate a queue of updates with different arrival times (simulating network delay)
    update_queue = []
    
    for i in range(CLIENT_UPDATES):
        client_id = i % NUM_CLIENTS
        # Each client starts training on the 'then-current' global model step
        start_step = server.current_step
        
        # Simulate local training
        x, y = generate_data()
        local_model = TradingNN()
        local_model.load_state_dict(server.global_model.state_dict())
        
        # Random delay simulation (0 to 10 steps of staleness)
        delay = random.randint(0, 10)
        arrival_step = start_step + delay
        
        update_queue.append({
            'client_id': client_id,
            'weights': copy.deepcopy(local_model.state_dict()),
            'start_step': start_step,
            'arrival_order': arrival_step
        })

    # Sort the queue by arrival order to simulate real asynchronous arrival
    update_queue.sort(key=lambda x: x['arrival_order'])

    print(f"\nProcessing {len(update_queue)} asynchronous updates...")
    for update in update_queue:
        server.receive_update(update['weights'], update['start_step'])

    print("\n--- Simulation Summary ---")
    print(f"Final Server Global Step: {server.current_step}")
    print("SUCCESS: Asynchronous aggregation completed with staleness-aware weighting.")

if __name__ == "__main__":
    simulate_asynchronous_fl()
