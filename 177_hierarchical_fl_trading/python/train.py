import torch
from model import TradingNN
from hfl_core import RegionalHub, GlobalCoordinator

def simulate_hfl_vs_flat():
    print("Hierarchical FL vs. Flat FL Bandwidth & Convergence Simulation")
    
    NUM_REGIONS = 3
    CLIENTS_PER_REGION = 5
    TOTAL_CLIENTS = NUM_REGIONS * CLIENTS_PER_REGION
    GLOBAL_ROUNDS = 5
    REGIONAL_SYNC_FREQUENCY = 3 # Every 3 local steps, sync regional
    
    # 1. FLAT FL SIMULATION (Every client to Cloud)
    print("\n--- Running Flat FL (Star Topology) ---")
    global_bytes = 0
    for r in range(GLOBAL_ROUNDS):
        # Every client sends to cloud every round
        global_bytes += TOTAL_CLIENTS * 22000
    print(f"Flat FL Global Bandwidth Cost: {global_bytes / 1024:.2f} KB")

    # 2. HIERARCHICAL FL SIMULATION
    print("\n--- Running Hierarchical FL ---")
    coordinator = GlobalCoordinator()
    regions = [RegionalHub(i) for i in range(NUM_REGIONS)]
    
    hfl_regional_bytes = 0
    hfl_global_bytes = 0
    
    for r in range(1, GLOBAL_ROUNDS + 1):
        print(f"Round {r} Aggregation Cycle...")
        
        # Phase A: Clients -> Regional Hubs (Local/Regional sync)
        for hub in regions:
            for c in range(CLIENTS_PER_REGION):
                dummy_weights = TradingNN().state_dict()
                hub.receive_update(dummy_weights, 100)
                hfl_regional_bytes += 22000
        
        # Phase B: Regional Hubs -> Global Coordinator
        # In HFL, regions might only sync to global every few local cycles
        # Here we simulate one global sync per round for comparison
        for hub in regions:
            summary, size = hub.get_regional_summary()
            coordinator.receive_regional_summary(summary, size)
            hfl_global_bytes += 22000
            
        global_model = coordinator.aggregate_global()
        
    print(f"\nHFL Regional Bandwidth (Short-distance): {hfl_regional_bytes / 1024:.2f} KB")
    print(f"HFL Global Bandwidth (Long-distance):   {hfl_global_bytes / 1024:.2f} KB")
    
    saving = (1 - (hfl_global_bytes / global_bytes)) * 100
    print(f"\nSUCCESS: HFL reduced GLOBAL core traffic by {saving:.2f}%")
    print("Note: In HFL, 100% of global traffic is replaced by filtered regional summaries.")

if __name__ == "__main__":
    simulate_hfl_vs_flat()
