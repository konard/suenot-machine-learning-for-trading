import hashlib
import time
import torch
from collections import OrderedDict

class Block:
    """
    Represents a single block in the federated training ledger.
    """
    def __init__(self, index, previous_hash, timestamp, model_update_hash, data_size):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.model_update_hash = model_update_hash
        self.data_size = data_size
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        content = f"{self.index}{self.previous_hash}{self.timestamp}{self.model_update_hash}{self.data_size}"
        return hashlib.sha256(content.encode()).hexdigest()

class DecentralizedLedger:
    """
    Simulated blockchain ledger for storing and auditing model updates.
    """
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_updates = []

    def create_genesis_block(self):
        return Block(0, "0", time.time(), "GENESIS", 0)

    def add_model_update(self, client_id, weights, data_size):
        """
        Records a model update hash on the ledger.
        In a real BcFL, we'd store the hash here and the weights in IPFS.
        """
        weights_flat = torch.cat([p.flatten() for p in weights.values()])
        weights_bytes = weights_flat.cpu().numpy().tobytes()
        model_hash = hashlib.sha256(weights_bytes).hexdigest()
        
        timestamp = time.time()
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), previous_block.hash, timestamp, model_hash, data_size)
        
        self.chain.append(new_block)
        print(f"Audit Log: Block #{new_block.index} mined for Client {client_id}. Hash: {model_hash[:10]}...")
        
        return new_block.hash

    def get_audit_trail(self):
        return [(b.index, b.model_update_hash, b.data_size) for b in self.chain]

class DecentralizedAggregator:
    """
    Logic for aggregating model updates retrieved from the ledger.
    """
    def aggregate(self, client_weights_list, client_data_sizes):
        total_data = sum(client_data_sizes)
        if total_data == 0: return None
        
        aggregated_weights = OrderedDict()
        first_weights = client_weights_list[0]
        
        for key in first_weights.keys():
            weighted_params = [
                weights[key] * (size / total_data)
                for weights, size in zip(client_weights_list, client_data_sizes)
            ]
            aggregated_weights[key] = torch.stack(weighted_params, dim=0).sum(dim=0)
            
        return aggregated_weights
