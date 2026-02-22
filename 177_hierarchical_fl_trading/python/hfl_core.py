import torch
from collections import OrderedDict

class HierarchicalAggregator:
    """
    Handles aggregation at both Regional and Global levels.
    """
    @staticmethod
    def aggregate(weights_list, data_sizes):
        """
        Performs weighted average of models.
        """
        total_data = sum(data_sizes)
        if total_data == 0: return None
        
        aggregated_weights = OrderedDict()
        first_weights = weights_list[0]
        
        for key in first_weights.keys():
            weighted_params = [
                weights[key] * (size / total_data)
                for weights, size in zip(weights_list, data_sizes)
            ]
            aggregated_weights[key] = torch.stack(weighted_params, dim=0).sum(dim=0)
            
        return aggregated_weights

class RegionalHub:
    """
    Represents an intermediate aggregator (e.g., London Office).
    """
    def __init__(self, region_id):
        self.region_id = region_id
        self.client_weights = []
        self.client_sizes = []
        self.network_bytes_in = 0

    def receive_update(self, weights, size):
        self.client_weights.append(weights)
        self.client_sizes.append(size)
        # Simulate bandwidth: weights are approx (input*hidden + hidden*hidden + hidden*output) floats
        # Roughly 20*64 + 64*64 + 64*1 = 1280 + 4096 + 64 = 5440 floats = ~21.7 KB
        self.network_bytes_in += 22000 

    def get_regional_summary(self):
        """
        Aggregates local client updates into a single regional model.
        """
        if not self.client_weights: return None, 0
        
        summary_weights = HierarchicalAggregator.aggregate(self.client_weights, self.client_sizes)
        total_size = sum(self.client_sizes)
        
        # Reset for next round
        self.client_weights = []
        self.client_sizes = []
        
        return summary_weights, total_size

class GlobalCoordinator:
    """
    The top-level cloud aggregator.
    """
    def __init__(self):
        self.regional_summaries = []
        self.regional_sizes = []
        self.network_bytes_in = 0

    def receive_regional_summary(self, weights, size):
        self.regional_summaries.append(weights)
        self.regional_sizes.append(size)
        self.network_bytes_in += 22000 # One model update size

    def aggregate_global(self):
        if not self.regional_summaries: return None
        
        global_weights = HierarchicalAggregator.aggregate(self.regional_summaries, self.regional_sizes)
        self.regional_summaries = []
        self.regional_sizes = []
        return global_weights
