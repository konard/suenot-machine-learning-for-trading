import torch
from collections import OrderedDict

class FedAvgAggregator:
    """
    Server-side component for aggregating client updates.
    """
    def __init__(self):
        pass

    def aggregate(self, client_weights, client_sample_counts):
        """
        Computes the weighted average of client models.
        
        Args:
            client_weights: List of state_dicts from clients.
            client_sample_counts: List of integers (number of samples per client).
        """
        total_samples = sum(client_sample_counts)
        global_dict = OrderedDict()

        # Iterate through model parameters
        for key in client_weights[0].keys():
            # Weighted average for each parameter
            weighted_params = [
                weights[key] * (count / total_samples)
                for weights, count in zip(client_weights, client_sample_counts)
            ]
            global_dict[key] = torch.stack(weighted_params, dim=0).sum(dim=0)

        return global_dict

class FederatedClient:
    """
    Simulates a decentralized client (e.g., a local trading desk).
    """
    def __init__(self, client_id, data, labels, model_fn):
        self.client_id = client_id
        self.data = data
        self.labels = labels
        self.model = model_fn()
        self.sample_count = len(data)

    def local_train(self, global_weights, epochs=5, lr=0.01):
        """
        Performs local SGD updates.
        """
        self.model.load_state_dict(global_weights)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(self.data)
            loss = criterion(outputs, self.labels)
            loss.backward()
            optimizer.step()
        
        return self.model.state_dict(), self.sample_count
