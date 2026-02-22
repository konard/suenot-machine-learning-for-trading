import torch

class FedProxClient:
    """
    FedProx Client with proximal term regularization.
    """
    def __init__(self, client_id, data, labels, model_fn):
        self.client_id = client_id
        self.data = data
        self.labels = labels
        self.model = model_fn()
        self.sample_count = len(data)

    def local_train(self, global_weights, mu=0.01, epochs=5, lr=0.01):
        """
        Performs local SGD updates with a proximal term.
        Target: min F_k(w) + (mu/2) * ||w - w_t||^2
        """
        # Load global model weights
        self.model.load_state_dict(global_weights)
        
        # Keep a copy of global weights for the proximal term
        global_params = [p.clone().detach() for p in self.model.parameters()]
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            outputs = self.model(self.data)
            loss_mse = criterion(outputs, self.labels)
            
            # Proximal Term: (mu/2) * ||w - w_t||^2
            proximal_term = 0.0
            for i, p in enumerate(self.model.parameters()):
                proximal_term += (p - global_params[i]).pow(2).sum()
            
            total_loss = loss_mse + (mu / 2.0) * proximal_term
            
            total_loss.backward()
            optimizer.step()
        
        return self.model.state_dict(), self.sample_count
