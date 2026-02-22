import torch
import copy

class PFLManager:
    """
    Manages global-to-local model adaptation (Fine-Tuning).
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    def fine_tune(self, model, data, labels, local_epochs=5):
        """
        Adapts a global model to local data through fine-tuning.
        """
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()
        
        for _ in range(local_epochs):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        return model

    def interpolate_models(self, global_model, local_model, alpha=0.5):
        """
        Blends global and local weights: weights = alpha * global + (1-alpha) * local
        """
        blended_model = copy.deepcopy(global_model)
        global_dict = global_model.state_dict()
        local_dict = local_model.state_dict()
        blended_dict = blended_model.state_dict()
        
        for key in global_dict.keys():
            blended_dict[key] = alpha * global_dict[key] + (1.0 - alpha) * local_dict[key]
            
        blended_model.load_state_dict(blended_dict)
        return blended_model
