import torch
import copy
import math

class AsynchronousServer:
    """
    Simulates an asynchronous federated server that handles stale updates.
    """
    def __init__(self, model, lr=0.1, staleness_func="polynomial"):
        self.global_model = model
        self.current_step = 0
        self.lr = lr
        self.staleness_func = staleness_func

    def calculate_staleness_weight(self, tau):
        """
        Calculates the weight decay factor based on update staleness (tau).
        """
        if self.staleness_func == "polynomial":
            return (tau + 1) ** (-0.5)
        elif self.staleness_func == "exponential":
            return math.exp(-0.5 * tau)
        else:
            return 1.0 / (tau + 1.0)

    def receive_update(self, client_weights, client_step):
        """
        Processes a single asynchronous update from a client.
        """
        tau = self.current_step - client_step
        weight = self.calculate_staleness_weight(tau)
        
        print(f"Server: Received update from Step {client_step} (Staleness: {tau}). Weight: {weight:.4f}")
        
        # Apply the update with staleness weighting: 
        # W_new = (1 - alpha*weight) * W_old + (alpha*weight) * W_client
        alpha = self.lr * weight
        
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = (1.0 - alpha) * global_dict[key] + alpha * client_weights[key]
            
        self.global_model.load_state_dict(global_dict)
        self.current_step += 1
        
        return self.current_step
