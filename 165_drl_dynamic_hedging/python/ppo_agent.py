import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ActorCritic(nn.Module):
    """
    Dual-headed Neural Network for Proximal Policy Optimization (PPO).
    State input shape: (4,)
    
    1. Actor Network: Outputs Mean & Log_Std mapping to a Gaussian Distribution
       over the optimal hedge ratio [-1, 1]
    2. Critic Network: Outputs a scalar Baseline Value function (V) predicting sum of future rewards.
    """
    def __init__(self, state_dim=4, action_dim=1):
        super(ActorCritic, self).__init__()
        
        # ACTOR: Evaluates the specific Action (Hedging Ratio) to take
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Restrict hedge output bounds between -1 and 1
        )
        # Trainable Log Standard Deviation parameter for Action Exploration 
        # Initialized slightly higher to encourage initial exploration
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim) + 0.5)

        # CRITIC: Predicts the Expected Return starting from this state (V-Value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1) # Single scalar return
        )

    def forward(self, state):
        # We generally do not call forward() directly in PPO, 
        # we call evaluate() or get_action()
        raise NotImplementedError

    def get_action(self, state):
        """
        Used during the ROLLOUT phase (interacting with the environment).
        Samples an action probabilistically to ensure exploration.
        """
        action_mean = self.actor_mean(state)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        
        dist = Normal(action_mean, action_std)
        action = dist.sample() # Probabilistic sampling
        
        action_logprob = dist.log_prob(action).sum(dim=-1)
        state_val = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        """
        Used during the TRAIN phase.
        Given a batch of states and old actions, calculate their new probabilities 
        against the newly updated Actor weights.
        """
        action_mean = self.actor_mean(state)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        
        dist = Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPOAgent:
    """
    Proximal Policy Optimization logic implementing Clipped Surrogate Objective 
    and Advantage normalization.
    """
    def __init__(self, state_dim=4, action_dim=1, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma       # Discount factor
        self.eps_clip = eps_clip # PPO clipping threshold
        
        # Policy Network
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Replay buffers
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, action_logprob, state_val = self.policy.get_action(state)

        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        self.state_values.append(state_val)

        return action.item()

    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.is_terminals.append(done)

    def optimize(self):
        """
        Optimizes the Actor and Critic networks using the stored episode trajectory.
        """
        # Monte Carlo estimate of Returns (Discounted Sum of Rewards)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Normalize Returns (Vital for financial optimization stability)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.state_values, dim=0)).detach()

        # Calculate ADVANTAGE: True Return - Critic Baseline Prediction
        # If > 0, the action was better than expected.
        advantages = rewards.detach() - old_state_values.detach()
        
        # Optimize Policy for K epochs:
        K_epochs = 40
        for _ in range(K_epochs):
            # Evaluate old actions and values under NEW policy weights
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Find the Ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # PPO Clipped Objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Final PPO Loss Function
            # Maximize Surrogate (Negative Loss) - Value Loss + Entropy Bonus (Exploration)
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Clear memory buffers for next episode
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()
