import torch
import numpy as np
from environment import DynamicHedgingEnv
from ppo_agent import PPOAgent

def train_ppo():
    """
    Main training loop for the PPO agent interacting with the Dynamic Hedging simulator.
    """
    print("Initializing Environment and PPO Agent...")
    env = DynamicHedgingEnv(n_steps=50, mu=0.00, sigma=0.02, transaction_cost=0.02)
    
    # State: [Price_Norm, Step_Return, Current_Hedge, Time_Left]
    # Action: Continuous Hedge Ratio [-1, 1]
    agent = PPOAgent(state_dim=4, action_dim=1, lr=5e-4, gamma=0.99, eps_clip=0.2)
    
    max_episodes = 2500
    update_timestep = 200 # Update policy every 4 episodes (4 * 50 steps)
    time_step = 0
    
    # Logging variables
    log_running_reward = 0
    log_running_episodes = 0
    
    print("Starting PPO Training...")
    
    for ep in range(1, max_episodes + 1):
        state = env.reset()
        current_ep_reward = 0
        
        while True:
            time_step += 1
            
            # Agent evaluates State -> Selects continuous Hedge Action
            action = agent.select_action(state)
            
            # Environment steps forward
            state, reward, done = env.step(action)
            
            # Agent stores experience
            agent.store_reward(reward, done)
            
            current_ep_reward += reward
            
            # Update PPO Policy if we collected enough experience points
            if time_step % update_timestep == 0:
                agent.optimize()
            
            if done:
                break
                
        log_running_reward += current_ep_reward
        log_running_episodes += 1
        
        # Log progress to terminal
        if ep % 100 == 0:
            avg_reward = log_running_reward / log_running_episodes
            print(f"Episode: {ep} \t Average Reward (Last 100): {avg_reward:.2f}")
            log_running_reward = 0
            log_running_episodes = 0
            
    # Save the trained Actor network for Rust Inference
    print("Training complete! Saving Actor network weights for Rust...")
    
    # We only need the actor weights for live fast-inference
    torch.save(agent.policy.actor_mean.state_dict(), "ppo_actor.pth")
    print("Saved -> ppo_actor.pth")

if __name__ == '__main__':
    train_ppo()
