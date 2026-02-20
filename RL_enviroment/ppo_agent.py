

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import json
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


class CallCenterEnvironment:
    
    
    def __init__(self):
        self.db_url = os.getenv('db_url', 'postgresql+psycopg2://postgres:root@localhost:5432/call_agent')
        self.engine = create_engine(self.db_url)
        self.feedback_table = 'customer_feedback'
        self.current_call_id = None
        self.call_history = deque(maxlen=1000)
        
    def fetch_call_episodes(self, batch_size=32):
      
        query = f"""
            SELECT feedback_id, customer_nic, connection_number, 
                   feedback_value, feedback_message, created_at 
            FROM {self.feedback_table}
            ORDER BY created_at DESC
            LIMIT {batch_size};
        """
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql_query(text(query), conn)
            return result
        except Exception as e:
            print(f"Error fetching call episodes: {str(e)}")
            return pd.DataFrame()
    
    def convert_feedback_to_reward(self, feedback_value):
        
        
        normalized = (feedback_value / 10.0) * 2 - 1
        return normalized
    
    def reset(self):
        """Reset environment state."""
        self.current_call_id = None
        return None
    
    def step(self, action, feedback_value):
       
        reward = self.convert_feedback_to_reward(feedback_value)
        done = True  # Each call is one episode
        info = {
            'feedback_value': feedback_value,
            'reward': reward,
            'timestamp': datetime.now().isoformat()
        }
        
        self.call_history.append(info)
        return reward, done, info


class PPOMemory:
    def __init__(self, batch_size=32):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.batch_size = batch_size
        
    def push(self, state, action, reward, log_prob, value, done):
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def is_full(self):
        return len(self.states) >= self.batch_size
    
    def get_batch(self):
        if not self.is_full():
            return None
        
        return {
            'states': torch.stack(self.states[:self.batch_size]),
            'actions': torch.stack(self.actions[:self.batch_size]),
            'rewards': torch.tensor(self.rewards[:self.batch_size], dtype=torch.float32),
            'log_probs': torch.stack(self.log_probs[:self.batch_size]),
            'values': torch.stack(self.values[:self.batch_size]),
        }
    
    def compute_returns(self, gamma=0.99, gae_lambda=0.95):
        returns = []
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1].item()
            
            delta = self.rewards[t] + gamma * next_value - self.values[t].item()
            gae = delta + gamma * gae_lambda * gae
            
            returns.insert(0, gae + self.values[t].item())
            advantages.insert(0, gae)
        
        return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=768, action_dim=256, hidden_dim=512):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        
        action_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return action_logits, value


class PPOAgent:
    def __init__(self, state_dim=768, action_dim=256, learning_rate=5e-6, device='cpu'):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # PPO hyperparameters
        self.clip_ratio = 0.2
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        
        # Training state
        self.memory = PPOMemory(batch_size=32)
        self.training_step = 0
        self.training_history = []
        
        # Config file
        self.config_file = './RL_enviroment/ppo_agent_config.json'
        self._initialize_config()
    
    def _initialize_config(self):
        """Initialize configuration file."""
        if not os.path.exists(self.config_file):
            config = {
                'agent_type': 'PPO',
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'learning_rate': self.learning_rate,
                'created_at': datetime.now().isoformat(),
                'training_episodes': 0,
                'total_reward': 0,
                'avg_reward': 0,
                'training_history': []
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
    
    def select_action(self, state):
        with torch.no_grad():
            action_logits, value = self.policy(state)
        
        # Sample action from policy
        action_probs = torch.nn.functional.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob, value
    
    def train_step(self, batch, num_epochs=4):
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_log_probs = batch['log_probs'].to(self.device)
        
        # Compute returns and advantages
        returns, advantages = self.memory.compute_returns()
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(num_epochs):
            # Forward pass
            action_logits, values = self.policy(states)
            values = values.squeeze(-1)
            
            # Compute new log probabilities
            action_probs = torch.nn.functional.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            
            # Compute policy loss (PPO objective)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = nn.functional.mse_loss(values, returns)
            
            # Compute entropy bonus (for exploration)
            entropy = action_dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        metrics = {
            'policy_loss': total_policy_loss / num_epochs,
            'value_loss': total_value_loss / num_epochs,
            'entropy': total_entropy / num_epochs,
            'total_loss': (total_policy_loss + total_value_loss) / num_epochs,
        }
        
        self.training_step += 1
        return metrics
    
    def train_on_feedback(self, num_episodes=100):
        print("\n" + "="*70)
        print("PPO TRAINING ON CUSTOMER FEEDBACK")
        print("="*70)
        
        env = CallCenterEnvironment()
        total_reward = 0
        episode_count = 0
        
        # Fetch feedback data
        feedback_data = env.fetch_call_episodes(batch_size=num_episodes)
        
        if feedback_data.empty:
            return {'status': 'no_data', 'episodes': 0}
        
        
        
        
        for idx, row in feedback_data.iterrows():
            feedback_value = row['feedback_value']
            reward = env.convert_feedback_to_reward(feedback_value)
            total_reward += reward
            episode_count += 1
            
            
            state = torch.randn(self.state_dim).to(self.device)
            
            
            with torch.no_grad():
                action, log_prob, value = self.select_action(state)
            
            self.memory.push(state, action, reward, log_prob, value, True)
            
            if self.memory.is_full():
                print(f"[2/3] Training on batch {len(self.training_history) + 1}...")
                batch = self.memory.get_batch()
                metrics = self.train_step(batch)
                
                self.training_history.append({
                    'step': self.training_step,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics,
                    'episodes_processed': episode_count
                })
                
              
                
                self.memory.clear()
        
       
        self._save_training_state(episode_count, total_reward / episode_count)
        
        
        
        return {
            'status': 'success',
            'episodes': episode_count,
            'avg_reward': total_reward / episode_count,
            'training_steps': self.training_step,
            'training_history': self.training_history[-5:]  # Last 5 training updates
        }
    
    def _save_training_state(self, episodes, avg_reward):
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            config['training_episodes'] += episodes
            config['total_reward'] += avg_reward * episodes
            config['avg_reward'] = config['total_reward'] / config['training_episodes']
            config['last_training'] = datetime.now().isoformat()
            config['training_history'] = self.training_history
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"Training state saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving training state: {str(e)}")
    
    def save_checkpoint(self, path='./RL_enviroment/ppo_agent.pt'):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'learning_rate': self.learning_rate,
            }
        }, path)

    
    def load_checkpoint(self, path='./RL_enviroment/ppo_agent.pt'):
        """Load model checkpoint."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_step = checkpoint['training_step']
         
    
    def get_training_stats(self):
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return config
        except:
            return None


# Convenience functions
def train_ppo_agent(num_episodes=100):
    
    agent = PPOAgent(state_dim=768, action_dim=256, learning_rate=5e-6)
    return agent.train_on_feedback(num_episodes=num_episodes)


def get_ppo_agent(device='cpu'):

    agent = PPOAgent(state_dim=768, action_dim=256, device=device)
    agent.load_checkpoint()
    return agent
