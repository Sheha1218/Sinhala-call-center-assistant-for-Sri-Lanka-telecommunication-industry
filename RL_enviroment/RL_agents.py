import gymnasium as gym
import random
import collections as deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm 



class DQN(nn.Module):
    def __init__(self,in_states,h1_nodes,out_actions):
        super().__init__()
        
        self.fc1 =nn.Linear(in_states,h1_nodes)
        self.out = nn.Linear(h1_nodes,out_actions)
        
        
    def forward(self,x):
        x =F.relu(self.fc1(x))
        x =self.out(x)
        return x
    

class ReplayMemory():
    def __init__(self,maxlen):
        self.memory = deque([],maxlen=maxlen)
        
    def append(self,transition):
        self.memory.append(transition)
    
    def sample(self, sample_size):
        return random.sample(self.memory,sample_size)


class FrozenLakeDQL():
    learing_rate =0.001
    discount_factor_g =0.9
    network_sync_rate =10
    replay_memory_size = 1000
    min_batch_size =32
    
    
    loss_fn =nn.MSELoss()
    optimizer =None
    
    Action = ['L','D','R','U']
    
    def train(self,episodes,render=False,is_slippery=False):
        env = gym.make('RL_enviroment\model.pth',map_name ='4*4',is_slippery=is_slippery,render_mode='human' if render else None)
        num_states = env.observation_space.n
        num_actions =env.action_space.n 
        
        epsilon =1
        memory = ReplayMemory(self.replay_memory_size)
        
        
        policy_dqn = DQN(in_states=num_states,h1_nodes=num_states,out_actions=num_actions)
        target_dqn = DQN(in_states=num_states,h1_nodes=num_states,out_actions=num_actions)
        
        target_dqn.load_state_dict(policy_dqn.state_dict())
        
        print('policy(random,before training):')
        
        self.print_dqn(policy_dqn)
        
        self.optimizer =optm.Adam(policy_dqn.parameters(),lr=self.learing_rate)
        
        rewards_per_episode = np.zeros(episodes)
        
        epsilon_history = []
        
        step_count =0
        
        for i in range(episodes):
            state = env.rest()[0]
            terminated =False
            truncated =False
            
            
            while(not terminated and not truncated):
                step_count +=1
                
                if random.random() < epsilon:
                    action =env.action_space.sample()
                
                else:
                    with torch.no_grad():
                        action =policy_dqn(self.state_to_dqn_input(state,num_states)).argmax().item()

                
                new_state,reward,terminated,truncated,_ =env.step(action)
                
                memory.append((state,action,new_state,reward,terminated))
                
                state =new_state
                
                step_count+=1
                
                if reward ==1:
                    rewards_per_episode[i] =1
                    
                if len(memory.memory)>self.mini_batch_size and np.sum(rewards_per_episode)>0:
                    mini_batch =memory.sample(self.min_batch_size)
                    self.optimizer(mini_batch,policy_dqn,target_dqn)
                    
                    epsilon=max(epsilon -1/epsilon,0)
                    epsilon_history.append(epsilon)
                    
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count =0
                        
            env.close()
            
            return torch.save(policy_dqn.state_dict(),'model.pth')
                
        
        
            
        
    
    

    
            