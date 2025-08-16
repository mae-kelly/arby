import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import math

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256, 128]):
        super(DQN, self).__init__()
        
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256, 128]):
        super(Actor, self).__init__()
        
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, action_size)
        
    def forward(self, state):
        x = self.network(state)
        actions = torch.tanh(self.output(x))
        return actions

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256, 128]):
        super(Critic, self).__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.ReLU()
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_size, hidden_sizes[0]),
            nn.ReLU()
        )
        
        combined_size = hidden_sizes[0] * 2
        
        layers = []
        prev_size = combined_size
        
        for hidden_size in hidden_sizes[1:]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.q_network = nn.Sequential(*layers)
        
    def forward(self, state, action):
        state_encoded = self.state_encoder(state)
        action_encoded = self.action_encoder(action)
        combined = torch.cat([state_encoded, action_encoded], dim=1)
        q_value = self.q_network(combined)
        return q_value

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.FloatTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, 
                 buffer_size=100000, batch_size=64, target_update=1000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.update_target_network()
        self.step_count = 0
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
        
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.long().unsqueeze(1))
        
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.update_target_network()
            
        return loss.item()

class DDPGAgent:
    def __init__(self, state_size, action_size, lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, tau=1e-3, buffer_size=100000, batch_size=64):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.actor = Actor(state_size, action_size).to(self.device)
        self.actor_target = Actor(state_size, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = Critic(state_size, action_size).to(self.device)
        self.critic_target = Critic(state_size, action_size).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        self.noise = OUNoise(action_size)
        
    def hard_update(self, target, source):
        target.load_state_dict(source.state_dict())
        
    def soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
            
    def act(self, state, add_noise=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy()
        self.actor.train()
        
        if add_noise:
            action += self.noise.sample()
            
        return np.clip(action, -1, 1)
        
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions)
        target_q_values = rewards + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        current_q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q_values, target_q_values.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.critic_target, self.critic, self.tau)
        self.soft_update(self.actor_target, self.actor, self.tau)
        
        return actor_loss.item(), critic_loss.item()

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        self.state = self.mu.copy()
        
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.standard_normal(len(self.state))
        self.state += dx
        return self.state

class PPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, 
                 epsilon_clip=0.2, k_epochs=4, value_coef=0.5, entropy_coef=0.01):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.k_epochs = k_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.memory = []
        
    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, state_value = self.policy(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
        return action.item(), action_dist.log_prob(action).item(), state_value.item()
        
    def remember(self, state, action, reward, log_prob, state_value, done):
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'log_prob': log_prob,
            'state_value': state_value,
            'done': done
        })
        
    def update(self):
        if not self.memory:
            return
        
        states = torch.FloatTensor([m['state'] for m in self.memory]).to(self.device)
        actions = torch.LongTensor([m['action'] for m in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([m['log_prob'] for m in self.memory]).to(self.device)
        old_state_values = torch.FloatTensor([m['state_value'] for m in self.memory]).to(self.device)
        
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed([m['reward'] for m in self.memory]), 
                               reversed([m['done'] for m in self.memory])):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.FloatTensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        advantages = rewards - old_state_values
        
        for _ in range(self.k_epochs):
            action_probs, state_values = self.policy(states)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values.squeeze(), rewards)
            
            total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
        self.memory.clear()
        
        return total_loss.item()

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCritic, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, state):
        shared_features = self.shared(state)
        
        action_probs = F.softmax(self.actor(shared_features), dim=-1)
        state_value = self.critic(shared_features)
        
        return action_probs, state_value

class TradingAgent:
    def __init__(self, algorithm='dqn', state_size=20, action_size=3, **kwargs):
        self.algorithm = algorithm
        self.state_size = state_size
        self.action_size = action_size
        
        if algorithm == 'dqn':
            self.agent = DQNAgent(state_size, action_size, **kwargs)
        elif algorithm == 'ddpg':
            self.agent = DDPGAgent(state_size, action_size, **kwargs)
        elif algorithm == 'ppo':
            self.agent = PPOAgent(state_size, action_size, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'portfolio_values': []
        }
        
    def preprocess_state(self, market_data, portfolio_data):
        features = []
        
        features.extend([
            market_data.get('price', 0),
            market_data.get('volume', 0),
            market_data.get('volatility', 0),
            market_data.get('rsi', 50),
            market_data.get('macd', 0),
            market_data.get('bb_position', 0.5),
            market_data.get('momentum_5', 0),
            market_data.get('momentum_10', 0)
        ])
        
        features.extend([
            portfolio_data.get('total_value', 0),
            portfolio_data.get('cash_ratio', 1.0),
            portfolio_data.get('position_size', 0),
            portfolio_data.get('unrealized_pnl', 0),
            portfolio_data.get('drawdown', 0)
        ])
        
        market_features = market_data.get('technical_indicators', {})
        for indicator in ['sma_10', 'sma_50', 'ema_12', 'ema_26', 'stoch_k', 'stoch_d', 'atr']:
            features.append(market_features.get(indicator, 0))
        
        return np.array(features[:self.state_size])
        
    def act(self, state, training=True):
        if self.algorithm in ['dqn']:
            return self.agent.act(state, training)
        elif self.algorithm == 'ddpg':
            action = self.agent.act(state, add_noise=training)
            return self.continuous_to_discrete_action(action)
        elif self.algorithm == 'ppo':
            action, log_prob, state_value = self.agent.act(state)
            return action, log_prob, state_value
            
    def continuous_to_discrete_action(self, continuous_action):
        if continuous_action[0] > 0.33:
            return 2  # Buy
        elif continuous_action[0] < -0.33:
            return 0  # Sell
        else:
            return 1  # Hold
            
    def remember(self, *args):
        self.agent.remember(*args)
        
    def train(self, episodes=1000, environment=None):
        for episode in range(episodes):
            state = environment.reset()
            total_reward = 0
            step = 0
            
            while True:
                if self.algorithm == 'ppo':
                    action, log_prob, state_value = self.act(state)
                    next_state, reward, done, info = environment.step(action)
                    
                    self.remember(state, action, reward, log_prob, state_value, done)
                    
                else:
                    action = self.act(state)
                    next_state, reward, done, info = environment.step(action)
                    
                    self.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                step += 1
                
                if done or step > 1000:
                    break
            
            if self.algorithm == 'dqn':
                if len(self.agent.replay_buffer) > self.agent.batch_size:
                    loss = self.agent.replay()
                else:
                    loss = 0
                    
            elif self.algorithm == 'ddpg':
                if len(self.agent.replay_buffer) > self.agent.batch_size:
                    actor_loss, critic_loss = self.agent.replay()
                    loss = actor_loss + critic_loss
                else:
                    loss = 0
                    
            elif self.algorithm == 'ppo':
                loss = self.agent.update()
            
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(total_reward)
            self.training_history['losses'].append(loss)
            self.training_history['portfolio_values'].append(info.get('portfolio_value', 0))
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.training_history['rewards'][-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")
                
        return self.training_history
        
    def save_model(self, filepath):
        if self.algorithm == 'dqn':
            torch.save({
                'q_network': self.agent.q_network.state_dict(),
                'target_network': self.agent.target_network.state_dict(),
                'optimizer': self.agent.optimizer.state_dict(),
                'epsilon': self.agent.epsilon
            }, filepath)
            
        elif self.algorithm == 'ddpg':
            torch.save({
                'actor': self.agent.actor.state_dict(),
                'critic': self.agent.critic.state_dict(),
                'actor_target': self.agent.actor_target.state_dict(),
                'critic_target': self.agent.critic_target.state_dict(),
                'actor_optimizer': self.agent.actor_optimizer.state_dict(),
                'critic_optimizer': self.agent.critic_optimizer.state_dict()
            }, filepath)
            
        elif self.algorithm == 'ppo':
            torch.save({
                'policy': self.agent.policy.state_dict(),
                'optimizer': self.agent.optimizer.state_dict()
            }, filepath)
            
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        
        if self.algorithm == 'dqn':
            self.agent.q_network.load_state_dict(checkpoint['q_network'])
            self.agent.target_network.load_state_dict(checkpoint['target_network'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
            self.agent.epsilon = checkpoint['epsilon']
            
        elif self.algorithm == 'ddpg':
            self.agent.actor.load_state_dict(checkpoint['actor'])
            self.agent.critic.load_state_dict(checkpoint['critic'])
            self.agent.actor_target.load_state_dict(checkpoint['actor_target'])
            self.agent.critic_target.load_state_dict(checkpoint['critic_target'])
            self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            
        elif self.algorithm == 'ppo':
            self.agent.policy.load_state_dict(checkpoint['policy'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer'])