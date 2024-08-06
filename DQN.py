import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(37, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 4)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class DuelingQNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(37, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 4)  
        self.value = nn.Linear(12,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        values = self.value(x)
        advantages = self.fc3(x)
        return values + (advantages - advantages.mean())
    

class DQNAgent:
    def __init__(self):
        self.q_network = QNetwork()
        self.target_network = QNetwork()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = deque(maxlen=1000)
        self.gamma = 0.99
        self.batch_size = 16
        self.update_target_network()
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.loss = nn.HuberLoss()
        self.valid_states = list(range(37))

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def one_hot(self, index):
        arr = np.zeros(37)
        arr[index] = 1
        return arr

    def remember(self, state, action, next_state, reward, terminated, truncated):
        self.memory.append((state, action, next_state, reward, terminated, truncated))

    def get_action(self, state):
        if random.random() < self.epsilon:  
            return random.choice(list(range(4)))
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def replay(self,e):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, next_state, reward, terminated, truncated in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            target = reward
            if not (terminated or truncated):
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                q_values = self.target_network(next_state)
                target += self.gamma * q_values.max().item()
            target_f = self.q_network(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            output = self.loss(target_f, self.q_network(state))
            output.backward()
            self.optimizer.step()

    
    def train(self, env, episodes):
        scores = []
        errors = []
        for e in range(episodes):
            print('episode: ', e, 'epsilon: ', self.epsilon)
            state = self.one_hot(env.reset()[0])
            done = False
            score = 0
            error = 0
            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = self.one_hot(next_state) if next_state<37 else -1
                self.remember(state, action, next_state, reward, terminated, truncated)
                state = next_state
                score += reward
                self.replay(e)
                done = terminated or truncated
            scores.append(score)
            
            for valid_state in self.valid_states:
                action_scores = agent.q_network(torch.FloatTensor(agent.one_hot(valid_state)).unsqueeze(0)).detach().numpy()[0]
                for i in range(4):
                    error += abs(action_scores[i] - true_values[valid_state][i])
            errors.append(error)
                    
            self.update_target_network()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        return scores, errors

    def load(self, name):
        self.q_network.load_state_dict(torch.load(name))
        self.target_network.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.target_network.state_dict(), name)


true_values = [
        [-15.0, -14.0, -14.0, -15.0],
        [-14.0, -13.0, -13.0, -15.0],
        [-13.0, -12.0, -12.0, -14.0],
        [-12.0, -11.0, -11.0, -13.0],
        [-11.0, -10.0, -10.0, -12.0],
        [-10.0, -9.0, -9.0, -11.0],
        [-9.0, -8.0, -8.0, -10.0],
        [-8.0, -7.0, -7.0, -9.0],
        [-7.0, -6.0, -6.0, -8.0],
        [-6.0, -5.0, -5.0, -7.0],
        [-5.0, -4.0, -4.0, -6.0],
        [-4.0, -4.0, -3.0, -5.0],
        [-15.0, -13.0, -13.0, -14.0],
        [-14.0, -12.0, -12.0, -14.0],
        [-13.0, -11.0, -11.0, -13.0],
        [-12.0, -10.0, -10.0, -12.0],
        [-11.0, -9.0, -9.0, -11.0],
        [-10.0, -8.0, -8.0, -10.0],
        [-9.0, -7.0, -7.0, -9.0],
        [-8.0, -6.0, -6.0, -8.0],
        [-7.0, -5.0, -5.0, -7.0],
        [-6.0, -4.0, -4.0, -6.0],
        [-5.0, -3.0, -3.0, -5.0],
        [-4.0, -3.0, -2.0, -4.0],
        [-14.0, -12.0, -14.0, -13.0],
        [-13.0, -11.0, -113.0, -13.0],
        [-12.0, -10.0, -113.0, -12.0],
        [-11.0, -9.0, -113.0, -11.0],
        [-10.0, -8.0, -113.0, -10.0],
        [-9.0, -7.0, -113.0, -9.0],
        [-8.0, -6.0, -113.0, -8.0],
        [-7.0, -5.0, -113.0, -7.0],
        [-6.0, -4.0, -113.0, -6.0],
        [-5.0, -3.0, -113.0, -5.0],
        [-4.0, -2.0, -113.0, -4.0],
        [-3.0, -2.0, -1.0, -3.0],
        [-13.0, -113.0, -14.0, -14.0]
    ]


if __name__ == '__main__':
    env = gym.make('CliffWalking-v0', render_mode='rgb_array')
    env.reset()
    agent = DQNAgent()
    episodes = 200 
    scores, errors = agent.train(env, episodes)

    plt.plot(errors)