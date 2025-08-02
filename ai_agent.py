import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Tuple, List
import matplotlib.pyplot as plt
from game_controller import Action


class DQNNetwork(nn.Module):
    """
    Deep Q-Network für das Reinforcement Learning
    """
    
    def __init__(self, input_channels: int = 4, num_actions: int = 5):
        super(DQNNetwork, self).__init__()
        
        # Convolutional Layers für Bildverarbeitung
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Berechnung der Größe nach Convolution (für 84x84 Input)
        conv_output_size = self._get_conv_output_size(84, 84)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        # Dropout für Regularization
        self.dropout = nn.Dropout(0.5)
    
    def _get_conv_output_size(self, h: int, w: int) -> int:
        """
        Berechnet die Ausgabegröße der Convolutional Layers
        """
        # Simuliere Forward Pass durch Conv Layers
        x = torch.zeros(1, 4, h, w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass durch das Netzwerk
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten für FC Layers
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ReplayMemory:
    """
    Replay Buffer für Experience Replay
    """
    
    def __init__(self, capacity: int = 100000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Fügt eine Erfahrung zum Replay Buffer hinzu
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sampelt eine Batch von Erfahrungen
        """
        batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.BoolTensor([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.memory)


class DQNAgent:
    """
    DQN Agent für das Reinforcement Learning
    """
    
    def __init__(self, input_channels: int = 4, num_actions: int = 5, 
                 learning_rate: float = 0.0001, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, 
                 epsilon_decay: int = 100000, memory_size: int = 100000,
                 batch_size: int = 32, target_update: int = 1000):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Hyperparameter
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Netzwerke
        self.q_network = DQNNetwork(input_channels, num_actions).to(self.device)
        self.target_network = DQNNetwork(input_channels, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Target Network mit Q-Network initialisieren
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Replay Memory
        self.memory = ReplayMemory(memory_size)
        
        # Training Statistics
        self.steps_done = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Wählt eine Aktion basierend auf Epsilon-Greedy Policy
        """
        if training and random.random() < self.epsilon:
            # Zufällige Aktion (Exploration)
            return random.randrange(self.num_actions)
        else:
            # Beste Aktion laut Q-Network (Exploitation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.max(1)[1].item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        Speichert eine Erfahrung im Replay Buffer
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """
        Führt einen Trainingsschritt durch
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Batch aus Replay Memory sampeln
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Q-Values für aktuelle Zustände
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Q-Values für nächste Zustände (vom Target Network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Loss berechnen
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        
        self.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_end:
            self.epsilon -= (self.epsilon_start - self.epsilon_end) / self.epsilon_decay
        
        # Target Network Update
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.losses.append(loss.item())
        return loss.item()
    
    def save_model(self, filepath: str):
        """
        Speichert das trainierte Modell
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses
        }, filepath)
        print(f"Modell gespeichert: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Lädt ein gespeichertes Modell
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps_done = checkpoint['steps_done']
            self.episode_rewards = checkpoint['episode_rewards']
            self.episode_lengths = checkpoint['episode_lengths']
            self.losses = checkpoint['losses']
            print(f"Modell geladen: {filepath}")
            return True
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}")
            return False
    
    def plot_training_stats(self):
        """
        Zeigt Trainingsstatistiken an
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode Rewards
        if self.episode_rewards:
            ax1.plot(self.episode_rewards)
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True)
        
        # Episode Lengths
        if self.episode_lengths:
            ax2.plot(self.episode_lengths)
            ax2.set_title('Episode Lengths')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Steps')
            ax2.grid(True)
        
        # Training Loss
        if self.losses:
            ax3.plot(self.losses)
            ax3.set_title('Training Loss')
            ax3.set_xlabel('Training Step')
            ax3.set_ylabel('Loss')
            ax3.grid(True)
        
        # Epsilon Decay
        ax4.plot(range(len(self.episode_rewards)), 
                [max(self.epsilon_end, self.epsilon_start - i * (self.epsilon_start - self.epsilon_end) / self.epsilon_decay) 
                 for i in range(len(self.episode_rewards))])
        ax4.set_title('Epsilon Decay')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_stats(self) -> dict:
        """
        Gibt aktuelle Trainingsstatistiken zurück
        """
        return {
            'episodes': len(self.episode_rewards),
            'total_steps': self.steps_done,
            'epsilon': self.epsilon,
            'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_length_last_100': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
            'avg_loss_last_100': np.mean(self.losses[-100:]) if self.losses else 0
        }
