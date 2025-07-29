import argparse
import numpy as np
import os
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
from ns3gym import ns3env
import csv

# DQN Model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, h2_nodes, h3_nodes, out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.fc3 = nn.Linear(h2_nodes, h3_nodes)
        self.out = nn.Linear(h3_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

# Replay Buffer
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    def append(self, transition):
        self.memory.append(transition)
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    def __len__(self):
        return len(self.memory)

# Training Class
class ResourceAllocation:
    learning_rate_a = 0.001
    discount_factor_g = 0.8
    network_sync_rate = 100
    replay_memory_size = 5000
    mini_batch_size = 128
    loss_fn = nn.MSELoss()

    def __init__(self, env, model_path=None):
        self.env = env
        self.optimizer = None
        self.model_path = model_path

    def train(self, episodes):
        # Determinación de dimensiones
        state_max   = np.array([4, 30, 4, 4])
        num_states  = 4  # ob_space.shape[0] en otro caso
        num_actions = 4*140*15

        # Tamaños de las capas ocultas
        l1, l2, l3 = 512, 256, 128

        # Inicializar redes
        policy_dqn = DQN(num_states, l1, l2, l3, num_actions)
        target_dqn = DQN(num_states, l1, l2, l3, num_actions)

        # Eliminada la carga de checkpoint para entrenar siempre desde cero
        print("[INFO] Iniciando entrenamiento desde cero.")

        # Sincronizar target con policy
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Optimizador
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        memory = ReplayMemory(self.replay_memory_size)

        epsilon = 1.0
        epsilon_history, rewards_per_episode = [], []
        step_count = step_count2 = 0

        for episode in range(episodes):
            print(f"============= Episodio {episode + 1} =============")
            obs = self.env.reset()
            state = torch.tensor(np.array(obs)/state_max, dtype=torch.float32)
            total_reward = 0
            done = False

            while not done:
                # Política ε-greedy
                if random.random() < epsilon:
                    action = random.randint(0, num_actions - 1)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state).argmax().item()

                next_obs, reward, done, _ = self.env.step(action)
                if next_obs is None:
                    continue

                next_state = torch.tensor(np.array(next_obs)/state_max, dtype=torch.float32)
                total_reward += reward

                memory.append((state, action, next_state, reward, done))
                state = next_state
                step_count += 1
                step_count2 += 1


                # Sincronización periódica
                if step_count2 > self.network_sync_rate:
                    step_count2 = 0
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                
            # Aprendizaje
            if len(memory) > self.mini_batch_size:
                batch = memory.sample(self.mini_batch_size)
                self.optimize(batch, policy_dqn, target_dqn)

            # Fin de episodio
            rewards_per_episode.append(total_reward / (step_count or 1))
            epsilon = max(0.0, epsilon - 1/episodes)
            epsilon_history.append(epsilon)
            step_count = 0

            # Guardado periódico cada 100 episodios
            if (episode + 1) % 100 == 0:
                guardar = f"Resource_Allocation{episode+1}.pt"
                torch.save(policy_dqn.state_dict(), guardar)
                print(f"[INFO] Modelo guardado en «{guardar}»")
                # Exportar CSV de métricas
                csv_name = f"datos{episode+1}.csv"
                with open(csv_name, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["reward","epsilon"])
                    writer.writerows(zip(rewards_per_episode, epsilon_history))

        self.env.close()
        return rewards_per_episode, epsilon_history

    def optimize(self, batch, policy_dqn, target_dqn):
        current_q, target_q = [], []
        for state, action, next_state, reward, done in batch:
            # Cálculo de Q* objetivo
            if done:
                y = torch.tensor([reward])
            else:
                with torch.no_grad():
                    y = torch.tensor([reward]) + self.discount_factor_g * target_dqn(next_state).max()

            q_pred = policy_dqn(state)
            current_q.append(q_pred)
            y_full = q_pred.clone()
            y_full[action] = y
            target_q.append(y_full)

        loss = self.loss_fn(torch.stack(current_q), torch.stack(target_q))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Punto de entrada
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=1000)
    args = parser.parse_args()

    # Configurar NS3-Gym
    port = 5555
    stepTime = 0.1
    env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=bool(args.start),
                        simSeed=0, simArgs={"--simTime": 40})
    env.reset()

    # Entrenador
    agent = ResourceAllocation(env)
    rewards, eps_history = agent.train(args.iterations)

    # Visualización final
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa media')
    plt.show()
