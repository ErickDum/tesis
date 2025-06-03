import argparse
import gymnasium as gym
import numpy as np
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
    network_sync_rate = 10
    replay_memory_size = 3000
    mini_batch_size = 32
    loss_fn = nn.MSELoss()

    def __init__(self, env):
        self.env = env
        self.optimizer = None

    def train(self, episodes):
        # Determine sizes based on environment's spaces
        num_states = 4#*30*5*5
        num_actions = 4*140*15
        ob_space = self.env.observation_space
        ac_space = self.env.action_space
        print("Observation space: ", ob_space, ob_space.dtype)
        print("Action space: ", ac_space, ac_space.dtype)
        #num_states = ob_space.shape[0]
        #num_actions = ac_space.n

        # Network sizes
        l1, l2, l3 = 512, 256, 128

        # Initialize networks and memory
        policy_dqn = DQN(num_states, l1, l2, l3, num_actions)
        target_dqn = DQN(num_states, l1, l2, l3, num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        memory = ReplayMemory(self.replay_memory_size)

        epsilon = 1.0
        epsilon_history, rewards_per_episode = [], []
        step_count = 0

        for episode in range(episodes):
            print(f"Episode {episode + 1}")
            obs = self.env.reset()
            state = torch.tensor(obs, dtype=torch.float32)
            total_reward = 0
            done = False

            while not done:
                # Epsilon-greedy action
                if random.random() < epsilon:
                    action = random.randint(0, num_actions - 1)
                else:
                    with torch.no_grad():
                        q_vals = policy_dqn(state)
                        action = q_vals.argmax().item()

                # Step environment
                next_obs, reward, done, info = env.step(action)
                if next_obs is None:
                    print("No next observation, ending step")
                    continue
                next_state = torch.tensor(next_obs, dtype=torch.float32)
                total_reward += reward

                # Store transition
                memory.append((state, action, next_state, reward, done))
                state = next_state
                step_count += 1

                # Learn
                if len(memory) > self.mini_batch_size:
                    batch = memory.sample(self.mini_batch_size)
                    self.optimize(batch, policy_dqn, target_dqn)

                # Sync target network
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            # End of episode
            rewards_per_episode.append(total_reward / step_count if step_count > 0 else 0)
            epsilon = epsilon - 1 / episodes
            epsilon_history.append(epsilon)
            step_count = 0

            # Guardado cada 1000 episodios (empezando desde episodio 1)
            if (episode + 1) % 1000 == 0:
                # Guardar red neuronal
                model_name = f"ResourceAllocation{episode+1}.pt"
                torch.save(policy_dqn.state_dict(), model_name)

                # Guardar CSV
                csv_name = f"datos{episode+1}.csv"
                with open(csv_name, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["reward", "epsilon"])
                    for r, e in zip(rewards_per_episode, epsilon_history):
                        writer.writerow([r, e])

        # Close env
        self.env.close()
        return rewards_per_episode, epsilon_history

    def optimize(self, batch, policy_dqn, target_dqn):
        current_q, target_q = [], []
        for state, action, next_state, reward, done in batch:
            # Compute target
            if done:
                target_val = torch.tensor([reward])
            else:
                with torch.no_grad():
                    next_q = target_dqn(next_state).max()
                    target_val = torch.tensor([reward]) + self.discount_factor_g * next_q

            # Current Q
            cur_q = policy_dqn(state)
            current_q.append(cur_q)

            # Build target Q vector
            tq = policy_dqn(state).clone()
            tq[action] = target_val
            target_q.append(tq)

        loss = self.loss_fn(torch.stack(current_q), torch.stack(target_q))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Main: parse args, setup ns3-gym, train agent
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=10000)
    args = parser.parse_args()

    # ns3-gym parameters
    port = 5555
    stepTime = 0.1
    seed = 0
    simTime = 40
    simArgs = {"--simTime": simTime}
    env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=bool(args.start), simSeed=seed, simArgs=simArgs)
    # Optionally: env = ns3env.Ns3Env()
    env.reset()
    agent = ResourceAllocation(env)
    # Train for given number of iterations
    rewards, eps_history = agent.train(args.iterations)

    # Optionally: plot results
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.show()