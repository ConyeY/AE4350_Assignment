# GLIDER LANDING SIMULATION
# REINFORCEMENT LEARNING
# LEARNING VISUALIZED
# BY: CONNY YU
# DATE: 2023-06-21

# Import Dependencies
import os, gym, random, time
from turtle import color, distance
from gym.spaces import Discrete, Box
import numpy as np
import matplotlib.pyplot as plt

# Lowering the logger level in gym to avoid warning
gym.logger.set_level(40)

# Defining the environment
class Glider_env():
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box( low=0, high=100, shape=(1,), dtype=np.float32)
        
        # Initial position Above Ground Level [m]
        self.state = 70

        # Distance left to Touch Down Point [m]
        self.dist_left = 800

    def step(self, action):
        done = False

        # ACTIONS being
        # [0] = No airbrakes, sinkrate 1/35 per meter in x-axis
        # [1] = 50% airbrake, sinkrate 1/11 per meter in x-axis
        # [2] = 100% airbrake, sinkrate 1/5 per meter in x-axis
        if action == 0:
            self.state -= 1/35*10
        elif action == 1:
            self.state -= 1/11*10
        else:
            self.state -= 1/5*10

        # Reduce distance left by 10 m
        self.dist_left -= 10


        # REWARD, more reward if glider in the middle cone, less reward if in outer cone
        if self.state <= 1/5 * self.dist_left and self.state > 1/7 * self.dist_left:
            reward = -3
        elif self.state <= 1/7 * self.dist_left and self.state >= 1/17 * self.dist_left:
            reward = -1
        elif self.state < 1/17 * self.dist_left and self.state >= 1/35 * self.dist_left:
            reward = -3
        else:
            reward = -50
            done = True

        if done == False:
            if self.dist_left <= 10 or self.state <= 0:
                if self.state <= 1/5 * self.dist_left and self.state >= 1/35 * self.dist_left:
                    reward = 1000
                done = True
            else:
                done = False         

        # Placeholder for info
        info = {}

        return self.state, reward, done, info, self.dist_left
    
    def reset(self):
        self.state = 70
        self.dist_left = 800
        return self.state, self.dist_left

    def start(self, height):
        self.state = 60 + height
        self.dist_left = 800
        return self.state, self.dist_left

    def render(self, x_pos, y_pos, counter):
        '''
        Take in x_pos and y_pos as lists
        '''
        # Guide lines for plot
        x_axis = np.linspace(-800,0)
        upper_bound = -1/5*np.linspace(-800,0)
        inner_upper_bound = -1/7*np.linspace(-800,0)
        inner_lower_bound = -1/17*np.linspace(-800,0)
        lower_bound = -1/35*np.linspace(-800,0)
        optimal_path = -1/11*np.linspace(-800,0)
        plt.xlabel('X-Distance [m]'), plt.ylabel('Height [m]'), plt.title(f'AI LEARNING IN PROGESS Episode: {counter}')
        plt.xlim([-800, 100]), plt.ylim([0, 80])
        plt.plot(x_pos, y_pos, label='AI path')
        plt.plot(x_axis, optimal_path, label='Optimal path', linestyle=':', color='grey')
        plt.plot(x_axis, upper_bound, label='Upper bound', linestyle='-.', color='black')
        plt.plot(x_axis, inner_upper_bound, label='Inner upper bound', linestyle='--', color='grey')
        plt.plot(x_axis, inner_lower_bound, label='Inner lower bound', linestyle='--', color='grey')
        plt.plot(x_axis, lower_bound, label='Lower bound', linestyle='-.', color='black')
        plt.pause(1e-4)

def enviroment_test(env):
    '''
    Quick function to test that the environment is working
    '''
    episodes = 5

    for episode in range(1,episodes+1):
        state = env.reset()
        done = False
        score = 0
        while not done:
            # env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score+=reward
            #print('reward:',reward)
            #print('state:', n_state)
        print('Iteration:', episode, 'Score:', score)
        print('The Environment is working fine.\n')

env = Glider_env()

# Training the agent

from IPython.display import clear_output

# Amount of episodes for learning 
# Converges around 20000
n_episodes = 3000

# Creating empty Q-table
Q_table = np.zeros((80*100+1, 800+1, 3))
#               Height, Distance left, Actions

# Hyperparameters
alpha = 0.1         # Learning Rate
gamma = 0.6         # Discount Factor
epsilon = 0.1

# Stats
rewards_per_episode = list()

# Guide lines for plot
x_axis = np.linspace(-800,0)
upper_bound = -1/5*np.linspace(-800,0)
inner_upper_bound = -1/7*np.linspace(-800,0)
inner_lower_bound = -1/17*np.linspace(-800,0)
lower_bound = -1/35*np.linspace(-800,0)
optimal_path = -1/11*np.linspace(-800,0)

time_start = time.time()
counter = 0

for i in range(1, n_episodes+1):
    state, distance_left = env.reset()
    # For plotting
    x_pos = [-distance_left]
    y_pos = [state]

    plt.ion()
    plt.figure(figsize=(6,4))
    plt.show()

    # Converting to correct format for Q-table
    state = round(state*100)
    
    total_reward = 0
    done = False

    while not done:
        # Render the training process
        env.render(x_pos, y_pos, counter)

        # Epsilon greedy algorithm
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(Q_table[state, distance_left]) # Exploit learned values
        next_state, reward, done, info, next_distance_left = env.step(action) 
        next_state = round(next_state*100)

        old_value = Q_table[state, distance_left, action]
        next_max = np.max(Q_table[next_state,next_distance_left])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        Q_table[state, distance_left, action] = new_value
        total_reward += reward
        state = next_state
        distance_left = next_distance_left
        
        # For plotting
        x_pos.append(-distance_left)
        y_pos.append(state/100)

    plt.close()
    counter += 1
    rewards_per_episode.append(total_reward)
        
    # Keeping track of the status of the learning
    if i % 5000 == 0:
        print(f"Episode: {i}")

print("Training finished.\n")
time_end = time.time()

print("The learning process took about:", round(time_end-time_start), "s\n")

print("Mean score per thousand episodes")

for i in range(0,20):
    print((i+1)*1000, ": mean episode score:",\
           np.mean(rewards_per_episode[1000*i:1000*(i+1)]))
