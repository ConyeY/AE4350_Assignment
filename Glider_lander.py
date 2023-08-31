# GLIDER LANDING SIMULATION
# REINFORCEMENT LEARNING
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
        self.state = 70 + random.randint(-10,10)

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
        self.state = 70 + random.randint(-10,10)
        self.dist_left = 800
        return self.state, self.dist_left

    def start(self):
        self.state = 70 + random.randint(-10,10)
        self.dist_left = 800
        return self.state, self.dist_left

    def render(self, x_pos, y_pos):
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
        plt.xlabel('X-Distance [m]'), plt.ylabel('Height [m]'), plt.title('AI LEARNING IN PROGESS')
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

# -------------------------------------------------------------------------------------
# Training the agent
# -------------------------------------------------------------------------------------

# Amount of episodes for learning 
n_episodes = 500000
env = Glider_env()

# Creating empty Q-table
Q_table = np.zeros((80*100+1, 800+1, 3))
#               Height, Distance left, Actions

# Hyperparameters
alpha = 0.1         # Learning Rate
gamma = 0.6/2         # Discount Factor
epsilon = 0.1       # Action choosing reference

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

for i in range(1, n_episodes+1):
    state, distance_left = env.reset()
    # For plotting
    x_pos = [-distance_left]
    y_pos = [state]


    # Converting to correct format for Q-table
    state = round(state*100)
    
    total_reward = 0
    done = False

    while not done:
        # Render the training process
        # env.render(x_pos, y_pos)

        # Epsilon greedy algorithm
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(Q_table[state, distance_left]) # Exploit learned values
        
        next_state, reward, done, info, next_distance_left = env.step(action) 
        next_state = round(next_state*100)
        old_value = Q_table[state, distance_left, action]
        next_max = np.max(Q_table[next_state,next_distance_left])
    
        # Updating of Q-Value
        Q_table[state, distance_left, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        total_reward += reward
        state = next_state
        distance_left = next_distance_left
        
        # For plotting
        x_pos.append(-distance_left)
        y_pos.append(state/100)

    rewards_per_episode.append(total_reward)
        
    # Keeping track of the status of the learning
    if i % 5000 == 0:
        print(f"Episode: {i}")

print("Training finished.\n")
time_end = time.time()
#plt.close()

print("The learning process took about:", round(time_end-time_start), "s\n")

print("Mean score per thousand episodes:")

mean_score = list()
for i in range(0, int(n_episodes/1000)):
    print((i+1)*1000, ": mean episode score:",\
           np.mean(rewards_per_episode[1000*i:1000*(i+1)]))
    mean_score.append(np.mean(rewards_per_episode[1000*i:1000*(i+1)]))
           
plt.figure()
plt.xlabel('Episodes x1000'), plt.ylabel('Reward'), plt.title('Average Reward as function of Episodes')
plt.plot(range(0, int(n_episodes/1000)), mean_score)
plt.plot(range(0, int(n_episodes/1000)), np.zeros(len(mean_score)), color='grey', linestyle='--')   # Guide line
plt.show()

# -------------------------------------------------------------------------------------
# Evaluation of agent's performance after Q-learning
# -------------------------------------------------------------------------------------

print('\nResults: ')

# Initiation
total_score, total_timestep, total_penalty = 0, 0, 0
episodes = 4
start_height = list()

for episode in range(episodes):
    state, distance_left = env.start()
    score, timestep, penalty = 0, 0, 0
    done = False
    start_height.append(state)
    x_distance_left = [-800]    # For plotting
    total_state = [state]

    while not done:
        # env.render()
        action = np.argmax(Q_table[state, distance_left]) # BEST ACTION
        # action = env.action_space.sample() # RANDOM ACTIONS
        state, reward, done, info, distance_left = env.step(action)
        state = round(state*100)
        score += reward
        timestep += 10
        if reward == -50:
            penalty += 1

        # For plotting
        x_distance_left.append(-800+timestep)
        total_state.append(state/100)

    print('Start height:', start_height[episode], 'm, Total Score:', score)

    plt.figure(figsize=(6,4))
    plt.xlabel('X-Distance [m]'), plt.ylabel('Height [m]'), plt.title(f'Start height: {start_height[episode]} m')
    plt.xlim([-800, 100]), plt.ylim([0, 80])
    plt.plot(x_distance_left,total_state, label='AI Path')
    plt.plot(x_axis, optimal_path, label='Optimal path', linestyle=':', color='grey')
    plt.plot(x_axis, upper_bound, label='Upper bound', linestyle='-.', color='black')
    plt.plot(x_axis, inner_upper_bound, label='Inner upper bound', linestyle='--', color='grey')
    plt.plot(x_axis, inner_lower_bound, label='Inner lower bound', linestyle='--', color='grey')
    plt.plot(x_axis, lower_bound, label='Lower bound', linestyle='-.', color='black')
    plt.legend()

    total_score += score
    total_timestep += timestep
    total_penalty += penalty
plt.show()

# Different results
print(f"\nResults after {episodes} episodes:")
print(f"Average timesteps per episode: {round(total_timestep / episodes)}")
print(f"Average penalty per episode: {round(total_penalty / episodes, 3)}")
print(f"Average score per episode: {round(total_score / episodes)}")