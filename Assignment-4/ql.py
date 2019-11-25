# Author: Saad Waraich
#GTID: 903459227
# https://github.com/dennybritz/reinforcement-learning
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import mdptoolbox, mdptoolbox.example
from gym.envs.toy_text.frozen_lake import generate_random_map
from collections import defaultdict, namedtuple
import itertools
import sys
import pandas as pd



smoothing_window=10

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, currEnv="forest"):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.nA))

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.nA)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        if currEnv == "lake":
            state = env.reset()
        else:
            state = np.random.randint(env.nS, size=1)[0]
        
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            if currEnv == "lake":
                next_state, reward, done, _ = env.step(action)
            else:
                next_state, reward, done = env.step(state, action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            state = next_state
    
    return Q, stats

def forest():
    class fEnv:
        def __init__(self, states, actions):
            self.nS = states
            self.nA = actions
            self.P = [[[] for x in range(actions)] for y in range(states)]

        def step(self,state,action):
            allP = []
            for i in envi.P[state][action]:
                allP.append(i[0])
            LS = envi.P[state][action][np.random.choice(envi.nS, 1, p=np.array(allP))[0]]
            next_state, reward, done = LS[1],LS[0], LS[2]
            return next_state, reward, done


    def forestEnv(states):
        np.random.seed(1055)
        env = fEnv(states, 2)
        P, R = mdptoolbox.example.forest(S=states, r2=40, p=0.5)

        for a in range(0,2):
            for s in range(0, states):
                for ss in range(0, states):
                    env.P[s][a].append([P[a][s][ss], ss, R[s][a], False])
        return env

    timeToRun = []
    # totalIters = []
    envi = forestEnv(15)
    for gamma in [0.001, 0.01, 0.1, 0.2, 0.3]:
        print("Running PI Forest time run gamma = " + str(gamma))
        tmp = []
        for s in [400,600,800,1000,1200,1400]:
            startTime=time.time()
            Q, stats = q_learning(envi, s, discount_factor=0.999)
            timeTaken=time.time() - startTime
            tmp.append(timeTaken)
        timeToRun.append(tmp)
        # totalIters.append(len(Vs))

    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Time to Run VS No. of Episodes, Forest - QL"
    plt.title(titleStr)
    plt.xlabel(" No. of Episodes")
    plt.ylabel("Time to Run")
    plt.plot([400,600,800,1000,1200,1400], timeToRun[0], label="epsilon = 0.001")
    plt.plot([400,600,800,1000,1200,1400], timeToRun[1], label="epsilon = 0.01")
    plt.plot([400,600,800,1000,1200,1400], timeToRun[2], label="epsilon = 0.1")
    plt.plot([400,600,800,1000,1200,1400], timeToRun[3], label="epsilon = 0.2")
    plt.plot([400,600,800,1000,1200,1400], timeToRun[4], label="epsilon = 0.3")
    # plt.plot(lengths, fitnessValMeanSA, '-o', label="simulated annealing")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-QL-TimesVSEpisdoes" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    allStats = []
    allQ = []
    envi = forestEnv(15)
    for gamma in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        print("Running QL Forest gamma = " + str(gamma))
        Q, stats = q_learning(envi, 1000, discount_factor=gamma)
        allStats.append(stats)
        allQ.append(Q)

    # Plot the episode length over time
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    plt.plot(allStats[0].episode_lengths, label="gamma = 0.6")
    plt.plot(allStats[1].episode_lengths, label="gamma = 0.7")
    plt.plot(allStats[2].episode_lengths, label="gamma = 0.8")
    plt.plot(allStats[3].episode_lengths, label="gamma = 0.9")
    plt.plot(allStats[4].episode_lengths, label="gamma = 0.95")
    plt.plot(allStats[5].episode_lengths, label="gamma = 0.99")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time for Gamma - Forest")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-QL-LengthVSTime-gamma" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    # Plot the episode reward over time
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    rewards_smoothed = []
    for stat in allStats:
        rewards_smoothed.append(pd.Series(stat.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean())
    plt.plot(rewards_smoothed[0], label="gamma = 0.6")
    plt.plot(rewards_smoothed[1], label="gamma = 0.7")
    plt.plot(rewards_smoothed[2], label="gamma = 0.8")
    plt.plot(rewards_smoothed[3], label="gamma = 0.9")
    plt.plot(rewards_smoothed[4], label="gamma = 0.95")
    plt.plot(rewards_smoothed[5], label="gamma = 0.99")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time  for Gamma (Smoothed over window size {})".format(smoothing_window) + " - Forest")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-QL-RewardVSTime-gamma" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    # Plot time steps and episode number
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    plt.plot(np.cumsum(allStats[0].episode_lengths), np.arange(len(allStats[0].episode_lengths)), label="gamma = 0.6")
    plt.plot(np.cumsum(allStats[1].episode_lengths), np.arange(len(allStats[1].episode_lengths)), label="gamma = 0.7")
    plt.plot(np.cumsum(allStats[2].episode_lengths), np.arange(len(allStats[2].episode_lengths)), label="gamma = 0.8")
    plt.plot(np.cumsum(allStats[3].episode_lengths), np.arange(len(allStats[3].episode_lengths)), label="gamma = 0.9")
    plt.plot(np.cumsum(allStats[4].episode_lengths), np.arange(len(allStats[4].episode_lengths)), label="gamma = 0.95")
    plt.plot(np.cumsum(allStats[5].episode_lengths), np.arange(len(allStats[5].episode_lengths)), label="gamma = 0.99")
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step for Gamma - Forest")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-QL-EpisodeVSTimeStep-gamma" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    allStats = []
    allQ = []
    envi = forestEnv(15)
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print("Running QL Forest alpha = " + str(alpha))
        Q, stats = q_learning(envi, 1000, discount_factor=0.999, alpha=alpha)
        allStats.append(stats)
        allQ.append(Q)

    # Plot the episode length over time
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    plt.plot(allStats[0].episode_lengths, label="alpha = 0.3")
    plt.plot(allStats[1].episode_lengths, label="alpha = 0.4")
    plt.plot(allStats[2].episode_lengths, label="alpha = 0.5")
    plt.plot(allStats[3].episode_lengths, label="alpha = 0.6")
    plt.plot(allStats[4].episode_lengths, label="alpha = 0.7")
    plt.plot(allStats[5].episode_lengths, label="alpha = 0.8")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time for Alpha - Forest")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-QL-LengthVSTime-alpha" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    # Plot the episode reward over time
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    rewards_smoothed = []
    for stat in allStats:
        rewards_smoothed.append(pd.Series(stat.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean())
    plt.plot(rewards_smoothed[0], label="alpha = 0.3")
    plt.plot(rewards_smoothed[1], label="alpha = 0.4")
    plt.plot(rewards_smoothed[2], label="alpha = 0.5")
    plt.plot(rewards_smoothed[3], label="alpha = 0.6")
    plt.plot(rewards_smoothed[4], label="alpha = 0.7")
    plt.plot(rewards_smoothed[5], label="alpha = 0.8")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time for Alpha (Smoothed over window size {})".format(smoothing_window) + " - Forest")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-QL-RewardVSTime-alpha" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    # Plot time steps and episode number
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    plt.plot(np.cumsum(allStats[0].episode_lengths), np.arange(len(allStats[0].episode_lengths)), label="alpha = 0.3")
    plt.plot(np.cumsum(allStats[1].episode_lengths), np.arange(len(allStats[1].episode_lengths)), label="alpha = 0.4")
    plt.plot(np.cumsum(allStats[2].episode_lengths), np.arange(len(allStats[2].episode_lengths)), label="alpha = 0.5")
    plt.plot(np.cumsum(allStats[3].episode_lengths), np.arange(len(allStats[3].episode_lengths)), label="alpha = 0.6")
    plt.plot(np.cumsum(allStats[4].episode_lengths), np.arange(len(allStats[4].episode_lengths)), label="alpha = 0.7")
    plt.plot(np.cumsum(allStats[5].episode_lengths), np.arange(len(allStats[5].episode_lengths)), label="alpha = 0.8")
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step for Alpha - Forest")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-QL-EpisodeVSTimeStep-alpha" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    allStats = []
    allQ = []
    envi = forestEnv(15)
    for epsilon in [0.01, 0.1, 0.2, 0.3]:
        print("Running QL Forest epsilon = " + str(epsilon))
        Q, stats = q_learning(envi, 1000, discount_factor=0.999, epsilon=epsilon)
        allStats.append(stats)
        allQ.append(Q)

    # Plot the episode length over time
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    plt.plot(allStats[0].episode_lengths, label="epsilon = 0.01")
    plt.plot(allStats[1].episode_lengths, label="epsilon = 0.1")
    plt.plot(allStats[2].episode_lengths, label="epsilon = 0.2")
    plt.plot(allStats[3].episode_lengths, label="epsilon = 0.3")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time for Epsilon - Forest")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-QL-LengthVSTime-epsilon" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    # Plot the episode reward over time
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    rewards_smoothed = []
    for stat in allStats:
        rewards_smoothed.append(pd.Series(stat.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean())
    plt.plot(rewards_smoothed[0], label="epsilon = 0.01")
    plt.plot(rewards_smoothed[1], label="epsilon = 0.1")
    plt.plot(rewards_smoothed[2], label="epsilon = 0.2")
    plt.plot(rewards_smoothed[3], label="epsilon = 0.3")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time for Epsilon (Smoothed over window size {})".format(smoothing_window) + " - Forest")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-QL-RewardVSTime-epsilon" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    # Plot time steps and episode number
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    plt.plot(np.cumsum(allStats[0].episode_lengths), np.arange(len(allStats[0].episode_lengths)), label="epsilon = 0.01")
    plt.plot(np.cumsum(allStats[1].episode_lengths), np.arange(len(allStats[1].episode_lengths)), label="epsilon = 0.1")
    plt.plot(np.cumsum(allStats[2].episode_lengths), np.arange(len(allStats[2].episode_lengths)), label="epsilon = 0.2")
    plt.plot(np.cumsum(allStats[3].episode_lengths), np.arange(len(allStats[3].episode_lengths)), label="epsilon = 0.3")
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step for Epsilon - Forest")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-QL-EpisodeVSTimeStep-epsilon" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

def lake():

    timeToRun = []
    np.random.seed(1055)
    envi = gym.make("FrozenLake-v0", desc=generate_random_map(size=30, p=0.8), is_slippery = False)
    for gamma in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        print("Running QL Lake time run gamma = " + str(gamma))
        tmp = []
        for s in [400,600,800,1000,1200,1400]:
            startTime=time.time()
            Q, stats = q_learning(envi, s, discount_factor=0.999, currEnv="lake")
            timeTaken=time.time() - startTime
            tmp.append(timeTaken)
        timeToRun.append(tmp)


    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Time to Run VS No. of Episodes, Lake - QL"
    plt.title(titleStr)
    plt.xlabel(" No. of Episodes")
    plt.ylabel("Time to Run")
    plt.plot([400,600,800,1000,1200,1400], timeToRun[0], label="epsilon = 0.001")
    plt.plot([400,600,800,1000,1200,1400], timeToRun[1], label="epsilon = 0.01")
    plt.plot([400,600,800,1000,1200,1400], timeToRun[2], label="epsilon = 0.1")
    plt.plot([400,600,800,1000,1200,1400], timeToRun[3], label="epsilon = 0.2")
    plt.plot([400,600,800,1000,1200,1400], timeToRun[4], label="epsilon = 0.3")
    # plt.plot(lengths, fitnessValMeanSA, '-o', label="simulated annealing")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-QL-TimesVSEpisdoes" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    allStats = []
    allQ = []
    np.random.seed(1055)
    envi = gym.make("FrozenLake-v0", desc=generate_random_map(size=30, p=0.8), is_slippery = False)
    for gamma in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        print("Running QL Lake gamma = " + str(gamma))
        Q, stats = q_learning(envi, 1000, discount_factor=gamma, currEnv="lake")
        allStats.append(stats)
        allQ.append(Q)

    # Plot the episode length over time
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    plt.plot(allStats[0].episode_lengths, label="gamma = 0.6")
    plt.plot(allStats[1].episode_lengths, label="gamma = 0.7")
    plt.plot(allStats[2].episode_lengths, label="gamma = 0.8")
    plt.plot(allStats[3].episode_lengths, label="gamma = 0.9")
    plt.plot(allStats[4].episode_lengths, label="gamma = 0.95")
    plt.plot(allStats[5].episode_lengths, label="gamma = 0.99")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time for Gamma - Lake")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-QL-LengthVSTime-gamma" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    # Plot the episode reward over time
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    rewards_smoothed = []
    for stat in allStats:
        rewards_smoothed.append(pd.Series(stat.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean())
    plt.plot(rewards_smoothed[0], label="gamma = 0.6")
    plt.plot(rewards_smoothed[1], label="gamma = 0.7")
    plt.plot(rewards_smoothed[2], label="gamma = 0.8")
    plt.plot(rewards_smoothed[3], label="gamma = 0.9")
    plt.plot(rewards_smoothed[4], label="gamma = 0.95")
    plt.plot(rewards_smoothed[5], label="gamma = 0.99")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time  for Gamma (Smoothed over window size {})".format(smoothing_window) + " - Lake")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-QL-RewardVSTime-gamma" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    # Plot time steps and episode number
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    plt.plot(np.cumsum(allStats[0].episode_lengths), np.arange(len(allStats[0].episode_lengths)), label="gamma = 0.6")
    plt.plot(np.cumsum(allStats[1].episode_lengths), np.arange(len(allStats[1].episode_lengths)), label="gamma = 0.7")
    plt.plot(np.cumsum(allStats[2].episode_lengths), np.arange(len(allStats[2].episode_lengths)), label="gamma = 0.8")
    plt.plot(np.cumsum(allStats[3].episode_lengths), np.arange(len(allStats[3].episode_lengths)), label="gamma = 0.9")
    plt.plot(np.cumsum(allStats[4].episode_lengths), np.arange(len(allStats[4].episode_lengths)), label="gamma = 0.95")
    plt.plot(np.cumsum(allStats[5].episode_lengths), np.arange(len(allStats[5].episode_lengths)), label="gamma = 0.99")
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step for Gamma - Lake")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-QL-EpisodeVSTimeStep-gamma" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    allStats = []
    allQ = []
    np.random.seed(1055)
    envi = gym.make("FrozenLake-v0", desc=generate_random_map(size=30, p=0.8), is_slippery = False)
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print("Running QL Lake alpha = " + str(alpha))
        Q, stats = q_learning(envi, 1000, discount_factor=0.999, alpha=alpha, currEnv="lake")
        allStats.append(stats)
        allQ.append(Q)

    # Plot the episode length over time
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    plt.plot(allStats[0].episode_lengths, label="alpha = 0.3")
    plt.plot(allStats[1].episode_lengths, label="alpha = 0.4")
    plt.plot(allStats[2].episode_lengths, label="alpha = 0.5")
    plt.plot(allStats[3].episode_lengths, label="alpha = 0.6")
    plt.plot(allStats[4].episode_lengths, label="alpha = 0.7")
    plt.plot(allStats[5].episode_lengths, label="alpha = 0.8")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time for Alpha - Lake")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-QL-LengthVSTime-alpha" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    # Plot the episode reward over time
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    rewards_smoothed = []
    for stat in allStats:
        rewards_smoothed.append(pd.Series(stat.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean())
    plt.plot(rewards_smoothed[0], label="alpha = 0.3")
    plt.plot(rewards_smoothed[1], label="alpha = 0.4")
    plt.plot(rewards_smoothed[2], label="alpha = 0.5")
    plt.plot(rewards_smoothed[3], label="alpha = 0.6")
    plt.plot(rewards_smoothed[4], label="alpha = 0.7")
    plt.plot(rewards_smoothed[5], label="alpha = 0.8")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time for Alpha (Smoothed over window size {})".format(smoothing_window) + " - Lake")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-QL-RewardVSTime-alpha" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    # Plot time steps and episode number
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    plt.plot(np.cumsum(allStats[0].episode_lengths), np.arange(len(allStats[0].episode_lengths)), label="alpha = 0.3")
    plt.plot(np.cumsum(allStats[1].episode_lengths), np.arange(len(allStats[1].episode_lengths)), label="alpha = 0.4")
    plt.plot(np.cumsum(allStats[2].episode_lengths), np.arange(len(allStats[2].episode_lengths)), label="alpha = 0.5")
    plt.plot(np.cumsum(allStats[3].episode_lengths), np.arange(len(allStats[3].episode_lengths)), label="alpha = 0.6")
    plt.plot(np.cumsum(allStats[4].episode_lengths), np.arange(len(allStats[4].episode_lengths)), label="alpha = 0.7")
    plt.plot(np.cumsum(allStats[5].episode_lengths), np.arange(len(allStats[5].episode_lengths)), label="alpha = 0.8")
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step for Alpha - Lake")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-QL-EpisodeVSTimeStep-alpha" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    allStats = []
    allQ = []
    np.random.seed(1055)
    envi = gym.make("FrozenLake-v0", desc=generate_random_map(size=30, p=0.8), is_slippery = False)
    for epsilon in [0.01, 0.1, 0.2, 0.3]:
        print("Running QL Forest epsilon = " + str(epsilon))
        Q, stats = q_learning(envi, 1000, discount_factor=0.999, epsilon=epsilon, currEnv="lake")
        allStats.append(stats)
        allQ.append(Q)

    # Plot the episode length over time
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    plt.plot(allStats[0].episode_lengths, label="epsilon = 0.01")
    plt.plot(allStats[1].episode_lengths, label="epsilon = 0.1")
    plt.plot(allStats[2].episode_lengths, label="epsilon = 0.2")
    plt.plot(allStats[3].episode_lengths, label="epsilon = 0.3")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time for Epsilon - Lake")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-QL-LengthVSTime-epsilon" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    # Plot the episode reward over time
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    rewards_smoothed = []
    for stat in allStats:
        rewards_smoothed.append(pd.Series(stat.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean())
    plt.plot(rewards_smoothed[0], label="epsilon = 0.01")
    plt.plot(rewards_smoothed[1], label="epsilon = 0.1")
    plt.plot(rewards_smoothed[2], label="epsilon = 0.2")
    plt.plot(rewards_smoothed[3], label="epsilon = 0.3")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time for Epsilon (Smoothed over window size {})".format(smoothing_window) + " - Lake")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-QL-RewardVSTime-epsilon" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    # Plot time steps and episode number
    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    plt.plot(np.cumsum(allStats[0].episode_lengths), np.arange(len(allStats[0].episode_lengths)), label="epsilon = 0.01")
    plt.plot(np.cumsum(allStats[1].episode_lengths), np.arange(len(allStats[1].episode_lengths)), label="epsilon = 0.1")
    plt.plot(np.cumsum(allStats[2].episode_lengths), np.arange(len(allStats[2].episode_lengths)), label="epsilon = 0.2")
    plt.plot(np.cumsum(allStats[3].episode_lengths), np.arange(len(allStats[3].episode_lengths)), label="epsilon = 0.3")
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step for Epsilon - Lake")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-QL-EpisodeVSTimeStep-epsilon" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

forest()
lake()

# policy, v = policy_improvement(env)
# print("Policy Probability Distribution:")
# print(policy)
# print("")

# print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
# print(np.reshape(np.argmax(policy, axis=1), env.shape))
# print("")

# print("Value Function:")
# print(v)
# print("")

# print("Reshaped Grid Value Function:")
# print(v.reshape(env.shape))
# print("")