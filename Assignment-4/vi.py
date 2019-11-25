# Author: Saad Waraich
#GTID: 903459227
# https://github.com/dennybritz/reinforcement-learning
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import mdptoolbox, mdptoolbox.example
from gym.envs.toy_text.frozen_lake import generate_random_map

def value_iteration(env, theta=0.0001, discount_factor=1.0):

    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    V = np.zeros(env.nS)
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10. 
            V[s] = best_action_value        
        # Check if we can stop 
        if delta < theta:
            break
    
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0
    
    return policy, V

def forest():
    class fEnv:
        def __init__(self, states, actions):
            self.nS = states
            self.nA = actions
            self.P = [[[] for x in range(actions)] for y in range(states)]


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
    totalIters = []
    for s in range(5,15,1):
        print("Running VI Forest states = " + str(s))
        envi = forestEnv(s)
        startTime=time.time()
        policy, Vs = value_iteration(envi, discount_factor=0.999)
        timeTaken=time.time() - startTime
        timeToRun.append(timeTaken)
        totalIters.append(len(Vs))

    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Convergence Runs VS No. of States, Forest - VI"
    plt.title(titleStr)
    plt.xlabel(" No. of States")
    plt.ylabel("Convergence Runs")
    plt.plot(range(5,15,1), totalIters)
    # plt.plot(lengths, fitnessValMeanSA, '-o', label="simulated annealing")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-VI-ConvergenceVSStates" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Time to Run VS No. of States, Forest - VI"
    plt.title(titleStr)
    plt.xlabel(" No. of States")
    plt.ylabel("Time to Run")
    plt.plot(range(5,15,1), timeToRun)
    # plt.plot(lengths, fitnessValMeanSA, '-o', label="simulated annealing")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-VI-TimesVSStates" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    totalIters = []
    allV = []
    envi = forestEnv(15)
    for gamma in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        print("Running VI Forest gamma = " + str(gamma))
        policy, Vs = value_iteration(envi, discount_factor=gamma)
        totalIters.append(len(Vs))
        allV.append(Vs)

    totalV = []
    for v in allV:
        tmp = []
        for v1 in v:
            tmp.append(np.sum(v1))
        totalV.append(tmp)
    print(totalV)

    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Convergence Runs VS Gamma Values, Forest - VI"
    plt.title(titleStr)
    plt.xlabel("Gamma Values")
    plt.ylabel("Convergence Runs")
    # plt.plot(range(5,15,1), totalIters)
    plt.plot([0.6], totalIters[0], 'o', label="gamma = 0.6")
    plt.plot([0.7], totalIters[1], 'o', label="gamma = 0.7")
    plt.plot([0.8], totalIters[2], 'o', label="gamma = 0.8")
    plt.plot([0.9], totalIters[3], 'o', label="gamma = 0.9")
    plt.plot([0.95], totalIters[4], 'o', label="gamma = 0.95")
    plt.plot([0.99], totalIters[5], 'o', label="gamma = 0.99")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-VI-ConvergenceVSGamma" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Runs VS Function Values, Forest - VI"
    plt.title(titleStr)
    plt.xlabel("Runs")
    plt.ylabel("Value Function")
    # plt.plot(range(5,15,1), totalIters)
    plt.plot(range(1,len(totalV[0])+1), totalV[0], label="gamma = 0.6")
    plt.plot(range(1,len(totalV[1])+1), totalV[1], label="gamma = 0.7")
    plt.plot(range(1,len(totalV[2])+1), totalV[2], label="gamma = 0.8")
    plt.plot(range(1,len(totalV[3])+1), totalV[3], label="gamma = 0.9")
    plt.plot(range(1,len(totalV[4])+1), totalV[4], label="gamma = 0.95")
    plt.plot(range(1,len(totalV[5])+1), totalV[5], label="gamma = 0.99")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-VI-ValueFunctionVSGamma" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

def lake():

    timeToRun = []
    totalIters = []
    for s in range(20,35,1):
        print("Running VI Forest states = " + str(s))
        envi = gym.make("FrozenLake-v0", desc=generate_random_map(size=s, p=0.8), is_slippery = False)
        startTime=time.time()
        policy, Vs = value_iteration(envi, discount_factor=0.999)
        timeTaken=time.time() - startTime
        timeToRun.append(timeTaken)
        totalIters.append(len(Vs))

    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Convergence Runs VS No. of States, Lake - VI"
    plt.title(titleStr)
    plt.xlabel(" No. of States")
    plt.ylabel("Convergence Runs")
    plt.plot(range(20,35,1), totalIters)
    # plt.plot(lengths, fitnessValMeanSA, '-o', label="simulated annealing")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-VI-ConvergenceVSStates" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Time to Run VS No. of States, Lake - VI"
    plt.title(titleStr)
    plt.xlabel(" No. of States")
    plt.ylabel("Time to Run")
    plt.plot(range(20,35,1), timeToRun)
    # plt.plot(lengths, fitnessValMeanSA, '-o', label="simulated annealing")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-VI-TimesVSStates" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    totalIters = []
    allV = []
    envi = gym.make("FrozenLake-v0", desc=generate_random_map(size=30, p=0.8), is_slippery = False)
    for gamma in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        print("Running VI Lake gamma = " + str(gamma))
        policy, Vs = value_iteration(envi, discount_factor=gamma)
        totalIters.append(len(Vs))
        allV.append(Vs)

    totalV = []
    for v in allV:
        tmp = []
        for v1 in v:
            tmp.append(np.sum(v1))
        totalV.append(tmp)
    print(totalV)

    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Convergence Runs VS Gamma Values, Lake - VI"
    plt.title(titleStr)
    plt.xlabel("Gamma Values")
    plt.ylabel("Convergence Runs")
    # plt.plot(range(5,15,1), totalIters)
    plt.plot([0.6], totalIters[0], 'o', label="gamma = 0.6")
    plt.plot([0.7], totalIters[1], 'o', label="gamma = 0.7")
    plt.plot([0.8], totalIters[2], 'o', label="gamma = 0.8")
    plt.plot([0.9], totalIters[3], 'o', label="gamma = 0.9")
    plt.plot([0.95], totalIters[4], 'o', label="gamma = 0.95")
    plt.plot([0.99], totalIters[5], 'o', label="gamma = 0.99")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-VI-ConvergenceVSGamma" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Runs VS Function Values, Lake - VI"
    plt.title(titleStr)
    plt.xlabel("Runs")
    plt.ylabel("Value Function")
    # plt.plot(range(5,15,1), totalIters)
    plt.plot(range(1,len(totalV[0])+1), totalV[0], label="gamma = 0.6")
    plt.plot(range(1,len(totalV[1])+1), totalV[1], label="gamma = 0.7")
    plt.plot(range(1,len(totalV[2])+1), totalV[2], label="gamma = 0.8")
    plt.plot(range(1,len(totalV[3])+1), totalV[3], label="gamma = 0.9")
    plt.plot(range(1,len(totalV[4])+1), totalV[4], label="gamma = 0.95")
    plt.plot(range(1,len(totalV[5])+1), totalV[5], label="gamma = 0.99")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-VI-ValueFunctionVSGamma" + ".png"
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