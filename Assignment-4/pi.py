# Author: Saad Waraich
#GTID: 903459227

import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import mdptoolbox, mdptoolbox.example
from gym.envs.toy_text.frozen_lake import generate_random_map


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI environment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
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
    
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    Vs = []
    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)
        
        # Will be set to false if we make any changes to the policy
        policy_stable = True
        
        # For each state...
        for s in range(env.nS):
            # The best action we would take under the current policy
            chosen_a = np.argmax(policy[s])
            
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)
            
            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
        Vs.append(V)
        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, Vs

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
        print("Running PI Forest states = " + str(s))
        envi = forestEnv(s)
        startTime=time.time()
        policy, Vs = policy_improvement(envi, discount_factor=0.999)
        timeTaken=time.time() - startTime
        timeToRun.append(timeTaken)
        totalIters.append(len(Vs))

    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Convergence Runs VS No. of States, Forest - PI"
    plt.title(titleStr)
    plt.xlabel(" No. of States")
    plt.ylabel("Convergence Runs")
    plt.plot(range(5,15,1), totalIters)
    # plt.plot(lengths, fitnessValMeanSA, '-o', label="simulated annealing")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-PI-ConvergenceVSStates" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Time to Run VS No. of States, Forest - PI"
    plt.title(titleStr)
    plt.xlabel(" No. of States")
    plt.ylabel("Time to Run")
    plt.plot(range(5,15,1), timeToRun)
    # plt.plot(lengths, fitnessValMeanSA, '-o', label="simulated annealing")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Forest-PI-TimesVSStates" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    totalIters = []
    allV = []
    envi = forestEnv(15)
    for gamma in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        print("Running PI Forest gamma = " + str(gamma))
        policy, Vs = policy_improvement(envi, discount_factor=gamma)
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
    titleStr="Convergence Runs VS Gamma Values, Forest - PI"
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
    saveFigPath="figures/" + "Forest-PI-ConvergenceVSGamma" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Runs VS Function Values, Forest - PI"
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
    saveFigPath="figures/" + "Forest-PI-ValueFunctionVSGamma" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

def lake():

    timeToRun = []
    totalIters = []
    for s in range(20,35,1):
        print("Running PI Forest states = " + str(s))
        envi = gym.make("FrozenLake-v0", desc=generate_random_map(size=s, p=0.8), is_slippery = False)
        startTime=time.time()
        policy, Vs = policy_improvement(envi, discount_factor=0.999)
        timeTaken=time.time() - startTime
        timeToRun.append(timeTaken)
        totalIters.append(len(Vs))

    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Convergence Runs VS No. of States, Lake - PI"
    plt.title(titleStr)
    plt.xlabel(" No. of States")
    plt.ylabel("Convergence Runs")
    plt.plot(range(20,35,1), totalIters)
    # plt.plot(lengths, fitnessValMeanSA, '-o', label="simulated annealing")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-PI-ConvergenceVSStates" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Time to Run VS No. of States, Lake - PI"
    plt.title(titleStr)
    plt.xlabel(" No. of States")
    plt.ylabel("Time to Run")
    plt.plot(range(20,35,1), timeToRun)
    # plt.plot(lengths, fitnessValMeanSA, '-o', label="simulated annealing")
    plt.legend(loc='best')
    saveFigPath="figures/" + "Lake-PI-TimesVSStates" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    totalIters = []
    allV = []
    envi = gym.make("FrozenLake-v0", desc=generate_random_map(size=30, p=0.8), is_slippery = False)
    for gamma in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        print("Running PI Lake gamma = " + str(gamma))
        policy, Vs = policy_improvement(envi, discount_factor=gamma)
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
    titleStr="Convergence Runs VS Gamma Values, Lake - PI"
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
    saveFigPath="figures/" + "Lake-PI-ConvergenceVSGamma" + ".png"
    plt.savefig(saveFigPath)
    plt.clf()
    plt.cla()
    plt.close()

    plt.style.use('seaborn-poster')
    plt.style.use('ggplot')
    titleStr="Runs VS Function Values, Lake - PI"
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
    saveFigPath="figures/" + "Lake-PI-ValueFunctionVSGamma" + ".png"
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