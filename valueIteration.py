from maze import *
import numpy as np

numStates = 112
numActions = 4
discount = 0.9


def valueIteration():
    env = Maze()
    initial_state = env.reset()

    #get the transition proability matrix
    transitions = np.zeros((numStates,numActions,numStates),dtype='float')

    #get the reward matrix
    rewards = np.zeros((numStates,numActions))

    #create the transition and reward matrices
    numIters = 500
    for i in range(0,numIters):
        for s in range(0,numStates):
            for a in range(0,numActions):
                #use step function to get the next state
                reward, next_state, done = env.step(s, a)
                #put the reward in the next state
                rewards[s,a] = reward
                #add one to the state transition matrix
                transitions[s,a,next_state] = transitions[s,a,next_state] + 1.0

    #standardize the transitions matrix
    for s in range(0,numStates):
        for a in range(0,numActions):
            transitions[s,a,:] = transitions[s,a,:] / np.sum(transitions[s,a,:])

    #initialize the value function states as zeros
    values = np.random.choice(a = ([0,1,2,3,4,5,6,7,8,9]), size=(112,))

    #initialize the policies randomly
    policies = np.random.choice(a = (0,1,2,3), size=(112,))

    #do value iteration
    for i in range (0,5000):
        # #evaluate the policy
        # values = policyEvaluation(transitions,rewards,values,policies)
        # #improve policy
        # policies, policy_stable = policyImprovement(transitions, rewards, values, policies)
        # if policy_stable:
        #     print "stable"
        #     break
        values, policies = valueIter(transitions, rewards, values, policies)

    #best policy found so now save all the Q values
    Qvals = np.zeros((numStates,numActions))
    for s in range(0,numStates):
        for a in range(0,numActions):
            reward = rewards[s,a]

            #get all the possible new states based on s and action (from policy pi)
            possibleNewStates = transitions[s,a,:]

            #get the s+1 state where probability != 0
            indiciesSPrime = np.where(possibleNewStates > 0)[0]
            probSPrime = possibleNewStates[indiciesSPrime]

            #find the new value estimation for state s based on all the possible s+1 states sPrime
            newValue = 0
            for i in range(0,len(indiciesSPrime)):
                #get the state sPrime
                sPrime = indiciesSPrime[i]
                #get the probability of that state sPrime
                prob = probSPrime[i]
                #get the value of sPrime
                valueSPrime = values[sPrime]
                #add the current value to the values for the other sPrimes
                newValue = newValue + prob*(reward + discount*valueSPrime)
            Qvals[s,a] = newValue
    #save the q values
    np.save('QValues',Qvals)

def valueIter(transitions, rewards, values, policies):
    epsilon = 1e-5
    delta = []
    while delta > epsilon:
        delta = 0
        # iterate over all the states
        for s in range(0, numStates):
            # get the current value v
            v = values[s,]
            # get the current action based on policy pi and state s
            action = policies[s,]

            # get all the possible new states based on s and action (from policy pi)
            possibleNewStates = transitions[s, action, :]

            # get the s+1 state where probability != 0
            indiciesSPrime = np.where(possibleNewStates > 0)[0]
            probSPrime = possibleNewStates[indiciesSPrime]

            # get the reward for state s
            reward = rewards[s, action]

            # find the new value estimation for state s based on all the possible s+1 states sPrime
            newValue = 0
            for i in range(0, len(indiciesSPrime)):
                # get the state sPrime
                sPrime = indiciesSPrime[i]
                # get the probability of that state sPrime
                prob = probSPrime[i]
                # get the value of sPrime
                valueSPrime = values[sPrime]
                # add the current value to the values for the other sPrimes
                newValue = newValue + prob * (reward + discount * valueSPrime)
            values[s] = newValue

            # update delta
            delta = max(delta, np.abs(v - values[s]))

    for s in range(0,numStates):
        oldAction = policies[s,]

        maxAVal = 0
        maxA = oldAction

        #iterate through all the possible action:
        for a in range(0,4):
            # get all the possible new states based on s and action (from policy pi)
            possibleNewStates = transitions[s,a,:]

            # get the s+1 state where probability != 0
            indiciesSPrime = np.where(possibleNewStates > 0)[0]
            probSPrime = possibleNewStates[indiciesSPrime]

            # get the reward for state s and action a
            reward = rewards[s, a]

            # find the new value estimation for state s based on all the possible s+1 states sPrime
            newValue = 0
            for i in range(0, len(indiciesSPrime)):
                # get the state sPrime
                sPrime = indiciesSPrime[i]
                # get the probability of that state sPrime
                prob = probSPrime[i]
                # get the value of sPrime
                valueSPrime = values[sPrime]
                # add the current value to the values for the other sPrimes
                newValue = newValue + prob * (reward + discount * valueSPrime)
            if newValue > maxAVal:
                maxAVal = newValue
                maxA = a

        policies[s] = maxA

    return values, policies

# def policyEvaluation(transitions, rewards, values, policies):
#     epsilon = 1e-5
#     delta = []
#     while delta > epsilon:
#         delta = 0
#         #iterate over all the states
#         for s in range(0,numStates):
#             #get the current value v
#             v = values[s,]
#             #get the current action based on policy pi and state s
#             action = policies[s,]
#
#             #get all the possible new states based on s and action (from policy pi)
#             possibleNewStates = transitions[s,action,:]
#
#             #get the s+1 state where probability != 0
#             indiciesSPrime = np.where(possibleNewStates > 0)[0]
#             probSPrime = possibleNewStates[indiciesSPrime]
#
#             #get the reward for state s
#             reward = rewards[s,action]
#
#             #find the new value estimation for state s based on all the possible s+1 states sPrime
#             newValue = 0
#             for i in range(0,len(indiciesSPrime)):
#                 #get the state sPrime
#                 sPrime = indiciesSPrime[i]
#                 #get the probability of that state sPrime
#                 prob = probSPrime[i]
#                 #get the value of sPrime
#                 valueSPrime = values[sPrime]
#                 #add the current value to the values for the other sPrimes
#                 newValue = newValue + prob*(reward + discount*valueSPrime)
#             values[s] = newValue
#
#             #update delta
#             delta = max(delta,np.abs(v-values[s]))
#     return values
#
#
# def policyImprovement(transitions, rewards, values, policies):
#     policy_stable = True
#     for s in range(0,numStates):
#         oldAction = policies[s,]
#
#         maxAVal = 0
#         maxA = oldAction
#
#         #iterate through all the possible action:
#         for a in range(0,4):
#             # get all the possible new states based on s and action (from policy pi)
#             possibleNewStates = transitions[s,a,:]
#
#             # get the s+1 state where probability != 0
#             indiciesSPrime = np.where(possibleNewStates > 0)[0]
#             probSPrime = possibleNewStates[indiciesSPrime]
#
#             # get the reward for state s and action a
#             reward = rewards[s, a]
#
#             # find the new value estimation for state s based on all the possible s+1 states sPrime
#             newValue = 0
#             for i in range(0, len(indiciesSPrime)):
#                 # get the state sPrime
#                 sPrime = indiciesSPrime[i]
#                 # get the probability of that state sPrime
#                 prob = probSPrime[i]
#                 # get the value of sPrime
#                 valueSPrime = values[sPrime]
#                 # add the current value to the values for the other sPrimes
#                 newValue = newValue + prob * (reward + discount * valueSPrime)
#             if newValue > maxAVal:
#                 maxAVal = newValue
#                 maxA = a
#
#         #check if the old action = new action
#         if oldAction != maxA:
#             policy_stable = False
#
#     return policies, policy_stable

if __name__ == "__main__":
    valueIteration()