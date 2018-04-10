import gym
import numpy as np
from maze import *
from matplotlib import pyplot as plt
from evaluation import *

env = gym.make("Acrobot-v1")
observation = env.reset()

#get the actionsSpace
actionSpace = env.action_space
numActions = 2
actions = np.linspace(0,numActions)

#get the observations space
#[position, velocity]
observationSpace = env.observation_space
obsSpaceLowerbound = observationSpace.low
obsSpaceUpperbound = observationSpace.high

#descritize the observation space after doing linear function approximation
numFeatures = 6
numBinsObs = 100
obsBinsFeat1 = np.linspace(obsSpaceLowerbound[0],obsSpaceUpperbound[0],numBinsObs)
obsBinsFeat2 = np.linspace(obsSpaceLowerbound[1],obsSpaceUpperbound[1],numBinsObs)
obsBinsFeat3 = np.linspace(obsSpaceLowerbound[2],obsSpaceUpperbound[2],numBinsObs)
obsBinsFeat4 = np.linspace(obsSpaceLowerbound[3],obsSpaceUpperbound[3],numBinsObs)
obsBinsFeat5 = np.linspace(obsSpaceLowerbound[4],obsSpaceUpperbound[4],numBinsObs)
obsBinsFeat6 = np.linspace(obsSpaceLowerbound[5],obsSpaceUpperbound[5],numBinsObs)

discount = 0.9
alpha = 0.05

#do the policy gradient
def REINFORCE():
    # for function approxomation
    theta = np.array([1, 1, 1, 1, 1, 1])

    featureVector = np.zeros((numFeatures, numActions))

    #baseline
    b = np.sum(2,)

    for i_episode in range(1000):
        observation = env.reset()
        #collect a set of trajectories by executing current policy
        time = 100
        trajectories = np.zeros((3,time)) #(state,action,reward)

        #collect a set the average of the observations
        scores = np.zeros((numFeatures,time))

        for t in range(time-1):
            #get the probabilities of the theta using softmax
            vector = np.multiply(observation, theta)
            vector = np.exp(vector)
            vector = vector / np.sum(vector)

            #find the best action
            action = pi(vector)

            #take a step
            observationPrime, reward, done, info = env.step(action)

            averageScore = np.zeros((numFeatures,))

            #add the obs to the average score
            averageScore = averageScore + observationPrime

            #calcualte the scores
            averageScore = averageScore / np.sum(averageScore)

            #get the next observation
            observationPrime, reward, done, info = env.step(action)

            #fill im trajectories
            trajectories[0,t] = np.inner(observation,theta)
            trajectories[1,t] = action
            trajectories[2,t] = reward

            #fill in the scores
            scores[:,t] = observation - averageScore

            observation = observationPrime

        #find gradient of J(theta)
        gradient = np.zeros((6,1))

        for t in range(time):
            #find gradient log pi
            #find Gt
            Gt = 0
            for tRest in range(t,time-1):
                Gt = Gt + discount**tRest + trajectories[2,tRest]

            #get the advantage
            At = Gt - b

            #refit the baseline
            b = np.linalg.norm(b - Gt)

            #calculate g-hat
            gHat = np.zeros((6,1))

            #calcualte the current gradient for time t
            currGradient = At * gHat

            #add the current gradient to the total gradient
            gradient = gradient + currGradient

        #reconfigure thetas
        theta = theta + alpha * gradient
        print theta


def pi(vector):
    max = np.max(vector)
    if max<2:
        return 0
    elif max >=2 & max < 5:
        return 1
    elif max >= 5:
        return 2


# def qLearningAcrobat():
#     qVals = np.random.choice(a = ([.1,.2,.3,.4,.5,.6,.7,.8,.9]), size=(numBinsObs,2))
#
#     learningRate = .20
#
#     for i_episode in range(1000):
#         observation = env.reset()
#         for t in range(100):
#             env.render()
#
#             #get the descritized states
#             pos = observation[0]
#             vel = observation[1]
#             posDes = np.digitize(pos, obsBinsPos)
#             velDes = np.digitize(vel, obsBinsVel)
#
#             # e-greedy probaility
#             e = 1.0 - t / (100)
#
#             # randomly pick action based on epsilon
#             if np.random.rand() < e:
#                 action = random.uniform(actionSpaceLowerBound, actionSpaceUpperBound)
#             else:
#                 # get the action derived from q for current state
#                 possibleActions = qVals[posDes,velDes, :]
#                 action = np.argmax(possibleActions)
#
#             # take that action and step
#             observationPrime, reward, done, info = env.step(action)
#
#             if (observation[0]>.3):
#                 reward = reward + 50
#
#             #descritize the action
#             actionDes = np.digitize(action,actionBins)
#
#             # get the current q value
#             currQVal = qVals[posDes,velDes,actionDes]
#
#             #descretize the observation primes
#             posPrime = np.digitize(observationPrime[0], obsBinsPos)
#             velPrime = np.digitize(observationPrime[1], obsBinsVel)
#
#             # get the possible qvalues for s'
#             possibleNewStates = qVals[posPrime,velPrime, :]
#
#             # get the best action based on the next state s'
#             actionPrime = np.argmax(possibleNewStates)
#
#             # get the q value of that state
#             maxQSPrime = qVals[posPrime,velPrime, actionPrime]
#
#             # update the q value table
#             qVals[posDes,velDes,actionDes] = currQVal + learningRate * (reward + discount * maxQSPrime - currQVal)
#
#             # set the current state equal to the next state
#             observation = observationPrime
#
#             if done:
#                 print("Episode finished after {} timesteps".format(t + 1))
#                 break
#
#     print np.argmax(qVals,axis=1)
#     print qVals
#
#     # f2, ax2 = plt.subplots()
#     # # repeat for different algs
#     # ax2.plot(range(0, numIter, 50),eval_steps)
#     # f3, ax3 = plt.subplots()
#     # # repeat for different algs
#     # ax3.plot(range(0,numIter,50),eval_reward)
#     # plt.show()

if __name__ == "__main__":
    REINFORCE()