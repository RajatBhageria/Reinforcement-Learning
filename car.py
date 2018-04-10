import gym
import numpy as np
from maze import *
from matplotlib import pyplot as plt
from evaluation import *

env = gym.make("MountainCarContinuous-v0")
observation = env.reset()
env._max_episode_seconds = 500
env._max_episode_steps = 1000

#get the actionsSpace
#[position]
actionSpace = env.action_space
actionSpaceLowerBound = actionSpace.low
actionSpaceUpperBound = actionSpace.high

#descritize the action space
numBinsActions = 100
actionBins = np.linspace(actionSpaceLowerBound,actionSpaceUpperBound,numBinsActions)

#get the observations space
#[position, velocity]
observationSpace = env.observation_space
obsSpaceLowerbound = observationSpace.low
obsSpaceUpperbound = observationSpace.high

#descritize the observation space
numBinsObs = 100
obsBinsPos = np.linspace(obsSpaceLowerbound[0],obsSpaceUpperbound[0],numBinsObs)
obsBinsVel = np.linspace(obsSpaceLowerbound[1],obsSpaceUpperbound[1],numBinsObs)
discount = 0.9

def qLearningMountain():
    qVals = np.random.choice(a = ([.1,.2,.3,.4,.5,.6,.7,.8,.9]), size=(numBinsObs,numBinsObs,numBinsActions))

    learningRate = .20

    for i_episode in range(1000):
        observation = env.reset()
        for t in range(100):
            env.render()

            #get the descritized states
            pos = observation[0]
            vel = observation[1]
            posDes = np.digitize(pos, obsBinsPos)
            velDes = np.digitize(vel, obsBinsVel)

            # e-greedy probaility
            e = 1.0 - t / (100)

            # randomly pick action based on epsilon
            if np.random.rand() < e:
                action = random.uniform(actionSpaceLowerBound, actionSpaceUpperBound)
            else:
                # get the action derived from q for current state
                possibleActions = qVals[posDes,velDes, :]
                action = np.argmax(possibleActions)

            # take that action and step
            observationPrime, reward, done, info = env.step(action)

            if (observation[0]>.3):
                reward = reward + 50

            #descritize the action
            actionDes = np.digitize(action,actionBins)

            # get the current q value
            currQVal = qVals[posDes,velDes,actionDes]

            #descretize the observation primes
            posPrime = np.digitize(observationPrime[0], obsBinsPos)
            velPrime = np.digitize(observationPrime[1], obsBinsVel)

            # get the possible qvalues for s'
            possibleNewStates = qVals[posPrime,velPrime, :]

            # get the best action based on the next state s'
            actionPrime = np.argmax(possibleNewStates)

            # get the q value of that state
            maxQSPrime = qVals[posPrime,velPrime, actionPrime]

            # update the q value table
            qVals[posDes,velDes,actionDes] = currQVal + learningRate * (reward + discount * maxQSPrime - currQVal)

            # set the current state equal to the next state
            observation = observationPrime

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    print np.argmax(qVals,axis=1)
    print qVals

    # f2, ax2 = plt.subplots()
    # # repeat for different algs
    # ax2.plot(range(0, numIter, 50),eval_steps)
    # f3, ax3 = plt.subplots()
    # # repeat for different algs
    # ax3.plot(range(0,numIter,50),eval_reward)
    # plt.show()

if __name__ == "__main__":
    qLearningMountain()