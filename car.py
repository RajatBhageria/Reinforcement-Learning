import gym
import numpy as np
from maze import *
from matplotlib import pyplot as plt
from evaluationCar import *
from evaluationREINFORCE import *

env = gym.make("MountainCarContinuous-v0")
observation = env.reset()
env._max_episode_steps = 2000
discount = 0.9

# get the actionsSpace
# [position]
actionSpace = env.action_space
actionSpaceLowerBound = actionSpace.low
actionSpaceUpperBound = actionSpace.high

# descritize the action space
numBinsActions = 100
actionBins = np.linspace(actionSpaceLowerBound, actionSpaceUpperBound, numBinsActions)

# get the observations space
# [position, velocity]
observationSpace = env.observation_space
obsSpaceLowerbound = observationSpace.low
obsSpaceUpperbound = observationSpace.high

# descritize the observation space
numBinsObs = 100
obsBinsPos = np.linspace(obsSpaceLowerbound[0], obsSpaceUpperbound[0], numBinsObs)
obsBinsVel = np.linspace(obsSpaceLowerbound[1], obsSpaceUpperbound[1], numBinsObs)

#do the policy gradient
def REINFORCECar():
    learningRate = .9

    eval_steps, eval_reward = [], []

    # for function approxomation
    theta = np.random.random(size=(2,numBinsActions))

    #baseline
    b = np.sum(2,)

    numIter = 1000

    for i_episode in range(numIter):
        #collect a set of trajectories by executing current policy
        time = 2000
        trajectories = np.zeros((3,time)) #(state,action,reward)

        #collect a set the average of the observations
        scores = np.zeros((2,time))

        observation = env.reset()

        for t in range(time-1):
            # find the value of phi*theta
            phiTheta = np.dot(observation, theta)

            # take exponentials for softmax
            phiThetaExp = np.exp(phiTheta)

            # find the probailities of each column
            probs = np.mean(phiThetaExp, axis=0) / np.mean(phiThetaExp)

            # e-greedy probaility
            e = 1.0 - t / (100)

            # randomly pick action based on epsilon
            if np.random.rand() < e:
                action = random.uniform(actionSpaceLowerBound, actionSpaceUpperBound)
            else:
                # find the appropriate action
                actionDes = int(np.argmax(probs))
                action = np.array(actionBins[actionDes]).reshape((1,))

            #take a step
            observationPrime, reward, done, info = env.step(action)

            #fill im trajectories
            trajectories[1,t] = action
            trajectories[2,t] = reward

            #fill in the scores
            scores[:,t] = (observation - observation/3).reshape(2,)

            observation = observationPrime

            env.render()

        # evaluation
        if (i_episode % 50 == 0):
            avg_step, avg_reward = evaluationREINFORCE(env, action)
            eval_steps.append(avg_step)
            eval_reward.append(avg_reward)

        #find gradient of J(theta)
        for t in range(time):
            #find Gt and update bt
            Gt = 0
            for tRest in range(t,time-1):
                Gt = Gt + ((discount**tRest) * trajectories[2,tRest])

            #get b
            b = np.mean(trajectories[2,:])

            #get the advantage
            At = Gt - b

            #correct action taken
            action = int(trajectories[1,t])

            #calculate g-hat, the gradient of logPi
            gHat = scores[:,t]

            #reconfigure thetas
            theta[:,action] = theta[:,action] #+ learningRate * discount * At * gHat

    f2, ax2 = plt.subplots()
    # repeat for different algs
    ax2.plot(range(0, numIter, 50), eval_steps)
    f2.suptitle('Evaluation Steps')
    f3, ax3 = plt.subplots()
    # repeat for different algs
    ax3.plot(range(0, numIter, 50), eval_reward)
    f3.suptitle('Evaluation Reward')
    plt.show()

def qLearningMountain():
    learningRate = .2
    eval_steps, eval_reward = [], []
    qVals = np.random.choice(a = np.linspace(0,100,100), size=(numBinsObs,numBinsObs,numBinsActions))

    numIter = 500

    for i_episode in range(numIter):
        observation = env.reset()
        for t in range(2000):
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
                #discretize the action
                actionDes = np.digitize(action, actionBins)
            else:
                # get the action derived from q for current state
                possibleActions = qVals[posDes,velDes, :]
                actionDes = np.argmax(possibleActions)
                action = np.array(actionBins[actionDes]).reshape((1,))

            # take that action and step
            observationPrime, reward, done, info = env.step(action)

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

        #evaluation
        if (i_episode % 50 == 0):
            avg_step, avg_reward = evaluationCar(env, qVals,50,numIter)
            eval_steps.append(avg_step)
            eval_reward.append(avg_reward)

    f2, ax2 = plt.subplots()
    # repeat for different algs
    ax2.plot(range(0, numIter, 50),eval_steps)
    f2.suptitle('Evaluation Steps')
    f3, ax3 = plt.subplots()
    # repeat for different algs
    ax3.plot(range(0,numIter,50),eval_reward)
    f3.suptitle('Evaluation Reward')
    plt.show()

if __name__ == "__main__":
    REINFORCECar()

    #qLearningMountain()