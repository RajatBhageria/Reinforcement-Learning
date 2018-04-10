import gym
import numpy as np
from maze import *
from matplotlib import pyplot as plt
from evaluationAcrobat import *

env = gym.make("Acrobot-v1")
observation = env.reset()

#get the actionsSpace
actionSpace = env.action_space
numActions = 3
actions = np.linspace(0,numActions)

#get the observations space
observationSpace = env.observation_space
obsSpaceLowerbound = observationSpace.low
obsSpaceUpperbound = observationSpace.high

#descritize the observation space after doing linear function approximation
numFeatures = 6
discount = 0.9
alpha = 0.3

eval_steps, eval_reward = [], []

#do the policy gradient
def REINFORCEAcrobat():
    # for function approxomation
    theta = np.random.random(size=(numFeatures,numActions))

    #baseline
    b = np.sum(2,)

    for i_episode in range(10000):
        #collect a set of trajectories by executing current policy
        time = 100
        trajectories = np.zeros((3,time)) #(state,action,reward)

        #collect a set the average of the observations
        scores = np.zeros((numFeatures,time))

        for t in range(time-1):
            observation = env.reset().reshape(6, 1)  # this is phi

            # find the value of phi*theta
            phiTheta = np.multiply(observation, theta)

            # take exponentials for softmax
            phiThetaExp = np.exp(phiTheta)

            # find the probailities of each column
            probs = np.mean(phiThetaExp, axis=0) / np.mean(phiThetaExp)

            # find the appropriate action
            action = int(np.argmax(probs))

            #take a step
            observationPrime, reward, done, info = env.step(action)

            #fill im trajectories
            trajectories[1,t] = action
            trajectories[2,t] = reward

            #fill in the scores
            scores[:,t] = (observation - observation/3).reshape(6,)

            observation = observationPrime

            env.render()

        #find gradient of J(theta)
        for t in range(time):
            #find Gt
            Gt = 0
            for tRest in range(t,time-1):
                Gt = Gt + discount**tRest + trajectories[2,tRest]

            #get the advantage
            At = Gt - b

            #refit the baseline
            b = np.linalg.norm(b - Gt)

            #correct action taken
            action = int(trajectories[1,t])

            #calculate g-hat, the gradient of logPi
            gHat = scores[:,t]

            #reconfigure thetas
            theta[:,action] = theta[:,action] + alpha * discount * At * gHat

#do the qLearning for the Acrobat
def qLearningAcrobat():
    numBinsObs = 10
    obsBinsFeat1 = np.linspace(obsSpaceLowerbound[0],obsSpaceUpperbound[0],numBinsObs)
    obsBinsFeat2 = np.linspace(obsSpaceLowerbound[1],obsSpaceUpperbound[1],numBinsObs)
    obsBinsFeat3 = np.linspace(obsSpaceLowerbound[2],obsSpaceUpperbound[2],numBinsObs)
    obsBinsFeat4 = np.linspace(obsSpaceLowerbound[3],obsSpaceUpperbound[3],numBinsObs)
    obsBinsFeat5 = np.linspace(obsSpaceLowerbound[4],obsSpaceUpperbound[4],numBinsObs)
    obsBinsFeat6 = np.linspace(obsSpaceLowerbound[5],obsSpaceUpperbound[5],numBinsObs)

    qVals = np.random.choice(a = np.linspace(0,5,10), size=(numBinsObs,numBinsObs,numBinsObs,numBinsObs,numBinsObs,numBinsObs,numActions))

    learningRate = .20

    numIter = 100

    for i_episode in range(numIter):
        observation = env.reset()
        for t in range(100):
            env.render()
            #get the descritized states
            feat1Des = np.digitize(observation[0], obsBinsFeat1)
            feat2Des = np.digitize(observation[1], obsBinsFeat2)
            feat3Des = np.digitize(observation[2], obsBinsFeat3)
            feat4Des = np.digitize(observation[3], obsBinsFeat4)
            feat5Des = np.digitize(observation[4], obsBinsFeat5)
            feat6Des = np.digitize(observation[5], obsBinsFeat6)

            # e-greedy probaility
            e = 1.0 - t / (100)

            # randomly pick action based on epsilon
            if np.random.rand() < e:
                action = np.random.choice(a = ([0,1,2]))
            else:
                # get the action derived from q for current state
                possibleActions = qVals[feat1Des,feat2Des,feat3Des,feat4Des,feat5Des,feat6Des, :]
                action = np.argmax(possibleActions)

            # take that action and step
            observationPrime, reward, done, info = env.step(action)

            # get the current q value
            currQVal = qVals[feat1Des,feat2Des,feat3Des,feat4Des,feat5Des,feat6Des,action]

            #descretize the observation primes
            feat1DesPrime = np.digitize(observationPrime[0], obsBinsFeat1)
            feat2DesPrime = np.digitize(observationPrime[1], obsBinsFeat2)
            feat3DesPrime = np.digitize(observationPrime[2], obsBinsFeat3)
            feat4DesPrime = np.digitize(observationPrime[3], obsBinsFeat4)
            feat5DesPrime = np.digitize(observationPrime[4], obsBinsFeat5)
            feat6DesPrime = np.digitize(observationPrime[5], obsBinsFeat6)

            # get the possible qvalues for s'
            possibleNewStates = qVals[feat1DesPrime,feat2DesPrime,feat3DesPrime,feat4DesPrime,feat5DesPrime,feat6DesPrime,:]

            # get the best action based on the next state s'
            actionPrime = np.argmax(possibleNewStates)

            # get the q value of that state
            maxQSPrime = qVals[feat1DesPrime,feat2DesPrime,feat3DesPrime,feat4DesPrime,feat5DesPrime,feat6DesPrime,actionPrime]

            # update the q value table
            qVals[feat1DesPrime, feat2DesPrime, feat3DesPrime, feat4DesPrime, feat5DesPrime, feat6DesPrime, action] \
                = currQVal + learningRate * (reward + discount * maxQSPrime - currQVal)

            # set the current state equal to the next state
            observation = observationPrime

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        #evaluation
        if (i_episode % 50 == 0):
            avg_step, avg_reward = evaluationAcrobat(env, qVals,50,numIter)
            eval_steps.append(avg_step)
            eval_reward.append(avg_reward)

    f2, ax2 = plt.subplots()
    # repeat for different algs
    ax2.plot(range(0, numIter, 50),eval_steps)
    f3, ax3 = plt.subplots()
    # repeat for different algs
    ax3.plot(range(0,numIter,50),eval_reward)
    plt.show()

if __name__ == "__main__":
    #REINFORCEAcrobat()

    qLearningAcrobat()