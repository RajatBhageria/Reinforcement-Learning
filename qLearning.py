from maze import *
import numpy as np
from matplotlib import pyplot as plt
from evaluation import *

numStates = 112
numActions = 4
discount = 0.9


def qLearning():
    env = Maze()
    initial_state = env.reset()

    #number of times to run algorithm
    numIter = 5000

    #get the optimal Q* values for the RMSE
    QStar = np.load('QValues.npy')
    RMSErrors = np.zeros((numIter,))

    qVals = np.random.choice(a = ([.1]), size=(numStates,numActions))
    qVals[[12,24,36,48,60,72,84,96,108]] = 0

    currState = initial_state

    learningRate = 0.10

    isDone = False
    i = 0

    #init for evaluation
    eval_steps, eval_reward = [], []

    while (i < numIter & ~isDone):
        # e-greedy probaility
        e = 1-i/numIter

        #randomly pick action based on epsilon
        if np.random.rand() < e:
            action = np.random.choice(a = ([0,1,2,3]))
            #action = ACTMAP[action]
        else:
            # get the action derived from q for current state
            possibleActions = qVals[currState, :]
            action = np.argmax(possibleActions)

        #take that action and step
        reward, next_state, done = env.step(currState, action)

        #get the current q value
        currQVal = qVals[currState,action]

        #get the possible qvalues for s'
        possibleNewStates = qVals[next_state,:]

        #get the best action based on the next state s'
        actionPrime = np.argmax(possibleNewStates)

        #get the q value of that state
        maxQSPrime = qVals[next_state,actionPrime]

        #update the q value table
        qVals[currState,action] = currQVal + learningRate*(reward + discount*maxQSPrime - currQVal)

        #set the current state equal to the next state
        currState = next_state

        #RMSE
        error = np.linalg.norm(np.sqrt(np.abs(np.square(qVals) - np.square(QStar))))
        RMSErrors[i] = error

        #evaluation
        if (i % 50 == 0):
            avg_step, avg_reward = evaluation(env, qVals)
            eval_steps.append(avg_step)
            eval_reward.append(avg_reward)

        #update counter for number of iters
        i = i + 1
        isDone = done

    print qVals
    #plot the RMSE
    plt.plot(RMSErrors)
    plt.show()

    # # Plot example #
    # f1, ax1 = plt.subplots()
    # # repeat for different algs
    # ax1.plot(range(0, numIter, 50),eval_steps)
    # f2, ax2 = plt.subplots()
    # # repeat for different algs
    # ax2.plot(range(0,numIter,50),eval_reward)
    # plt.show()

if __name__ == "__main__":
    qLearning()