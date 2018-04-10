from maze import *
import numpy as np
from matplotlib import pyplot as plt
from evaluation import *
from value_plot import value_plot

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

    qVals = np.random.choice(a = np.linspace(0,2.8,1000), size=(numStates,numActions))
    qVals[[12,24,36,48,60,72,84,96,108]] = 0

    currState = initial_state

    learningRate = .05

    #init for evaluation
    eval_steps, eval_reward = [], []

    #number of iterations
    for i in range(0,numIter):
        done = False
        # numVisits for the epsilon function
        numVisits = np.zeros((numStates,))

        #episodes
        while (~done):
            # e-greedy probaility
            e = 100/(100 + numVisits[currState])
            #increase numVisits
            numVisits[currState] = numVisits[currState] + 1

            #randomly pick action based on epsilon
            if np.random.rand() <= e:
                action = np.random.choice(a = ([0,1,2,3]))
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
            qVals[currState,action] = currQVal + learningRate*(reward + discount * maxQSPrime - currQVal)

            #set the current state equal to the next state
            currState = next_state

            #RMSE
            error = np.linalg.norm(np.sqrt(np.abs(np.square(qVals) - np.square(QStar))))
            RMSErrors[i] = error

            if done:
                break

        #evaluation
        if (i % 50 == 0):
            avg_step, avg_reward = evaluation(env, qVals)
            eval_steps.append(avg_step)
            eval_reward.append(avg_reward)

    print qVals

    # #plot the RMSE and evaluaton
    f1, ax1 = plt.subplots()
    ax1.plot(RMSErrors)
    f1.suptitle('RMSE Values')
    f2, ax2 = plt.subplots()
    # repeat for different algs
    ax2.plot(range(0, numIter, 50),eval_steps)
    f2.suptitle('Evaluation Steps')
    f3, ax3 = plt.subplots()
    # repeat for different algs
    ax3.plot(range(0,numIter,50),eval_reward)
    f2.suptitle('Evaluation Reward')
    plt.show()

    value_plot(qVals,env,True,True)


if __name__ == "__main__":
    qLearning()