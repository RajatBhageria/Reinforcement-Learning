from maze import *
import numpy as np

numStates = 112
numActions = 4
discount = 0.9

def qLearning():
    env = Maze()
    initial_state = env.reset()

    qVals = np.random.choice(a = ([1,2,3,4,5,6,7,8,9]), size=(numStates,numActions))
    qVals[[12,24,36,48,60,72,84,96,108]] = 0
    numIter = 5000

    currState = initial_state

    learningRate = 0.05

    isDone = False
    i = 0

    while (i < numIter & ~isDone):
        #get the action derived from q for the init_state
        possibleActions = qVals[currState,:]
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
        qVals[currState,action] = currQVal + learningRate*(reward+discount*maxQSPrime-currQVal)

        #set the current state equal to the next state
        currState = next_state

        #update counter for number of iters
        i = i + 1
        isDone = done

    print qVals

if __name__ == "__main__":
    qLearning()