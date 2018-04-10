# University of Pennsylvaina
# ESE650 Fall 2018
# Heejin Chloe Jeong

import numpy as np

def evaluationCar(env, Q_table, step_bound = 50, num_itr = 1000):
    """
    Semi-greedy evaluation for discrete state and discrete action spaces and an episodic environment.

    Input:
        env : an environment object.
        Q : A numpy array. Q values for all state and action pairs.
            Q.shape = (the number of states, the number of actions)
        step_bound : the maximum number of steps for each iteration
        num_itr : the number of iterations

    Output:
        Total number of steps taken to finish an episode (averaged over num_itr trials)
        Cumulative reward in an episode (averaged over num_itr trials)

    """
    # descritize the action space
    actionSpace = env.action_space
    actionSpaceLowerBound = actionSpace.low
    actionSpaceUpperBound = actionSpace.high

    numBinsActions = 100
    actionBins = np.linspace(actionSpaceLowerBound, actionSpaceUpperBound, numBinsActions)

    #discretize the observation space
    observationSpace = env.observation_space
    obsSpaceLowerbound = observationSpace.low
    obsSpaceUpperbound = observationSpace.high

    # descritize the observation space
    numBinsObs = 100
    obsBinsPos = np.linspace(obsSpaceLowerbound[0], obsSpaceUpperbound[0], numBinsObs)
    obsBinsVel = np.linspace(obsSpaceLowerbound[1], obsSpaceUpperbound[1], numBinsObs)

    total_step = 0
    total_reward = 0
    itr = 0
    while(itr<num_itr):
        step = 0
        np.random.seed()
        state = env.reset()
        pos = state[0]
        vel = state[1]
        posDes = np.digitize(pos, obsBinsPos)
        velDes = np.digitize(vel, obsBinsVel)
        reward = 0.0
        done = False
        while((not done) and (step < step_bound)):
            if np.random.rand() < 0.05:
                action = np.random.uniform(actionSpaceLowerBound, actionSpaceUpperBound)
            else:
                # get the action derived from q for current state
                actionDes = np.argmax(Q_table[posDes,velDes])
                action = np.array(actionBins[actionDes]).reshape((1,))

            state_n, r, done, info = env.step(action)
            state = state_n
            reward += r
            step +=1
        total_reward += reward
        total_step += step
        itr += 1
    return total_step/float(num_itr), total_reward/float(num_itr)
