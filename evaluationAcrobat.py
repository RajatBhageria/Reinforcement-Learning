# University of Pennsylvaina
# ESE650 Fall 2018
# Heejin Chloe Jeong

import numpy as np

def evaluationAcrobat(env, Q_table, step_bound = 50, num_itr = 1000):
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

    #discretize the observation space
    observationSpace = env.observation_space
    obsSpaceLowerbound = observationSpace.low
    obsSpaceUpperbound = observationSpace.high
    numBinsObs = 10
    obsBinsFeat1 = np.linspace(obsSpaceLowerbound[0],obsSpaceUpperbound[0],numBinsObs)
    obsBinsFeat2 = np.linspace(obsSpaceLowerbound[1],obsSpaceUpperbound[1],numBinsObs)
    obsBinsFeat3 = np.linspace(obsSpaceLowerbound[2],obsSpaceUpperbound[2],numBinsObs)
    obsBinsFeat4 = np.linspace(obsSpaceLowerbound[3],obsSpaceUpperbound[3],numBinsObs)
    obsBinsFeat5 = np.linspace(obsSpaceLowerbound[4],obsSpaceUpperbound[4],numBinsObs)
    obsBinsFeat6 = np.linspace(obsSpaceLowerbound[5],obsSpaceUpperbound[5],numBinsObs)

    total_step = 0
    total_reward = 0
    itr = 0
    while(itr<num_itr):
        step = 0
        np.random.seed()
        state = env.reset()
        feat1Des = np.digitize(state[0], obsBinsFeat1)
        feat2Des = np.digitize(state[1], obsBinsFeat2)
        feat3Des = np.digitize(state[2], obsBinsFeat3)
        feat4Des = np.digitize(state[3], obsBinsFeat4)
        feat5Des = np.digitize(state[4], obsBinsFeat5)
        feat6Des = np.digitize(state[5], obsBinsFeat6)
        reward = 0.0
        done = False
        while((not done) and (step < step_bound)):
            if np.random.rand() < 0.05:
                action = np.random.choice(a = ([0,1,2]))
            else:
                # get the action derived from q for current state
                action = np.argmax(Q_table[feat1Des,feat2Des,feat3Des,feat4Des,feat5Des,feat6Des,:])
            state_n, r, done, info = env.step(action)
            state = state_n
            reward += r
            step +=1
        total_reward += reward
        total_step += step
        itr += 1
    return total_step/float(num_itr), total_reward/float(num_itr)
