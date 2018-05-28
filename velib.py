import numpy as np

## intensity of department
Lam = np.array([2.8, 3.7, 5.5, 3.5, 4.6], dtype=float)

## probability of heading j-th station from i-th one
Prob = np.array([
    [0, 0.2, 0.3, 0.2, 0.3], 
    [0.2, 0, 0.3, 0.2, 0.3],
    [0.2, 0.25, 0, 0.25, 0.3],
    [0.15, 0.2, 0.3, 0, 0.35],
    [0.2, 0.25, 0.35, 0.2, 0]
], dtype=float)

## intensity of arrival toward stations
Mu = np.array([
    [0, 3, 5, 7, 7],
    [2, 0, 2, 5, 5],
    [4, 2, 0, 3, 3],
    [8, 6, 4, 0, 2],
    [7, 7, 5, 2, 0]
], dtype=float)
Mu = np.divide(Mu, 60.0)
Mu = np.divide(1, Mu, where=Mu!=0)

## initial state
State = np.array([
    [20, 1, 0, 0, 0],
    [1, 15, 1, 0, 0],
    [0, 1, 17, 1, 0],
    [0, 0, 1, 13, 1],
    [0, 0, 0, 1, 18]
])

## intensity of staying & rate of transition
def transRate(state, lam=Lam, mu=Mu, p=Prob):
    '''
    parameters:
        state: current state of the system --type matrix
        lam: intensity of department of stations --type vector
        mu: intensity of arrival toward stations --type matrix
        p: probability of heading j-th station from i-th one --type matrix
    return:
        stayLam: intensity of staying at current state --type float
        rate: rate of transition --type matrix
    '''
    lamExt = np.transpose(np.ones((5, 5)) * Lam)
    notEmpty = [np.sign(state[i][i]) for i in range(5)]
    depart = np.multiply(np.multiply(lamExt, notEmpty), p)
    arrive = np.multiply(state, mu)
    rate = np.concatenate((depart, arrive), axis=1)
    stayLam = np.sum(rate)

    return stayLam, rate

## lasting time of the state ~ EXP
def last(lam):
    return np.random.exponential(1/lam)

## next state
def nextState(rate, state):
    '''
    parameters:
        rate: rate of transition --type matrix
        state: current state of the system --type matrix
    return:
        nextS: next state of the system -- type matrix
    '''
    choice = np.random.choice(np.arange(50), p=rate.flatten()/np.sum(rate))
    i, j = choice / 10, choice % 10
    action = np.zeros((5, 5))
    if j <= 4:
        action[i][i], action[i][j] = -1, 1
    else:
        action[i][j-5], action[j-5][j-5] = -1, 1    
#     print("rate")
#     print(rate)
#     print("action (%d %d, %d)" % (choice, i, j))
    nextS = np.add(state, action)

    return nextS 
    
## simulation
def simulate(init=State, lam=Lam, mu=Mu, p=Prob, t=1.0):
    '''
    parameters:
        init: initial state of the system --type matrix
        lam: intensity of department of stations --type vector
        mu: intensity of arrival toward stations --type matrix
        p: probability of heading j-th station from i-th one --type matrix
        t: lasting time(hour) of this simulation --type float
    return:
        states: list of time and corresponding state of the system --type list
    '''
    time = 0.0
    states = [[time, init]]
    while time <= t:
#         print("time %f" % time)
#         print(states[-1][1])
        stayLam, rate = transRate(state=states[-1][1], lam=Lam, mu=Mu, p=Prob)
        ## lasting time of the current state
        lastTime = last(lam=stayLam)
        time = time + lastTime
        ## time updated & state to be changed
        state = nextState(rate, states[-1][1])
        states.append([time, state])        
#         print("state")
#         print(state)
#         print("***************************")    
    states[-1][0] = t
    
    return states    

## start simulation during T hour(s)
T = 1.0
print("Simulation during %.2f hour(s)." % T)
states = simulate(t=T)
for state in states:    
    print("time = %f" % state[0])
    print("state")
    print(state[1])
    print("-------------------------")