import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
  
import cPickle as pickle

class QMap (object):
    def __init__ (self):
        # QMap - Keeps the map of (state,action) to q value
        self.qmap = {} 
        
    def getQValue (self, state, action):
        ''' Returns the Q value for given action in state'''
        if (state,action) not in self.qmap.keys(): # First Access
            return 0.0 
        else:
            return self.qmap[(state,action)]  
        
    def setQValue (self, state, action, value):
        ''' Seets the Q value for given action in state '''
        self.qmap[(state, action)] = value
    
    def getMaxQVal (self, state):
        ''' Returns max Q value for a state for all actions'''
        actions = [None, 'forward', 'left', 'right']
        return max([self.getQValue(state,action) for action in actions])

    def getMaxQAction (self, state):
        ''' Returns action with max Q value for a state'''
        actions = [None, 'forward', 'left', 'right']
        maxVal = self.getMaxQVal(state)
        maxActions = [action for action in actions if self.getQValue(state,action) == maxVal] # Find all pairs in case of draw
        return random.choice(maxActions)

class Stats(object):
    def __init__(self):
        self.stats = [] # Keeps all stats for each episode
        self.cntNegreward = 0    # Number of negative rewards (bad driving) in this trial
        self.lastDeadLineVal = 0 # Value of the deadline last time the agent was run. 1 means agent did not meet deadline

    def reset(self):
        self.stats.append((self.lastDeadLineVal, self.cntNegreward))
        self.cntNegreward = 0
        self.lastDeadLineVal = 0

    #def dump(self, fileName):
    #    with open(fileName,'wb') as f:
    #        pickle.dump(stats, f)


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.qMap = QMap()
        self.alpha = 0.1
        self.gamma = 0.9
    
    def set_stats(self, stats):
        self.stats = stats

    def set_alpha_gamma(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
    
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.stats.reset()

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (self.next_waypoint,inputs['light'], inputs['oncoming'])
        
        # TODO: Select action according to your policy
        #action = random.choice([None, 'forward', 'left', 'right'])
        #action = self.next_waypoint
        action = self.qMap.getMaxQAction(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # Update stats
        self.stats.lastDeadLineVal  = deadline
        if reward == -1:
            self.stats.cntNegreward += 1

        # TODO: Learn policy based on state, action, reward
        nextInputs = self.env.sense(self)
        nextState = (self.next_waypoint, nextInputs['light'], nextInputs['oncoming']) 
        qValNew = (1 - self.alpha) * self.qMap.getQValue(self.state, action) +\
                        self.alpha * (reward + self.gamma * self.qMap.getMaxQVal(nextState))
        self.qMap.setQValue(self.state, action, qValNew)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, nextInput = {} reward = {}, nextwaypoint = {}".format(deadline, inputs, action, nextState, reward, self.next_waypoint)  # [debug]
        #print self.qMap.qmap

def run():
    """Run the agent for a finite number of trials."""

    allStats = {}
    for i in range(0,11):
        for j in range(0,11):
            alpha = i / 10.0
            gamma = j / 10.0
    
            # Set up environment and agent
            stat = Stats()
            e = Environment()  # create environment (also adds some dummy traffic)
            a = e.create_agent(LearningAgent)  # create agent
            e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
            # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

            a.set_stats(stat)
            a.set_alpha_gamma(alpha, gamma)
  
            # Now simulate it
            sim = Simulator(e, update_delay=0.1, display=False)  # create simulator (uses pygame when display=True, if available)
            # NOTE: To speed up simulation, reduce update_delay and/or set display=False

            sim.run(n_trials=100)  # run for a specified number of trials
            # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
          
            allStats[(i,j)] = stat

    with open('results.pickle','wb') as f:
        pickle.dump(allStats, f)
  
if __name__ == '__main__':
    run()
