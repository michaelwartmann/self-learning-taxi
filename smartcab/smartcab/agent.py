import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from analysis import test_reporter
import csv
import os

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.state = []
        self.q = {}
        self.alpha = 0.001 # learning rate should be small to avoid oscillation after many trials
        self.epsilon = 0.02
        self.gamma = 0.9
        self.actions = [None, 'forward', 'left', 'right']
        self.action = None
        self.max_q = 0.0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required    
    
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        
        # Update state
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])
        # save previous state for Q table update later        
        previous_state = self.state
        
        #Select action according to hard coded traffic rules and always follow planner (optimal policy)        
        # This code has been used as a benchmarking agent, taken from environment.py
#        action_okay = True
#        if self.next_waypoint == 'right':
#            if inputs['light'] == 'red' and inputs['left'] == 'forward':
#                action_okay = False
#        elif self.next_waypoint == 'forward':
#            if inputs['light'] == 'red':
#                action_okay = False
#        elif self.next_waypoint == 'left':
#            if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
#                action_okay = False
  
#        self.action = None
#        if action_okay:
#            self.action = self.next_waypoint
       
       
       # Select action according to q learning policy (basis of code taken and adjusted from https://github.com/studywolf)
        if random.random() < self.epsilon:
            self.action = random.choice(self.actions)
        else:
            # retrieve the maximum q-value
            self.max_q = max([self.q.get((self.state, a), 0.0) for a in self.actions])
            # find the best actions for the given state
            best_actions = [a for a in self.actions if self.q.get((self.state, a), 0) == self.max_q]
            # randomly pick one of the best actions
            self.action = random.choice(best_actions)
            

        # Execute action and get reward
        reward = self.env.act(self, self.action)

        # Learn policy based on state, action, reward
        # Sense next state        
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)                
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])
        
        # retrieve the maximum q-value for the next state
        self.max_q = max([self.q.get((self.state, a), 0.0) for a in self.actions])        
        
        # Update Q learning table last state and last action       
        self.q[previous_state, self.action] = (1 - self.alpha) * self.q.get((previous_state,self.action),0.0) + self.alpha *( reward + self.gamma * self.max_q) 
        # write q-table into csv file for analysis       
        with open('policy.csv', 'wb') as f:  # Just use 'w' mode in 3.x
            w = csv.DictWriter(f, self.q.keys())
            w.writeheader()
            w.writerow(self.q)
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, self.action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    # define global variable n_trials to be used for learning rate alpha
    global n_trials    
    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    #test_reporter()

if __name__ == '__main__':
    run()
