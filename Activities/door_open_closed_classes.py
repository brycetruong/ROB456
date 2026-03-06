#!/usr/bin/env python3

import numpy as np


# This file has two classes, one for the "world" and one for the "sensors"
#    This may be a bit overkill for this assignment, but we'll be using this general structure over
#. and over again, so might as well become familiar with it now.
#  World class keeps track of the "ground truth" of the world
#      - if the door is open or closed
#  it also handles the robot "opening" and "closing the door"
#.     - probabilities of the robot successfully opening/closing the door
#
#. And sensor queries: Ask the sensor if the door is "open"
# 
#  Sensor class handles generating "True/False" queries. The sensor class has access to the World class
#   so it can know the "ground truth" (if the door is ACTUALLY open or closed)

# GUIDE: The GUIDES are labeled with part 1 through part 3; you don't need to do it all at once
#. Also make sure to look for GUIDES/part in all of the classes


class DoorGroundTruth:
    # Define the actions the robot can take
    #   Defined both as a text string and as a number
    #. You can access this variable as DoorGroundTruth.actions
    actions = {"Open": 0, "Close": 1}

    def __init__(self, door_open_state: bool):
        """Create a door that can be opened and closed
        @param door_open_state - set to True if door is open, otherwise, false"""

        # GUIDE  Part 1: Store door state
        self.door_state = door_open_state

        # GUIDE  Part 3: Store probabilities of door being opened if open action taken, etc
        #           (the transition table)
        #.         - set to uniform probabilities initially
        # We will store: key=(start_state, action), value=probability_of_resulting_state_being_OPEN (True)
        self.transition_probs = {}
        for start_state in [True, False]:
            for action in DoorGroundTruth.actions:
                self.transition_probs[(start_state, action)] = 0.5

    def set_probability(self, door_initial_state : bool, action : str, door_final_state : bool, prob : float):
        """Set the probability that the door will be in the final state if it started in the initial state and took
           the given action
           This is filling in the transition table 
        @param door_initial_state : boolean, if True, door started open
        @param action : one of "Open" or "closed" - action the robot took
        @param door_final_state : boolean, if True the door ends open
        @param prob - probability that the door will end in the given state, given the starting state and the action"""

        # Probability values are always between 0 and 1
        assert 0.0 <= prob <= 1.0

        # Action is one of "Open" or "closed" (checks the dictionary)
        assert action in DoorGroundTruth.actions

        # GUIDE: Part 3: Update your transition table
        # We normalize everything to store: P(Result=Open | Start, Action)
        if door_final_state is True:
            # If the user is setting the prob of ending up Open, store it directly
            self.transition_probs[(door_initial_state, action)] = prob
        else:
            # If the user is setting the prob of ending up Closed, P(Open) = 1.0 - P(Closed)
            self.transition_probs[(door_initial_state, action)] = 1.0 - prob

    def robot_tries_to_open_door(self):
        """ The robot tries (once) to open the door, and succeeds (or fails) based on the probabilities in
        your transition table, the starting state, and the action (Open)
        @ return the new state of the door"""

        # GUIDE:  Part 3: 
        #.  Randomly sample from the random variable for "open the door given the current door state" 
        #.    WHICH random variable depends on whether or not the door is actually open
        # (Possibly) change the state of the door to open and (possibly) change it to closed 
        #.  The latter can happen if there is some probability of the robot accidentally closing
        #.    The door when it is open 

        action = "Open"
        # 1. Get the probability that the door ends up OPEN (True) for the current state + action
        prob_ends_open = self.transition_probs[(self.door_state, action)]
        
        # 2. Sample
        if np.random.uniform() < prob_ends_open:
            self.door_state = True
        else:
            self.door_state = False

        return self.get_door_state()

    def robot_tries_to_close_door(self):
        """ The robot tries (once) to close the door, and succeeds (or fails) based on the probabilities"""

        # GUIDE: Part 3: 
        #.  Same as opening, but this time try closing 
        
        action = "Close"
        # 1. Get the probability that the door ends up OPEN (True) for the current state + action
        prob_ends_open = self.transition_probs[(self.door_state, action)]
        
        # 2. Sample
        if np.random.uniform() < prob_ends_open:
            self.door_state = True
        else:
            self.door_state = False

        return self.get_door_state()

    def get_door_state(self):
        """ GUIDE Part 1: Return the door state as a boolean (Open - True/Closed - False)"""
        return self.door_state

    def __str__(self):
        """ Once you fill in get_door_state, this will print nicely """
        return f"Door state is: {self.get_door_state()}"


class DoorSensor():
    def __init__(self):
        # GUIDE: Part 2: Initialize probabilities to uniform probabilities
        #. I.e., whether or not the door is open or closed, 0.5 probability of saying door is open (or closed)
        #.  The methods will be used to set the probabilities to something other than uniform
        
        # We need two variables:
        # 1. P(return True | Door is Open)
        self.prob_true_if_open = 0.5
        # 2. P(return False | Door is Closed)
        self.prob_false_if_closed = 0.5

    def set_return_true_if_open_probability(self, prob: float):
        """ Set the probability of the sensor returning True if the door is open
        @param prob - the probability value (between 0 and 1) """

        # Probability values are always between 0 and 1
        assert 0.0 <= prob <= 1.0

        # GUIDE: Part 2: Set the random variable to the probability value
        self.prob_true_if_open = prob

    def set_return_false_if_closed_probability(self, prob: float):
        """ Set the probability of the sensor returning False if the door is closed
        @param prob - the probability value (between 0 and 1) """

        # Probability values are always between 0 and 1
        assert 0.0 <= prob <= 1.0

        # GUIDE: Set the random variable to the probability value
        self.prob_false_if_closed = prob
        

    def sample_sensor(self, door_ground_truth : DoorGroundTruth):
        """ Sample the sensor
        @param door_ground_truth - has if the door is actually open or not
        @return True or False"""

        # GUIDE: Part 2: Simulate the sensor
        #. Reminder: There are two random variables here, one for if the door is open, one for if it
        #.  is closed. First determine which to sample from, THEN do the same thing you did in the
        #.  first problem in the jupyter notebook.
        
        actual_state = door_ground_truth.get_door_state()
        val = np.random.uniform() # Random roll 0.0 to 1.0
        
        if actual_state is True: # Door is Open
            # We want to return True with probability self.prob_true_if_open
            if val < self.prob_true_if_open:
                return True
            else:
                return False
        else: # Door is Closed
            # We want to return False with probability self.prob_false_if_closed.
            # This implies returning True with probability (1 - self.prob_false_if_closed).
            prob_return_true = 1.0 - self.prob_false_if_closed
            if val < prob_return_true:
                return True
            else:
                return False


# Check if the door and sensor are working correctly
def test_combo(prob_true_if_open: float, prob_false_if_closed: float):
    # Make the sensor
    door_sensor = DoorSensor()
    door_sensor.set_return_true_if_open_probability(prob=prob_true_if_open)
    door_sensor.set_return_false_if_closed_probability(prob=prob_false_if_closed)

    # Do both an open and a closed door
    n_total = 1000
    for b_door_gt, exp_val in zip([True, False], [prob_true_if_open, 1.0 - prob_false_if_closed]):
        # Make the door, with door either open or closed
        door_gt = DoorGroundTruth(door_open_state=b_door_gt)

        count_true = 0
        for _ in range(0, n_total):
            # Just count the true values
            if door_sensor.sample_sensor(door_gt):
                count_true += 1

        if not np.isclose(count_true / n_total, exp_val, atol=0.05):
            print(f"For door in state {door_gt}, expected {exp_val}, got {count_true / n_total}")
            return False

    return True


if __name__ == '__main__':

    # GUIDE: These are the tests; they're the same ones as in the jupyter notebook
    #.  Once the code for Part 1 is all written the first test will pass and so on
    
    # Tests, in order
    # part 1: Create a DoorGroundTruth instance and check that it is has the door state set correctly

    door_start_open = DoorGroundTruth(True)
    door_start_closed = DoorGroundTruth(False)

    # You can print variables out, btw - or look in the variable window to see what the values are
    print(f"Door state open: {door_start_open}")
    print(f"Door state closed: {door_start_closed}")

    # Check
    assert door_start_open.get_door_state() == True
    assert door_start_closed.get_door_state() == False

    print("Part 1 passed")

    # Part 2: Check sensor; if the first one(s) work but the second don't, you have one of your if
    #. statements backwards
    assert test_combo(0.5, 0.5)
    assert test_combo(0.7, 0.5)
    assert test_combo(0.5, 0.8)
    assert test_combo(0.85, 0.71)

    print("Part 2 passed")

    # Part 3: Check actions.
    n_samples = 100
    b_ret = True
    for door_start_state in [True, False]:
        for action in DoorGroundTruth.actions:
            for door_end_state in [True, False]:
                for prob in [0.0, 0.2, 0.8, 1.0]:
                    count_in_state = 0
                    for _ in range(0, n_samples):
                        my_door = DoorGroundTruth(door_open_state=door_start_state)
                        my_door.set_probability(door_initial_state=door_start_state,
                                                action=action,
                                                door_final_state=door_end_state,
                                                prob=prob)
                        if action == "Open":
                            my_door.robot_tries_to_open_door()
                        else:
                            my_door.robot_tries_to_close_door()
                        if my_door.get_door_state() == door_end_state:
                            count_in_state += 1                    

                    prob_in_state = count_in_state / n_samples
                    if not np.isclose(prob_in_state, prob, atol=0.1):
                        print(f"Failed start state {door_start_state}, door end state {door_end_state}, action {action}")
                        print(f" Expected {prob}, got {prob_in_state}")
                        b_ret = False
    assert b_ret
    print(f"Part 3 passed")