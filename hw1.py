
import random
import numpy as np


class Agent:
    def __init__(self, start_position, rewards, grid):
        self.position = start_position
        self.possible_actions = ['up', 'down', 'left', 'right']
        self.v_value_function_matrix = self.initialize_value_function(grid)
        self.rewards = {
             "H": rewards[0], 
             "F": rewards[1],  
             "G": rewards[2]  
         }
        # self.visited_positions = set()
        self.counter_steps = 0

    def move_up(self, grid):
        self.counter_steps += 1
        new_position = (self.position[0] - 1, self.position[1])
        if self.is_valid_move(grid, new_position):
            self.position = new_position

            # self.visited_positions.add(new_position)

    def move_down(self, grid):
        self.counter_steps += 1
        new_position = (self.position[0] + 1, self.position[1])
        if self.is_valid_move(grid, new_position):
            self.position = new_position
            # self.visited_positions.add(new_position)

    def move_left(self, grid):
        self.counter_steps += 1
        new_position = (self.position[0], self.position[1] - 1)
        if self.is_valid_move(grid, new_position):
            self.position = new_position
            # self.visited_positions.add(new_position)

    def move_right(self, grid):
        self.counter_steps += 1
        new_position = (self.position[0], self.position[1] + 1)
        if self.is_valid_move(grid, new_position):
            self.position = new_position
            # self.visited_positions.add(new_position)

    def is_valid_move(self, grid, new_position):
        rows = len(grid)
        cols = len(grid[0])
        if (0 <= new_position[0] < rows) and (0 <= new_position[1] < cols):
            return True
        return False
    

    def initialize_value_function(self, grid):
        rows = len(grid)
        cols = len(grid[0])
        V = np.zeros((rows, cols))
        return V  

    def estimate_value_function(self, grid, discount_factor, state):
        
        if grid[state[0]][state[1]] in ["G","H"]:
            return 0
        
        neighbors = [
            (state[0] + 1, state[1]),  # Down
            (state[0], state[1] + 1),  # Right
            (state[0], state[1] - 1),  # Left
            (state[0] - 1, state[1])   # Up
        ]
        total_value = 0
        for neighbor in neighbors:
            if self.is_valid_move(grid, neighbor):
                reward = self.get_state_reward(neighbor, grid)
                
                total_value += 0.25 * (self.v_value_function_matrix[neighbor[0], neighbor[1]] * discount_factor + reward)
            else:
                reward = self.get_state_reward(state, grid)
                total_value += 0.25 * (self.v_value_function_matrix[state[0], state[1]] * discount_factor + reward)
        return total_value
    


    def get_state_reward(self, state, grid):
        bonus = 0
        # # if not state in self.visited_positions  :
        #     bonus = 0.1
        if grid[state[0]][state[1]] == "H":
                return self.rewards["H"] + bonus
        elif grid[state[0]][state[1]] == "S":
                return self.rewards["F"] + bonus
        elif grid[state[0]][state[1]] == "F":
                return self.rewards["F"] + bonus
        else:
                print(self.counter_steps * -theta)
                print("counter:"+str(self.counter_steps))
                return self.rewards["G"] + self.counter_steps * -0.1



    def policy_iteration(self, grid, discount_factor, theta):
        rows = len(grid)
        cols = len(grid[0])
        policy_stable = False
        iteration_count = 0  

        while not policy_stable:
            policy_stable = True
            for i in range(rows):
                for j in range(cols):
                    old_action = grid[i][j]
                    if old_action in ['S', 'G']:
                        continue
                    old_value = self.v_value_function_matrix[i][j]
                    self.v_value_function_matrix[i][j] = self.estimate_value_function(grid, discount_factor, (i, j))
                    if abs(old_value - self.v_value_function_matrix[i][j]) > theta:
                        policy_stable = False

            iteration_count += 1  # Increment iteration count
            print(f"Iteration {iteration_count}:")
            for row in self.v_value_function_matrix:
                print(row)

        print("Policy iteration completed.")
        return self.v_value_function_matrix
    
    def get_optimal_policy(self, value_function):
        rows, cols = value_function.shape
        optimal_policy = np.empty((rows, cols), dtype='<U5')  # Initialize the optimal policy array

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] in ['H', 'G']:  # Skip hole and goal states
                    continue
                
                max_action = None
                max_value = float('-inf')

                # Check all possible actions and select the one with the maximum value
                for action in self.possible_actions:
                    self.counter_steps+=1

                    new_position = self.get_new_position((i, j), action)
                  
                    if self.is_valid_move(grid, new_position):
                        current_reward = self.get_state_reward(new_position, grid)
                       
                        new_value = value_function[new_position[0], new_position[1]] + current_reward
                        if new_value > max_value:
                            max_value = new_value
                            max_action = action

                        if grid[new_position[0]][new_position[1]] == 'G':
                                max_action = action

                                optimal_policy[i, j] = action    

                optimal_policy[i, j] = max_action

        return optimal_policy
    def get_new_position(self, position, action):
        if action == 'up':
            return (position[0] - 1, position[1])
        elif action == 'down':
            return (position[0] + 1, position[1])
        elif action == 'left':
            return (position[0], position[1] - 1)
        elif action == 'right':
            return (position[0], position[1] + 1)
    
    
    def play_game(self, grid):
        # Get the optimal policy using policy iteration
        optimal_value_function = self.policy_iteration(grid, discount_factor=0.9, theta=1e-4)
        optimal_policy = self.get_optimal_policy(optimal_value_function)

        # Start the game from the start position ('S')
        current_position = (0, 0)
        while True:
            # Print current state
            print("Current position:", current_position)
            print(grid[current_position[0]][current_position[1]])
            # Check if current position is goal ('G') or hole ('H')
            if grid[current_position[0]][current_position[1]] == 'G':
                print("Congratulations! You reached the goal.")
                break
            elif grid[current_position[0]][current_position[1]] == 'H':
                print("Oops! You fell into a hole.")
                break

            # Get the optimal action from the policy
            action = optimal_policy[current_position]

            # Perform the action and update the current position
            if action == 'up':
                current_position = (current_position[0] - 1, current_position[1])
            elif action == 'down':
                current_position = (current_position[0] + 1, current_position[1])
            elif action == 'left':
                current_position = (current_position[0], current_position[1] - 1)
            elif action == 'right':
                current_position = (current_position[0], current_position[1] + 1)
            print("Action taken:", action)

class StochasticAgent(Agent):
    def __init__(self, start_position, rewards, grid):

        super().__init__(start_position, rewards, grid)


    def stochastic_move_up(self, grid):
        self.counter_steps += 1
        outcomes = ['up', 'left', 'right']
        probabilities = [0.8, 0.1, 0.1]
        action = np.random.choice(outcomes, p=probabilities)
        if action == 'up':
            new_position = (self.position[0] - 1, self.position[1])
            if self.is_valid_move(grid, new_position):
                self.position = new_position
        elif action == 'left':
            new_position = (self.position[0], self.position[1] - 1)
            if self.is_valid_move(grid, new_position):
                self.position = new_position
        elif action == 'right':
            new_position = (self.position[0], self.position[1] + 1)
            if self.is_valid_move(grid, new_position):
                self.position = new_position
        return self.position

    def stochastic_move_down(self, grid):
        self.counter_steps += 1
        outcomes = ['down', 'left', 'right']
        probabilities = [0.7, 0.15, 0.15]
        action = np.random.choice(outcomes, p=probabilities)
        if action == 'down':
            new_position = (self.position[0] + 1, self.position[1])
            if self.is_valid_move(grid, new_position):
                self.position = new_position
        elif action == 'left':
            new_position = (self.position[0], self.position[1] - 1)
            if self.is_valid_move(grid, new_position):
                self.position = new_position
        elif action == 'right':
            new_position = (self.position[0], self.position[1] + 1)
            if self.is_valid_move(grid, new_position):
                self.position = new_position
        return self.position
        

    def stochastic_move_left(self, grid):
        self.counter_steps += 1
        outcomes = ['left', 'up', 'down']
        probabilities = [0.4, 0.3, 0.3]
        action = np.random.choice(outcomes, p=probabilities)
        if action == 'left':
            new_position = (self.position[0], self.position[1] - 1)
            if self.is_valid_move(grid, new_position):
                self.position = new_position
        elif action == 'up':
            new_position = (self.position[0] - 1, self.position[1])
            if self.is_valid_move(grid, new_position):
                self.position = new_position
        elif action == 'down':
            new_position = (self.position[0] + 1, self.position[1])
            if self.is_valid_move(grid, new_position):
                self.position = new_position
        return self.position
        

    def stochastic_move_right(self, grid):
        self.counter_steps += 1
        outcomes = ['right', 'up', 'down']
        probabilities = [0.6, 0.3, 0.1]
        action = np.random.choice(outcomes, p=probabilities)
        if action == 'right':
            new_position = (self.position[0], self.position[1] + 1)
            if self.is_valid_move(grid, new_position):
                self.position = new_position
        elif action == 'up':
            new_position = (self.position[0] - 1, self.position[1])
            if self.is_valid_move(grid, new_position):
                self.position = new_position
        elif action == 'down':
            new_position = (self.position[0] + 1, self.position[1])
            if self.is_valid_move(grid, new_position):
                self.position = new_position

        return self.position
        

    def stochastic_estimate_value_function(self, grid, discount_factor, state):
        
        if grid[state[0]][state[1]] in ["G", "H"]:
            return 0
            
        neighbors = [
            (state[0] + 1, state[1]),  # Down
            (state[0], state[1] + 1),  # Right
            (state[0], state[1] - 1),  # Left
            (state[0] - 1, state[1])   # Up
        ]
        total_value = 0
        for neighbor in neighbors:
            if self.is_valid_move(grid, neighbor):
                reward = self.get_state_reward(neighbor, grid)
                
                # Define transition probabilities based on action
                if neighbor[0] == state[0] + 1:  # Down
                    p_success = 0.7
                    p_opposite = 0.15
                    p_perpendicular = 0.15
                elif neighbor[0] == state[0] - 1:  # Up
                    p_success = 0.8
                    p_opposite = 0.1
                    p_perpendicular = 0.1
                elif neighbor[1] == state[1] + 1:  # Right
                    p_success = 0.6
                    p_opposite = 0.3
                    p_perpendicular = 0.1
                elif neighbor[1] == state[1] - 1:  # Left
                    p_success = 0.4
                    p_opposite = 0.3
                    p_perpendicular = 0.3
                
                total_value += p_success *(self.v_value_function_matrix[neighbor[0], neighbor[1]] * discount_factor + reward)
                total_value += p_opposite * (self.v_value_function_matrix[state[0], state[1]] * discount_factor + reward)
                total_value += p_perpendicular*(self.v_value_function_matrix[state[0], state[1]] * discount_factor + reward)
                total_value = total_value*0.25
            else:
                reward = self.get_state_reward(state, grid)
                total_value += 0.25 * (self.v_value_function_matrix[state[0], state[1]] * discount_factor + reward)
        return total_value  

    def policy_iteration(self, grid, discount_factor, theta):
        rows = len(grid)
        cols = len(grid[0])
        policy_stable = False
        iteration_count = 0  

        while not policy_stable:
            policy_stable = True
            for i in range(rows):
                for j in range(cols):
                    old_action = grid[i][j]
                    if old_action in ['S', 'G']:
                        continue
                    old_value = self.v_value_function_matrix[i][j]
                    self.v_value_function_matrix[i][j] = self.stochastic_estimate_value_function(grid, discount_factor, (i, j))
                    if abs(old_value - self.v_value_function_matrix[i][j]) > theta:
                        policy_stable = False

            iteration_count += 1  # Increment iteration count
            print(f"Iteration {iteration_count}:")
            for row in self.v_value_function_matrix:
                print(row)

        print("Policy iteration completed.")
        return self.v_value_function_matrix         

    # random_policey= 0.25
    # random_policy * (action_probility*(dicount*valumat[i,j]+ reward))
    
    def play_game(self, grid):
        # Get the optimal policy using policy iteration
        optimal_value_function = self.policy_iteration(grid, discount_factor=0.9, theta=1e-4)
        optimal_policy = self.get_optimal_policy(optimal_value_function)

        # Start the game from the start position ('S')
        current_position = (0, 0)
        while True:
            # Print current state
            print("Current position:", current_position)
            # Check if current position is goal ('G') or hole ('H')
            if grid[current_position[0]][current_position[1]] == 'G':
                print("Congratulations! You reached the goal.")
                break
            elif grid[current_position[0]][current_position[1]] == 'H':
                print("Oops! You fell into a hole.")
                break

            # Get the optimal action from the policy
            action = optimal_policy[current_position]

            # Perform the action and update the current position
            if action == 'up':
                current_position = self.stochastic_move_up(grid)
            elif action == 'down':
                current_position = self.stochastic_move_down(grid)
            elif action == 'left':
                current_position = self.stochastic_move_left(grid)
            elif action == 'right':
                current_position = self.stochastic_move_right(grid)


  
grid = [
    ['S', 'F', 'F'],
    ['F', 'H', 'F'],
    ['F', 'H', 'G']
]

# # Create an agent with start position (0, 0)
rewards = [-1, 0, 10]
# agent = Agent((0, 0), rewards, grid)
# discount_factor = 0.9
theta = 1e-4
# agent.play_game(grid)

stochastic_agent = StochasticAgent((0, 0), rewards, grid)

# # # # # Call the play_game method of the stochastic_agent
stochastic_agent.play_game(grid)
