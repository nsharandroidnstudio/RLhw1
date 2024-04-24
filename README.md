# RLhw1
Here's a draft for your README file:

---

# Grid World RL Agent

This project implements a Reinforcement Learning (RL) agent to navigate a grid world environment. The agent aims to reach the goal while avoiding hazards. Two types of agents are provided: deterministic and stochastic.

## Features

- **Deterministic Agent**: This agent uses a deterministic policy to navigate the grid world. It employs the Policy Iteration algorithm to determine the optimal policy.
- **Stochastic Agent**: The stochastic agent, on the other hand, operates in an environment with uncertain outcomes. It considers probabilistic transitions and uses a modified version of Policy Iteration to handle stochasticity.

## Usage

1. **Environment Setup**: Define the grid world environment, including the start position, rewards, and obstacles.
   
   ```python
   grid = [
       ['S', 'F', 'F'],
       ['F', 'H', 'F'],
       ['F', 'H', 'G']
   ]
   rewards = [-1, 0, 10]  # Rewards for Hole, Safe, and Goal states respectively
   ```

2. **Deterministic Agent**:

   ```python
   agent = Agent((0, 0), rewards, grid)
   agent.play_game(grid)
   ```

3. **Stochastic Agent**:

   ```python
   stochastic_agent = StochasticAgent((0, 0), rewards, grid)
   stochastic_agent.play_game(grid)
   ```

## Dependencies

- Python 3.x
- NumPy

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize it further to fit your project's specifics!
