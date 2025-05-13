# Reinforcement Learning for Circuit Synthesis from Frequency Response

This repository presents a novel Reinforcement Learning (RL) approach to synthesize analog circuits that match a given frequency response. The goal is to automate the design process of analog circuits by learning a policy that builds circuits based on a target frequency response graph using the Proximal Policy Optimization (PPO) algorithm.

---

## Project Overview

In analog electronics, a common challenge is the lack of a known circuit topology for a given frequency response. This project formulates the problem as a **Markov Decision Process (MDP)** and applies **Reinforcement Learning** to synthesize a circuit that replicates the target response.

- **State Space**: Includes both the current circuit configuration and a low-dimensional target graph descriptor.
- **Action Space**: The agent chooses a component type, value, and connection nodes.
- **Reward Function**: Encourages high correlation with the target response while penalizing unnecessary complexity.
- **Environment**: Custom Gymnasium environment integrated with **LTSpice** for simulation.

---

## Directory Structure
```plaintext
├── CircuitEnv/               # Custom Gym environment
│   └── CustomEnv.py
├── train_graphs/             # Training frequency response CSVs
├── test_graphs/              # Testing frequency response CSVs
├── scripts/
│   ├── buildckt_simulate.py  # Script to generate circuit frequency responses
│   ├── train.py              # Training script using PPO
│   └── test.py               # Testing script for synthesized circuits
├── models/                   # Saved models
├── utils/                    # Utility functions
└── README.md                 # This file
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install gymnasium stable-baselines3 networkx numpy
```

- Install LTSpice and note the path to the executable (e.g., LTspice.exe on Windows).
  
### 2. Configure Paths

- Update the LTSpice executable path in CustomEnv.py:
- ltspice_path = "path_to_LTspice_executable"

### 3. Running the Project and Testing the Model

- Place target frequency response CSVs inside the test_graphs/ folder.
- Run the test script:
  ```python scripts/test.py```

### 4. Training the Model

- Generate training frequency responses:
  ```bash
  python scripts/buildckt_simulate.py
  ```
- Or use pre-generated CSVs in train_graphs/.
- Run the training script:
  ```bash
  python scripts/train.py
  ```

### 5. Results

- The RL agent successfully learned to synthesize circuits whose frequency response closely matches the target. A sample result:
    - Green: Target frequency response
    - Red: Synthesized circuit's frequency response

- While the trends match, exact magnitudes vary due to simplified state space and reward formulation.

### 6. Limitations and Future Work
- Limitations:
    - Dimensionality reduction of the target response may lead to loss of fine details.
    - Reward function relies on correlation, not pointwise accuracy.

### Future Improvements:

  - Use richer state representations (e.g., more segments, local extrema).
  - Incorporate better similarity metrics (e.g., MSE, frequency-weighted errors).
  - Extend to more complex/non-passive components.
  - Explore alternate RL methods (e.g., DQN, A2C).

### Contributing

- Feel free to raise issues or open pull requests if you'd like to contribute to the project or suggest improvements!

### License
- This project is licensed under the MIT License.


