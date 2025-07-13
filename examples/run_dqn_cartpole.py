"""
Example Script: Run DQN on CartPole.

This script demonstrates how to train a DQN agent on the CartPole environment using the Modular DRL Framework, as described in the DRL survey and project engineering blueprint.

Detailed Description:
The script loads the DQN algorithm, environment, and hyperparameters from YAML configuration files. It initializes the agent, environment, and logger, then runs the training loop, logging results and saving checkpoints for reproducibility. The example serves as a template for running other discrete control experiments.

Key Concepts/Algorithms:
- DQN algorithm for discrete action spaces
- Modular configuration and reproducibility
- Logging and checkpointing of training progress

Important Parameters/Configurations:
- Environment: CartPole-v1
- Algorithm: DQN (parameters from `configs/dqn_config.yaml`)
- Logging and checkpoint directories

Expected Inputs/Outputs:
- Inputs: YAML config files, environment spec
- Outputs: training logs, saved models, evaluation metrics

Dependencies:
- torch, numpy, gym, PyYAML, matplotlib, tensorboard, tqdm
- src/algorithms/dqn.py, src/networks/q_network.py, src/buffers/replay_buffer.py, src/utils/logger.py

Author: REL Project Team
Date: 2025-07-13
"""
# Example: Run DQN on CartPole
