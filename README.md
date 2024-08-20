# Splendor-AI

Welcome to the Splendor-AI repository! This project allows you to build an RL-based AI to play the board game Splendor.

## Overview

- **Reinforcement Learning Approach**: Capable of human-level performance after just minutes of training.
- **Brute Force Approach**: Approaches human-level performance by searching thousands of game states per second, bypassing unnecessary logic during the random search.

## Features

- **Double Dueling Deep Q-Network (DDQN)**: Mitigates overestimation bias by decoupling action selection from Q-value evaluation.
- **Efficient Game Environment**: Includes a robust game environment capable of simulating hundreds of games per second as most operations are vectorized in Numpy.
- **Appealing Visualization**: Generates frames of the games at each state along with card counts and move descriptions.
- **Modular Design**: Allows easy swapping and customization of components, strategies and model design.

## Getting Started

To get started with the project, clone the repository and follow the setup instructions below. After setting things up, you can run `run.py` with your preferred settings!  
This project utilizes a VSCode DevContainer and a Docker environment, so all dependencies are automatically managed.

### Requirements
- **Docker** - Ensure Docker is installed and running on your system
- **VSCode (optional)** - For seamlessly using the DevContainer.
- **Cuda GPU compatibility (optional)** - The code is optimized for GPU execution using CUDA. If your GPU is not CUDA-compatible, you may need to make minor adjustments to the code.

```bash
git clone https://github.com/BreckEmert/Splendor-AI
cd Splendor-AI
docker-compose up --build
```
