# Splendor-AI

Welcome to the Splendor-AI repository! This project allows you to build an RL-based AI to play the board game Splendor.

## Overview

- **Reinforcement Learning Approach**: Capable of human-level performance.  In its current state though, it can only learn well in an easier version of the game with more starting resources.
- **Brute Force Approach**: Approaches acceptable performance by searching thousands of game states per second.  I use it as a jump-off point for the replay pool.

## Features

- **DDQN RL approach**: An expansion of Q-learning that mitigates overestimation bias by decoupling action selection from Q-value evaluation.
- **Efficient Game Environment**: Includes a game environment that simulates hundreds of games per second as most operations are vectorized in Numpy.
- **Visualization**: Generates frames of the games at each state along with card counts and move descriptions.
- **Modular Design**: I built this in a fully object-oriented, clean and commented way.  Although the overall game is too complex to fully redo, you should have an easy time changing any specific set of features you want.

## Getting Started

To get started with the project, clone the repository and follow the setup instructions below. After setting things up, you can run `run.py` with your preferred settings!  
This project utilizes a VSCode DevContainer and a Docker environment, so all dependencies are automatically managed.

## Requirements
- **Docker** - Ensure Docker is installed and running on your system.
- **VSCode** - I use a DevContainer for other parts of the setup.
- **Cuda GPU compatibility (optional)** - The code is optimized for GPU execution using CUDA.

```bash
git clone https://github.com/BreckEmert/Splendor-AI
cd Splendor-AI
docker-compose up --build
```
