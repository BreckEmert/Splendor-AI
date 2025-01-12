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

To get started with the project, clone the repository and follow the setup instructions below. After setting things up, you'll run 'run.py' to use it.
Depending on your setup, various filepathing may need to be changed.  I've done a good job making all of the filepathing dynamic, and os.makedirs, but occasionally there are some hardcoded paths, such as in the .dockerfile.  Please check the run.py, and .devcontainer folder to make sure things are good if you have issues.

## Requirements
**Note:** You *can* just run 'run.py' after cloning.  However, if you plan to use this project extensively, it may be worth using the Docker+DevContainer setup with CUDA:
- **Docker** - Ensure Docker is installed and running on your system.
- **VSCode** - I use a DevContainer for other parts of the setup, so you'll need the DevContainers extension.
- **Cuda GPU compatibility (optional)** - The code is optimized for GPU execution using CUDA.  I use a GPU-compatible TensorFlow version.

```bash
git clone https://github.com/BreckEmert/Splendor-AI
```

Next, open VSCode and open the project location (Splendor subfolder).  It should prompt you to reopen in a container, and launch the Docker accordingly.

(extra, probably not needed): running the docker with -dit straight from a WSL terminal allows you to host the terminal on VSCode instead of the terminal where this is ran:
```wsl2
docker run -dit --gpus all --rm -v **your-GitHub-path**/Splendor-AI/Splendor:/workspace -p 3000:3000 --name splendor-dev splendor-dev
```