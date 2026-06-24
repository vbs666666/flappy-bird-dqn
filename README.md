# Flappy Bird DQN

This project trains a Deep Q Network (DQN) agent to play Flappy Bird using PyTorch and Pygame.

## Structure

- `train.py`: trains the DQN agent and writes TensorBoard metrics.
- `evaluate_checkpoints.py`: evaluates saved checkpoints across multiple games.
- `play_trained_model.py`: runs one trained model in the Pygame environment.
- `src/deep_q_network.py`: DQN model definition.
- `src/flappy_bird.py`: Flappy Bird environment.
- `src/utils.py`: image preprocessing.
- `assets/sprites/`: game sprites.

## Setup

```bash
uv venv .venv
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -r requirements.txt
```

For RTX 5060 or other GPUs requiring recent PyTorch builds, install a CUDA wheel compatible with your driver. For CUDA 12.8:

```bash
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

## Training

```bash
UV_CACHE_DIR=.uv-cache SDL_VIDEODRIVER=dummy uv run --python .venv/bin/python python train.py
```

TensorBoard logs are written to `tensorboard/`, and checkpoints are written to `trained_models/`.

## Evaluation

```bash
UV_CACHE_DIR=.uv-cache uv run --python .venv/bin/python python evaluate_checkpoints.py
UV_CACHE_DIR=.uv-cache uv run --python .venv/bin/python python play_trained_model.py --checkpoint_iter 2000000
```

Generated logs, checkpoints, virtual environments, and caches are intentionally ignored by Git.
