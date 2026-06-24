# Repository Guidelines

## Project Structure & Module Organization

This repository implements a PyTorch Deep Q Network for Flappy Bird. Core reusable code lives in `src/`: `deep_q_network.py` defines the model, `flappy_bird.py` contains the Pygame environment, and `utils.py` holds image preprocessing helpers. `train.py` is the main training entry point. `evaluate_checkpoints.py` evaluates many saved models and plots scores; `play_trained_model.py` runs one trained model for visual inspection. Game images are stored in `assets/sprites/`; generated outputs such as `runs/`, `tensorboard/`, and trained model checkpoints should be treated as local artifacts.

## Build, Test, and Development Commands

Create and activate a virtual environment before installing dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pygame tensorboard tqdm opencv-python
```

Run a short smoke training pass:

```bash
python train.py --num_iters 10 --batch_size 4
```

Run a GUI/demo script when checking gameplay behavior:

```bash
python GUI.py
```

View TensorBoard logs after training:

```bash
tensorboard --logdir tensorboard
```

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Follow existing module naming with lowercase snake_case for files, functions, and variables, and PascalCase for classes such as `DeepQNetwork` and `FlappyBird`. Keep model and environment logic inside `src/`; avoid duplicating full environment code in new scripts unless the script is explicitly experimental. Prefer concise comments for non-obvious game or reinforcement-learning behavior.

## Testing Guidelines

There is no formal test framework configured. For changes to preprocessing, model shapes, or environment stepping, add focused `pytest` tests under a new `tests/` directory when practical. Name tests `test_<feature>.py`. At minimum, run the short `train.py --num_iters 10` smoke test and any affected demo script before submitting changes.

## Commit & Pull Request Guidelines

This checkout does not expose Git history, so use clear imperative commit messages such as `Fix sprite asset paths` or `Add DQN smoke test`. Pull requests should include a short summary, commands run, expected output artifacts, and screenshots or recordings for visible gameplay or GUI changes. Link related issues when available and call out any dependency, checkpoint, or asset-path changes.

## Security & Configuration Tips

Do not commit large generated files from `runs/`, `tensorboard/`, or model checkpoint directories. Keep asset paths relative to the repository root so scripts work from a fresh clone.
