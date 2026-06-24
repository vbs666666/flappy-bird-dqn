import argparse
import os
import time

import pygame
import torch
from torch.utils.tensorboard import SummaryWriter

from src.flappy_bird import FlappyBird
from src.utils import pre_processing

# 这个脚本用于演示单个训练好的模型。
# 它只加载一个 checkpoint，让模型在 Pygame 窗口中实际玩一局，
# 将单局过程写入 TensorBoard，最后输出本局最高分和运行耗时。


def get_args():
    parser = argparse.ArgumentParser("Run a trained DQN model on Flappy Bird")
    parser.add_argument("--image_size", type=int, default=84, help="Input image width and height")
    parser.add_argument("--save_path", type=str, default="trained_models", help="Directory containing checkpoints")
    parser.add_argument("--log_path", type=str, default="tensorboard_play", help="TensorBoard log directory")
    parser.add_argument("--checkpoint_iter", type=int, default=2000000, help="Checkpoint iteration to load")
    parser.add_argument("--model_path", type=str, default=None, help="Explicit model path; overrides checkpoint_iter")
    return parser.parse_args()


def resolve_model_path(opt):
    if opt.model_path:
        return opt.model_path
    return os.path.join(opt.save_path, f"flappy_bird_{opt.checkpoint_iter}.pth")


def play_one_game(opt):
    torch.manual_seed(123)
    model_path = resolve_model_path(opt)
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()
    writer = SummaryWriter(log_dir=opt.log_path)

    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    max_score = 0
    total_reward = 0.0
    step = 0
    start_time = time.time()

    while not terminal:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                writer.close()
                return max_score, time.time() - start_time

        with torch.no_grad():
            prediction = model(state)[0]
            action = torch.argmax(prediction).item()

        next_image, reward, terminal = game_state.next_frame(action)
        total_reward += reward
        step += 1
        max_score = max(max_score, game_state.score)
        writer.add_scalar("Play/Score", max_score, step)
        writer.add_scalar("Play/StepReward", reward, step)
        writer.add_scalar("Play/TotalReward", total_reward, step)
        writer.add_scalar("Play/Action", action, step)
        writer.add_scalar("Play/MaxQ", torch.max(prediction).item(), step)
        writer.add_scalar("Play/ElapsedTime", time.time() - start_time, step)

        next_image = pre_processing(
            next_image[:game_state.screen_width, :int(game_state.base_y)],
            opt.image_size,
            opt.image_size,
        )
        next_image = torch.from_numpy(next_image)
        state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

    elapsed_time = time.time() - start_time
    writer.add_scalar("Play/FinalScore", max_score, 0)
    writer.add_scalar("Play/FinalTotalReward", total_reward, 0)
    writer.add_scalar("Play/FinalElapsedTime", elapsed_time, 0)
    writer.close()
    return max_score, elapsed_time


if __name__ == "__main__":
    opt = get_args()
    score, elapsed_time = play_one_game(opt)
    print(f"Game over! Max score: {score}, Time elapsed: {elapsed_time:.2f} seconds")
