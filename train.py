import argparse
import os
import shutil
from collections import deque
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image-size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=2000000)       # 迭代次数
    parser.add_argument("--replay_memory_size", type=int, default=50000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--save_path", type=str, default="trained_models")
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between TensorBoard step-metric writes")
    parser.add_argument("--print_interval", type=int, default=100, help="Steps between console status prints")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    model = DeepQNetwork()
    if os.path.exists(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    os.makedirs(opt.save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=opt.log_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)

    if torch.cuda.is_available():       # GPU运行
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    replay_memory = []
    episode_reward = 0.0
    episode_rewards = deque(maxlen=100)
    episode_scores = deque(maxlen=100)
    episode = 0
    episode_length = 0
    action_count = [0, 0]
    iter = 0
    while iter < opt.num_iters:
        prediction = model(state)[0]
        epsilon = opt.final_epsilon + (
            (opt.num_iters - iter) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_iters)
        u = random()
        random_action = u <= epsilon
        if random_action:
            print("Perform a random action")
            action = randint(0, 1)
        else:
            action = torch.argmax(prediction).item()

        next_state, reward, terminal = game_state.next_frame(action)
        episode_reward += reward
        episode_length += 1
        action_count[action] += 1
        finished_score = game_state.last_score if terminal else game_state.score
        next_image = pre_processing(next_state[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():  # GPU运行
            model.cuda()
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        replay_memory.append([state, action, reward, next_state, terminal])
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        current_prediction_batch = model(state_batch)
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
            y_batch = torch.cat(
                tuple(reward if terminal else reward + opt.gamma * torch.max(prediction)
                      for reward, terminal, prediction in zip(reward_batch, terminal_batch, next_prediction_batch)))
        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = next_state
        iter += 1
        if iter % opt.print_interval == 0 or terminal:
            print("Iteration: {}/{}, Action: {}, loss: {}, Epsilon: {}, Reward: {}, Q_value: {}".format(
                iter,
                opt.num_iters,
                action,
                loss,
                epsilon, reward, torch.max(prediction)))
        if iter % opt.log_interval == 0 or terminal:
            writer.add_scalar('Train/Loss', loss.item(), iter)
            writer.add_scalar('Train/Epsilon', epsilon, iter)
            writer.add_scalar('Train/StepReward', reward, iter)
            writer.add_scalar('Train/CurrentScore', finished_score, iter)
            writer.add_scalar('Train/ReplayMemorySize', len(replay_memory), iter)
            writer.add_scalar('Train/Action', action, iter)
            writer.add_scalar('Train/FlapActionRate', action_count[1] / sum(action_count), iter)
            writer.add_scalar('Train/CurrentMaxQ', torch.max(prediction).item(), iter)
            writer.add_scalar('Train/BatchMeanQ', q_value.mean().item(), iter)
            writer.add_scalar('Train/BatchMaxQ', q_value.max().item(), iter)
            writer.add_scalar('Train/TDTargetMean', y_batch.mean().item(), iter)

        if terminal:
            episode += 1
            episode_rewards.append(episode_reward)
            episode_scores.append(finished_score)
            writer.add_scalar('Episode/Reward', episode_reward, episode)
            writer.add_scalar('Episode/Score', finished_score, episode)
            writer.add_scalar('Episode/AvgReward100', sum(episode_rewards) / len(episode_rewards), episode)
            writer.add_scalar('Episode/AvgScore100', sum(episode_scores) / len(episode_scores), episode)
            writer.add_scalar('Episode/Length', episode_length, episode)
            episode_reward = 0.0
            episode_length = 0

        if iter % 10000 == 0:
            torch.save(model, "{}/flappy_bird_{}.pth".format(opt.save_path, iter))
    torch.save(model, "{}/flappy_bird.pth".format(opt.save_path))
    writer.close()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
