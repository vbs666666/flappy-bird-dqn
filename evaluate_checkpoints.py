import argparse
from statistics import pstdev

import torch
from torch.utils.tensorboard import SummaryWriter

from src.flappy_bird import FlappyBird
from src.utils import pre_processing
import matplotlib.pyplot as plt

# 这个脚本用于批量评估多个训练 checkpoint。
# 它会依次加载不同迭代次数保存的模型，让每个模型玩多局游戏，
# 计算平均得分，写入 TensorBoard，并绘制“训练迭代次数-平均得分”曲线。


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--save_path", type=str, default="trained_models")
    parser.add_argument("--log_path", type=str, default="tensorboard_eval")
    parser.add_argument("--num_games", type=int, default=5, help="Number of games to average for each checkpoint")
    parser.add_argument("--max_score", type=int, default=350, help="Stop a game early after this score")

    args = parser.parse_args()
    return args


def run_one_game(model, opt):
    game_state = FlappyBird()
    image, _, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    while not terminal and game_state.score < opt.max_score:
        with torch.no_grad():
            prediction = model(state)[0]
            action = torch.argmax(prediction).item()

        next_image, _, terminal = game_state.next_frame(action)
        next_image = pre_processing(
            next_image[:game_state.screen_width, :int(game_state.base_y)],
            opt.image_size,
            opt.image_size,
        )
        next_image = torch.from_numpy(next_image)
        state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

    return game_state.score


def evaluate_checkpoint(opt, i):
    torch.manual_seed(123)
    model = torch.load("./{}/flappy_bird_{}0000.pth".format(opt.save_path, i),
                       map_location=lambda storage, loc: storage)
    model.eval()
    scores = [run_one_game(model, opt) for _ in range(opt.num_games)]
    return scores


if __name__ == "__main__":
    opt = get_args()
    writer = SummaryWriter(log_dir=opt.log_path)
    iteration = []
    game_scores = []
    for i in range(5, 201, 5):
        step = i * 10000
        scores = evaluate_checkpoint(opt, i)
        avg_score = sum(scores) / len(scores)
        iteration.append(step)
        game_scores.append(avg_score)
        writer.add_scalar("Eval/AvgScore", avg_score, step)
        writer.add_scalar("Eval/MaxScore", max(scores), step)
        writer.add_scalar("Eval/MinScore", min(scores), step)
        writer.add_scalar("Eval/ScoreStd", pstdev(scores) if len(scores) > 1 else 0, step)
        writer.add_scalar("Eval/NumGames", len(scores), step)
        print("迭代", step, "平均奖励", avg_score)

    plt.figure(figsize=(20, 8), dpi=80)
    plt.ylabel('Average Score')
    plt.xlabel('Iterations')
    plt.plot(iteration, game_scores)
    plt.savefig("iteration-game_score.jpg")
    writer.close()
