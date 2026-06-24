from itertools import cycle
from pygame import Rect, init, time, display
from pygame.event import pump
from pygame.image import load
from pygame.surfarray import array3d, pixels_alpha
from pygame.transform import rotate
import numpy as np
from random import randint


class FlappyBird(object):
    init()
    fps_clock = time.Clock()
    screen_width = 288
    screen_height = 512
    screen = display.set_mode((screen_width, screen_height))
    display.set_caption("Flappy Bird")
    base_image = load('./assets/sprites/base.png').convert_alpha()
    background_image = load('./assets/sprites/background-black.png').convert()
    pipe_images = [rotate(load('./assets/sprites/pipe-green.png').convert_alpha(), 180),
                   load('./assets/sprites/pipe-green.png').convert_alpha()]
    bird_images = [load('./assets/sprites/redbird-upflap.png').convert_alpha(),
                   load('./assets/sprites/redbird-midflap.png').convert_alpha(),
                   load('./assets/sprites/redbird-downflap.png').convert_alpha()]
    bird_hitmask = [pixels_alpha(image).astype(bool) for image in bird_images]
    pipe_hitmask = [pixels_alpha(image).astype(bool) for image in pipe_images]

    # 鸟的参数
    min_velocity_y = -8
    max_velocity_y = 10
    downward_speed = 1
    upward_speed = -5

    bird_index_generator = cycle([0, 1, 2, 1])

    def __init__(self):
        self.fps = 40000       # 帧率

        self.iter = self.bird_index = self.score = 0
        self.last_score = 0

        self.bird_width = self.bird_images[0].get_width()           # 鸟的宽度
        self.bird_height = self.bird_images[0].get_height()         # 鸟的高度
        self.pipe_width = self.pipe_images[0].get_width()           # 管道的宽度
        self.pipe_height = self.pipe_images[0].get_height()         # 管道的高度
        self.pipe_gap_size = 5 * self.bird_width                    # 管道与管道之间的间隔
        self.pipe_velocity_x = -4                                   # 管道左移的速度（注意是负值）

        self.bird_x = int(self.screen_width / 5)
        self.bird_y = int((self.screen_height - self.bird_height) / 2)

        self.base_x = 0
        self.base_y = self.screen_height * 0.79
        self.base_shift = self.base_image.get_width() - self.background_image.get_width()

        pipes = [self.generate_pipe(), self.generate_pipe()]
        pipes[0]["x_upper"] = pipes[0]["x_lower"] = self.screen_width
        pipes[1]["x_upper"] = pipes[1]["x_lower"] = int(self.screen_width * 1.5)
        self.pipes = pipes

        self.current_velocity_y = 0
        self.is_flapped = False

    def generate_pipe(self):
        x = self.screen_width + 10
        gap_y = randint(2, 10) * 10 + int(self.base_y / 5)
        return {"x_upper": x, "y_upper": gap_y - self.pipe_height, "x_lower": x, "y_lower": gap_y + self.pipe_gap_size}

    def is_collided(self):
        if self.bird_height + self.bird_y + 1 >= self.base_y:
            return True
        bird_bbox = Rect(self.bird_x, self.bird_y, self.bird_width, self.bird_height)
        pipe_boxes = []
        for pipe in self.pipes:
            pipe_boxes.append(Rect(pipe["x_upper"], pipe["y_upper"], self.pipe_width, self.pipe_height))
            pipe_boxes.append(Rect(pipe["x_lower"], pipe["y_lower"], self.pipe_width, self.pipe_height))
            if bird_bbox.collidelist(pipe_boxes) == -1:
                return False
            for i in range(2):
                cropped_bbox = bird_bbox.clip(pipe_boxes[i])
                min_x1 = cropped_bbox.x - bird_bbox.x
                min_y1 = cropped_bbox.y - bird_bbox.y
                min_x2 = cropped_bbox.x - pipe_boxes[i].x
                min_y2 = cropped_bbox.y - pipe_boxes[i].y
                if np.any(self.bird_hitmask[self.bird_index][min_x1: min_x1 + cropped_bbox.width,
                          min_y1: min_y1 + cropped_bbox.height] * self.pipe_hitmask[i][
                                                                  min_x2: min_x2 + cropped_bbox.width,
                                                                  min_y2: min_y2 + cropped_bbox.height]):
                    return True
        return False

    def next_frame(self, action):
        pump()              # 更新事件队列，处理输入事件
        reward = 0.1        # 奖励值（默认0.1）
        terminal = False    # 终止标志（默认False）

        if action == 1:  # 输入动作为1向上飞
            self.current_velocity_y = self.upward_speed  # 更新鸟的垂直速度为向上的速度
            self.is_flapped = True  # 标记鸟拍打翅膀

        # 更新得分
        bird_center_x = self.bird_x + self.bird_width / 2  # 鸟的中心横坐标
        for pipe in self.pipes:
            pipe_center_x = pipe["x_upper"] + self.pipe_width / 2  # 管道的中心横坐标
            if pipe_center_x < bird_center_x < pipe_center_x + 5:
                self.score += 1  # 鸟通过管道加分
                reward = 1  # 奖励
                break

        # 更新鸟的动画帧
        if (self.iter + 1) % 3 == 0:
            self.bird_index = next(self.bird_index_generator)  # 鸟动画效果
        self.iter = (self.iter + 1) % 3

        # 地面动画效果
        self.base_x = -((-self.base_x + 100) % self.base_shift)

        # 更新鸟的位置
        if self.current_velocity_y < self.max_velocity_y and not self.is_flapped:
            self.current_velocity_y += self.downward_speed  # 鸟未拍打翅膀就下降

        if self.is_flapped:
            self.is_flapped = False  # 拍打翅膀后重置标志

        self.bird_y += min(self.current_velocity_y, self.bird_y - self.current_velocity_y - self.bird_height)  # 更新鸟的y坐标

        if self.bird_y < 0:  # 确保鸟不会飞跑（上溢）
            self.bird_y = 0

        # 更新管道的位置
        for pipe in self.pipes:
            pipe["x_upper"] += self.pipe_velocity_x  # 更新上管道的x坐标
            pipe["x_lower"] += self.pipe_velocity_x  # 更新下管道的x坐标

        # 更新管道列表
        if 0 < self.pipes[0]["x_lower"] < 5:
            self.pipes.append(self.generate_pipe())  # 添加新管道
        if self.pipes[0]["x_lower"] < -self.pipe_width:
            del self.pipes[0]  # 删除屏幕外的管道

        # 检查碰撞
        if self.is_collided():
            terminal = True  # 游戏结束
            reward = -1  # 负奖励
            last_score = self.score
            self.__init__()  # 重新初始化游戏状态
            self.last_score = last_score

        # 绘制所有内容
        self.screen.blit(self.background_image, (0, 0))  # 绘制背景
        self.screen.blit(self.base_image, (self.base_x, self.base_y))  # 绘制地面
        self.screen.blit(self.bird_images[self.bird_index], (self.bird_x, self.bird_y))  # 绘制鸟
        for pipe in self.pipes:
            self.screen.blit(self.pipe_images[0], (pipe["x_upper"], pipe["y_upper"]))  # 绘制上管道
            self.screen.blit(self.pipe_images[1], (pipe["x_lower"], pipe["y_lower"]))  # 绘制下管道

        # 当前帧的图像转换为数组
        image = array3d(display.get_surface())

        # 更新显示
        display.update()
        self.fps_clock.tick(self.fps)  # 控制帧率

        return image, reward, terminal  # 当前帧的图像、奖励值和终止标志


# def test(i):
#     game_state = FlappyBird()
#     game_state.next_frame(0)
#
#     while True:
#         action = randint(0, 1)
#         game_state.next_frame(action)
#
#
# if __name__ == "__main__":
#     for i in range(5):
#         test(i)
