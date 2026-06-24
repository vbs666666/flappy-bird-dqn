from random import randint
from src.flappy_bird import FlappyBird

def test(i):
    game_state = FlappyBird()
    game_state.next_frame(0)

    while True:
        action = randint(0, 1)#Exploration
        game_state.next_frame(action)


if __name__ == "__main__":
    for i in range(5):
        test(i)