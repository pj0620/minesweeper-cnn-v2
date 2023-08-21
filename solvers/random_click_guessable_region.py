import random
import numpy as np
from game import Game

NUM_GAMES = 1000
N = 10
n_bombs = 10
PRINT_BOARD = False

wins = 0
num_moves = 0
for iter in range(NUM_GAMES):
    click_x = 0
    click_y = 0
    g = Game(N=N, n_bombs=n_bombs, seed=iter)
    game_over, won_loss = g.click(0, 0, print_wl=False)
    num_moves += 1

    while not game_over:
        game_input, correct_output = g.get_input_output_representation()

        while g.guessable_mask[click_x][click_y] == 0:
            click_x = random.randint(0, N - 1)
            click_y = random.randint(0, N - 1)

        num_moves += 1

        game_over, won_loss = g.click(click_x, click_y, print_wl=False)

    if won_loss == 'win':
        wins += 1

print("{:.2f}% win rate, {} avg moves per game".format(100.0 * wins / NUM_GAMES, num_moves / NUM_GAMES))