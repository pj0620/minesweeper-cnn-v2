import random
from game import Game
from tqdm import tqdm

NUM_GAMES = 1000
# NUM_GAMES = 1
N = 10
n_bombs = 10
PRINT_BOARD = False

def get_start_point(game):
    sx, sy = 0, 0
    while game.bomb[sx][sy] == 1:
        sx = random.randint(0, N-1)
        sy = random.randint(0, N-1)
    return sx, sy

def get_next_click_in_guessable_mask(game, unvisited):
    rx, ry = -1, -1
    for x, y in unvisited:

        # skip if not in guessable mask
        if game.guessable_mask[x][y] == 1:
            rx, ry = x, y
            break

    if rx != -1:
        unvisited.remove((rx, ry))

    return rx, ry

def get_any_next_click(unvisited):
    return unvisited.pop()

wins = 0
num_moves = 0
for iter in tqdm(range(NUM_GAMES), desc='Playing Games', ncols=100):
# for iter in [42]:
    g = Game(N=N, n_bombs=n_bombs, seed=iter)
    click_x, click_y = get_start_point(g)
    game_over, won_loss = g.click(click_x, click_y, print_wl=False)
    num_moves += 1

    # get list of unvisited nodes
    unvisited = []
    for x in range(0, N):
        for y in range(0, N):
            # don't add bomb locations to unvisited cells
            if g.bomb[x][y] == 1:
                continue
            unvisited.append((x, y))
    # remove start click
    unvisited.remove((click_x, click_y))

    while not game_over:
        game_input, correct_output = g.get_input_output_representation()

        click_x, click_y = get_next_click_in_guessable_mask(g, unvisited)

        if click_x == -1:
            click_x, click_y = get_any_next_click(unvisited)

        num_moves += 1
        game_over, won_loss = g.click(click_x, click_y, print_wl=False)

    if won_loss == 'win':
        wins += 1

print("{:.2f}% win rate, {} avg moves per game".format(100.0 * wins / NUM_GAMES, num_moves / NUM_GAMES))