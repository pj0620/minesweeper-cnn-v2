from keras.models import load_model
import numpy as np
from game import Game
from tqdm import tqdm

from scipy.signal import convolve2d

from utils.hash_utils import array_hash

model_file = '/model/ten_by_ten_rcnn_v10.2.h5'

# Load the saved model from the file
model = load_model(model_file)

NUM_GAMES = 100
N = 10
SEEDS_OFFSET = 0
n_bombs = 10
DEBUG_PRINT = False
GUESS_SIZE = 9

COMPUTE_FULL_PROB_MAP = False
PROB_ONE_THRESHOLD = 0.99999

r = lambda vect: np.where(vect > 0, 1, vect).astype(np.int16)
con = lambda A, B: convolve2d(A, B, mode='same')

cache = {}

wins = 0
lost_on_first_click = 0
num_moves = 0
losing_nonfirstclick_games = []

for iter in tqdm(range(NUM_GAMES), desc='Playing Games', ncols=100):
    if iter % 1000 == 0: print(f"iter: {iter}")

    g = Game(N=N, n_bombs=n_bombs, seed=SEEDS_OFFSET + iter)
    game_over, won_loss = g.click(0,0,
        # iter % 10, (iter // 10) % 10,
        print_wl=DEBUG_PRINT)

    if game_over and (won_loss == 'loss'):
        lost_on_first_click += 1

    while not game_over:

        if DEBUG_PRINT: print(('-' * 40) + '\n\n')

        usable_values = g.known * r(con(1 - g.known, g.gaussian_kernal)) * g.values

        # scale all values so they are in the range [0,1]
        #   note: max value of any cell in minesweeper is 8
        usable_values = usable_values.astype(np.float16) / 8.

        guessable_mask = r(con(g.known, g.gaussian_kernal)) * (1 - g.known)

        padding_size = (GUESS_SIZE - 1) // 2

        usable_values_padded = np.pad(usable_values, (padding_size,))
        known_values_padded = np.pad(g.known, (padding_size,))

        # this channel will be
        # 1 -> outside board
        # 0 -> inside board
        border_channel = np.zeros((N, N))
        border_channel = np.pad(border_channel, (padding_size,), constant_values=1)

        final_board = np.stack([usable_values_padded, border_channel, known_values_padded], axis=2)

        if DEBUG_PRINT:
            print("input_data")
            print(final_board)

        click_x = -1
        click_y = -1
        found_100_percent_prob = False
        probs = np.zeros((10, 10))
        for cx, cy in zip(*np.where(guessable_mask == 1)):

            # get section of padded board centered at (cx,cy) in padded board
            board_section = final_board[cx: cx + GUESS_SIZE, cy: cy + GUESS_SIZE, :]
            board_section = board_section.reshape((1, GUESS_SIZE, GUESS_SIZE, 3))

            board_hash = array_hash(board_section)

            bomb_prob = 0
            if board_hash in cache:
                bomb_prob = cache[board_hash]
            else:
                bomb_prob = model.predict(board_section, verbose=0)
                cache[board_hash] = bomb_prob

            probs[cx, cy] = bomb_prob

            if (not COMPUTE_FULL_PROB_MAP) and bomb_prob[0, 0] > PROB_ONE_THRESHOLD:
                found_100_percent_prob = True
                click_x = cx
                click_y = cy
                break

        np.set_printoptions(precision=2, threshold=np.inf, suppress=True)

        num_moves += 1

        highest_prob = float("-inf")
        if not found_100_percent_prob:
            for x, y in zip(*np.where(guessable_mask == 1)):
                if highest_prob < probs[x, y]:
                    if DEBUG_PRINT:
                        print(f'found {(x, y)} has higher prob {probs[x, y]}')
                    highest_prob = probs[x, y]
                    click_x = x
                    click_y = y

        if DEBUG_PRINT:
            print('clicking')
            g.print_board(marker=(click_x, click_y))
        game_over, won_loss = g.click(click_x, click_y, print_wl=False)

    if won_loss == 'win':
        wins += 1
    else:
        losing_nonfirstclick_games.append(SEEDS_OFFSET + iter)

print("{:.2f}% win rate, {} avg moves per game, {:.2f}% lost on first click".format(
    100.0 * wins / NUM_GAMES,
    num_moves / NUM_GAMES,
    100.0 * lost_on_first_click / NUM_GAMES
))

print(f"losing non-first-click loss games: {losing_nonfirstclick_games}")
