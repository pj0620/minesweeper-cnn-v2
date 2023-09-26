from keras.models import load_model
import numpy as np
from game import Game
from tqdm import tqdm
import random

from scipy.signal import convolve2d

from utils.hash_utils import array_hash

MODEL_DIR = '/Users/pj/Projects/Minesweeper/minesweeper-cnn/ms/model'
MODEL_FILE = 'ten_by_ten_rcnn.h5'

# Load the saved model from the file
model = load_model(MODEL_DIR + '/' + MODEL_FILE)

cache = {}

wins = 0
lost_on_first_click = 0
num_moves = 0
losing_nonfirstclick_games = []

NUM_GAMES = 100
N = 10
SEEDS_OFFSET = 1_000_000_000
n_bombs = 10
DEBUG_PRINT = False
GUESS_SIZE = 9

COMPUTE_FULL_PROB_MAP = False
PROB_ONE_THRESHOLD = 0.99999

r = lambda vect: np.where(vect > 0, 1, vect).astype(np.int16)
con = lambda A, B: convolve2d(A, B, mode='same')


def get_safe_click_probs(final_board: np.array, guessable_mask: np.array) -> np.array:
    global model
    global cache
    probs = np.zeros((10, 10))
    for cx, cy in zip(*np.where(guessable_mask == 1)):

        # get section of padded board centered at (cx,cy) in padded board
        board_section = final_board[cx: cx + GUESS_SIZE, cy: cy + GUESS_SIZE, :]
        board_section = board_section.reshape((1, GUESS_SIZE, GUESS_SIZE, 3))

        board_hash = array_hash(board_section)

        if board_hash in cache:
            bomb_prob = cache[board_hash]
        else:
            bomb_prob = model.predict(board_section, verbose=0)
            cache[board_hash] = bomb_prob

        probs[cx, cy] = bomb_prob

        if COMPUTE_FULL_PROB_MAP and (bomb_prob[0, 0] > PROB_ONE_THRESHOLD):
            break
            
    return probs


def get_safe_click_from_rcnn(final_board: np.array, guessable_mask: np.array):
    probs = get_safe_click_probs(final_board, guessable_mask)

    highest_prob = float("-inf")
    click_x = -1
    click_y = -1
    for x, y in zip(*np.where(guessable_mask == 1)):
        if highest_prob < probs[x, y]:
            if DEBUG_PRINT:
                print(f'found {(x, y)} has higher prob {probs[x, y]}')
            highest_prob = probs[x, y]
            click_x = x
            click_y = y
    
    return click_x, click_y
    

def should_use_guessable_region(known: np.array) -> bool:
    total_unknown_cells = known.sum()
    return total_unknown_cells > 2


def get_click_outside_guessable_region(guessable_mask: np.array, known: np.array):
    not_guess_mask = guessable_mask + known

    safe_next_moves = np.where(not_guess_mask == 0)

    if len(safe_next_moves[0]) == 0:
        return -1, -1

    # only click one for now
    rand_idx = random.randint(0, len(safe_next_moves[0]) - 1)
    return safe_next_moves[0][rand_idx], safe_next_moves[1][rand_idx]


def get_next_move(g: Game):
    global num_moves

    usable_values_mask = g.known * r(con(1 - g.known, g.gaussian_kernal))
    usable_values = usable_values_mask * g.values

    # scale all values so they are in the range [0,1]
    #   note: max value of any cell in minesweeper is 8
    usable_values_scaled = usable_values.astype(np.float16) / 8.

    guessable_mask = r(con(g.known, g.gaussian_kernal)) * (1 - g.known)

    padding_size = (GUESS_SIZE - 1) // 2

    usable_values_padded = np.pad(usable_values_scaled, (padding_size,))
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

    if should_use_guessable_region(known=g.known):
        return get_safe_click_from_rcnn(final_board, guessable_mask)
    else:
        print('\nclicking outside of guessable region')
        click_x, click_y = get_click_outside_guessable_region(guessable_mask, g.known)
        if click_x == -1:
            print('\nno cells outside of guessable region, falling back to guessable region')
            return get_safe_click_from_rcnn(final_board, guessable_mask)
        else:
            return click_x, click_y


np.set_printoptions(precision=2, threshold=np.inf, suppress=True)

for iter in range(NUM_GAMES):
    print(f"Playing Game #{iter}")

    g = Game(N=N, n_bombs=n_bombs, seed=SEEDS_OFFSET + iter)
    game_over, won_loss = g.click(0, 0, print_wl=True)

    if game_over and (won_loss == 'loss'):
        lost_on_first_click += 1

    while not game_over:

        if DEBUG_PRINT: print(('-' * 40) + '\n\n')

        num_moves += 1
        click_x, click_y = get_next_move(g)

        print(f'({click_x},{click_y}),', end='')
        if DEBUG_PRINT:
            g.print_board(marker=(click_x, click_y))
        game_over, won_loss = g.click(click_x, click_y, print_wl=True)

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
