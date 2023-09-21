import numpy as np
from keras.src.saving.saving_api import load_model
from scipy.signal import convolve2d

from game import Game
from utils.hash_utils import array_hash

N = 10
GUESS_SIZE = 9

model_file = '/model/ten_by_ten_rcnn_v10.h5'

# Load the saved model from the file
model = load_model(model_file)

g = Game()

g.bomb = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
g.values = [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,3,3,4,1,1,0,0,0,0,0,3,3,4,1,2,1,1,0,0,1,4,3,4,1,2,1,1,0,0,1,3,2,2,1,2,2,1,0,0,1,2,1,1,1,1,1,0,0,0]
g.known = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1]

g.bomb = np.array(g.bomb).reshape((10, 10))
g.values = np.array(g.values).reshape((10, 10))
g.known = np.array(g.known).reshape((10, 10))

r = lambda vect: np.where(vect > 0, 1, vect).astype(np.int16)
con = lambda A, B: convolve2d(A, B, mode='same')

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

    bomb_prob = model.predict(board_section, verbose=0)

    probs[cx, cy] = bomb_prob

np.set_printoptions(precision=2, threshold=np.inf, suppress=True)
print('probs')
print(probs)

highest_prob = float("-inf")
if not found_100_percent_prob:
    for x, y in zip(*np.where(guessable_mask == 1)):
        if highest_prob < probs[x, y]:
            print(f'found {(x, y)} has higher prob {probs[x, y]}')
            highest_prob = probs[x, y]
            click_x = x
            click_y = y

# if DEBUG_PRINT:
#     print('clicking')
#     g.print_board(marker=(click_x, click_y))

