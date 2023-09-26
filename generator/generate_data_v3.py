import xxhash as xxhash

from game import Game
from PIL import Image
import numpy as np
import os
import shutil
import random
from tqdm import tqdm
from scipy.signal import convolve2d

from utils.file_utils import remove_and_recreate_directory
from utils.hash_utils import array_hash

NUM_GAMES = 500000
N = 10
N_BOMBS = 10

GUESS_SIZE = 9
assert GUESS_SIZE % 2 == 1

# how often should it print board
PRINT_RATE = 1000000000
NEW_GAME_PRINT_RATE = 1000000000

PARTITION_SIZE = 1_000_000

image_count = 0
DATA_DIR = '../data/10_by_10_rcnn'
NUMPY_DATA_DIR = f"{DATA_DIR}/numpy"

one_output_input_arrs = []
zero_output_input_arrs = []
previously_seen_hashes = set()
previously_seen_g_hashes = set()

def save_input_output_images(game, idx):
    input_arr, output_arr = game.get_input_output_representation()

    input_img = Image.fromarray(32 * input_arr.astype(np.uint8))
    output_img = Image.fromarray(255 * output_arr.astype(np.uint8))

    # Save the image to a file
    input_img.save(f'{DATA_DIR}/input/image_{idx}.png')
    output_img.save(f'{DATA_DIR}/output/image_{idx}.png')

def get_start_point(game):
    sx = random.randint(0, N - 1)
    sy = random.randint(0, N - 1)
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

def get_hash(g: Game):
    input_arr, output_arr = g.get_input_output_representation()

    in_hash = array_hash(input_arr)
    out_hash = array_hash(output_arr)
    combined_hash = xxhash.xxh64(in_hash + out_hash).hexdigest()

    return combined_hash

def get_samples_of_board(game: Game):
    global previously_seen_g_hashes

    r = lambda vect: np.where(vect > 0, 1, vect).astype(np.int16)
    con = lambda A, B: convolve2d(A, B, mode='same')

    usable_values = game.known * r(con(1 - game.known, game.gaussian_kernal)) * game.values

    # scale all values so they are in the range [0,1]
    #   note: max value of any cell in minesweeper is 8
    usable_values = usable_values.astype(np.float16) / 8.

    guessable_mask = r(con(game.known, game.gaussian_kernal)) * (1 - game.known)

    padding_size = (GUESS_SIZE - 1) // 2

    usable_values_padded = np.pad(usable_values, (padding_size,))
    known_values_padded = np.pad(g.known, (padding_size,))

    # this channel will be
    # 1 -> outside board
    # 0 -> inside board
    border_channel = np.zeros((N, N))
    border_channel = np.pad(border_channel, (padding_size,), constant_values=1)

    final_board = np.stack([usable_values_padded, border_channel, known_values_padded], axis=2)

    one_output_inputs_game = []
    zero_output_inputs_game = []
    for cx, cy in zip(*np.where(guessable_mask == 1)):

        # get section of padded board centered at (cx,cy) in padded board
        board_section = final_board[cx: cx + GUESS_SIZE, cy: cy + GUESS_SIZE, :]

        # is there a bomb at (cx,cy)
        is_bomb = 1 - game.bomb[cx, cy]

        section_hash = array_hash(board_section)
        if section_hash not in previously_seen_g_hashes:
            if is_bomb == 1:
                one_output_inputs_game.append(board_section)
            else:
                zero_output_inputs_game.append(board_section)
            previously_seen_g_hashes.add(section_hash)

    return one_output_inputs_game, zero_output_inputs_game

def do_next_move(g: Game):
    global one_output_input_arrs
    global zero_output_input_arrs
    global previously_seen_hashes

    # store input output representation if we have not this game board before
    g_hash = get_hash(g)
    if g_hash in previously_seen_hashes:
        return
    else:
        previously_seen_hashes.add(g_hash)

    one_output_inputs_game, zero_output_inputs_game = get_samples_of_board(g)
    one_output_input_arrs += one_output_inputs_game
    zero_output_input_arrs += zero_output_inputs_game

    safe_next_moves = g.get_safe_clicks_in_guessable_region()

    if len(safe_next_moves[0]) == 0:
        return

    # only click one for now
    rand_idx = random.randint(0, len(safe_next_moves[0]) - 1)
    next_clicks = [
        (safe_next_moves[0][rand_idx], safe_next_moves[1][rand_idx])
    ]

    for safe_x, safe_y in next_clicks:
        next_g = g

        # big performance hit for >1 sampling. Please look into this
        if len(next_clicks) > 1:
            next_g = g.copy()

        game_over, won_loss = next_g.click(safe_x, safe_y, print_wl=False)

        if game_over:
            continue
        else:
            do_next_move(next_g)


def save_shuffle_partition(one_arrs_partition, zero_arrs_partition, partition_key):
    assert len(one_arrs_partition) == len(zero_arrs_partition)

    total_cases = len(one_arrs_partition)

    ones_np_array = np.array(one_arrs_partition)
    zeros_np_array = np.array(zero_arrs_partition)

    combined_input = np.concatenate([ones_np_array, zeros_np_array], axis=0)

    print(f'{zeros_np_array.shape=}, {zeros_np_array.shape=}, {combined_input.shape=}')

    all_ones = np.ones((total_cases,))
    all_zeros = np.zeros((total_cases,))

    combined_output = np.concatenate([all_ones, all_zeros], axis=0)

    print(f'{all_ones.shape=}, {all_zeros.shape}, {combined_output.shape=}')

    print('shuffling partition')

    shuffled_indices = np.random.permutation(len(combined_output))
    shuffled_inputs = combined_input[shuffled_indices]
    shuffled_outputs = combined_output[shuffled_indices]

    print(f'partition {partition_key}: {len(one_arrs_partition)} one cases')
    print(f'partition {partition_key}: {len(zero_arrs_partition)} zero cases')

    remove_and_recreate_directory(f'{NUMPY_DATA_DIR}/part={partition_key}')

    np.save(f'{NUMPY_DATA_DIR}/part={partition_key}/input.npy', shuffled_inputs)
    np.save(f'{NUMPY_DATA_DIR}/part={partition_key}/output.npy', shuffled_outputs)


PRINT_BOARD = False

wins = 0
num_moves = 0
for game_num in tqdm(range(NUM_GAMES), desc='Playing Games', ncols=100):
    g = Game(N=N, n_bombs=N_BOMBS, seed=game_num)
    click_x, click_y = get_start_point(g)
    game_over, won_loss = g.click(click_x, click_y, print_wl=False)
    num_moves += 1

    if (game_num + 1) % NEW_GAME_PRINT_RATE == 0:
        print(f"new game #{game_num}")
        g.print_board()

    do_next_move(g)

remove_and_recreate_directory(NUMPY_DATA_DIR)

total_final_cases = min(len(one_output_input_arrs), len(zero_output_input_arrs))
print(f'final list has {len(one_output_input_arrs)} one output cases')
print(f'final list has {len(zero_output_input_arrs)} zero output cases')
print(f'final list will be {total_final_cases} output cases')

idx = 0
# needed since we are removing from two arrays
half_partition_size = PARTITION_SIZE // 2
while idx * PARTITION_SIZE < total_final_cases:
    slice_end = min((idx + 1) * half_partition_size, total_final_cases // 2)

    zero_cases = zero_output_input_arrs[idx * half_partition_size: slice_end]
    one_cases = one_output_input_arrs[idx * half_partition_size: slice_end]

    save_shuffle_partition(one_cases, zero_cases, idx * PARTITION_SIZE)

    idx += 1



print("{:.2f}% win rate, {} avg moves per game".format(100.0 * wins / NUM_GAMES, num_moves / NUM_GAMES))
