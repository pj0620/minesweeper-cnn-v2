import xxhash as xxhash

from game import Game
from PIL import Image
import numpy as np
import os
import shutil
import random
from tqdm import tqdm

NUM_GAMES = 200000
N = 10
N_BOMBS = 10

# how often should it print board
PRINT_RATE = 1000000000
NEW_GAME_PRINT_RATE = 1000000000

image_count = 0
DATA_DIR = '/Users/pj/Projects/Minesweeper/minesweeper-cnn/ms/data/10_by_10'

input_arrs = []
output_arrs = []
previously_seen_hashes = set()

def remove_and_recreate_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Use shutil.rmtree to remove the directory and its contents
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' removed.")

    # Recreate the directory
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' recreated.")

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

def array_hash(arr):
    return xxhash.xxh64(arr.tobytes()).hexdigest()

def get_hash(g: Game):
    input_arr, output_arr = g.get_input_output_representation()

    in_hash = array_hash(input_arr)
    out_hash = array_hash(output_arr)
    combined_hash = xxhash.xxh64(in_hash + out_hash).hexdigest()

    return combined_hash


def do_next_move(g: Game):
    global input_arrs
    global output_arrs
    global previously_seen_hashes

    # store input output representation if we have not this game board before
    #  ** calls get_input_output_representation **
    g_hash = get_hash(g)
    if g_hash in previously_seen_hashes:
        return
    input_arr, output_arr = g.get_input_output_representation()
    input_arrs.append(input_arr)
    output_arrs.append(output_arr)
    previously_seen_hashes.add(g_hash)

    if (len(input_arrs) + 1) % PRINT_RATE == 0:
        print(f"unique games: {len(input_arrs)}")
        g.print_board()
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

    # print(safe_next_moves)
    # g.print_board(flags=markers)


numpy_output_dir = f"{DATA_DIR}/numpy"

remove_and_recreate_directory(f"{DATA_DIR}/input")
remove_and_recreate_directory(f"{DATA_DIR}/output")
remove_and_recreate_directory(numpy_output_dir)

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

input_np_array = np.array(input_arrs)
output_np_array = np.array(output_arrs)

print(f'input_np_array.shape = {input_np_array.shape}')
print(f'output_np_array.shape = {output_np_array.shape}')

np.save(f'{numpy_output_dir}/input.npy', input_np_array)
np.save(f'{numpy_output_dir}/output.npy', output_np_array)

print("{:.2f}% win rate, {} avg moves per game".format(100.0 * wins / NUM_GAMES, num_moves / NUM_GAMES))
