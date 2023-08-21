import xxhash as xxhash

from game import Game
from PIL import Image
import numpy as np
import os
import shutil
import random
from tqdm import tqdm

NUM_GAMES = 1
N = 10
N_BOMBS = 10

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

def array_hash(arr):
    return xxhash.xxh64(arr.tobytes()).hexdigest()

def add_if_unique(g: Game):
    global input_arrs
    global output_arrs
    global previously_seen_hashes

    input_arr, output_arr = g.get_input_output_representation()

    in_hash = array_hash(input_arr)
    out_hash = array_hash(output_arr)
    combined_hash = xxhash.xxh64(in_hash + out_hash).hexdigest()

    print('previously_seen_hashes: ' + str(len(previously_seen_hashes)))

    if combined_hash not in previously_seen_hashes:
        input_arrs.append(input_arr)
        output_arrs.append(output_arr)
        previously_seen_hashes.add(combined_hash)


def do_next_move(g: Game):
    # store input output representation if we have not this game board before
    #  ** calls get_input_output_representation **
    add_if_unique(g)

    g.print_board()
    safe_next_moves = g.get_safe_clicks_in_guessable_region()

    markers = [(x, y) for x,y in zip(safe_next_moves[0], safe_next_moves[1])]
    print(safe_next_moves)
    g.print_board(flags=markers)



numpy_output_dir = f"{DATA_DIR}/numpy"

remove_and_recreate_directory(f"{DATA_DIR}/input")
remove_and_recreate_directory(f"{DATA_DIR}/output")
remove_and_recreate_directory(numpy_output_dir)

PRINT_BOARD = False

wins = 0
num_moves = 0
for game_num in tqdm(range(NUM_GAMES), desc='Playing Games', ncols=100):
    g = Game(N=N, n_bombs=N_BOMBS, seed=iter)
    click_x, click_y = get_start_point(g)
    game_over, won_loss = g.click(click_x, click_y, print_wl=False)
    num_moves += 1

    # get list of unvisited nodes
    # todo: optimize
    # unvisited = []
    # for x in range(0, N):
    #     for y in range(0, N):
    #         # don't add bomb locations to unvisited cells
    #         if g.bomb[x][y] == 1:
    #             continue
    #         unvisited.append((x, y))
    # # remove start click
    # unvisited.remove((click_x, click_y))

    do_next_move(g)

input_np_array = np.array(input_arrs)
output_np_array = np.array(output_arrs)

print(f'input_np_array.shape = {input_np_array.shape}')
print(f'output_np_array.shape = {output_np_array.shape}')

np.save(f'{numpy_output_dir}/input.npy', input_np_array)
np.save(f'{numpy_output_dir}/output.npy', output_np_array)

print("{:.2f}% win rate, {} avg moves per game".format(100.0 * wins / NUM_GAMES, num_moves / NUM_GAMES))
