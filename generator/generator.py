import random
from tqdm import tqdm
from game import Game
from utils.file_utils import remove_and_recreate_directory
import numpy as np

class GameDataGenerator:
    def __init__(self, num_games, n, num_bombs):
        self.num_games = num_games
        self.n = n
        self.num_bombs = num_bombs

        self.print_rate = 10000000000
        self.new_game_print_rate = 10000000000

        self.input_arrs = []
        self.output_arrs = []
        self.previously_seen_hashes = set()

    def get_start_point(self, game: Game):
        sx = random.randint(0, self.n - 1)
        sy = random.randint(0, self.n - 1)
        while game.bomb[sx][sy] == 1:
            sx = random.randint(0, self.n - 1)
            sy = random.randint(0, self.n - 1)
        return sx, sy

    def do_next_move(self, g: Game):
        # store input output representation if we have not this game board before
        #  ** calls get_input_output_representation **
        g_hash = g.get_hash()
        if g_hash in self.previously_seen_hashes:
            return
        input_arr, output_arr = g.get_input_output_representation()
        self.input_arrs.append(input_arr)
        self.output_arrs.append(output_arr)
        self.previously_seen_hashes.add(g_hash)

        if (len(self.input_arrs) + 1) % self.print_rate == 0:
            print(f"unique games: {len(self.input_arrs)}")
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
                self.do_next_move(next_g)

    def generate_game_data(self):
        num_moves = 0
        for game_num in tqdm(range(self.num_games), desc='Playing Games', ncols=100):
            g = Game(N=self.n, n_bombs=self.num_bombs, seed=game_num)
            click_x, click_y = self.get_start_point(g)
            game_over, won_loss = g.click(click_x, click_y, print_wl=False)
            num_moves += 1

            if (game_num + 1) % self.new_game_print_rate == 0:
                print(f"new game #{game_num}")
                g.print_board()

            self.do_next_move(g)

    def write_results_to_numpy(self, data_dir):
        numpy_output_dir = f"{data_dir}/numpy"

        remove_and_recreate_directory(f"{data_dir}/input")
        remove_and_recreate_directory(f"{data_dir}/output")
        remove_and_recreate_directory(numpy_output_dir)

        input_np_array = np.array(self.input_arrs)
        output_np_array = np.array(self.output_arrs)

        print(f'input_np_array.shape = {input_np_array.shape}')
        print(f'output_np_array.shape = {output_np_array.shape}')

        np.save(f'{numpy_output_dir}/input.npy', input_np_array)
        np.save(f'{numpy_output_dir}/output.npy', output_np_array)
