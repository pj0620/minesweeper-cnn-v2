import random
from typing import Callable, Any

import numpy as np
from scipy.signal import convolve2d


class Game:
    def __init__(self, N=10, n_bombs=10, seed=None):
        self.N = N
        self.total_unknown = N * N - N

        # convert the following to use numpy arrays
        self.bomb = np.zeros((N, N), dtype=np.int16)
        self.known = np.zeros((N, N), dtype=np.int16)

        self.gaussian_kernal = np.ones((3, 3))

        # assert that there are no more bombs than there are cells
        assert n_bombs <= N * N

        if seed is not None:
            random.seed(seed)

        placed = 0
        while placed < n_bombs:
            x = random.randint(0, N - 1)
            y = random.randint(0, N - 1)

            if self.bomb[x][y] == 0:
                self.bomb[x][y] = 1
                placed += 1

        self.values = self.compute_values()

    def compute_values(self):
        return convolve2d(self.bomb, self.gaussian_kernal, mode='same').astype(np.int16)

    # input: ch0 -> known values, ch1 known locations
    # output: ch0 -> guessable bombs
    def get_input_output_representation(self):
        r = lambda vect: np.where(vect > 0, 1, vect).astype(np.int16)
        con = lambda A, B: convolve2d(A, B, mode='same')

        self.usable_values = self.known * r(con(1 - self.known, self.gaussian_kernal)) * self.values
        self.guessable_mask = r(con(self.known, self.gaussian_kernal)) * (1 - self.known)
        guessable_safe_locations = self.guessable_mask * (1 - self.bomb)

        return np.dstack((self.usable_values, self.guessable_mask)), guessable_safe_locations

    # ** must be called after get_input_output_representation **
    # clicks safe spot in guessable region
    # returns True -> game over, False -> game not over
    def get_safe_clicks_in_guessable_region(self):
        return np.where((self.guessable_mask == 1) & (self.bomb == 0))

    # returns game_over, won_loss
    def click(self, x, y, print_wl=True):
        # print you lost in red text if user clicked a bomb
        if self.bomb[x][y] == 1:
            if print_wl:
                print("\033[31mYou lost :(\033[0m")
            return True, 'loss'

        # # propagate click according to rules of minesweeper
        self.propagate_click(x, y)

        if self.total_unknown <= 0:
            if print_wl:
                print("\033[32mYou Won!!\033[0m")
            return True, 'win'

        return False, 'null'

    def propagate_click(self, x, y):
        if x < 0 or x >= self.N:
            return
        if y < 0 or y >= self.N:
            return

        # if the cell is already known, do nothing
        if self.known[x][y] == 1:
            return
        self.known[x][y] = 1
        self.total_unknown -= 1

        # if the cell is known to be safe, propagate click
        if self.values[x][y] == 0:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    self.propagate_click(x + i, y + j)

    def copy(self):
        # start off game with no bombs
        new_game = Game(self.N, 0)

        # manually copy over bombs/values from this game
        new_game.bomb = np.copy(self.bomb)
        new_game.values = np.copy(self.values)
        new_game.N = self.N
        new_game.total_unknown = self.total_unknown

        return new_game

    def print_bombs(self):
        self.print_board_values('BOMBS', lambda x, y: f'{self.bomb[x][y]} ')

    def print_known(self):
        self.print_board_values('KNOWN', lambda x, y: f'{self.known[x][y]} ')

    def print_values(self):
        self.print_board_values('VALUES', lambda x, y: f'{self.values[x][y]} ')

    def print_guessable_mask(self):
        self.print_board_values('MASK', lambda x, y: f'{self.guessable_mask[x][y]} ')

    def print_board(self, marker=None, flags=None):
        def print_value(x, y):
            if (marker is not None) and (marker == (x, y)):
                return f"\033[32;5m📍\033[0m"
            elif (flags is not None) and ((x, y) in flags):
                return '🚩'
            elif self.known[x][y] == 0:
                return '🟩'
            elif self.values[x][y] > 0:
                # emojis = "0️⃣1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣8️⃣9️⃣"
                emojis = "０１２３４５６７８９️"
                return str(emojis[int(self.values[x][y])])
            else:
                return '⬛'

        self.print_board_values('BOARD', print_value)

    def print_board_values(self, title: str, operation: Callable[[int, int], Any]) -> None:
        title_space = f' {title} '
        print("#" * (self.N - len(title_space) // 2) + title_space + "#" * (self.N - len(title_space) // 2))
        for x in range(self.N):
            for y in range(self.N):
                print(operation(x, y), end="")
            print("|")
