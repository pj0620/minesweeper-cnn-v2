from keras.models import load_model
import numpy as np
from game import Game
import matplotlib.pyplot as plt
from tqdm import tqdm

model_file = '/Users/pj/Projects/Minesweeper/minesweeper-cnn/ms/model/ten_by_ten.h5'

# Load the saved model from the file
model = load_model(model_file)

NUM_GAMES = 10
# NUM_GAMES = 2
N = 10
n_bombs = 10
DEBUG_PRINT = False

wins = 0
num_moves = 0
for iter in tqdm(range(NUM_GAMES), desc='Playing Games', ncols=100):
# for iter in range(NUM_GAMES):
    if iter % 1000 == 0: print(f"iter: {iter}")

    g = Game(N=N, n_bombs=n_bombs, seed=iter)
    game_over, won_loss = g.click(0, 0, print_wl=DEBUG_PRINT)

    while not game_over:
        if DEBUG_PRINT: print(('-' * 40) + '\n\n')

        game_input, correct_output = g.get_input_output_representation()

        input_data = game_input.reshape(1, 10, 10, 2)

        if DEBUG_PRINT:
            print("input_data")
            print(game_input)
        predicted_output = model.predict(input_data, verbose=0)[0, :, :, 0]

        np.set_printoptions(precision=2, threshold=np.inf, suppress=True)

        if DEBUG_PRINT:
            print("predicted_output")
            print(predicted_output)

        num_moves += 1

        click_x = -1
        click_y = -1
        highest_prob = float("-inf")
        if DEBUG_PRINT: g.print_guessable_mask()
        for x in range(0, g.N):
            for y in range(0, g.N):
                if int(g.guessable_mask[x][y]) == 0:
                    continue

                if highest_prob < predicted_output[x, y]:
                    if DEBUG_PRINT: print(f'found {(x, y)} has higher prob {predicted_output[x, y]}')
                    highest_prob = predicted_output[x, y]
                    click_x = x
                    click_y = y

        if DEBUG_PRINT:
            print('clicking')
            g.print_board(marker=(click_x, click_y))
        game_over, won_loss = g.click(click_x, click_y, print_wl=False)

    if won_loss == 'win':
        wins += 1

print("{:.2f}% win rate, {} avg moves per game".format(100.0 * wins / NUM_GAMES, num_moves / NUM_GAMES))
