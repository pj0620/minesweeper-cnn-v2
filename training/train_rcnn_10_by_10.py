from keras import Input, Model
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from tensorflow.python.keras import callbacks

from data_loader import load_data
from game import Game
import os

from models import *

GUESS_SIZE = 9
BATCH_SIZE = 32
EPOCHES = 30
N = 10

PERCENT_TRAIN = 0.9
TOTAL_DATA_OVERRIDE = 2_000_000
PARTITION_SIZE = 1_000_000

TEST_GAME_SEED = 102

LOAD_FROM_PNGS = False

DATA_DIR = '../data/10_by_10_rcnn'
NUMPY_DIR = f'{DATA_DIR}/numpy'
MODEL_DIR = '../model'
MODEL_FILE = 'ten_by_ten_rcnn.h5'


def verify_write_permissions(dir: str):
    path = dir + '/test.txt'

    # Create an empty file
    with open(path, 'w') as file:
        pass

    # Delete the file
    os.remove(path)

    print('verified user has access to create files in: ' + dir)


def test_one_game(model: Model):
    g = Game(N=10, seed=TEST_GAME_SEED)
    g.click(0, 0)

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

    g.print_board()

    game_input, correct_output = g.get_input_output_representation()

    print("shape(game_input): " + str(game_input.shape))

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(correct_output, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(probs, cmap='gray')
    plt.show()


verify_write_permissions(MODEL_DIR)

train_input_images, test_input_images, train_output_images, test_output_images = load_data(
    NUMPY_DIR,
    TOTAL_DATA_OVERRIDE,
    PARTITION_SIZE,
    PERCENT_TRAIN)

model = get_model_v15()
model.summary()

# Set the number of steps per epoch and validation steps
# These values depend on the size of your dataset and batch size
steps_per_epoch = len(train_input_images) // BATCH_SIZE
validation_steps = len(test_input_images) // BATCH_SIZE

earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min", patience=4,
                                        restore_best_weights=True,
                                        min_delta=0)

# Train the model using the combined generator for both input and output
history = model.fit(
    x=train_input_images,
    y=train_output_images,
    # steps_per_epoch=steps_per_epoch,
    epochs=EPOCHES,
    validation_data=(test_input_images, test_output_images),
    # validation_steps=validation_steps,
    callbacks=[earlystopping]
)

# save model
model.save(MODEL_DIR + '/' + MODEL_FILE)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.show()

test_one_game(model)
