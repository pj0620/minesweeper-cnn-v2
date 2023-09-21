from keras import Input, Model
from keras.layers import Conv2D
import matplotlib.pyplot as plt
import numpy as np
from keras.src.layers import Dropout, Flatten, Dense, Reshape, BatchNormalization, MaxPooling2D
from keras.src.regularizers import l2
from scipy.signal import convolve2d
from tensorflow.python.keras import callbacks

from game import Game
import os

GUESS_SIZE = 9
BATCH_SIZE = 32
EPOCHES = 30
N = 10

PERCENT_TRAIN = 0.9
TOTAL_DATA_OVERRIDE = 10_000_000

TEST_GAME_SEED = 102

LOAD_FROM_PNGS = False

DATA_DIR = '/Users/pj/Projects/Minesweeper/minesweeper-cnn/ms/data/10_by_10_rcnn'
MODEL_DIR = '/Users/pj/Projects/Minesweeper/minesweeper-cnn/ms/model'
MODEL_FILE = 'ten_by_ten_rcnn.h5'

def verify_write_permissions(dir: str):
    path = dir + '/test.txt'

    # Create an empty file
    with open(path, 'w') as file:
        pass

    # Delete the file
    os.remove(path)

    print('verified user has access to create files in: ' + dir)

def get_model_v1():
    inp = Input(shape=(GUESS_SIZE, GUESS_SIZE, 3))

    # Convolutional layers
    c1 = Conv2D(16, (5, 5), activation='relu', padding='same', input_shape=(10, 10, 1))(inp)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(10, 10, 1))(c1)
    # b1 = BatchNormalization()(c1)
    # p1 = MaxPooling2D((2, 2))(b1)
    d1 = Dropout(0.25)(c2)  # added dropout layer

    # Flatten the feature maps
    f = Flatten()(d1)  # use output from dropout layer

    # Output layer with 10x10 output
    d1 = Dense(100, activation='sigmoid')(f)
    d1d = Dropout(0.25)(d1)
    d2 = Dense(50, activation='sigmoid')(d1d)
    d2d = Dropout(0.25)(d2)
    df = Dense(1, activation='sigmoid')(d2d)

    model = Model(inputs=inp, outputs=df)

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def get_model_v2():
    inp = Input(shape=(GUESS_SIZE, GUESS_SIZE, 3))

    # Convolutional layers
    c1 = Conv2D(32, (5, 5), activation='relu', padding='same')(inp)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    # b1 = BatchNormalization()(c1)
    # p1 = MaxPooling2D((2, 2))(b1)
    d1 = Dropout(0.25)(c2)  # added dropout layer

    # Flatten the feature maps
    f = Flatten()(d1)  # use output from dropout layer

    # Output layer with 10x10 output
    d1 = Dense(100, activation='sigmoid')(f)
    d1d = Dropout(0.25)(d1)
    d2 = Dense(50, activation='sigmoid')(d1d)
    d2d = Dropout(0.25)(d2)
    df = Dense(1, activation='sigmoid')(d2d)

    model = Model(inputs=inp, outputs=df)

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# 78%, training data 9/1/2023 13:37 ET
def get_model_v3():
    inp = Input(shape=(GUESS_SIZE, GUESS_SIZE, 3))

    # Convolutional layers
    c1 = Conv2D(32, (5, 5), activation='relu', padding='same')(inp)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    # b1 = BatchNormalization()(c1)
    # p1 = MaxPooling2D((2, 2))(b1)
    d1 = Dropout(0.25)(c2)  # added dropout layer

    # Flatten the feature maps
    f = Flatten()(d1)  # use output from dropout layer

    # Output layer with 10x10 output
    d1 = Dense(100, activation='relu')(f)
    d1d = Dropout(0.25)(d1)
    d2 = Dense(50, activation='relu')(d1d)
    d2d = Dropout(0.25)(d2)
    df = Dense(1, activation='sigmoid')(d2d)

    model = Model(inputs=inp, outputs=df)

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def get_model_v4():
    inp = Input(shape=(GUESS_SIZE, GUESS_SIZE, 3))

    # Convolutional layers
    c1 = Conv2D(30, (5, 5), activation='relu', padding='same')(inp)
    c2 = Conv2D(15, (3, 3), activation='relu', padding='same')(c1)
    # b1 = BatchNormalization()(c1)
    # p1 = MaxPooling2D((2, 2))(b1)
    d1 = Dropout(0.25)(c2)  # added dropout layer

    # Flatten the feature maps
    f = Flatten()(d1)  # use output from dropout layer

    # Output layer with 10x10 output
    d1 = Dense(100, activation='relu')(f)
    d1d = Dropout(0.25)(d1)
    d2 = Dense(50, activation='relu')(d1d)
    d2d = Dropout(0.25)(d2)
    df = Dense(1, activation='sigmoid')(d2d)

    model = Model(inputs=inp, outputs=df)

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# 76% win rate, 1_000_000 data points
# 78% win rate, 10_000_000 data points, 10 epochs
# 77% win rate, 10_000_000 data points, 15 epochs
# 76% win rate, 20_000_000 data points, 10 epochs
# 78% win rate, 20_000_000 data points, 15 epochs
def get_model_v5():
    inp = Input(shape=(GUESS_SIZE, GUESS_SIZE, 3))

    # Convolutional layers
    c1 = Conv2D(35, (3,  3), activation='relu', padding='same')(inp)
    # c2 = Conv2D(15, (3, 3), activation='relu', padding='same')(c1)
    # b1 = BatchNormalization()(c1)
    # p1 = MaxPooling2D((2, 2))(b1)
    d1 = Dropout(0.25)(c1)  # added dropout layer

    # Flatten the feature maps
    f = Flatten()(d1)  # use output from dropout layer

    # Output layer with 10x10 output
    d1 = Dense(50, activation='relu')(f)
    d1d = Dropout(0.25)(d1)
    d2 = Dense(25, activation='relu')(d1d)
    d2d = Dropout(0.25)(d2)
    df = Dense(1, activation='sigmoid')(d2d)

    model = Model(inputs=inp, outputs=df)

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def get_model_v6():
    inp = Input(shape=(GUESS_SIZE, GUESS_SIZE, 3))

    # Convolutional layers
    c1 = Conv2D(35, (3, 3), activation='relu', padding='same')(inp)
    # c2 = Conv2D(15, (3, 3), activation='relu', padding='same')(c1)
    # b1 = BatchNormalization()(c1)
    # p1 = MaxPooling2D((2, 2))(b1)
    d1 = Dropout(0.25)(c1)  # added dropout layer

    # Flatten the feature maps
    f = Flatten()(d1)  # use output from dropout layer

    # Output layer with 10x10 output
    d1 = Dense(50, activation='relu')(f)
    d1d = Dropout(0.25)(d1)
    d2 = Dense(25, activation='relu')(d1d)
    d2d = Dropout(0.25)(d2)
    df = Dense(1, activation='sigmoid')(d2d)

    model = Model(inputs=inp, outputs=df)

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# 80.00% win rate, 1_000_000 data points, 10 epochs
# 75.00% win rate, 10_000_000 data points, 10 epochs
# 79.00% win rate, 10_000_000 data points, 15 epochs
def get_model_v7():
    inp = Input(shape=(GUESS_SIZE, GUESS_SIZE, 3))

    # Convolutional layers
    c1 = Conv2D(32, (5, 5), activation='relu', padding='valid')(inp)
    b1 = BatchNormalization()(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='valid')(b1)
    b2 = BatchNormalization()(c2)
    p1 = MaxPooling2D((2, 2))(b2)
    d0 = Dropout(0.25)(p1)

    # Flatten the feature maps
    f = Flatten()(d0)

    # Fully connected layers
    d1 = Dense(50, activation='relu', kernel_regularizer=l2(0.01))(f)
    d1d = Dropout(0.25)(d1)
    d2 = Dense(25, activation='relu', kernel_regularizer=l2(0.01))(d1d)
    d2d = Dropout(0.25)(d2)
    df = Dense(1, activation='sigmoid')(d2d)

    model = Model(inputs=inp, outputs=df)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# 80.00% win rate, 5_000_000 data points, 14 epochs
def get_model_v8():
    inp = Input(shape=(GUESS_SIZE, GUESS_SIZE, 3))

    # Convolutional layers
    c1 = Conv2D(50, (5, 5), activation='relu', padding='valid')(inp)
    b1 = BatchNormalization()(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='valid')(b1)
    b2 = BatchNormalization()(c2)
    p1 = MaxPooling2D((2, 2))(b2)
    d0 = Dropout(0.25)(p1)

    # Flatten the feature maps
    f = Flatten()(d0)

    # Fully connected layers
    d1 = Dense(75, activation='relu', kernel_regularizer=l2(0.01))(f)
    d1d = Dropout(0.25)(d1)
    d2 = Dense(50, activation='relu', kernel_regularizer=l2(0.01))(d1d)
    d2d = Dropout(0.25)(d2)
    df = Dense(1, activation='sigmoid')(d2d)

    model = Model(inputs=inp, outputs=df)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# 79% win rate, 5_000_000 data points, 16 epochs
# 72.00% win rate, 10_000_000 data points, ~27 epochs
def get_model_v9():
    inp = Input(shape=(GUESS_SIZE, GUESS_SIZE, 3))

    # Convolutional layers
    c1 = Conv2D(32, (5, 5), activation='relu', padding='valid')(inp)
    b1 = BatchNormalization()(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='valid')(b1)
    b2 = BatchNormalization()(c2)
    p1 = MaxPooling2D((2, 2))(b2)
    d0 = Dropout(0.25)(p1)

    # Flatten the feature maps
    f = Flatten()(d0)

    # Fully connected layers
    d = Dense(100, activation='relu', kernel_regularizer=l2(0.01))(f)
    d = Dropout(0.25)(d)
    d = Dense(75, activation='relu', kernel_regularizer=l2(0.01))(d)
    d = Dropout(0.25)(d)
    d = Dense(50, activation='relu', kernel_regularizer=l2(0.01))(d)
    d = Dropout(0.25)(d)
    d = Dense(1, activation='sigmoid')(d)

    model = Model(inputs=inp, outputs=d)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# 79.00% win rate, 5_000_000 data_points, 21 epochs
# 76.00% win rate, 10_000_data_points, 28 epochs
def get_model_v10():
    inp = Input(shape=(GUESS_SIZE, GUESS_SIZE, 3))

    # Convolutional layers
    c1 = Conv2D(32, (5, 5), activation='relu', padding='valid')(inp)
    b1 = BatchNormalization()(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='valid')(b1)
    b2 = BatchNormalization()(c2)
    p1 = MaxPooling2D((2, 2))(b2)
    d0 = Dropout(0.5)(p1)

    # Flatten the feature maps
    f = Flatten()(d0)

    # Fully connected layers
    d = Dense(120, activation='relu', kernel_regularizer=l2(0.01))(f)
    d = Dropout(0.5)(d)
    d = Dense(90, activation='relu', kernel_regularizer=l2(0.01))(d)
    d = Dropout(0.5)(d)
    d = Dense(60, activation='relu', kernel_regularizer=l2(0.01))(d)
    d = Dropout(0.5)(d)
    d = Dense(1, activation='sigmoid')(d)

    model = Model(inputs=inp, outputs=d)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def get_model_v11():
    inp = Input(shape=(GUESS_SIZE, GUESS_SIZE, 3))

    # Convolutional layers
    c1 = Conv2D(40, (5, 5), activation='relu', padding='valid')(inp)
    b1 = BatchNormalization()(c1)
    c2 = Conv2D(40, (3, 3), activation='relu', padding='valid')(b1)
    b2 = BatchNormalization()(c2)
    p1 = MaxPooling2D((2, 2))(b2)
    d0 = Dropout(0.5)(p1)

    # Flatten the feature maps
    f = Flatten()(d0)

    # Fully connected layers
    d = Dense(150, activation='relu', kernel_regularizer=l2(0.01))(f)
    d = Dropout(0.5)(d)
    d = Dense(110, activation='relu', kernel_regularizer=l2(0.01))(d)
    d = Dropout(0.5)(d)
    d = Dense(70, activation='relu', kernel_regularizer=l2(0.01))(d)
    d = Dropout(0.5)(d)
    d = Dense(1, activation='sigmoid')(d)

    model = Model(inputs=inp, outputs=d)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_data_from_saved_np():
    global TOTAL_DATA_OVERRIDE

    numpy_dir = f'{DATA_DIR}/numpy'
    inputs = np.load(f'{numpy_dir}/input.npy')
    outputs = np.load(f'{numpy_dir}/output.npy')

    inputs = inputs[:TOTAL_DATA_OVERRIDE]
    outputs = outputs[:TOTAL_DATA_OVERRIDE]

    # Get the number of samples (size of 0th dimension of the input array)
    num_samples = inputs.shape[0]
    assert inputs.shape[0] == outputs.shape[0]

    shuffled_indices = np.random.permutation(num_samples)
    shuffled_inputs = inputs[shuffled_indices]
    shuffled_outputs = outputs[shuffled_indices]

    num_train_samples = int(num_samples * PERCENT_TRAIN)

    train_inputs = shuffled_inputs[:num_train_samples]
    test_inputs = shuffled_inputs[num_train_samples:]
    train_outputs = shuffled_outputs[:num_train_samples]
    test_outputs = shuffled_outputs[num_train_samples:]

    print(f"total datapoints: {shuffled_inputs.shape}")
    print(f"train set size = {train_inputs.shape}")
    print(f"test set size = {test_inputs.shape}")

    print(f"(min, max) = {(shuffled_inputs.min(), shuffled_inputs.max())}")

    train_outputs = np.expand_dims(train_outputs, axis=-1)
    test_outputs = np.expand_dims(test_outputs, axis=-1)

    return train_inputs, test_inputs, train_outputs, test_outputs


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

train_input_images, test_input_images, train_output_images, test_output_images = get_data_from_saved_np()

model = get_model_v11()
model.summary()

# Set the number of steps per epoch and validation steps
# These values depend on the size of your dataset and batch size
steps_per_epoch = len(train_input_images) // BATCH_SIZE
validation_steps = len(test_input_images) // BATCH_SIZE

earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min", patience=2,
                                        restore_best_weights=True)

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
