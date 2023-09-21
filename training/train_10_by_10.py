from keras import Input, Model
from keras.layers import Conv2D
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.src.layers import Dropout, Flatten, Dense, Reshape, BatchNormalization
from tqdm import tqdm
from game import Game
import os

N = 10
BATCH_SIZE = 32
EPOCHES = 10

PERCENT_TRAIN = 0.8
NUM_FILES_OVERRIDE = 10000

LOAD_FROM_PNGS = False

DATA_DIR = '/Users/pj/Projects/Minesweeper/minesweeper-cnn/ms/data/10_by_10'

def get_model_v1():
    inp = Input(shape=(N, N, 2))

    # Convolutional layers
    c1 = Conv2D(16, (5, 5), activation='relu', padding='same', input_shape=(10, 10, 1))(inp)
    # b1 = BatchNormalization()(c1)
    # p1 = MaxPooling2D((2, 2))(b1)
    d1 = Dropout(0.25)(c1)  # added dropout layer

    # c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(d1)  # use output from dropout layer
    # b2 = BatchNormalization()(c2)
    # p2 = MaxPooling2D((2, 2))(b2)
    # d2 = Dropout(0.25)(p2)  # added dropout layer

    # Flatten the feature maps
    f = Flatten()(d1)  # use output from dropout layer

    # Fully connected layer
    # d = Dense(50, activation='relu')(f)
    # nd = BatchNormalization()(d)
    # d3 = Dropout(0.25)(d)  # added dropout layer

    # Output layer with 10x10 output
    # d4 = Dense(100, activation='sigmoid')(d3)  # use output from dropout layer
    df = Dense(100, activation='sigmoid')(f)  # use output from dropout layer
    output = Reshape((N, N, 1))(df)

    # output = d2

    model = Model(inputs=inp, outputs=output)

    model.compile(optimizer='adam', loss='mean_squared_error')
    # model.compile(optimizer='adam', loss='binary_crossentropy')

    return model

def get_model_v2():
    inp = Input(shape=(N, N, 2))

    # Convolutional layers
    c1 = Conv2D(10, (5, 5), activation='relu', padding='same', input_shape=(10, 10, 1))(inp)
    # b1 = BatchNormalization()(c1)
    # p1 = MaxPooling2D((2, 2))(b1)
    # d1 = Dropout(0.25)(b1)  # added dropout layer

    c2 = Conv2D(10, (3, 3), activation='relu', padding='same')(c1)  # use output from dropout layer
    # b2 = BatchNormalization()(c2)
    # p2 = MaxPooling2D((2, 2))(b2)
    # d2 = Dropout(0.25)(p2)  # added dropout layer

    # Flatten the feature maps
    f = Flatten()(c2)  # use output from dropout layer

    # Fully connected layer
    # d = Dense(50, activation='relu')(f)
    # nd = BatchNormalization()(d)
    # d3 = Dropout(0.25)(nd)  # added dropout layer

    # Output layer with 10x10 output
    # d4 = Dense(100, activation='sigmoid')(d3)  # use output from dropout layer
    df = Dense(100, activation='sigmoid')(f)  # use output from dropout layer
    output = Reshape((N, N, 1))(df)

    # output = d2

    model = Model(inputs=inp, outputs=output)

    model.compile(optimizer='adam', loss='mean_squared_error')
    # model.compile(optimizer='adam', loss='binary_crossentropy')

    return model


def get_data_from_pngs():
    # Set the data directories for input and output images
    input_directory = '/Users/pj/Projects/Minesweeper/minesweeper-cnn/ms/data/10_by_10/input'
    output_directory = '/Users/pj/Projects/Minesweeper/minesweeper-cnn/ms/data/10_by_10/output'

    # Create empty lists to store input and output image data
    input_images = []
    output_images = []

    # Get a list of image filenames from the input directory
    input_filenames = list(os.listdir(input_directory))[:NUM_FILES_OVERRIDE]

    # Loop through the filenames and load the corresponding images
    print(f'found {len(input_filenames)} example images')
    for i, filename in tqdm(enumerate(input_filenames), desc='Loading Images', ncols=100):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        # Load and resize input image
        input_img = Image.open(input_path)
        input_img = input_img.resize((N, N))

        # Load and resize output image
        output_img = Image.open(output_path)
        output_img = output_img.resize((N, N))

        # Convert images to NumPy arrays
        input_array = np.array(input_img)
        output_array = np.array(output_img)

        # Normalize pixel values to [0, 1] (optional)
        input_array = input_array / float(255 // 8)
        output_array = output_array / 255.0

        # Append the arrays to the respective lists
        input_images.append(input_array)
        output_images.append(output_array)

    input_images = np.array(input_images)
    output_images = np.array(output_images)

    # Convert the lists to NumPy arrays
    # Generate a permutation of indices
    indices = np.random.permutation(len(input_images))

    # Apply this permutation to input_images and output_images
    input_images = input_images[indices]
    output_images = output_images[indices]

    # Now split the data into training/validation sets
    train_split = int(len(input_images) * PERCENT_TRAIN)

    train_input_images, test_input_images = input_images[:train_split], input_images[train_split:]
    train_output_images, test_output_images = output_images[:train_split], output_images[train_split:]

    NUMPY_DIR = '/Users/pj/Projects/Minesweeper/minesweeper-cnn/ms/data/10_by_10/numpy'
    np.save(f'${NUMPY_DIR}/train_input_images.npy', train_input_images)
    np.save(f'${NUMPY_DIR}/test_input_images.npy', test_input_images)
    np.save(f'${NUMPY_DIR}/train_output_images.npy', train_output_images)
    np.save(f'${NUMPY_DIR}/test_output_images.npy', test_output_images)

    return train_input_images, test_input_images, train_output_images, test_output_images


def get_data_from_saved_np():
    NUMPY_DIR = f'{DATA_DIR}/numpy'

    inputs = np.load(f'{NUMPY_DIR}/input.npy')
    outputs = np.load(f'{NUMPY_DIR}/output.npy')

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

    train_outputs = np.expand_dims(train_outputs, axis=-1)
    test_outputs = np.expand_dims(test_outputs, axis=-1)

    return train_inputs, test_inputs, \
           train_outputs, test_outputs


# get_data_from_saved_np()
# exit()
train_input_images, test_input_images, train_output_images, test_output_images = None, None, None, None
if LOAD_FROM_PNGS:
    train_input_images, test_input_images, train_output_images, test_output_images = get_data_from_pngs()
else:
    train_input_images, test_input_images, train_output_images, test_output_images = get_data_from_saved_np()

# train_input_images = train_input_images[:10000]
# test_input_images = test_input_images[:10000]
# train_output_images = train_output_images[:10000]
# test_output_images = test_output_images[:10000]

model = get_model_v2()
model.summary()

# Set the number of steps per epoch and validation steps
# These values depend on the size of your dataset and batch size
steps_per_epoch = int(len(train_input_images) * PERCENT_TRAIN) // BATCH_SIZE
validation_steps = int(len(test_input_images) * (1.0 - PERCENT_TRAIN)) // BATCH_SIZE

# Train the model using the combined generator for both input and output
history = model.fit(
    x=train_input_images,
    y=train_output_images,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHES,
    validation_data=(test_input_images, test_output_images),
    validation_steps=validation_steps
)

# save model
model_filename = '/Users/pj/Projects/Minesweeper/minesweeper-cnn/ms/model/ten_by_ten.h5'
model.save(model_filename)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.show()

g = Game(N=N, seed=13)
g.click(0, 0)

g.print_board()

game_input, correct_output = g.get_input_output_representation()

print("shape(game_input): " + str(game_input.shape))

input_data = np.expand_dims(game_input, axis=0)
predicted_output = model.predict(input_data)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(correct_output, cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(predicted_output[0], cmap='gray')
plt.show()
