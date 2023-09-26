from keras import Input, Model
from keras.src.layers import Dropout, Flatten, Dense, Reshape, BatchNormalization, MaxPooling2D
from keras.src.regularizers import l2
from keras.layers import Conv2D

GUESS_SIZE = 9
BATCH_SIZE = 32
EPOCHES = 30
N = 10

PERCENT_TRAIN = 0.9
TOTAL_DATA_OVERRIDE = 10_000_000

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

# 72.00% win rate, 10_000_000 data points, 9 epoches, patience=3
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

# v12.0: 62.00% win rate, 12.9 avg moves per game, 9.00% lost on first click, 10_000_000 data points, 8 epochs,
#       patience=4, min_delta=0.002
# v12.1: 73.00% win rate, 14.3 avg moves per game, 9.00% lost on first click, 10_000_000 data points, 11 epochs,
#       patience=4, min_delta=0
def get_model_v12():
    inp = Input(shape=(GUESS_SIZE, GUESS_SIZE, 3))

    # Convolutional layers
    c = Conv2D(40, (5, 5), activation='relu', padding='valid')(inp)
    c = Conv2D(40, (3, 3), activation='relu', padding='valid')(c)
    c = Dropout(0.5)(c)

    # Flatten the feature maps
    f = Flatten()(c)

    # Fully connected layers
    d = Dense(200, activation='relu', kernel_regularizer=l2(0.01))(f)
    d = Dropout(0.5)(d)
    d = Dense(100, activation='relu', kernel_regularizer=l2(0.01))(d)
    d = Dropout(0.5)(d)
    d = Dense(50, activation='relu', kernel_regularizer=l2(0.01))(d)
    d = Dropout(0.5)(d)
    d = Dense(1, activation='sigmoid')(d)

    model = Model(inputs=inp, outputs=d)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model

# v13.0: 68.00% win rate, 13.09 avg moves per game, 9.00% lost on first click, 10_000_000 data points, 22 epochs
#       patience=4, min_delta=0
# v13.1: 66.00% win rate, 13.59 avg moves per game, 9.00% lost on first click, > 20 epochs, 4 patience
def get_model_v13():
    inp = Input(shape=(GUESS_SIZE, GUESS_SIZE, 3))

    # Convolutional layers
    c = Conv2D(32, (5, 5), activation='relu', padding='valid')(inp)
    c = Conv2D(64, (3, 3), activation='relu', padding='valid')(c)
    c = Dropout(0.5)(c)

    # Flatten the feature maps
    f = Flatten()(c)

    # Fully connected layers
    d = Dense(200, activation='relu', kernel_regularizer=l2(0.01))(f)
    d = Dropout(0.5)(d)
    d = Dense(200, activation='relu', kernel_regularizer=l2(0.01))(d)
    d = Dropout(0.5)(d)
    d = Dense(100, activation='relu', kernel_regularizer=l2(0.01))(d)
    d = Dropout(0.5)(d)
    d = Dense(1, activation='sigmoid')(d)

    model = Model(inputs=inp, outputs=d)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model


# v14.0: 79.00% win rate, 14.57 avg moves per game, 9.00% lost on first click, 2_000_000 data points,
#       12 epochs
def get_model_v14():
    inp = Input(shape=(GUESS_SIZE, GUESS_SIZE, 3))

    # Convolutional layers
    c = Conv2D(32, (5, 5), activation='relu', padding='valid')(inp)
    c = Conv2D(64, (3, 3), activation='relu', padding='valid')(c)
    c = Dropout(0.5)(c)

    # Flatten the feature maps
    f = Flatten()(c)

    # Fully connected layers
    d = Dense(200, activation='relu', kernel_regularizer=l2(0.01))(f)
    d = Dropout(0.5)(d)
    d = Dense(200, activation='relu', kernel_regularizer=l2(0.01))(d)
    d = Dropout(0.5)(d)
    d = Dense(100, activation='relu', kernel_regularizer=l2(0.01))(d)
    d = Dropout(0.5)(d)
    d = Dense(1, activation='sigmoid')(d)

    model = Model(inputs=inp, outputs=d)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# ** increase 5x5 kernels
# v15.0: 85.00% win rate, 15.42 avg moves per game, 9.00% lost on first click, 2_000_000 data points,
#       30 epochs
def get_model_v15():
    inp = Input(shape=(GUESS_SIZE, GUESS_SIZE, 3))

    # Convolutional layers
    c = Conv2D(53, (5, 5), activation='relu', padding='valid')(inp)
    c = Conv2D(64, (3, 3), activation='relu', padding='valid')(c)
    c = Dropout(0.5)(c)

    # Flatten the feature maps
    f = Flatten()(c)

    # Fully connected layers
    d = Dense(200, activation='relu', kernel_regularizer=l2(0.01))(f)
    d = Dropout(0.5)(d)
    d = Dense(100, activation='relu', kernel_regularizer=l2(0.01))(d)
    d = Dropout(0.5)(d)
    d = Dense(1, activation='sigmoid')(d)

    model = Model(inputs=inp, outputs=d)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

