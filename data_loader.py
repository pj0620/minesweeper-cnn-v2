import numpy as np


def load_data(data_dir: str, total_samples_to_load: int, partition_size: int, per_train: float):
    partitions_to_load = total_samples_to_load // partition_size

    inputs_all = []
    outputs_all = []
    for part in range(partitions_to_load):
        print(f'loading partition {part}')
        inputs_part = np.load(f'{data_dir}/part={part * partition_size}/input.npy')
        outputs_part = np.load(f'{data_dir}/part={part * partition_size}/output.npy')

        outputs_part = np.expand_dims(outputs_part, axis=-1)

        print(f'{inputs_part.shape=}')
        print(f'{outputs_part.shape=}')

        inputs_all.append(inputs_part)
        outputs_all.append(outputs_part)

    inputs = np.concatenate(inputs_all, axis=0)
    outputs = np.concatenate(outputs_all, axis=0)

    inputs = np.expand_dims(inputs, axis=-1)
    outputs = np.expand_dims(outputs, axis=-1)

    print(f'all data: {inputs.shape=} {outputs.shape=}')

    num_samples = inputs.shape[0]
    num_train_samples = int(num_samples * per_train)

    print(f'{num_samples=} {num_train_samples=}')

    train_inputs = inputs[:num_train_samples]
    test_inputs = inputs[num_train_samples:]
    train_outputs = outputs[:num_train_samples]
    test_outputs = outputs[num_train_samples:]

    print(f"train set size: {train_inputs.shape=} {train_outputs.shape=}")
    print(f"test set size: {test_inputs.shape=} {test_outputs.shape=}")

    print(f"inputs (min, max) = {(inputs.min(), inputs.max())}")

    return train_inputs, test_inputs, train_outputs, test_outputs

# legacy
# def get_data_from_saved_np():
    # global TOTAL_DATA_OVERRIDE
    #
    # numpy_dir = f'{DATA_DIR}/numpy'
    # inputs = np.load(f'{numpy_dir}/input.npy')
    # outputs = np.load(f'{numpy_dir}/output.npy')
    #
    # inputs = inputs[:TOTAL_DATA_OVERRIDE]
    # outputs = outputs[:TOTAL_DATA_OVERRIDE]
    #
    # # Get the number of samples (size of 0th dimension of the input array)
    # num_samples = inputs.shape[0]
    # assert inputs.shape[0] == outputs.shape[0]
    #
    # shuffled_indices = np.random.permutation(num_samples)
    # shuffled_inputs = inputs[shuffled_indices]
    # shuffled_outputs = outputs[shuffled_indices]
    #
    # num_train_samples = int(num_samples * PERCENT_TRAIN)
    #
    # train_inputs = shuffled_inputs[:num_train_samples]
    # test_inputs = shuffled_inputs[num_train_samples:]
    # train_outputs = shuffled_outputs[:num_train_samples]
    # test_outputs = shuffled_outputs[num_train_samples:]
    #
    # print(f"total datapoints: {shuffled_inputs.shape}")
    # print(f"train set size = {train_inputs.shape}")
    # print(f"test set size = {test_inputs.shape}")
    #
    # print(f"(min, max) = {(shuffled_inputs.min(), shuffled_inputs.max())}")
    #
    # train_outputs = np.expand_dims(train_outputs, axis=-1)
    # test_outputs = np.expand_dims(test_outputs, axis=-1)
    #
    # return train_inputs, test_inputs, train_outputs, test_outputs

