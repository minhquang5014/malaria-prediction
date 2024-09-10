import tensorflow_datasets as tfds

dataset, dataset_info = tfds.load(
    'malaria',
    with_info=True,
    as_supervised=True,
    shuffle_files=True,
    split=['train'])

TRAIN_RATIO = 0.9
VAL = 0.05
TEST = 0.05

def splits(dataset, train, val, test):
    # dataset = tf.data.Dataset.range(40)
    dataset_size = len(dataset)
    # print(list(dataset.as_numpy_iterator()))

    train_dataset = dataset.take(int(train * dataset_size))
    # print(list(train_dataset.as_numpy_iterator()))

    val_dataset = dataset.skip(int(train * dataset_size)).take(int(val * dataset_size))
    # print(list(val_dataset.as_numpy_iterator()))

    test_dataset = dataset.skip(int((train + val) * dataset_size)).take(int(test * dataset_size))
    # print(list(test_dataset.as_numpy_iterator()))
    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL, TEST)