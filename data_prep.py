import tensorflow_datasets as tfds
from tensorflow.image import resize
import tensorflow as tf
def sample_data():
    dataset, dataset_info = tfds.load(
        'malaria',
        with_info=True,
        as_supervised=True,
        shuffle_files=True,
        split=['train'])
    return dataset, dataset_info

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

def resizing_and_rescaling(image, label, IMG_SIZE=224):
    resized_image = resize(image, (IMG_SIZE, IMG_SIZE))
    resized_image = tf.cast(resized_image, tf.float32)
    normalized_image = resized_image / 255.0
    return normalized_image, label

