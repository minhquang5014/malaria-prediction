from data_prep import sample_data, splits, cutmix, resizing_and_rescaling
from data_visualizer import visualizer, visualize_cutmix
import tensorflow as tf
BATCH_SIZE = 32

# in here, we got the dataset and information of the dataset
dataset, dataset_info = sample_data()

# split the data into train dataset, validation data, and test set
train_dataset, val_dataset, test_dataset = splits(dataset[0])

# visualizer(val_dataset, dataset_info, 25)

"""no need to mix up the data here
train_dataset_1 = train_dataset.shuffle(buffer_size = 4096,).map(resizing_and_rescaling)
train_dataset_2 = train_dataset.shuffle(buffer_size = 4096,).map(resizing_and_rescaling)

mixed_dataset = tf.data.Dataset.zip(train_dataset_1, train_dataset_2)"""

train_dataset = (
    train_dataset
    .shuffle(buffer_size=8, reshuffle_each_iteration=True)
    .map(resizing_and_rescaling)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
val_dataset = (
    val_dataset
    .shuffle(buffer_size=8, reshuffle_each_iteration=True)
    .map(resizing_and_rescaling)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# print(train_dataset)
# print(val_dataset)

