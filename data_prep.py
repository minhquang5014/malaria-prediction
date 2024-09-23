import tensorflow_datasets as tfds
from tensorflow.image import resize, rot90, random_flip_left_right, random_flip_up_down
import tensorflow as tf
import tensorflow_probability as tfp

def sample_data():
    dataset, dataset_info = tfds.load(
        'malaria',
        with_info=True,
        as_supervised=True,
        shuffle_files=True,
        split=['train'])
    return dataset, dataset_info

def splits(dataset, train=0.9, val=0.05, test=0.05):
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

@tf.function
def augment(image, label):
    image, label = resizing_and_rescaling(image, label)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32))
    image = random_flip_left_right(image)
    image = random_flip_up_down(image)
    return image, label

def calculate_lamda(concentration1=0.2, concentration0=0.2):
    lamda = tfp.distributions.Beta(concentration1, concentration0).sample(1)[0]
    return lamda

def mixup(train_dataset_1, train_dataset_2):
    """Mixup data augmentation - this function is for mixing up tha dataset with a weight"""
    (image_1, label_1), (image_2, label_2) = train_dataset_1, train_dataset_2
    lamda = calculate_lamda()
    image = lamda*image_1 + (1-lamda) * image_2
    label = lamda * tf.cast(label_1, dtype=tf.float32) + (1-lamda)*tf.case(label_2, dtype=tf.float32)
    return image, label

def box(IMG_SIZE=224):
    """Creating the bbox for cropping a region of interest in an image 
    and padding it to another image"""
    x = tf.cast(tfp.distributions.Uniform(0, IMG_SIZE).sample(1)[0], dtype=tf.int32)
    y = tf.cast(tfp.distributions.Uniform(0, IMG_SIZE).sample(1)[0], dtype=tf.int32)
    lamda = calculate_lamda()
    w = tf.cast(IMG_SIZE * tf.math.sqrt(1-lamda))
    h = tf.cast(IMG_SIZE * tf.math.sqrt(1-lamda))

    x = tf.clip_by_value(x - w, 0, IMG_SIZE)
    y = tf.clip_by_value(y - h, 0, IMG_SIZE)

    bottom_x = tf.clip_by_value(x + w, 0, IMG_SIZE)
    bottom_y = tf.clip_by_value(y + h, 0, IMG_SIZE)

    w = bottom_x - x
    if w == 0:
        w = 1
    h = bottom_y - y
    if h == 0:
        h = 1 
    return y, x, h, w

# dataset, dataset_info = sample_data()
# train_dataset, val_dataset, test_dataset = splits(dataset[0])
# print(dataset_info)



