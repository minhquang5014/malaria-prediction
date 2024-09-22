import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, BatchNormalization, Layer

class FeatureExtractor(Layer):
    """This class inherited from Layer. 
    It is mainly for extracting main main features in the image, 
    using conv2d, batch norm, and maxpooling"""
    def __init__(self, filters, kernel_size, strides, padding, activation, pool_size):
        super(FeatureExtractor, self).__init__()
        self.conv1 = Conv2D(filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding, activation=activation)
        self.batch1 = BatchNormalization()
        self.pool1 = MaxPooling2D(pool_size=pool_size, strides=2*strides)
        self.conv2 = Conv2D(filters=2*filters, kernel_size=kernel_size,
                            strides=strides, padding=padding, activation=activation)
        self.batch2 = BatchNormalization()
        self.pool2 = MaxPooling2D(pool_size=pool_size, strides=2*strides)
    def call(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.pool2(x)
        return x
    
class Subclass_model(Model):
    """This class is for dense layer, the final dense layer using the sigmoid function for binary prediction
    https://en.wikipedia.org/wiki/Sigmoid_function
    """
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor(16, 3, 1, "valid", "relu", 2)
        self.flatten = Flatten()
        self.dense1 = Dense(20, activation="relu")
        self.batch1 = BatchNormalization()
        self.dropout = Dropout(0.2)
        self.dense2 = Dense(1, activation="sigmoid")
    def call(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batch1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x
    
sub_model = Subclass_model()
IMG_SIZE = 224
sample_input = tf.random.normal((1, IMG_SIZE, IMG_SIZE, 3))
# _ = sub_model(sample_input)
# sub_model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
output = sub_model(sample_input)
sub_model.summary()