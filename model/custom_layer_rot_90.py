from tensorflow.keras import Layer
import tensorflow as tf
class RotNinty(Layer):
    def __init__(self):
        super().__init__()
    
    @tf.function
    def call(self, image):
        return tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32))
    