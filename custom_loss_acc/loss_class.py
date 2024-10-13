import tensorflow as tf
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, factor):
        super(CustomLoss, self).__init__()
        self.factor = factor
    def call(self, y_true, y_pred):
        return self.factor * tf.keras.losses.BinaryCrossentropy(y_true, y_pred)