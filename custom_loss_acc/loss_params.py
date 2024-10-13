import tensorflow as tf

def custom_bce(factor):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy()
        return bce(y_true, y_pred) * factor
    return loss