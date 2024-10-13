import tensorflow as tf

def custom_accuracy(y_true, y_pred):
    accuracy = tf.keras.metrics.BinaryAccuracy()
    return accuracy(y_true, y_pred)