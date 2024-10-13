import tensorflow as tf

def custom_accuracy(factor):
  def metric(y_true, y_pred):
    accuracy = tf.keras.metrics.BinaryAccuracy()
    return accuracy(y_true, y_pred) * factor
  return metric