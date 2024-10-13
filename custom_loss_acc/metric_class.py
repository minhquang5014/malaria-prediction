import tensorflow as tf
class CustomAcc(tf.keras.metrics.Metric):
    def __init__(self, name='custom_acc', factor=1):
        super(self).__init__()
        self.factor = factor
        self.accuracy = self.add_weight(name='name', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        accuracy = tf.keras.metrics.BinaryAccuracy(tf.cast(y_true, dtype=tf.float32), y_pred) * self.factor
        self.accuracy.assign(tf.math.count_nonzero(accuracy, dtype=tf.float32)/tf.cast(len(accuracy), dtype=tf.float32))

    def result(self):
        return self.accuracy

    def reset_states(self):
        self.accuracy.assign(0)