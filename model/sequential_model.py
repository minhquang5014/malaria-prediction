from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, BatchNormalization

def sequential_model(IMG_SIZE = 224):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

model = sequential_model()
model.summary()