from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, BatchNormalization

def functional_api(IMG_SIZE=224):
    func_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_model")
    x = Conv2D(16, kernel_size=3, activation='relu')(func_input)

    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    x = Conv2D(32, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    x= Flatten()(x)
    x= Dense(20, activation='relu')(x)
    x= BatchNormalization()(x)
    x= Dropout(0.2)(x)
    x= Dense(1, activation='sigmoid')(x)

    func_model = Model(inputs=func_input, outputs=x, name="func_model")
    return func_model

func_model = functional_api()
func_model.summary()