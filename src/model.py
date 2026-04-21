from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


def build_model(input_shape):

    model = Sequential()

    model.add(
        Conv1D(
            filters=32,
            kernel_size=3,
            activation='relu',
            input_shape=input_shape
        )
    )

    model.add(
        MaxPooling1D(pool_size=2)
    )

    model.add(
        LSTM(64)
    )

    model.add(
        Dropout(0.3)
    )

    model.add(
        Dense(1, activation='sigmoid')
    )

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model