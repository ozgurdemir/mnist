from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


class Model:

    @staticmethod
    def basic_cnn(channels, width, height, num_classes):
        """ A basic CNN """
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=5, strides=1,
                         padding='same', data_format='channels_first',
                         input_shape=(channels, width, height), activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))
        model.add(Dropout(rate=0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
