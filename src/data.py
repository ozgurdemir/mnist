from keras.datasets import mnist
from keras.utils import np_utils
import logging

class Data:

    @staticmethod
    def mnist():
        logging.info("Loading Mnist dataset")
        return mnist.load_data()

    @staticmethod
    def normalize(X_train, X_test):
        """" Normalize values from 0-255 to 0-1 """
        logging.info("Normalizing to 0-1")
        x_train = X_train / 255
        x_test = X_test / 255
        return x_train, x_test

    @staticmethod
    def categorize(y):
        logging.info("Categorizing target labels")
        return np_utils.to_categorical(y)

    #todo: experiment with other normalizations