from keras.datasets import mnist
from keras.utils import np_utils
import logging


class Data:

    @staticmethod
    def mnist():
        logging.info("Loading Mnist data set")
        return mnist.load_data()

    @staticmethod
    def normalize(x):
        """" Normalize values from 0-255 to 0-1 """
        logging.info("Normalizing to 0-1")
        return x / 255

    @staticmethod
    def categorize(y):
        logging.info("Categorizing target labels")
        return np_utils.to_categorical(y)
