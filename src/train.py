import argparse
import logging

from data import Data
from model import Model


def train():
    parser = argparse.ArgumentParser(description='A Keras dnn for digit recognition')
    parser.add_argument('--model', type=str, help='Path to save model to')
    parser.add_argument('--batchSize', type=int, help='Training batch size', default=512)
    parser.add_argument('--epochs', type=int, help='Training epochs', default=10)
    parser.add_argument('--verbose', type=int, help='Verbosity', default=1)
    args = parser.parse_args()

    log_format = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logging.info(args)

    (X_train, y_train), (X_test, y_test) = Data.mnist()
    (X_train, X_test) = Data.normalize(X_train, X_test)

    width = 28
    height = 28
    channels = 1  # gray scale

    # preprocess labels
    y_train = Data.categorize(y_train)
    y_test = Data.categorize(y_test)
    num_classes = y_test.shape[1]

    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], channels, width, height).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], channels, width, height).astype('float32')

    logging.info("Training model")
    model = Model.basic_cnn(channels, width, height, num_classes)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.epochs, batch_size=args.batchSize,
              verbose=args.verbose, shuffle=True)

    logging.info("Predicting validation set")
    scores = model.evaluate(X_test, y_test, verbose=1, batch_size=1)
    print("%S: %.4f %S: %.2f%%" % (model.metrics_names[0], model.metrics_names[1], scores[0], scores[1] * 100))


if __name__ == "__main__":
    train()
