import argparse
import logging

from data import Data
from model import Model


def train():
    parser = argparse.ArgumentParser(description='A Keras DNN for digit recognition')
    parser.add_argument('--model', type=str, help='Path to save model to')
    parser.add_argument('--image', type=str, help='Path to model image')
    parser.add_argument('--batchSize', type=int, help='Training batch size', default=512)
    parser.add_argument('--epochs', type=int, help='Training epochs', default=10)
    parser.add_argument('--verbose', type=int, help='Verbosity', default=1)
    args = parser.parse_args()

    log_format = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logging.info(args)

    (X_train, y_train), (X_test, y_test) = Data.mnist()
    X_train = Data.normalize(X_train)
    X_test = Data.normalize(X_test)

    (width, height) = X_train[0].shape
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

    if args.image:
        model.plot(args.image)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.epochs, batch_size=args.batchSize,
              verbose=args.verbose, shuffle=True)

    logging.info("Predicting validation set")
    scores = model.evaluate(X_test, y_test, verbose=args.verbose, batch_size=args.batchSize)
    logging.info("%s: %.4f %s: %.2f%%" % (model.metrics_names[0], scores[0], model.metrics_names[1], scores[1] * 100))

    logging.info("Saving model")
    model.save_weights(args.model)


if __name__ == "__main__":
    train()
