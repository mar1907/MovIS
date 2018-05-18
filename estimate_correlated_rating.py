from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse

import correlated_movie_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

DNN = correlated_movie_data.DNN


def main(argv):
    args = parser.parse_args(argv[1:])

    (train_x, train_y), (validate_x, validate_y), (test_x, test_y) = correlated_movie_data.load_data()

    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    if DNN:
        predictor = tf.estimator.DNNClassifier(  # DNN classifier
            feature_columns=my_feature_columns,
            # Two hidden layers of 10 nodes each.
            hidden_units=[10, 10],
            # The model must choose between 3 classes.
            n_classes=11)
    else:
        predictor = tf.estimator.LinearRegressor(   # Linear Regression estimator
            feature_columns=my_feature_columns
        )

    predictor.train(
        input_fn=lambda: correlated_movie_data.train_input_fn(train_x, train_y,
                                                   args.batch_size),
        steps=args.train_steps
    )

    eval_result = predictor.evaluate(
        input_fn=lambda: correlated_movie_data.eval_input_fn(validate_x, validate_y,
                                                  args.batch_size)
    )

    if DNN:
        print('\nEvaluation set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    else:
        print('\nEvaluation set average loss: {average_loss:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
