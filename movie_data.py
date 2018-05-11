import pandas as pd
import tensorflow as tf

COLUMNS = ['budget', 'popularity', 'revenue', 'runtime',
           'vote_count', 'release_date', 'Action', 'Adventure', 'Fantasy',
           'Science_Fiction', 'Crime', 'Drama', 'Thriller', 'Animation', 'Family',
           'Western', 'Comedy', 'Romance', 'Horror', 'Mystery', 'History', 'War',
           'Music', 'Documentary', 'Foreign', 'TV_Movie', 'vote_average']


def load_data():
    train = pd.read_csv("scaled_movies_train.csv", names=COLUMNS, header=0)
    train_x, train_y = train, train.pop('vote_average')

    validate = pd.read_csv("scaled_movies_validate.csv")
    validate_x, validate_y = validate, validate.pop('vote_average')

    test = pd.read_csv("scaled_movies_test.csv")
    test_x, test_y = test, test.pop('vote_average')

    return (train_x, train_y), (validate_x, validate_y), (train_x, test_y)


def train_input_fn(features, predictions, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), predictions))

    dataset = dataset.shuffle(4000).repeat().batch(batch_size)

    return dataset


def eval_input_fn(features, predictions, batch_size):
    features = dict(features)
    if predictions is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, predictions)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset
