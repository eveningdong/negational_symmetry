"""The basic functions for QNN classification on MNIST."""
import cirq
import collections
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq


def get_mnist():
    """Return MNIST training set of digit a and b."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test


def prepare_classical_data(
        x, y,
        num_qubits=16,
        a=3,
        b=6,
        invert=False):
    """Prepare classical data.

    Parameters:
    ----------
    invert : bool
        Invert the image for the test set. Default to be False in supervised
        learning, change to True to create a domain shift.
    """
    side = get_side(num_qubits)
    x, y = preprocess_classical_data(x, y, side=side, a=a, b=b, invert=invert)
    return x, y


def convert_to_circuit(image, side):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(side, side)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit


def prepare_quantum_data(
        x, y,
        num_qubits=16,
        a=3,
        b=6,
        invert=False):
    """Prepare quantum data.

    Parameters:
    ----------
    invert : bool
        Invert the image for the test set. Default to be False in supervised
        learning, change to True to create a domain shift.
    """
    side = get_side(num_qubits)
    x, y = preprocess_classical_data(
        x, y, side=side, a=a, b=b, invert=invert)

    x_circ = [convert_to_circuit(_x, side) for _x in x]
    x_tfcirc = tfq.convert_to_tensor(x_circ)

    y = 2.0 * y - 1.0

    return x_tfcirc, y


def get_side(num_qubits):
    """Return the square root of the number of qubits."""
    side = int(np.sqrt(num_qubits))
    assert side**2 == num_qubits, 'side should be an integer!'
    return side


def preprocess_classical_data(x, y, side=4, a=3, b=6, invert=False):
    """Preprocess classical data.

    Parameters:
    ----------
    x : numpy array
    y : numpy array
    num_qubits : int
    a : int
        The first digit
    b : int
        The second digit
    invert : bool
        Default to be False in supervised learning, change to True to create a
        domain shift.
    """
    # Filter out two digits.
    x = x[..., np.newaxis] / 255.0
    x, y = filter(x, y, a, b)

    # Resize the images.
    x_small = tf.image.resize(x, (side, side)).numpy()

    # Binarize the images.
    x_bin = np.array(x_small > 0.5, dtype=np.float32)

    # Remove ambiguous images.
    x_nocon, y_nocon = remove_ambiguous(x_bin, y, a, b)

    if invert:
        x_nocon = 1 - x_nocon

    return x_nocon, y_nocon


def filter(x, y, a=3, b=6):
    """Filter out two digits.

    Parameters:
    ----------
    a : int
        The first digit
    b : int
        The second digit
    """
    keep = (y == a) | (y == b)
    x, y = x[keep], y[keep]
    y = y == a
    return x, y


def remove_ambiguous(xs, ys, a=3, b=6):
    """Remove ambiguous images.

    Parameters:
    ----------
    a : int
        The first digit
    b : int
        The second digit
    """
    mapping = collections.defaultdict(set)
    # Determine the set of labels for each unique image:
    for x, y in zip(xs, ys):
        mapping[tuple(x.flatten())].add(y)

    new_x = []
    new_y = []
    for x, y in zip(xs, ys):
        labels = mapping[tuple(x.flatten())]
        if len(labels) == 1:
            new_x.append(x)
            new_y.append(list(labels)[0])
        else:
            # Throw out images that match more than one label.
            pass

    return np.array(new_x), np.array(new_y)


def hinge_accuracy(y_true, y_pred):
    """Hinge loss."""
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)
    return tf.reduce_mean(result)
