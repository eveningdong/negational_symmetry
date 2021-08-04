"""The models for QNN classification on MNIST."""
import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

import utils


class CircuitLayerBuilder():
    """Building block for QNN."""

    def __init__(self, data_qubits, readout):
        """__init__."""
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        """Add layer."""
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)


def create_quantum_model(num_qubits=16, arch=['XX', 'ZZ']):
    """Create a QNN model circuit and readout operation to go along with it."""
    h = utils.get_side(num_qubits)
    data_qubits = cirq.GridQubit.rect(h, h)    # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(
        data_qubits=data_qubits,
        readout=readout)

    # Then add layers
    for i, layer in enumerate(arch):
        if layer == 'XX':
            builder.add_layer(circuit, cirq.XX, "xx{}".format(i))
        elif layer == 'ZZ':
            builder.add_layer(circuit, cirq.ZZ, "zz{}".format(i))
        else:
            raise ValueError

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)


def create_qnn(num_qubits=16, arch=['XX', 'ZZ']):
    """Create Keras model of QNN."""
    model_circuit, model_readout = create_quantum_model(
        num_qubits=16, arch=['XX', 'ZZ'])
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        tfq.layers.PQC(model_circuit, model_readout)], name='qnn')

    # For simplicity, we use Adam optimizer here. Parameter shift can be
    # implemented by TensorFlow-Quantum, which converges much slower.
    model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[utils.hinge_accuracy])

    return model


def create_quantum_model_feat(num_qubits=16, arch=['XX', 'ZZ']):
    """Create a QNN model circuit and readout operation to go along with it."""
    h = utils.get_side(num_qubits)
    data_qubits = cirq.GridQubit.rect(h, h)    # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(
        data_qubits=data_qubits,
        readout=readout)

    # Then add layers
    for i, layer in enumerate(arch):
        if layer == 'XX':
            builder.add_layer(circuit, cirq.XX, "xx{}".format(i))
        elif layer == 'ZZ':
            builder.add_layer(circuit, cirq.ZZ, "zz{}".format(i))
        else:
            raise ValueError

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout), [cirq.Z(qubit) for qubit in data_qubits]


def create_qnn_feat(num_qubits=16, arch=['XX', 'ZZ']):
    """Create Keras model of QNN."""
    model_circuit, model_readout, model_feature = create_quantum_model_feat(
        num_qubits=16, arch=['XX', 'ZZ'])
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        tfq.layers.PQC(model_circuit, model_feature)], name='qnn_feat')

    return model


def create_dnn(num_qubits=16, model_name='dnn'):
    """A simple DNN model."""
    h = utils.get_side(num_qubits)
    model = tf.keras.Sequential(name=model_name)
    model.add(tf.keras.layers.Flatten(input_shape=(h, h, 1)))
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model


def create_cnn(num_qubits=16, model_name='cnn'):
    """Create a simple CNN model."""
    h = utils.get_side(num_qubits)
    model = tf.keras.Sequential(name=model_name)
    model.add(tf.keras.layers.Reshape((1, h**2), input_shape=(h, h, 1)))
    model.add(tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    return model
