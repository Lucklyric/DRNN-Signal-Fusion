import numpy as np
import sys

sys.path.append("..")


class BatchGenerator(object):
    def __init__(self, inputs, outputs, batch_size, num_unrollings):
        self._inputs = inputs
        self._outputs = outputs
        self._n_samples = len(outputs)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        n_segment = self._n_samples // batch_size
        self._cursor = [offset * n_segment for offset in range(batch_size)]
        self._last_input_batch, self._last_output_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch_inputs = np.zeros(shape=(self._batch_size, 11), dtype=np.float32)
        batch_outputs = np.zeros(shape=(self._batch_size, 2), dtype=np.float32)
        for b in range(self._batch_size):
            batch_inputs[b] = self._inputs[self._cursor[b]]
            batch_outputs[b] = self._outputs[self._cursor[b]]
            self._cursor[b] = (self._cursor[b] + 1) % self._n_samples
        return batch_inputs, batch_outputs

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches_input = [self._last_input_batch]
        batches_output = [self._last_output_batch]
        for step in range(self._num_unrollings):
            batch_inputs, batch_outputs = self._next_batch()
            batches_input.append(batch_inputs)
            batches_output.append(batch_outputs)
        self._last_output_batch = batches_output[-1]
        self._last_input_batch = batches_input[-1]
        return batches_input, batches_output


def parse_two_sensors(raw_data):
    sensor_one_inputs = []
    sensor_two_inputs = []
    outputs = []

    for index in range(len(raw_data)):
        if index % 2 == 0:
            sensor_one_inputs.append(raw_data[index, 0:11].astype(np.float32))
        else:
            sensor_two_inputs.append(raw_data[index, 0:11].astype(np.float32))
            if raw_data[index, 11:12] == 0:
                outputs.append([1, 0])
            else:
                outputs.append([0, 1])
    return sensor_one_inputs, sensor_two_inputs, outputs


def prepare_data():
    data_train = np.loadtxt("../data/twosensors_train.csv", delimiter=",")
    data_test = np.loadtxt("../data/twosensors_test.csv", delimiter=",")

    inputs_s1, inputs_s2, outputs = parse_two_sensors(data_train)
    print(np.shape(inputs_s1), np.shape(inputs_s2), np.shape(outputs))

    input = np.concatenate((inputs_s1, inputs_s2), axis=1)
    # data_set = BatchGenerator(inputs_s1, outputs, 128, 20)
    print(np.shape(input))
    return input, outputs


def windows(data, window_size):
    start = 0
    while start < len(data) - window_size:
        yield start, start + window_size
        start += (window_size / 2)


def extract_segments(inputs, outputs, window_size):
    segments = []
    labels = np.empty((0, 2))

    for (start, end) in windows(inputs, window_size):
        signal = inputs[start:end]
        segments = np.append(segments, [signal])
        labels = np.append(labels, [outputs[end]])
    return segments, labels
