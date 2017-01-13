import numpy as np
import sys

sys.path.append("..")


class BatchGenerator(object):
    def __init__(self, inputs, outputs, batch_size, time_step):
        self._inputs = inputs
        self._outputs = outputs
        self._batch_size = batch_size
        self._time_step = time_step
        self._start_cursor = 0

    def get_batch(self):
        reset = False
        if (self._start_cursor + self._time_step * self._batch_size) > len(self._inputs):
            self._start_cursor = 0
            reset = True

        xs = self._inputs[self._start_cursor:self._start_cursor + self._time_step * self._batch_size].reshape(
            (self._batch_size, self._time_step, 22))
        ys = np.asarray(
            self._outputs[self._start_cursor:self._start_cursor + self._time_step * self._batch_size]).reshape(
            (self._batch_size, self._time_step, 2))
        self._start_cursor += self._time_step
        return xs, ys, reset


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


def prepare_data(batch_size, time_step):
    data_train = np.loadtxt("../data/twosensors_train.csv", delimiter=",")
    data_test = np.loadtxt("../data/twosensors_test.csv", delimiter=",")

    train_inputs_s1, train_inputs_s2, train_outputs = parse_two_sensors(data_train)
    test_inputs_s1, test_inputs_s2, test_outputs = parse_two_sensors(data_test)

    train_inputs = np.concatenate((train_inputs_s1, train_inputs_s2), axis=1)
    test_inputs = np.concatenate((test_inputs_s1, test_inputs_s2), axis=1)
    # data_set = BatchGenerator(inputs_s1, outputs, 128, 20)
    train_data = BatchGenerator(train_inputs, train_outputs, batch_size, time_step)
    test_data = BatchGenerator(test_inputs, test_outputs, batch_size, time_step)
    return train_data, test_data
