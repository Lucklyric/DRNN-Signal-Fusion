import numpy as np
import sys

sys.path.append("..")


class FusionDataSet(object):
    def __init__(self, inputs, outputs, error_means, batch_size, time_step):
        self._inputs = inputs
        self._outputs = outputs
        self._error_means = error_means
        self._batch_size = batch_size
        self._time_step = time_step
        self._start_cursor = 0
        self._s1_error_mean = np.average(np.asarray(error_means)[:, 0])
        self._s2_error_mean = np.average(np.asarray(error_means)[:, 1])
        self.worst_error_mean, self.best_error_mean = self.cal_worst_best()

    def get_batch(self):
        reset = 0
        if ((self._start_cursor + self._time_step * self._batch_size) > len(self._inputs)):
            self._start_cursor = 0
            reset = 1

        xs = self._inputs[self._start_cursor:self._start_cursor + self._time_step * self._batch_size].reshape(
            (self._batch_size, self._time_step, 24))
        ys = np.asarray(
            self._outputs[self._start_cursor:self._start_cursor + self._time_step * self._batch_size]).reshape(
            (self._batch_size, self._time_step, 2))
        self._start_cursor += self._time_step
        return xs, ys, reset

    def cal_worst_best(self):
        worst = 0
        best = 0
        n_sample = len(self._error_means)
        for index in range(len(self._error_means)):
            best += self._error_means[index][np.argmax(self._outputs[index])]
            worst += self._error_means[index][np.argmin(self._outputs[index])]
        return worst / n_sample, best / n_sample

    def evaluate(self, outputs):
        sum_error = 0
        n_sample = len(outputs)
        for index in range(len(outputs)):
            sum_error += self._error_means[index][outputs[index]]

        mean_error = sum_error / n_sample
        return sum_error / n_sample, 1-(mean_error - self.best_error_mean) / (
            self.worst_error_mean - self.best_error_mean)


def parse_two_sensors(raw_data):
    sensor_one_inputs = []
    sensor_two_inputs = []
    outputs = []
    sensor_error_means = []
    for index in range(len(raw_data)):
        if raw_data[index][13:14] == np.inf:
            print ("catach", index)
        if index % 2 == 0:
            sensor_one_inputs.append(raw_data[index, 0:12].astype(np.float32))
        else:
            sensor_two_inputs.append(raw_data[index, 0:12].astype(np.float32))
            if raw_data[index, 12:13] == 0:
                outputs.append([1, 0])
            else:
                outputs.append([0, 1])
            sensor_error_means.append(
                [raw_data[index - 1, 13:14].astype(np.float32), raw_data[index, 13:14].astype(np.float32)])
    return sensor_one_inputs, sensor_two_inputs, outputs, sensor_error_means


def prepare_data(batch_size, time_step):
    data_train = np.loadtxt("../data/twosensors2-new-normal.csv", delimiter=",")
    data_test = np.loadtxt("../data/twosensors-new-normal.csv", delimiter=",")

    train_inputs_s1, train_inputs_s2, train_outputs, train_error_means = parse_two_sensors(data_train)
    test_inputs_s1, test_inputs_s2, test_outputs, test_error_means = parse_two_sensors(data_test)

    train_inputs = np.concatenate((train_inputs_s1, train_inputs_s2), axis=1)
    test_inputs = np.concatenate((test_inputs_s1, test_inputs_s2), axis=1)
    # data_set = BatchGenerator(inputs_s1, outputs, 128, 20)
    train_data = FusionDataSet(train_inputs, train_outputs, train_error_means, batch_size, time_step)
    test_data = FusionDataSet(test_inputs, test_outputs, test_error_means, batch_size, time_step)
    return train_data, test_data
