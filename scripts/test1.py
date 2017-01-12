from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from keras.callbacks import ModelCheckpoint

import data_util

data_dim = 11
timesteps = 10
nb_classes = 2

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(1000, return_sequences=True, activation="sigmoid", inner_activation="hard_sigmoid",
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(1000, return_sequences=True, activation="sigmoid", inner_activation="hard_sigmoid"))
model.add(LSTM(1000, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(1000))  # return a single vector of dimension 32
model.add(Dense(512, activation="relu"))  # returns a sequence of vectors of dimension 32
model.add(Dense(152, activation="relu"))  # returns a sequence of vectors of dimension 32
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# # generate dummy training data
# x_train = np.random.random((1000, timesteps, data_dim))
# y_train = np.random.random((1000, nb_classes))
#
# # generate dummy validation data
# x_val = np.random.random((100, timesteps, data_dim))
# y_val = np.random.random((100, nb_classes))
raw_data = data_util.prepare_data()
train_input, train_output, test_input, test_output = raw_data
segments, labels = data_util.extract_segments(train_input, train_output, 10)
print (np.shape(segments))
print (np.shape(labels))

segments_test, labels_test = data_util.extract_segments(test_input, test_output, 10)
x_train = segments
y_train = labels
x_val = segments_test
y_val = labels_test
# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(x_train, y_train,
          batch_size=64, nb_epoch=100,
          validation_data=(x_val, y_val),callbacks=callbacks_list)
