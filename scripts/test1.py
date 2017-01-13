from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import data_util

model = Sequential()
model.add(LSTM(500, input_shape=(1, 22),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(500, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
train_data, test_data = data_util.prepare_data(1, 1)
x_train = train_data._inputs
y_train = np.asarray(train_data._outputs)
x_val = test_data._inputs
y_val = np.asarray(test_data._outputs)
trainX = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
trainY = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
testX = np.reshape(x_val, (x_val.shape[0], 1, x_val.shape[1]))
testY = np.reshape(y_val, (y_val.shape[0], 1, y_val.shape[1]))
model.fit(trainX, y_train, nb_epoch=1000, batch_size=60, verbose=2, validation_data=(testX, y_val))

#
# x_train = np.reshape(train_data._inputs,(len(train_data._inputs),1,22))
# y_train = train_data._outputs
# x_val = np.reshape(test_data._inputs,(len(test_data._inputs),1,22))
# y_val = test_data._outputs
# # generate dummy training data
# x_train = np.random.random((1000, timesteps, data_dim))
# y_train = np.random.random((1000, nb_classes))
#
# # generate dummy validation data
# x_val = np.random.random((100, timesteps, data_dim))
# y_val = np.random.random((100, nb_classes))
# train_data, test_data = data_util.prepare_data(1, timesteps)
#
# x_train = np.reshape(train_data._inputs,(len(train_data._inputs),1,22))
# y_train = train_data._outputs
# x_val = np.reshape(test_data._inputs,(len(test_data._inputs),1,22))
# y_val = test_data._outputs
# # checkpoint
# filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
# model.fit(x_train, y_train,
#           batch_size=1, nb_epoch=100,
#           validation_data=(x_val, y_val),callbacks=callbacks_list)
