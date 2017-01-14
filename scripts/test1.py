from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import data_util

model = Sequential()
model.add(Dense(output_dim=200, input_dim=22, activation="relu"))
model.add(Dense(output_dim=200, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(output_dim=300, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(output_dim=300, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(output_dim=300, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(output_dim=100, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(output_dim=2, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_data, test_data= data_util.prepare_data(1, 1)
x_train = train_data._inputs
y_train = np.asarray(train_data._outputs)
x_val = test_data._inputs
y_val = np.asarray(test_data._outputs)

model.fit(x_train, y_train, nb_epoch=100, batch_size=32, validation_data=(x_val, y_val))
classes = model.predict_classes(x_val, batch_size=32)

print()
print(test_data.best_error_mean, test_data.worst_error_mean, test_data._s1_error_mean, test_data._s2_error_mean)
print (test_data.evaluate(classes))
