from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import data_util
import tensorflow as tf

# with tf.name_scope("name1"):
#     with tf.variable_scope("a_variable_scope") as scope:
#         initializer = tf.constant_initializer(value=3)
#         var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=initializer)
#         scope.reuse_variables()
#         var3_reuse = tf.get_variable(name='var3', )
#         var4 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
#         var4_reuse = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(var3.name)  # a_variable_scope/var3:0
#     print(sess.run(var3))  # [ 3.]
#     print(var3_reuse.name)  # a_variable_scope/var3:0
#     print(sess.run(var3_reuse))  # [ 3.]
#     print(var4.name)  # a_variable_scope/var4:0
#     print(sess.run(var4))  # [ 4.]
#     print(var4_reuse.name)  # a_variable_scope/var4_1:0
#     print(sess.run(var4_reuse))  # [ 4.]

model = Sequential()
# model.add(Dense(output_dim=200, input_dim=22, activation="relu"))
# model.add(Dense(output_dim=200, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(output_dim=500, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(output_dim=500, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(output_dim=500, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(output_dim=200, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(output_dim=2, activation="softmax"))
model.add(Dense(output_dim=200, input_dim=22, activation="relu"))
model.add(Dense(output_dim=200, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(output_dim=300, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(output_dim=300, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(output_dim=300, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(output_dim=100, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(output_dim=2, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_data, test_data = data_util.prepare_data(1, 1)
x_train = train_data._inputs
y_train = np.asarray(train_data._outputs)
x_val = test_data._inputs
y_val = np.asarray(test_data._outputs)

length = np.shape(x_val)[0]
model.fit(x_train, y_train, nb_epoch=80, batch_size=32,
          validation_data=(x_val, y_val))
classes = model.predict_classes(x_val, batch_size=32)

print()
print(test_data.best_error_mean, test_data.worst_error_mean, test_data._s1_error_mean, test_data._s2_error_mean)
print (test_data.evaluate(classes))
