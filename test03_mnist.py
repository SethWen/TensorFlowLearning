"""  
    author: Shawn
    time  : 9/27/18 11:56 PM
    desc  :
    update: Shawn 9/27/18 11:56 PM      
"""

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
result = model.evaluate(x_test, y_test)
print(result)

# Save weights to a TensorFlow Checkpoint file
model.save_weights('./model')

# # Restore the model's state,
# # this requires a model with the same architecture.
# model.load_weights('my_model')
