"""  
    author: Shawn
    time  : 10/1/18 7:19 PM
    desc  :
    update: Shawn 10/1/18 7:19 PM      
"""

from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image

characters = string.digits + string.ascii_uppercase
print(characters)

width, height, n_len, n_class = 170, 80, 4, len(characters)


def gen_one():
    generator = ImageCaptcha(width=width, height=height)
    random_str = ''.join([random.choice(characters) for j in range(4)])
    img = generator.generate_image(random_str)
    plt.imshow(img)
    plt.title(random_str)
    plt.show()


def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(height, width, 3)),
    # keras.layers.Convolution2D(32 * 2 ** 1, 3, 3, activation=tf.nn.relu),
    # keras.layers.Convolution2D(32 * 2 ** 1, 3, 3, activation=tf.nn.relu),
    # keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(n_class, activation=tf.nn.softmax)
])

# Constructing a deep convolutional neural network
# input_tensor = keras.Input((height, width, 3))
# x = input_tensor
# for i in range(4):
#     x = keras.layers.Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)
#     x = keras.layers.Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)
#     x = keras.layers.MaxPooling2D((2, 2))(x)
#
# x = keras.layers.Flatten()(x)
# x = keras.layers.Dropout(0.25)(x)
# x = [keras.layers.Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(4)]
# model = keras.Model(input=input_tensor, output=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


def show_network():
    keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    pass


if __name__ == '__main__':
    show_network()
    # X, y = next(gen(1))
    # plt.imshow(X[0])
    # plt.title(decode(y))
    # plt.show()
    pass
