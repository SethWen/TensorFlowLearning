import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 这句倒入比较耗时，怎么破
import tensorflow as tf


def test1():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))


def test2():
    a = [1., 2., 3.]  # a rank 1 tensor; this is a vector with shape [3]
    b = [[1., 2., 3.], [4., 5., 6.]]  # a rank 2 tensor; a matrix with shape [2, 3]
    c = [[[1., 2., 3.]], [[7., 8., 9.]]]  # a rank 3 tensor with shape [2, 1, 3]


def test3():
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)  # also tf.float32 implicitly
    # print(node1, node2)
    sess = tf.Session()
    print(sess.run(tf.add(node1, node2)))


def test4():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b  # + provides a shortcut for tf.add(a, b)
    sess = tf.Session()
    # print(sess.run(adder_node, {a: 3, b: 4.5}))
    print(sess.run(adder_node * 3, {a: [1, 3], b: [2, 4]}))


def test5():
    k = tf.Variable([1], dtype=tf.float32)
    b = tf.Variable([1], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    liner_model = k * x + b

    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    fixk = tf.assign(k, [-1.])
    fixb = tf.assign(b, [1.])
    sess.run([fixk, fixb])
    # print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    y = tf.placeholder(tf.float32)
    # 差的平方
    squared_deltas = tf.square(liner_model - y)
    # 差的平方和
    loss = tf.reduce_sum(squared_deltas)
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


def test6():
    pass


if __name__ == '__main__':
    # test4()
    test5()
