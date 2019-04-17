import tensorflow as tf
tf.enable_eager_execution()


def main(x, y, w1, w2, b1, b2):
    """Use eager execution to calculate gradients."""
    x = tf.transpose(tf.constant([x], dtype=tf.float32))
    y = tf.transpose(tf.constant([y], dtype=tf.float32))

    w1 = tf.constant(w1, dtype=tf.float32)
    w2 = tf.constant(w2, dtype=tf.float32)
    b1 = tf.transpose(tf.constant([b1], dtype=tf.float32))
    b2 = tf.transpose(tf.constant([b2], dtype=tf.float32))

    with tf.GradientTape(persistent=True) as t:
        t.watch(x)
        t.watch(y)
        t.watch(w1)
        t.watch(w2)
        h1 = tf.add(tf.matmul(w1, x), b1)
        a1 = tf.sigmoid(h1)
        h2 = tf.add(tf.matmul(w2, a1), b2)
        y_est = tf.sigmoid(h2)
        j = tf.multiply(tf.squared_difference(y_est, y), 0.5)
        # j = tf.multiply(tf.square(tf.norm(tf.add(y_est, tf.negative(y)))), 0.5)

        # j = tf.square(tf.norm(tf.add(y_est, tf.negative(y))))

    print(f'dj/dy_est =\n{t.gradient(j, y_est)}')
    print(f'dj/da1 =\n{t.gradient(j, a1)}')
    print(f'dj/dw1 =\n{t.gradient(j, w1)}')
    print(f'dj/dw2 =\n{t.gradient(j, w2)}')
    del t
    pass


if __name__ == '__main__':
    # main(x=[2, 0], y=[0, 1], w1=[[-1, 3], [3, 1]], w2=[[-2, -3], [1, 4]], b1=[2, -5], b2=[1, 2])  #
    main(x=[.3, .9], y=[1, 0], w1=[[.1, .8], [.4, .6]], w2=[[.2, .4], [.9, .5]], b1=[0, 0], b2=[0, 0])





