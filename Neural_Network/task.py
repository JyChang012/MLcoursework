from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

keras = tf.keras

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4


def task1():
    train_data = pd.read_csv('./Data/Task1/horseColicTraining.txt', sep='\t', header=None)
    test_data = pd.read_csv('./Data/Task1/horseColicTest.txt', sep='\t', header=None)

    train_features = train_data.iloc[:, :-1]
    train_labels = train_data.iloc[:, -1]

    cls = LogisticRegression(C=2)

    cls.fit(train_features, train_labels)
    score = cls.score(test_data.iloc[:, :-1], test_data.iloc[:, -1])
    print(score)


def task1_neural_net():
    train_data = pd.read_csv('./Data/Task1/horseColicTraining.txt', sep='\t', header=None)
    test_data = pd.read_csv('./Data/Task1/horseColicTest.txt', sep='\t', header=None)

    train_features = train_data.iloc[:, :-1].values
    train_labels = train_data.iloc[:, -1].values

    test_features = test_data.iloc[:, :-1].values
    test_labels = test_data.iloc[:, -1].values

    callbacks = [keras.callbacks.EarlyStopping(patience=10)]

    model = keras.Sequential([tf.layers.Dense(5, activation='relu', input_shape=(21,), kernel_regularizer='l2'),
                              tf.layers.Dense(5, activation='relu', kernel_regularizer='l2'),
                              tf.layers.Dense(5, activation='relu', kernel_regularizer='l2'),
                              tf.layers.Dense(5, activation='relu', kernel_regularizer='l2'),
                              tf.layers.Dense(5, activation='relu', kernel_regularizer='l2'),
                              tf.layers.Dense(5, activation='relu', kernel_regularizer='l2'),
                              tf.layers.Dense(1, activation='sigmoid', kernel_regularizer='l2')])

    model.compile(optimizer='adam', loss=tf.losses.sigmoid_cross_entropy, metrics=['accuracy'])
    model.fit(train_features, train_labels, epochs=10000, batch_size=32, validation_split=0.2, callbacks=callbacks)
    print(model.evaluate(test_features, test_labels, batch_size=32))


def task2(max_features=5000, depth=60):
    with open('./Data/Task2/train/train_texts.dat', 'rb') as f:
        train_text = pk.load(f)

    with open('./Data/Task2/test/test_texts.dat', 'rb') as f:
        test_text = pk.load(f)

    vectorizer = TfidfVectorizer(max_features=max_features)
    train_features = vectorizer.fit_transform(train_text)  # (11314, 10000)
    test_features = vectorizer.transform(test_text)  # (7532, 10000)
    train_labels = pd.read_csv('./Data/Task2/train/train_labels.txt', header=None, dtype='int32').values.reshape(
        (11314,))
    # 20 unique classes

    # train_labels = train_labels
    # train_features = train_features

    model = keras.Sequential()
    # must declare input_shape
    model.add(tf.layers.Dense(200, activation='relu', kernel_regularizer='l2', input_shape=(max_features,)))

    callbacks = [keras.callbacks.EarlyStopping(patience=10)]

    for width in np.linspace(200, 20, depth).astype(np.int):
        model.add(tf.layers.Dense(width, activation='relu', kernel_regularizer='l2'))

    model.add(tf.layers.Dense(20, activation='softmax', kernel_regularizer='l2'))

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    model.fit(train_features, train_labels, batch_size=32, epochs=100, validation_split=0.2, callbacks=callbacks)
    pass


def task2_lstm(vo_size=10000, maxlen=500):
    with open('./Data/Task2/train/train_texts.dat', 'rb') as f:
        train_text = pk.load(f)

    with open('./Data/Task2/test/test_texts.dat', 'rb') as f:
        test_text = pk.load(f)

    train_labels = pd.read_csv('./Data/Task2/train/train_labels.txt', header=None, dtype='int32').values.reshape(
        (11314,))

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vo_size,
                                                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n1234567890')
    tokenizer.fit_on_texts(train_text + test_text)

    train_seq = np.array(tokenizer.texts_to_sequences(train_text))
    test_seq = tokenizer.texts_to_sequences(test_text)

    # pad the seq
    train_seq = np.array(keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=maxlen, value=0, padding='post',
                                                                    truncating='post'))

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(vo_size, 16),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(20, activation='softmax')
    # ])

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vo_size, 32),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv1D(filters=64,
                               kernel_size=5,
                               padding='valid',
                               activation='relu',
                               strides=1),
        keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(20, activation='softmax')
    ])

    callbacks = [keras.callbacks.EarlyStopping(patience=10)]

    model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit(train_seq, train_labels, epochs=100, batch_size=32, callbacks=callbacks,
                           validation_split=0.2)

    pass


def task2_lstm_post(vo_size=10000, maxlen=500):
    with open('./Data/Task2/train/train_texts.dat', 'rb') as f:
        train_text = pk.load(f)

    with open('./Data/Task2/test/test_texts.dat', 'rb') as f:
        test_text = pk.load(f)

    train_labels = pd.read_csv('./Data/Task2/train/train_labels.txt', header=None, dtype='int32').values.reshape(
        (11314,))

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vo_size,
                                                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n1234567890')
    tokenizer.fit_on_texts(train_text + test_text)

    train_seq = np.array(tokenizer.texts_to_sequences(train_text))
    test_seq = np.array(tokenizer.texts_to_sequences(test_text))

    # pad the seq
    train_seq = np.array(keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=maxlen, value=0, padding='post',
                                                                    truncating='post'))
    test_seq = np.array(keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=maxlen, value=0, padding='post',
                                                                   truncating='post'))

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(vo_size, 16),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(20, activation='softmax')
    # ])

    # for j, epochs in enumerate(range(40, 29, -1)):
    #     model = tf.keras.Sequential([
    #         tf.keras.layers.Embedding(vo_size, 32),
    #         tf.keras.layers.Dropout(0.25),
    #         tf.keras.layers.Conv1D(filters=64,
    #                                kernel_size=5,
    #                                padding='valid',
    #                                activation='relu',
    #                                strides=1),
    #         keras.layers.MaxPooling1D(pool_size=4),
    #         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    #         tf.keras.layers.Dense(64, activation='relu'),
    #         tf.keras.layers.Dense(20, activation='softmax')
    #     ])
#
    #     callbacks = [keras.callbacks.ModelCheckpoint('./lstm2_post.ckpt', save_weights_only=True, verbose=1)]
#
    #     model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #     print(model.summary())
    #     # with tf.Session(config=config) as sess:
    #     #     sess.run(model.fit(train_seq, train_labels, epochs=1, batch_size=32))  # nearly optimized
    #     #     y_est = sess.run(model.predict(test_seq, batch_size=32))
#
    #     model.fit(train_seq, train_labels, epochs=epochs, batch_size=32)
    #     y_est = model.predict(test_seq, batch_size=32)
#
    #     y_est_rst = np.argmax(y_est, axis=1)
#
    #     # y_train_est = np.argmax(model.predict(train_seq, batch_size=32), axis=1)
    #     # print(f'mine {(np.sum(y_train_est == train_labels)) / train_labels.shape[0]}')
    #     # print(f'eval {model.evaluate(train_seq, train_labels, batch_size=32)}')
#
    #     model.save('/home/changjy/pycharm_mapping/pycharm_project_148/Neural_Network/my_lstm_post.h5')
    #     np.savetxt(
    #         f'/home/changjy/pycharm_mapping/pycharm_project_148/Neural_Network/畅嘉宇-U201613248-第六章神经网络{j + 1}.txt',
    #         y_est_rst, fmt='%1d')

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vo_size, 32),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv1D(filters=64,
                               kernel_size=5,
                               padding='valid',
                               activation='relu',
                               strides=1),
        keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(20, activation='softmax')
    ])

    callbacks = [keras.callbacks.ModelCheckpoint('./lstm2_post.ckpt', save_weights_only=True, verbose=1)]

    model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    # with tf.Session(config=config) as sess:
    #     sess.run(model.fit(train_seq, train_labels, epochs=1, batch_size=32))  # nearly optimized
    #     y_est = sess.run(model.predict(test_seq, batch_size=32))

    model.fit(train_seq, train_labels, epochs=35, batch_size=32)
    y_est = model.predict(test_seq, batch_size=32)

    y_est_rst = np.argmax(y_est, axis=1)

    # y_train_est = np.argmax(model.predict(train_seq, batch_size=32), axis=1)
    # print(f'mine {(np.sum(y_train_est == train_labels)) / train_labels.shape[0]}')
    # print(f'eval {model.evaluate(train_seq, train_labels, batch_size=32)}')

    model.save('/home/changjy/pycharm_mapping/pycharm_project_148/Neural_Network/my_lstm_post.h5')
    np.savetxt(
        '/home/changjy/pycharm_mapping/pycharm_project_148/Neural_Network/rst.txt',
        y_est_rst, fmt='%1d')
    pass


if __name__ == '__main__':
    task2_lstm()
