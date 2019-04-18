import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


keras = tf.keras


class MyLayer(tf.layers.Layer):
    """A custom tf layer."""
    def __init__(self, output_dim, **kwargs):
        """Define the output shape of the kernel and pass other arguments to super().__init__"""
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Define the kernel."""
        shape = tf.TensorShape((input_shape[1], self.output_dim))  # shape is the shape of the kernel
        self.kernel = self.add_weight(name='kernel', shape=shape, initializer='uniform', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        """Define the forward pass."""
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        """Get the config of the custom layer."""
        base_config = super().get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod  # only bound to the class, can be called by both the class or the instance.
    def from_config(cls, config):
        """Return a instance initialized by config."""
        return cls(**config)


def main():
    train_data = pd.read_csv('./data/dataAll.csv', index_col=None)
    test_featues = pd.read_csv('./data/testAll.csv', index_col=None)

    train_data = shuffle(train_data)

    train_features = train_data.iloc[:, :-2]  # (3863, 30)
    train_labels = train_data.iloc[:, -2:]  # (3863, 2)

    model = keras.Sequential([MyLayer(2, input_shape=(30,))])  # Do not forget to specify input_shape
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='mse',
                  metrics=['mse'])
    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        # Write TensorBoard logs to `./logs` directory
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]

    model.fit(train_features.values, train_labels.values, batch_size=32, epochs=5, callbacks=callbacks,
              validation_split=0.2)
    pass


if __name__ == '__main__':
    main()


