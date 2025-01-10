'''
MIT License

Copyright (c) [2025] [Fernando Labra Caso]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import tensorflow as tf
import numpy as np

'''
Common Tensorflow Utilities (Dataset and Network Definition)
'''


def data():
    size = 1000
    f = lambda x: 13*x**3 + 67*x**2 + 21*x + 5
    x = np.random.randn(size)
    y = np.array([f(xi) for xi in x])
    return x, y


class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size=32):
        self.x = np.array(x, dtype=np.float32).reshape(-1, 1)
        self.y = np.array(y, dtype=np.float32).reshape(-1, 1)
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


class CustomNet(tf.keras.Model):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.fc1 = tf.keras.layers.Dense(5, activation='relu')
        self.fc2 = tf.keras.layers.Dense(5, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.fc3(x)
