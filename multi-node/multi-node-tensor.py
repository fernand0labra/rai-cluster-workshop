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

import os
import tensorflow as tf

from cluster.utils.tensor import data, CustomDataset, CustomNet
from tensorflow.distribute.experimental import CommunicationOptions, CommunicationImplementation

'''
Multi-Node Tensorflow Code Snippet (Data Parallelism)

This code is intended to showcase the usage of the tensorflow framework for distributed training
of a model on a cluster setup with the use of SLURM workload manager

The cluster is expected to be formed of several machines with each holding a GPU resource

References:
https://www.tensorflow.org/guide/distributed_training
http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-tf-multi-eng.html
https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
'''


cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()                        # Build multi-worker environment from Slurm variables
communication_options = CommunicationOptions(implementation=CommunicationImplementation.NCCL)   # Use NCCL communication protocol

strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver, communication_options)   # Declare distribution strategy
n_workers = int(os.environ['SLURM_NTASKS'])  # Get total number of workers (processes)
print('Number of workers: {}'.format(n_workers))

EPOCHS = 12                                                             # Number of training epochs
BATCH_SIZE_PER_GPU = 64                                                 # Number of instances per GPU (node)
MODEL_PATH = 'custom_model.keras'                                       # Define zip model name
CHECKPOINT_DIR = '~/SPEAR/cluster/examples/multi-gpu/tensor/ckpt'       # Define the checkpoint directory to store the checkpoints.
BATCH_SIZE = BATCH_SIZE_PER_GPU * n_workers                             # Total size of the batch

dataset = data()
train_size = int(0.8 * data.__len__())                                  # Calculate nr of train samples
train_data = CustomDataset(*data[:train_size], batch_size=BATCH_SIZE)   # Instantiate datasets
val_data = CustomDataset(*data[train_size:], batch_size=BATCH_SIZE)

with strategy.scope():  # Instantiate model on a distributed communication system (data parallelism)
    model = CustomNet()
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy'])
    
checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}")        # Define the name of the checkpoint files.

callbacks = [  # Put all the callbacks together.
    tf.keras.callbacks.TensorBoard(log_dir='~/SPEAR/cluster/examples/multi-gpu/tensor/logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
]

model.fit(train_data, epochs=EPOCHS, callbacks=callbacks)               # Train model
model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_DIR))          # Load last saved checkpoint

eval_loss, eval_acc = model.evaluate(val_data)                          # Evaluate accuracy
print('Eval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))

model.save(MODEL_PATH)  # To zip .keras file


''' 
# Loading and evaluating model
# with strategy.scope():  # Distributed inference

model = tf.keras.models.load_model(MODEL_PATH)
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy'])

eval_loss, eval_acc = model.evaluate(val_data)

print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))
'''