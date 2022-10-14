

import tensorflow as tf
import numpy as np
import horovod.tensorflow.keras as hvd

import argparse


parser = argparse.ArgumentParser(description='cifar10 classification models, tensorflow horovod test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=256, help='')

args = parser.parse_args()

hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                 input_shape=(32,32,3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10))

optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)

optimizer = hvd.DistributedOptimizer(optimizer)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
         optimizer=optimizer,metrics=['accuracy'])

callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

### This next line will attempt to download the CIFAR10 dataset from the internet if you don't already have it stored in ~/.keras/datasets.
### Run this line on a login node prior to submitting your job, or manually download the data from
### https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz, rename to "cifar-10-batches-py.tar.gz" and place it under ~/.keras/datasets

(x_train, y_train),_ = tf.keras.datasets.cifar10.load_data()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)

model.fit(dataset, epochs=2, callbacks=callbacks, verbose=2) # verbose=2 to avoid printing a progress bar to *.out files.
