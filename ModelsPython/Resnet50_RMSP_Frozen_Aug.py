# -*- coding: utf-8 -*-

## Imports
# General
import os
import csv
import json
import random
from shutil import copyfile  # Making copy of this file instance (including param settings used)
import numpy as np
from datetime import datetime

# Tensorflow
import tensorflow as tf
keras = tf.keras
from tensorflow.keras.datasets import cifar100
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


## Network input specs
IMG_SIZE  = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

## Hyperparameters
NUM_CLASSES = 100
EPOCHS = 300
BATCH_SIZE = 32
ARCHITECTURE = 'ResNet50'
NODES_HIDDEN_0 = 512
#NODES_HIDDEN_1 = 512
BASE_TRAINABLE = False
REGULARIZER = 'l2' # 'None' | 'l1' | 'l2' 
REGULARIZATZION_STRENGTH = '0.01'
AUGMENTATION = 1
PATIENCE = 20
OPTIMIZER = 'Adam'

## For documentation purposes - Add all parameters set above to this dict
params = dict(
    img_size = IMG_SIZE,
    img_shape = (IMG_SIZE, IMG_SIZE, 3),
    num_classes = NUM_CLASSES,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    architecture = ARCHITECTURE,
    nodes_hidden_0 = NODES_HIDDEN_0,
    #nodes_hidden_1 = NODES_HIDDEN_1,
    base_trainable = BASE_TRAINABLE,
    regularizer = REGULARIZER,
    augmentation = AUGMENTATION,
    regularization_strength = REGULARIZATZION_STRENGTH,
    patience_early_stopping = PATIENCE,
    optimizer = OPTIMIZER,
)

## Build model components from string-specifications

architecture = eval('tf.keras.applications.' + ARCHITECTURE)
regularizer = None if REGULARIZER is 'None' else eval('regularizers.' + REGULARIZER + '(' + REGULARIZATZION_STRENGTH + ')')

## Set path for saving training progress & data
now = datetime.now()
TIME_STAMP = now.strftime("_%Y_%d_%m__%H_%M_%S__%f")
MODEL_ID = 'Model_' + TIME_STAMP + '/'

DATA_STORAGE_PATH = 'Results/'
TRAINED_MODELS = 'Trained_Models/'
MODEL_ARCHITECTURE = ARCHITECTURE + '/'
path = DATA_STORAGE_PATH + TRAINED_MODELS + MODEL_ARCHITECTURE + MODEL_ID + '/'
TB_LOG_DIR = path + 'Tensorboard' + '/'

## Create folder
if not os.path.exists(path):
    os.makedirs(path)
    print('Created dir: ' + path)
else:
    path = None
    raise Exception('PATH EXISTS!')

## Save settings used during model run

# Save as csv - nicer to read for human
with open(path+'params.csv', 'w') as f:
    for key in params.keys():
        f.write("%s,%s\n"%(key,params[key]))
f.close()

# Save as json - easier to import again later
with open(path+'params.json', 'w') as f:
    json.dump(params, f)
f.close()

# Save a copy of this file - Including current param settings - For convenience during param sweeping
src = os.path.realpath(__file__)
dst = path + 'model_copy.py'
copyfile(src, dst)

## Obtain dataset
(x_train, y_train), (x_test_val, y_test_val) = cifar100.load_data(label_mode='fine')

print("Shape of X_test_val ", x_test_val.shape)
# Normalize
x_train, x_test_val = x_train / 255.0, x_test_val / 255.0

test_data_size = int(len(y_test_val) / 2)
rand_idx = random.sample(range(0, 9999), test_data_size)

# x_test = np.zeros((test_data_size,32,32,3))
x_test = np.asarray([x_test_val[x] for x in rand_idx])
y_test = np.asarray([y_test_val[x] for x in rand_idx])

x_val = np.delete(x_test_val, rand_idx, 0)
y_val = np.delete(y_test_val, rand_idx, 0)

print("Shape of Training dataset ", x_train.shape, y_train.shape)
print("Shape of Validation dataset ", x_val.shape, y_val.shape)
print("Shape of Test dataset  ", x_test.shape, y_test.shape)

## Helper function

def accuracy_score(y_test, y_prediction):
    score = 0
    for i in range(0, len(y_test)):
        if (y_test[i] == np.argmax(y_prediction[i])):
            score = score + 1
    return (score / len(y_test))

## Data Generators setup

# For data generators

def preprocessing_function(x):
    """
      Can be used for data augmentation.
    """
    return x

# Training generator

if AUGMENTATION:
    augmentations = {"rotation_range": 15, 
                    "width_shift_range": 0.1, 
                    "height_shift_range": 0.1, 
                    "horizontal_flip": True,
                    # ...
                    }
else:
    augmentations = {}

# The 1./255 is to convert from uint8 to float32 in range [0,1].
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(#rescale=1./255, # Done already
                                                     #preprocessing_function=preprocessing_function  # Pre-processing function may be passed here
                                                     **augmentations
                                                     )

train_data_gen = train_image_generator.flow(x_train,
                                            y_train,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

## Definition of callbacks adjusted from https://www.tensorflow.org/guide/keras/train_and_evaluate

early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',    # Stop training when `val_loss` is no longer improving
        min_delta=0,               # "no longer improving" being defined as "no better than 0 less"
        patience=PATIENCE,         # "no longer improving" being further defined as "for at least 2 epochs"
        verbose=0,                 # Quantity of printed output
        mode='max',                # In 'max' mode, training will stop when the quantity monitored has stopped increasing;
        #baseline=None,
        #restore_best_weights=False
        )

model_saving_callback = ModelCheckpoint(
        filepath=path+'best_model.h5',
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        save_best_only=True,
        monitor='val_accuracy',
        # mode: one of {auto, min, max}. If `save_best_only=True`, the decision to
        # overwrite the current save file is made based on either the maximization
        # or the minimization of the monitored quantity. For `val_acc`, this
        # should be `max`, for `val_loss` this should be `min`, etc. In `auto`
        # mode, the direction is automatically inferred from the name of the
        # monitored quantity.
        verbose=0)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TB_LOG_DIR, update_freq='epoch') #, write_graph=False

# Join list of required callbacks
callbacks = [model_saving_callback, tensorboard_callback, early_stopping_callback]

## Helper Layer

# Resizing from (32,32,3) to (224,224,3)

class Resizer(layers.Layer):
  def __init__(self):
    super(Resizer, self).__init__()

  def build(self, input_shapes):
    pass
  
  def call(self, input):
    return tf.image.resize(input, (IMG_SIZE, IMG_SIZE))

## Obtain & Compile Model

# Reset tf sessions
tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.

# Create the base model from the pre-trained model specified by variable ARCHITECTURE
#Include Top = false 
base_model = architecture(weights="imagenet", include_top=False, input_shape=IMG_SHAPE)

base_model.trainable = BASE_TRAINABLE

# Print summary base model
#print('Base model:')
#base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  # Suggested on TF tutorial page... 
flatten_operation = layers.Flatten()
hidden_dense_layer_0 = layers.Dense(NODES_HIDDEN_0, activation='relu', kernel_regularizer=regularizer)
#hidden_dense_layer_1 = layers.Dense(NODES_HIDDEN_1, activation='relu', kernel_regularizer=regularizer)
prediction_layer = layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizer)

# Construct overall model
model = tf.keras.Sequential([
  Resizer(),
  base_model,
  global_average_layer,
  flatten_operation,
  hidden_dense_layer_0,
  prediction_layer
])

# Compile model & make some design choices
model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=0.0001,
                                              rho=0.9,
                                              momentum=0.0,
                                              epsilon=1e-07,
                                              centered=False,
                                              name='RMSprop'
                                              ),
              loss='sparse_categorical_crossentropy',  # Capable of working with regularization
              metrics=['accuracy', 'sparse_categorical_crossentropy'])


# Construct computational graph with proper dimensions
inputs = np.random.random([1] + list(IMG_SHAPE)).astype(np.float32)
model(inputs)

# Print summary overall model
print('Overall model:')
model.summary()

## Perform training

history = model.fit(
                    x=train_data_gen,
                    #y=None,
                    #batch_size=None,
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=callbacks,
                    #validation_split=0.0,
                    validation_data=(x_val, y_val),
                    #shuffle=True,
                    #class_weight=None,
                    #sample_weight=None,
                    initial_epoch=0,
                    steps_per_epoch=20,
                    #validation_steps=18,
                    #validation_freq=1,
                    #max_queue_size=5,
                    #workers=1,
                    #use_multiprocessing=False,
                    #**kwargs
                    )

# Assess training outcome based on test-data
predictions = model.predict(x_test)

print("Predictions are", np.argmax(predictions[0]))
print("Y_test is ",y_test)
test_accuracy = accuracy_score(y_test, predictions)

print("Final Test Accuracy ", test_accuracy)

# Save the entire model as a final model to a HDF5 file.
name = 'final_model'
model.save(path+name+'.h5')

# Record training progress
with open(path+'training_progress.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy", "sparse_categorical_crossentropy"])
    for line in range(len(history.history['loss'])): 
        epoch = str(line+1)
        writer.writerow([epoch,
                         history.history["loss"][line], 
                         history.history["accuracy"][line], 
                         history.history["val_loss"][line], 
                         history.history["val_accuracy"][line], 
                         history.history["sparse_categorical_crossentropy"][line]
                         ])
    # Save some more important bits/summary
    writer.writerow(["End of training. Summary:"])
    writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy", "sparse_categorical_crossentropy"])
    # Max accuracy
    writer.writerow(["Max accuracy row:"])
    x = np.argmax(history.history["accuracy"])
    writer.writerow([str(x+1),
                         history.history["loss"][x], 
                         history.history["accuracy"][x], 
                         history.history["val_loss"][x], 
                         history.history["val_accuracy"][x], 
                         history.history["sparse_categorical_crossentropy"][x]
                         ])
    # Max val_accuracy
    writer.writerow(["Max val_accuracy row:"])
    x = np.argmax(history.history["val_accuracy"])
    writer.writerow([str(x+1),
                         history.history["loss"][x], 
                         history.history["accuracy"][x], 
                         history.history["val_loss"][x], 
                         history.history["val_accuracy"][x], 
                         history.history["sparse_categorical_crossentropy"][x]
                         ])
    file.close()

print('Done.')

