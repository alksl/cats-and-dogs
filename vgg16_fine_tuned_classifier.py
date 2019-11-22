import os
import numpy as np
import json
import tensorflow as tf
from argparse import ArgumentParser
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from pathlib import Path

def parse_args():
    parser = ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=20)

    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    return parser.parse_known_args()


def build_model():
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    model = tf.keras.models.Sequential()
    model.add(conv_base)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    for layer in conv_base.layers:
        if layer.name.startswith('block5'):
            layer.trainable = True
        else:
            layer.trainable = False

    return model


args, extra_args = parse_args()
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    args.train,
    target_size=(150, 150),
    batch_size=args.batch_size,
    class_mode='binary',
)

validation_generator = validation_datagen.flow_from_directory(
    args.test,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
)

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=1e-5),
    loss='binary_crossentropy',
    metrics=['acc'],
)

training_run = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=args.epochs,
    validation_data=validation_generator,
    validation_steps=50,
)

tf.contrib.saved_model.save_keras_model(model, args.model_dir)
