import os
import numpy as np
import json
from argparse import ArgumentParser
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
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
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

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
    optimizer=optimizers.RMSprop(lr=1e-5),
    loss='binary_crossentropy',
    metrics=['acc'],
)

training_run = model.fit_generator(
    train_generator,
    steps_per_epoch=1,
    epochs=args.epochs,
    validation_data=validation_generator,
    validation_steps=50,
)

model_dir = Path(args.model_dir)
model_file = 'vgg16_fine_tuned_augmented.h5'
history_file = "vgg16_fine_tuned_augmented_history.json"

model.save(str(model_dir.joinpath(model_file)))
with open(model_dir.joinpath(history_file), mode='w') as f:
    json.dump(training_run.history, f)
