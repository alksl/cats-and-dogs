import numpy as np
import json
from argparse import ArgumentParser
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from pathlib import Path

parser = ArgumentParser(prog='train.py')
parser.add_argument('--augment', type=bool, default=False, required=False)
parser.add_argument('--fine-tune', type=bool, default=False, required=False)
args = parser.parse_args()
print("Arguments: ", args)

if args.augment:
    augment_partial = '_augmented'
    augmentation_args = {
        'rescale': 1./255,
        'rotation_range': 40,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
    }
else:
    augment_partial = ''
    augmentation_args = {'rescale': 1./255}


train_datagen = ImageDataGenerator(**augmentation_args)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_dir = Path("./subset/train")
validation_dir= Path("./subset/validation")

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
)

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

model.compile(
    optimizer=optimizers.RMSprop(lr=2e-5),
    loss='binary_crossentropy',
    metrics=['acc'],
)

conv_base.summary()
model.summary()

training_run = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
)

model_file = 'models/vgg16_fine_tuned_{0}{1}.h5'.format(
    model.name,
    augment_partial,
)
history_file = "models/vgg16_fine_tuned_{0}{1}_history.json".format(
    model.name,
    augment_partial,
)
model.save(model_file)
with open(history_file, mode='w') as f:
    json.dump(training_run.history, f)
