from argparse import ArgumentParser
from keras.preprocessing.image import ImageDataGenerator
from models import models
from pathlib import Path
import json

parser = ArgumentParser(prog='train.py')
parser.add_argument('--augment', type=bool, default=False, required=False)
parser.add_argument('model', type=str, choices=models.keys())
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

model = models[args.model]
training_run = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
)

model_file = 'models/cats_and_dogs_{0}{1}.h5'.format(
    model.name,
    augment_partial,
)
history_file = "models/cats_and_dogs_{0}{1}_history.json".format(
    model.name,
    augment_partial,
)
model.save(model_file)
with open(history_file, mode='w') as f:
    json.dump(training_run.history, f)

