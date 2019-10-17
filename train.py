from model import model
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import json

train_datagen = ImageDataGenerator(rescale=1./255)
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

training_run = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
)

model.save('models/cats_and_dogs_small_1.h5')
with open('models/cats_and_dogs_small_1_history.json', mode='w') as f:
    json.dump(training_run.history, f)

