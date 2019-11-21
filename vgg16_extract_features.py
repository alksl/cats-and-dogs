import numpy as np
import json
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

BATCH_SIZE=20

def extract_features(conv_base, generator, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        batch_start = i * BATCH_SIZE
        batch_end = (i + 1) * BATCH_SIZE
        features[batch_start:batch_end] = features_batch
        labels[batch_start:batch_end] = labels_batch
        i += 1
        if i * BATCH_SIZE >= sample_count:
            break
    return features, labels

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = Path("./subset/train")
validation_dir= Path("./subset/validation")
test_dir = Path("./subset/test")

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=BATCH_SIZE,
    class_mode='binary',
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=BATCH_SIZE,
    class_mode='binary',
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=BATCH_SIZE,
    class_mode='binary',
)

print("Extracting train features")
train_features, train_labels = extract_features(conv_base, train_generator, BATCH_SIZE * 100)

print("Extracting validation features")
validation_features, validation_labels = extract_features(conv_base, validation_generator, BATCH_SIZE * 50)

print("Extracting test features")
test_features, test_labels = extract_features(conv_base, test_generator, BATCH_SIZE * 50)

np.save(Path("./subset/train_vgg16_features"), train_features)
np.save(Path("./subset/train_labels"), train_labels)

np.save(Path("./subset/validation_vgg16_features"), validation_features)
np.save(Path("./subset/validation_labels"), validation_labels)

np.save(Path("./subset/test_vgg16_features"), test_features)
np.save(Path("./subset/test_labels"), test_labels)
