import numpy as np
import json
from pathlib import Path
from keras import models
from keras import layers
from keras import optimizers

flattened_dim = 4 * 4 * 512

train_features = np.load(Path("./subset/train_vgg16_features.npy"))
train_labels = np.load(Path("./subset/train_labels.npy"))

validation_features = np.load(Path("./subset/validation_vgg16_features.npy"))
validation_labels = np.load(Path("./subset/validation_labels.npy"))

test_features = np.load(Path("./subset/test_vgg16_features.npy"))
test_labels = np.load(Path("./subset/test_labels.npy"))

train_features = np.reshape(train_features, (2000, flattened_dim))
validation_features = np.reshape(validation_features, (1000, flattened_dim))
test_features = np.reshape(test_features, (1000, flattened_dim))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=flattened_dim))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer=optimizers.RMSprop(lr=2e-5),
    loss='binary_crossentropy',
    metrics=['acc'],
)

training_run = model.fit(
    train_features,
    train_labels,
    epochs=30,
    batch_size=20,
    validation_data=(validation_features, validation_labels),
)

model_file = 'models/vgg16_features.h5'
history_file = "models/vgg16_features_history.json"
model.save(model_file)
with open(history_file, mode='w') as f:
    json.dump(training_run.history, f)
