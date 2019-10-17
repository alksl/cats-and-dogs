from keras import layers, models, optimizers

small_1 = models.Sequential(name='small_1')
small_1.add(
    layers.Conv2D(32, (3, 3), activation='relu',
    input_shape=(150, 150, 3)),
)
small_1.add(layers.MaxPooling2D((2, 2)))
small_1.add(layers.Conv2D(64, (3, 3), activation='relu'))
small_1.add(layers.MaxPooling2D((2, 2)))
small_1.add(layers.Conv2D(128, (3, 3), activation='relu'))
small_1.add(layers.MaxPooling2D((2, 2)))
small_1.add(layers.Conv2D(128, (3, 3), activation='relu'))
small_1.add(layers.MaxPooling2D((2, 2)))
small_1.add(layers.Flatten())
small_1.add(layers.Dense(512, activation='relu'))
small_1.add(layers.Dense(1, activation='sigmoid'))

small_1.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc'],
)

models = {
    'small_1': small_1,
}
