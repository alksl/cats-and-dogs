import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser(prog='display_acc.py')
parser.add_argument("file", type=str)
args = parser.parse_args()

with open(args.file, mode='r') as f:
    history = json.load(f)

acc = history['acc']
val_acc = history['val_acc']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
