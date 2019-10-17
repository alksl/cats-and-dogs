import os, shutil
from pathlib import Path

def copy_file(dataset, animal, index):
    filename = "{0}.{1}.jpg".format(animal, index)
    shutil.copyfile(
        data_dir.joinpath("train", filename),
        dataset.joinpath(animal, filename),
    )

data_dir = Path("./data")
subset_dir = Path("./subset")

train_dir = subset_dir.joinpath("train")
validation_dir = subset_dir.joinpath("validation")
test_dir = subset_dir.joinpath("test")

for directory in [train_dir, validation_dir, test_dir]:
    directory.mkdir(exist_ok=True)
    directory.joinpath("cat").mkdir(exist_ok=True)
    directory.joinpath("dog").mkdir(exist_ok=True)


for i in range(1000):
    copy_file(train_dir, "cat", i)
    copy_file(train_dir, "dog", i)

for i in range(1000, 1500):
    copy_file(validation_dir, "cat", i)
    copy_file(validation_dir, "dog", i)

for i in range(1500, 2000):
    copy_file(test_dir, "cat", i)
    copy_file(test_dir, "dog", i)
